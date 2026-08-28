"""Top-level simulation runner: assembles Hamiltonians, noise, and solvers."""

import math

import numpy as np
import qutip

from tiqs.chain.lamb_dicke import lamb_dicke_parameters
from tiqs.chain.normal_modes import normal_modes
from tiqs.constants import TWO_PI
from tiqs.cooling.sympathetic import (
    apply_sympathetic_cooling,
    coolant_participation,
    sympathetic_cooling_rate,
    sympathetic_doppler_nbar,
)
from tiqs.gates.molmer_sorensen import ms_gate_duration, ms_gate_hamiltonian
from tiqs.hilbert_space.builder import HilbertSpace
from tiqs.hilbert_space.operators import OperatorFactory
from tiqs.hilbert_space.states import StateFactory
from tiqs.interaction.hamiltonian import carrier_hamiltonian
from tiqs.noise.laser_noise import laser_phase_noise_op
from tiqs.noise.motional import motional_dephasing_op, motional_heating_ops
from tiqs.noise.photon_scattering import raman_scattering_ops
from tiqs.noise.qubit import qubit_dephasing_op, spontaneous_emission_op
from tiqs.potential import mode_hamiltonian
from tiqs.simulation.config import SimulationConfig
from tiqs.species.ion import IonSpecies
from tiqs.species.protocol import Species


class SimulationRunner:
    """Orchestrates a full trapped-ion simulation from configuration.

    Computes normal modes, Lamb-Dicke parameters, builds the Hilbert space,
    constructs operators, and provides methods to run standard operations
    (carrier pulses, MS gates) with the configured noise model.

    Attributes
    ----------
    modes : NormalModeResult
        Normal modes of the chain in every available direction.
    hs : HilbertSpace
        Composite Hilbert space ``[qubit_0, ..., mode_0, ...]``.
    ops : OperatorFactory
        Operator factory for ``hs``.
    sf : StateFactory
        State factory for ``hs``.
    eta : np.ndarray
        Lamb-Dicke parameters of the **axial** modes, shape
        ``(n_ions, n_ions)`` -- one column per axial normal mode, not
        per simulated mode, so ``eta`` spans all ``n_ions`` axial modes
        even when ``config.n_modes`` is smaller. Signs are the raw
        eigenvector signs from ``normal_modes``; only relative signs
        within a column are physical.
    """

    def __init__(self, config: SimulationConfig):
        """Initialize the runner from a simulation configuration.

        Computes normal modes, Lamb-Dicke parameters, builds the
        Hilbert space and operator factories, and pre-builds the
        list of collapse operators specified by the configuration.

        Parameters
        ----------
        config : SimulationConfig
            Complete description of the physical system, gate
            parameters, and noise model.
        """
        self.config = config

        species_list = (
            config.species
            if isinstance(config.species, list)
            else [config.species] * config.n_ions
        )

        masses = np.array([s.mass_kg for s in species_list])
        self.modes = normal_modes(config.n_ions, config.trap, masses)

        self.hs = HilbertSpace(
            n_ions=config.n_ions,
            n_modes=config.n_modes,
            n_fock=config.n_fock,
        )
        self.ops = OperatorFactory(self.hs)
        self.sf = StateFactory(self.hs)

        self.eta = lamb_dicke_parameters(
            self.modes, species_list, self._k_effs(species_list), "axial"
        )

        if config.coolant_indices is not None:
            axial = self.modes.modes["axial"]
            p = coolant_participation(axial, config.coolant_indices)[
                : config.n_modes
            ]
            f = axial.freqs[: config.n_modes]
            coolant_species = self._coolant_species(species_list)
            self._cooling_rates = sympathetic_cooling_rate(coolant_species, p)
            self._n_bar_cooled = sympathetic_doppler_nbar(
                coolant_species, f, p
            )
        else:
            self._cooling_rates = None
            self._n_bar_cooled = None

        self._c_ops = self._build_collapse_operators()
        self._anharmonic_H = self._build_anharmonic_correction()

    def _k_effs(self, species_list: list[Species]) -> list[float]:
        """Per-ion effective laser wavevectors in rad/m.

        Returns ``config.k_eff`` when supplied (broadcast to every
        ion if scalar), otherwise derives the wavevector from each
        species' laser transition.

        Parameters
        ----------
        species_list : list[Species]
            Per-ion species, length ``n_ions``.

        Returns
        -------
        list[float]
            Effective wavevector for each ion in rad/m.

        Raises
        ------
        ValueError
            If ``config.k_eff`` is a list whose length differs from
            the number of ions.
        """
        k_eff = self.config.k_eff
        if k_eff is None:
            return [self._species_k_eff(s) for s in species_list]
        if isinstance(k_eff, list):
            if len(k_eff) != len(species_list):
                raise ValueError(
                    f"k_eff list length {len(k_eff)} != n_ions "
                    f"{len(species_list)}"
                )
            return [float(k) for k in k_eff]
        return [float(k_eff)] * len(species_list)

    @staticmethod
    def _species_k_eff(species: Species) -> float:
        r"""Effective wavevector from a species' laser properties.

        Single-beam optical qubits use $k = 2\pi/\lambda$;
        counter-propagating Raman beams use $k_\text{eff} = 2k$.

        Parameters
        ----------
        species : Species
            The trapped particle species.

        Returns
        -------
        float
            Effective wavevector in rad/m.

        Raises
        ------
        ValueError
            If the species has no laser transition to derive a
            wavevector from. Spin-motion coupling is then not
            optical -- a magnetic-field gradient gives
            $k_\text{eff} = g\,\mu_B\,(\partial B/\partial z)
            /(\hbar\,\omega_m)$ (Mintert and Wunderlich,
            *Phys. Rev. Lett.* **87**, 257904 (2001)), which depends
            on the mode frequency -- so the caller must supply
            ``SimulationConfig.k_eff`` rather than have one guessed.
        """
        if isinstance(species, IonSpecies):
            if species.qubit_wavelength is not None:
                return TWO_PI / species.qubit_wavelength
            if species.raman_wavelength is not None:
                return 2 * TWO_PI / species.raman_wavelength
        name = getattr(species, "symbol", type(species).__name__)
        raise ValueError(
            f"Cannot derive an effective wavevector for {name}: no "
            f"optical-qubit or Raman wavelength is defined. Set "
            f"SimulationConfig.k_eff explicitly. For gradient-coupled "
            f"species (e.g. electrons) use k_eff = "
            f"g*mu_B*(dB/dz)/(hbar*omega_mode)."
        )

    def _coolant_species(self, species_list: list[Species]) -> Species:
        """The single species shared by every coolant ion.

        The sympathetic cooling rate factors one recoil frequency
        (hence one wavevector and one mass) out of the sum over
        coolant ions, which is only valid when the coolant ions are
        all the same species.

        Parameters
        ----------
        species_list : list[Species]
            Per-ion species, length ``n_ions``.

        Returns
        -------
        Species
            The common coolant species.

        Raises
        ------
        ValueError
            If the coolant ions are not all the same species.
        """
        indices = self.config.coolant_indices
        first = species_list[indices[0]]
        for idx in indices[1:]:
            if species_list[idx] != first:
                raise ValueError(
                    f"coolant ions must all be the same species: ion "
                    f"{indices[0]} is "
                    f"{getattr(first, 'symbol', type(first).__name__)} "
                    f"but ion {idx} is "
                    f"{getattr(species_list[idx], 'symbol', 'unknown')}. "
                    f"The cooling rate is derived from a single recoil "
                    f"frequency."
                )
        return first

    def _build_anharmonic_correction(self) -> qutip.Qobj | None:
        """Build the anharmonic Hamiltonian correction.

        For each mode with a configured potential, computes
        ``H_correction = H_potential - omega_m * n``, the difference
        between the full potential Hamiltonian and the harmonic part
        already accounted for in the interaction picture. ``omega_m``
        is always the physical normal-mode frequency, so a potential
        whose own ``omega`` differs from the mode frequency
        contributes a residual linear term.

        For ``DuffingPotential`` this correction commutes with
        the free Hamiltonian and is valid in the interaction
        picture. For ``ArbitraryPotential`` the simulation should
        use the Schrodinger picture.

        Potentials are assumed to apply to axial modes (mode indices
        correspond to positions in the axial frequency array, which
        ``SimulationConfig`` guarantees to be in range).

        Returns
        -------
        qutip.Qobj or None
            Summed anharmonic correction Hamiltonian, or ``None``
            if no potentials are configured.
        """
        if not self.config.potentials:
            return None
        axial_freqs = self.modes.modes["axial"].freqs
        H_correction = qutip.qzero(self.hs.dims)
        for mode_idx, potential in self.config.potentials.items():
            H_full = mode_hamiltonian(potential, self.ops, mode_idx)
            H_harmonic = axial_freqs[mode_idx] * self.ops.number(mode_idx)
            H_correction = H_correction + (H_full - H_harmonic)
        return H_correction

    def _build_collapse_operators(self) -> list[qutip.Qobj]:
        """Assemble collapse operators from the noise configuration.

        Includes motional heating and motional dephasing (per mode)
        and qubit dephasing, spontaneous emission, Raman photon
        scattering and laser phase noise (per ion), for whichever
        configuration fields are set. ``t1_qubit`` is forwarded to
        ``qubit_dephasing_op`` so the two qubit channels together
        decay coherences at exactly $1/T_2$.

        Model scope: heating and motional dephasing are applied
        independently and with the same rate to every simulated
        mode. Spatially uniform field noise instead heats only the
        centre-of-mass mode, and $N_I$ times faster (Brownnutt et
        al., *Rev. Mod. Phys.* **87**, 1419 (2015), Eqs. 22-23), so
        set ``heating_rates`` per mode when the noise geometry
        matters. Qubit dephasing is likewise one independent channel
        per ion, not the collective channel that common-mode
        magnetic-field noise produces. Addressing crosstalk and
        laser intensity noise are Hamiltonian terms rather than
        collapse operators and are not assembled here; build them
        with ``crosstalk_hamiltonian`` / ``laser_intensity_noise_op``
        and add them to the Hamiltonian passed to the solver.

        Returns
        -------
        list[qutip.Qobj]
            Collapse operators for the Lindblad master equation.
        """
        c_ops = []
        cfg = self.config

        if cfg.heating_rates is not None:
            h_rates = cfg.heating_rates
        elif cfg.heating_rate is not None and cfg.heating_rate > 0:
            h_rates = [cfg.heating_rate] * cfg.n_modes
        else:
            h_rates = []
        for m, r in enumerate(h_rates):
            if r > 0:
                c_ops.extend(motional_heating_ops(self.ops, m, r))

        if cfg.motional_dephasing_rates is not None:
            for m, r in enumerate(cfg.motional_dephasing_rates):
                if r > 0:
                    c_ops.append(motional_dephasing_op(self.ops, m, r))

        # t1 = inf leaves gamma_phi = 1/T2; a finite t1_qubit is also
        # added as spontaneous emission below, which decays coherences
        # at 1/(2*T1), so it must be subtracted here to honour T2.
        t1 = cfg.t1_qubit if cfg.t1_qubit is not None else math.inf
        scattering_rate = (
            cfg.photon_scattering_rate
            if cfg.photon_scattering_rate is not None
            else 0.0
        )
        linewidth = (
            cfg.laser_phase_linewidth
            if cfg.laser_phase_linewidth is not None
            else 0.0
        )
        for i in range(cfg.n_ions):
            if cfg.t2_qubit is not None:
                c_ops.append(
                    qubit_dephasing_op(self.ops, i, cfg.t2_qubit, t1=t1)
                )
            # Only include spontaneous emission when explicitly requested
            # via t1_qubit. The species default T1 (e.g., 1.168 s for Ca40
            # optical qubit) is not automatically included for sesolve runs.
            if t1 < math.inf:
                c_ops.append(spontaneous_emission_op(self.ops, i, t1))
            if scattering_rate > 0:
                c_ops.extend(
                    raman_scattering_ops(self.ops, i, scattering_rate)
                )
            if linewidth > 0:
                c_ops.append(laser_phase_noise_op(self.ops, i, linewidth))

        return c_ops

    def _initial_state(self) -> qutip.Qobj:
        """Build the default initial state.

        Returns a thermal motional state (density matrix) when
        ``n_bar_initial > 0`` or collapse operators are present,
        otherwise returns a pure ground state (ket).

        Returns
        -------
        qutip.Qobj
            Initial state for the simulation.
        """
        if self.config.n_bar_initial_per_mode is not None:
            n_bars = self.config.n_bar_initial_per_mode
        else:
            n_bars = [self.config.n_bar_initial] * self.config.n_modes
        if any(nb > 0 for nb in n_bars) or self._c_ops:
            return self.sf.thermal_state(n_bar=n_bars)
        return self.sf.ground_state()

    def _solve(self, H, tlist, psi0=None):
        """Dispatch to the appropriate QuTiP solver.

        Selects ``sesolve``, ``mesolve``, or ``mcsolve`` based on
        the configured solver name and whether collapse operators
        are present: ``"sesolve"`` is upgraded to ``mesolve``
        whenever the noise model produced collapse operators.
        ``mcsolve`` runs ``config.ntraj`` trajectories.

        Parameters
        ----------
        H : qutip.Qobj or list
            System Hamiltonian (static or time-dependent).
        tlist : array_like
            Times at which to evaluate the state.
        psi0 : qutip.Qobj or None, optional
            Initial state. If ``None``, built automatically via
            ``_initial_state``.

        Returns
        -------
        qutip.Result
            Solver result containing the time-evolved state.
        """
        if psi0 is None:
            psi0 = self._initial_state()

        opts = dict(self.config.solver_options)
        if opts.get("max_step", 0) <= 0 and len(tlist) > 1:
            opts["max_step"] = (tlist[-1] - tlist[0]) / (len(tlist) * 2)

        if self._anharmonic_H is not None:
            if isinstance(H, list):
                H = [self._anharmonic_H, *H]
            else:
                H = H + self._anharmonic_H

        solver = self.config.solver
        if solver == "sesolve" and not self._c_ops:
            return qutip.sesolve(H, psi0, tlist, options=opts)
        elif solver == "mcsolve":
            return qutip.mcsolve(
                H,
                psi0,
                tlist,
                c_ops=self._c_ops,
                ntraj=self.config.ntraj,
                options=opts,
            )
        else:
            return qutip.mesolve(
                H, psi0, tlist, c_ops=self._c_ops, options=opts
            )

    def _check_ion(self, ion: int):
        """Validate a single ion index.

        Parameters
        ----------
        ion : int
            Index of the target ion.

        Raises
        ------
        ValueError
            If *ion* is outside ``range(n_ions)``.
        """
        if ion < 0 or ion >= self.config.n_ions:
            raise ValueError(
                f"ion index {ion} out of range [0, {self.config.n_ions})"
            )

    def run_carrier_pulse(
        self,
        ion: int,
        theta: float,
        rabi_frequency: float = TWO_PI * 100e3,
        duration: float | None = None,
        n_steps: int = 200,
    ) -> qutip.Result:
        """Run a carrier rotation (single-qubit gate) on the specified ion.

        Parameters
        ----------
        ion : int
            Index of the target ion.
        theta : float
            Rotation angle in radians. Used to derive the default duration
            as ``abs(theta) / rabi_frequency``.
        rabi_frequency : float, optional
            Carrier Rabi frequency in rad/s (default 2*pi * 100 kHz).
            Must be positive; reverse the rotation sense with the
            sign of *theta* or with the pulse phase instead.
        duration : float or None, optional
            Pulse duration in seconds. If ``None``, computed from *theta*.
        n_steps : int, optional
            Number of time steps for the solver (default 200).

        Returns
        -------
        qutip.Result
            Solver result containing the time-evolved state.

        Raises
        ------
        ValueError
            If *ion* is out of range, ``rabi_frequency <= 0``, or
            *duration* is negative.
        """
        self._check_ion(ion)
        if rabi_frequency <= 0:
            raise ValueError(
                f"rabi_frequency must be > 0, got {rabi_frequency}"
            )
        if duration is None:
            duration = abs(theta) / rabi_frequency
        elif duration < 0:
            raise ValueError(f"duration must be >= 0, got {duration}")
        H = carrier_hamiltonian(self.ops, ion, rabi_frequency, phase=0.0)
        tlist = np.linspace(0, duration, n_steps)
        return self._solve(H, tlist)

    def run_ms_gate(
        self,
        ions: list[int],
        mode: int = 0,
        detuning: float | None = None,
        loops: int = 1,
        n_steps: int = 500,
    ) -> qutip.Result:
        r"""Run a Molmer-Sorensen entangling gate.

        Uses the time-dependent Hamiltonian from
        ``ms_gate_hamiltonian`` with Rabi frequency calibrated so
        that the geometric phase accumulates to $\pi/4$ over the
        gate duration, producing a maximally entangling gate.

        For the MS gate the geometric phase is

        $$
        \phi = 4\pi K\,\operatorname{sign}(\delta)\,
        \frac{\eta_i\,\eta_j\,\Omega^2}{\delta^2}
        $$

        where $K$ is the number of loops. Only $\Omega^2$ enters, so
        maximal entanglement $|\phi| = \pi/4$ fixes
        $|\Omega| = |\delta| / (4\sqrt{K}\sqrt{|\eta_i\eta_j|})$ --
        the *magnitude* of $\eta_i\eta_j$ sets the required drive
        strength, while its sign and the sign of $\delta$ set the
        sign of the geometric phase. The output Bell state is
        $(|00\rangle + s\,i\,|11\rangle)/\sqrt{2}$ with
        $s = \operatorname{sign}(\delta\,\eta_i\,\eta_j)$, so
        odd-parity modes (opposite participation signs) produce the
        conjugate Bell state. Pass that sign to
        ``bell_state_fidelity``.

        Model scope: the Hamiltonian is the idealized single-mode
        spin-dependent force. The zeroth-order carrier term of the
        bichromatic drive, the counter-rotating sideband halves, the
        spectator normal modes, the Debye-Waller and AC-Stark
        corrections and pulse shaping are all absent, and the pulse
        is square. Reported fidelities are therefore upper bounds;
        the omitted carrier and spectator-mode errors grow with
        $\delta$ and reach $10^{-4}$-$10^{-2}$ for fast gates
        (Sorensen and Molmer, *Phys. Rev. A* **62**, 022311 (2000),
        Sec. IV).

        Parameters
        ----------
        ions : list[int]
            Indices of the two distinct ions to entangle.
        mode : int, optional
            Motional mode index (default 0, the COM mode). Any mode
            in ``range(n_modes)`` is supported.
        detuning : float or None, optional
            Detuning from the motional sideband in rad/s. If
            ``None``, defaults to ``2*pi * 1 kHz``. Must be nonzero.
        loops : int, optional
            Number of phase-space loops (default 1, must be >= 1).
        n_steps : int, optional
            Number of time steps for the solver (default 500).

        Returns
        -------
        qutip.Result
            Solver result containing the time-evolved state.

        Raises
        ------
        ValueError
            If ``ions`` does not contain exactly two distinct
            in-range indices, if *mode* is outside
            ``range(n_modes)``, if ``loops < 1``, if *detuning* is
            zero, or if either ion has no participation in *mode*.
        """
        if len(ions) != 2:
            raise ValueError(
                f"run_ms_gate Rabi calibration is valid for exactly "
                f"2 ions, got {len(ions)}. For N > 2 ions, construct "
                f"the Hamiltonian manually with ms_gate_hamiltonian."
            )
        if len(set(ions)) != 2:
            raise ValueError(
                f"run_ms_gate needs two distinct ions, got {ions}. "
                f"Driving one ion twice squeezes it instead of "
                f"entangling a pair."
            )
        for ion in ions:
            self._check_ion(ion)
        if mode < 0 or mode >= self.config.n_modes:
            raise ValueError(
                f"mode {mode} out of range [0, {self.config.n_modes})"
            )
        if loops < 1:
            raise ValueError(f"loops must be >= 1, got {loops}")

        if detuning is None:
            detuning = TWO_PI * 1e3
        if detuning == 0:
            raise ValueError(
                "detuning must be nonzero: the gate duration is "
                "2*pi*loops/|detuning|."
            )

        eta_ions = [float(self.eta[i, mode]) for i in ions]
        eta_product = eta_ions[0] * eta_ions[1]
        if abs(eta_product) < np.finfo(float).eps:
            raise ValueError(
                f"ions {ions} cannot be entangled through mode {mode}: "
                f"Lamb-Dicke parameters {eta_ions} give a vanishing "
                f"product, so at least one ion has no participation in "
                f"that mode."
            )
        eta_geom = np.sqrt(abs(eta_product))

        Omega = detuning / (4 * eta_geom * np.sqrt(loops))
        tau = ms_gate_duration(detuning, loops)

        H = ms_gate_hamiltonian(
            self.ops,
            ions=ions,
            mode=mode,
            eta=eta_ions,
            rabi_frequency=Omega,
            detuning=detuning,
        )
        tlist = np.linspace(0, tau, n_steps)
        return self._solve(H, tlist)

    def run_sympathetic_cooling(
        self,
        rho: qutip.Qobj,
        duration: float,
        cooling_rates: np.ndarray | None = None,
        n_bar_target: np.ndarray | None = None,
    ) -> qutip.Qobj:
        """Apply sympathetic cooling to a density matrix.

        Uses the cooling rates and target phonon numbers computed
        from ``config.coolant_indices``, or accepts explicit
        overrides.

        Parameters
        ----------
        rho : qutip.Qobj
            Input density matrix.
        duration : float
            Cooling duration in seconds.
        cooling_rates : np.ndarray or None
            Per-mode cooling rates in 1/s. If ``None``, uses
            rates from ``coolant_indices``.
        n_bar_target : np.ndarray or None
            Per-mode target phonon numbers. If ``None``, uses
            the sympathetic Doppler limit.

        Returns
        -------
        qutip.Qobj
            Density matrix after cooling.

        Raises
        ------
        ValueError
            If no ``coolant_indices`` configured and no explicit
            rates provided.
        """
        rates = (
            cooling_rates if cooling_rates is not None else self._cooling_rates
        )
        targets = (
            n_bar_target if n_bar_target is not None else self._n_bar_cooled
        )
        if rates is None or targets is None:
            raise ValueError(
                "No coolant_indices configured and no explicit "
                "rates provided. Set coolant_indices in "
                "SimulationConfig or pass cooling_rates and "
                "n_bar_target."
            )
        return apply_sympathetic_cooling(
            rho, self.ops, rates, targets, duration
        )

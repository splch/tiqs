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
from tiqs.noise.motional import motional_heating_ops
from tiqs.noise.qubit import qubit_dephasing_op, spontaneous_emission_op
from tiqs.potential import mode_hamiltonian
from tiqs.simulation.config import SimulationConfig
from tiqs.species.ion import IonSpecies


class SimulationRunner:
    """Orchestrates a full trapped-ion simulation from configuration.

    Computes normal modes, Lamb-Dicke parameters, builds the Hilbert space,
    constructs operators, and provides methods to run standard operations
    (carrier pulses, MS gates) with the configured noise model.
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

        # Derive effective wavevector from species laser properties.
        # 400 nm fallback is a typical UV wavelength for species without
        # a defined laser transition (e.g. electrons with gradient coupling).
        k_effs = [self._species_k_eff(s) for s in species_list]
        self.eta = lamb_dicke_parameters(
            self.modes, species_list, k_effs, "axial"
        )

        if config.coolant_indices is not None:
            axial = self.modes.modes["axial"]
            self._coolant_participation = coolant_participation(
                axial, config.coolant_indices
            )
            coolant_species = species_list[config.coolant_indices[0]]
            n_m = min(config.n_modes, len(axial.freqs))
            self._cooling_rates = sympathetic_cooling_rate(
                coolant_species,
                self._coolant_participation[:n_m],
            )
            self._n_bar_cooled = sympathetic_doppler_nbar(
                coolant_species,
                axial.freqs[:n_m],
                self._coolant_participation[:n_m],
            )
        else:
            self._coolant_participation = None
            self._cooling_rates = None
            self._n_bar_cooled = None

        self._c_ops = self._build_collapse_operators()
        self._anharmonic_H = self._build_anharmonic_correction()

    @staticmethod
    def _species_k_eff(species) -> float:
        """Effective wavevector from a species' laser properties."""
        if isinstance(species, IonSpecies):
            if species.qubit_wavelength is not None:
                return TWO_PI / species.qubit_wavelength
            elif species.raman_wavelength is not None:
                return 2 * TWO_PI / species.raman_wavelength
        return TWO_PI / 400e-9

    def _build_anharmonic_correction(self) -> qutip.Qobj | None:
        """Build the anharmonic Hamiltonian correction.

        For each mode with a configured potential, computes
        ``H_correction = H_potential - omega * n``, the difference
        between the full potential Hamiltonian and the harmonic part
        already accounted for in the interaction picture.

        For ``DuffingPotential`` this correction commutes with
        the free Hamiltonian and is valid in the interaction
        picture. For ``ArbitraryPotential`` the simulation should
        use the Schrodinger picture.

        Potentials are assumed to apply to axial modes (mode indices
        correspond to positions in the axial frequency array).

        Returns
        -------
        qutip.Qobj or None
            Summed anharmonic correction Hamiltonian, or ``None``
            if no potentials are configured.
        """
        if not self.config.potentials:
            return None
        axial_freqs = self.modes.modes["axial"].freqs
        H_correction = 0 * self.ops.identity()
        for mode_idx, potential in self.config.potentials.items():
            H_full = mode_hamiltonian(potential, self.ops, mode_idx)
            omega = (
                axial_freqs[mode_idx]
                if mode_idx < len(axial_freqs)
                else potential.omega
            )
            H_harmonic = omega * self.ops.number(mode_idx)
            H_correction = H_correction + (H_full - H_harmonic)
        return H_correction

    def _build_collapse_operators(self) -> list[qutip.Qobj]:
        """Assemble collapse operators from the noise configuration.

        Includes motional heating (per mode), qubit dephasing (per
        ion), and spontaneous emission (per ion) when the
        corresponding configuration fields are set.

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

        for i in range(cfg.n_ions):
            if cfg.t2_qubit is not None:
                c_ops.append(qubit_dephasing_op(self.ops, i, cfg.t2_qubit))
            # Only include spontaneous emission when explicitly requested
            # via t1_qubit. The species default T1 (e.g., 1.168 s for Ca40
            # optical qubit) is not automatically included for sesolve runs.
            if cfg.t1_qubit is not None and cfg.t1_qubit < math.inf:
                c_ops.append(
                    spontaneous_emission_op(self.ops, i, cfg.t1_qubit)
                )

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
        are present.

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
                H = [H[0] + self._anharmonic_H, *H[1:]]
            else:
                H = H + self._anharmonic_H

        solver = self.config.solver
        if solver == "sesolve" and not self._c_ops:
            return qutip.sesolve(H, psi0, tlist, options=opts)
        elif solver == "mcsolve":
            return qutip.mcsolve(
                H, psi0, tlist, c_ops=self._c_ops, ntraj=100, options=opts
            )
        else:
            return qutip.mesolve(
                H, psi0, tlist, c_ops=self._c_ops, options=opts
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
        duration : float or None, optional
            Pulse duration in seconds. If ``None``, computed from *theta*.
        n_steps : int, optional
            Number of time steps for the solver (default 200).

        Returns
        -------
        qutip.Result
            Solver result containing the time-evolved state.
        """
        H = carrier_hamiltonian(self.ops, ion, rabi_frequency, phase=0.0)
        if duration is None:
            duration = abs(theta) / rabi_frequency
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
        \phi = 4\pi K\left(\frac{\eta\,\Omega}{\delta}\right)^2
        $$

        where $K$ is the number of loops. For maximally entangling:
        $\phi = \pi/4$, giving
        $\eta\,\Omega = \delta / (4\sqrt{K})$.
        Hence $\Omega = \delta / (4\,\bar{\eta}\,\sqrt{K})$.

        Parameters
        ----------
        ions : list[int]
            Indices of the two ions to entangle.
        mode : int, optional
            Motional mode index (default 0, the COM mode).
        detuning : float or None, optional
            Detuning from the motional sideband in rad/s. If
            ``None``, defaults to ``2*pi * 1 kHz``.
        loops : int, optional
            Number of phase-space loops (default 1).
        n_steps : int, optional
            Number of time steps for the solver (default 500).

        Returns
        -------
        qutip.Result
            Solver result containing the time-evolved state.

        Raises
        ------
        ValueError
            If ``ions`` does not contain exactly two indices.
        """
        if len(ions) != 2:
            raise ValueError(
                f"run_ms_gate Rabi calibration is valid for exactly "
                f"2 ions, got {len(ions)}. For N > 2 ions, construct "
                f"the Hamiltonian manually with ms_gate_hamiltonian."
            )
        eta_ions = [float(self.eta[i, mode]) for i in ions]
        eta_avg = np.mean(eta_ions)

        if detuning is None:
            detuning = TWO_PI * 1e3

        Omega = detuning / (4 * eta_avg * np.sqrt(loops))
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
            rho, self.ops, self.config.n_modes, rates, targets, duration
        )

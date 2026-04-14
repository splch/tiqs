"""Top-level simulation runner: assembles Hamiltonians, noise, and solvers."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from tiqs.pulses import Pulse
import qutip

from tiqs.chain.lamb_dicke import lamb_dicke_parameters
from tiqs.chain.normal_modes import normal_modes
from tiqs.constants import TWO_PI
from tiqs.gates.calibration import (
    calibrate_multi_tone_ms,
    calibrate_single_tone_ms,
)
from tiqs.gates.molmer_sorensen import (
    ms_gate_duration,
    ms_gate_hamiltonian,
    ms_multimode_hamiltonian,
)
from tiqs.hilbert_space.builder import HilbertSpace
from tiqs.hilbert_space.operators import OperatorFactory
from tiqs.hilbert_space.states import StateFactory
from tiqs.interaction.hamiltonian import carrier_hamiltonian
from tiqs.noise.motional import motional_heating_ops
from tiqs.noise.qubit import qubit_dephasing_op, spontaneous_emission_op
from tiqs.simulation.config import SimulationConfig


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

        self.modes = normal_modes(config.n_ions, config.trap)

        self.hs = HilbertSpace(
            n_ions=config.n_ions,
            n_modes=config.n_modes,
            n_fock=config.n_fock,
        )
        self.ops = OperatorFactory(self.hs)
        self.sf = StateFactory(self.hs)

        if config.gradient is not None:
            k_eff = config.gradient.effective_k(config.species)
        elif config.species.qubit_wavelength is not None:
            k_eff = TWO_PI / config.species.qubit_wavelength
        elif config.species.raman_wavelength is not None:
            k_eff = 2 * TWO_PI / config.species.raman_wavelength
        else:
            k_eff = TWO_PI / 400e-9
        self.eta = lamb_dicke_parameters(
            self.modes, config.species, k_eff, "axial"
        )

        self._c_ops = self._build_collapse_operators()

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

        if cfg.heating_rate is not None and cfg.heating_rate > 0:
            for m in range(cfg.n_modes):
                c_ops.extend(
                    motional_heating_ops(self.ops, m, cfg.heating_rate)
                )

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
            if cfg.microwave_linewidth is not None:
                from tiqs.noise.microwave_noise import (
                    microwave_phase_noise_op,
                )

                c_ops.append(
                    microwave_phase_noise_op(
                        self.ops, i, cfg.microwave_linewidth
                    )
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
        n_bar = self.config.n_bar_initial
        if n_bar > 0 or self._c_ops:
            return self.sf.thermal_state(n_bar=[n_bar] * self.config.n_modes)
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
        mode: int | None = None,
        modes: list[int] | None = None,
        detuning: float | None = None,
        loops: int = 1,
        drive: str = "single_tone",
        n_steps: int = 500,
    ) -> qutip.Result:
        r"""Run a Molmer-Sorensen entangling gate.

        When ``mode`` is given (or neither ``mode`` nor ``modes``),
        the original single-mode Hamiltonian is used.  When
        ``modes`` is given, the multi-mode Hamiltonian sums over
        all listed modes with calibrated parameters.

        Parameters
        ----------
        ions : list[int]
            Indices of the two ions to entangle.
        mode : int or None, optional
            Single motional mode index.  Mutually exclusive with
            *modes*.
        modes : list[int] or None, optional
            Motional mode indices for multi-mode gate.  Mutually
            exclusive with *mode*.
        detuning : float or None, optional
            Detuning from the motional sideband in rad/s.
        loops : int, optional
            Number of phase-space loops (default 1).
        drive : str, optional
            ``"single_tone"`` or ``"multi_tone"`` (only used in
            multi-mode path).
        n_steps : int, optional
            Number of solver time steps (default 500).

        Returns
        -------
        qutip.Result
            Solver result containing the time-evolved state.

        Raises
        ------
        ValueError
            If both *mode* and *modes* are given, or if *ions*
            does not contain exactly two indices.
        """
        if mode is not None and modes is not None:
            raise ValueError(
                "Specify either 'mode' (single-mode) or 'modes' "
                "(multi-mode), not both."
            )

        if mode is not None or modes is None:
            return self._run_ms_single_mode(
                ions,
                0 if mode is None else mode,
                detuning,
                loops,
                n_steps,
            )

        return self._run_ms_multi_mode(
            ions,
            modes,
            detuning,
            loops,
            drive,
            n_steps,
        )

    def _run_ms_single_mode(self, ions, mode, detuning, loops, n_steps):
        """Single-mode MS gate (original code path)."""
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

    def _run_ms_multi_mode(
        self,
        ions,
        modes,
        detuning,
        loops,
        drive,
        n_steps,
    ):
        """Multi-mode MS gate using calibrated parameters."""
        if len(ions) != 2:
            raise ValueError(
                "Multi-mode auto-calibration requires exactly "
                f"2 ions, got {len(ions)}."
            )

        mode_freqs = self.modes.axial_freqs[modes]
        eta_slice = self.eta[np.ix_(ions, modes)]

        if drive == "single_tone":
            params = calibrate_single_tone_ms(
                eta=eta_slice,
                mode_frequencies=mode_freqs,
                target_mode=0,
                ion_pair=(0, 1),
                loops=loops,
                detuning=detuning,
            )
        elif drive == "multi_tone":
            params = calibrate_multi_tone_ms(
                eta=eta_slice,
                mode_frequencies=mode_freqs,
                ion_pair=(0, 1),
                loops=loops,
                base_detuning=detuning,
            )
        else:
            raise ValueError(f"Unknown drive scheme: {drive}")

        H = ms_multimode_hamiltonian(
            self.ops,
            ions=ions,
            modes=modes,
            eta=eta_slice,
            rabi_frequency=params["rabi_frequency"],
            detunings=params["detunings"],
        )
        tau = params["gate_time"]
        tlist = np.linspace(0, tau, n_steps)
        return self._solve(H, tlist)

    def run_smooth_ms_gate(
        self,
        ions: list[int],
        mode: int = 0,
        detuning: float | None = None,
        ramp_amplitude: float | None = None,
        loops: int = 1,
        n_steps: int = 1000,
    ) -> qutip.Result:
        r"""Run a smooth (adiabatically ramped) MS gate.

        The detuning ramps as
        $\delta(t) = \delta_0 + A\cos(2\pi t/\tau)$, starting
        far-detuned and sweeping close to the mode at $t = \tau/2$.
        This suppresses residual spin-motion entanglement without
        requiring ground-state cooling.

        Parameters
        ----------
        ions : list[int]
            Indices of the two ions to entangle.
        mode : int
            Motional mode index.
        detuning : float or None
            Mean detuning $\delta_0$ (rad/s).  Default
            ``2*pi * 1 kHz``.
        ramp_amplitude : float or None
            Sinusoidal ramp amplitude *A* (rad/s).  Default
            ``0.8 * detuning``.
        loops : int
            Phase-space loops.
        n_steps : int
            Time steps (default 1000).

        Returns
        -------
        qutip.Result
        """
        from tiqs.gates.molmer_sorensen import ms_gate_hamiltonian_pulsed
        from tiqs.pulses import smooth_ms_pulse

        if len(ions) != 2:
            raise ValueError(f"Expected 2 ions, got {len(ions)}")

        eta_ions = [float(self.eta[i, mode]) for i in ions]
        eta_avg = np.mean(eta_ions)

        if detuning is None:
            detuning = TWO_PI * 1e3
        if ramp_amplitude is None:
            ramp_amplitude = 0.8 * detuning

        Omega = detuning / (4 * eta_avg * np.sqrt(loops))
        pulse = smooth_ms_pulse(
            delta_0=detuning,
            ramp_amplitude=ramp_amplitude,
            rabi_frequency=Omega,
            loops=loops,
        )

        tlist = np.linspace(0, pulse.duration, n_steps)
        H = ms_gate_hamiltonian_pulsed(
            self.ops,
            ions=ions,
            mode=mode,
            eta=eta_ions,
            pulse=pulse,
            tlist=tlist,
        )
        return self._solve(H, tlist)

    def run_pulsed_ms_gate(
        self,
        ions: list[int],
        pulse: Pulse,
        mode: int = 0,
        n_steps: int = 1000,
    ) -> qutip.Result:
        """Run an MS gate with an arbitrary pulse profile.

        Parameters
        ----------
        ions : list[int]
        pulse : Pulse
        mode : int
        n_steps : int
        """
        from tiqs.gates.molmer_sorensen import ms_gate_hamiltonian_pulsed

        eta_ions = [float(self.eta[i, mode]) for i in ions]
        tlist = np.linspace(0, pulse.duration, n_steps)
        H = ms_gate_hamiltonian_pulsed(
            self.ops,
            ions=ions,
            mode=mode,
            eta=eta_ions,
            pulse=pulse,
            tlist=tlist,
        )
        return self._solve(H, tlist)

    def run_gradient_zz_gate(
        self,
        ions: list[int],
        mode: int = 0,
        detuning: float | None = None,
        loops: int = 1,
        n_steps: int = 500,
    ) -> qutip.Result:
        r"""Run a ZZ entangling gate via magnetic gradient coupling.

        Uses the light-shift Hamiltonian with gradient-derived
        Lamb-Dicke parameters.  This is the native gate for
        gradient-coupled qubits (no dressing required).

        Parameters
        ----------
        ions : list[int]
        mode : int
        detuning : float or None
        loops : int
        n_steps : int
        """
        from tiqs.gates.light_shift import light_shift_gate_hamiltonian

        if self.config.gradient is None:
            raise ValueError(
                "run_gradient_zz_gate requires a gradient in SimulationConfig."
            )
        if len(ions) != 2:
            raise ValueError(f"Expected 2 ions, got {len(ions)}")

        eta_ions = [float(self.eta[i, mode]) for i in ions]
        eta_avg = np.mean(np.abs(eta_ions))

        if detuning is None:
            detuning = TWO_PI * 1e3

        Omega = detuning / (4 * eta_avg * np.sqrt(loops))
        tau = ms_gate_duration(detuning, loops)

        H = light_shift_gate_hamiltonian(
            self.ops,
            ions=ions,
            mode=mode,
            eta=eta_ions,
            rabi_frequency=Omega,
            detuning=detuning,
        )
        tlist = np.linspace(0, tau, n_steps)
        return self._solve(H, tlist)

    def run_gradient_ms_gate(
        self,
        ions: list[int],
        mode: int = 0,
        detuning: float | None = None,
        dressing_rabi_frequency: float | None = None,
        loops: int = 1,
        n_steps: int = 500,
    ) -> qutip.Result:
        r"""Run an MS (XX) gate via gradient + dressing.

        Requires a strong dressing drive to rotate the native
        $\sigma_z$ force into $\sigma_x$.

        Parameters
        ----------
        ions : list[int]
        mode : int
        detuning : float or None
        dressing_rabi_frequency : float or None
        loops : int
        n_steps : int
        """
        from tiqs.gates.microwave_ms import microwave_ms_gate_hamiltonian

        if self.config.gradient is None:
            raise ValueError(
                "run_gradient_ms_gate requires a gradient in SimulationConfig."
            )
        if len(ions) != 2:
            raise ValueError(f"Expected 2 ions, got {len(ions)}")

        eta_ions = [float(self.eta[i, mode]) for i in ions]
        eta_avg = np.mean(np.abs(eta_ions))

        if detuning is None:
            detuning = TWO_PI * 1e3

        Omega = detuning / (4 * eta_avg * np.sqrt(loops))
        tau = ms_gate_duration(detuning, loops)

        if dressing_rabi_frequency is None:
            dressing_rabi_frequency = 100 * eta_avg * Omega

        H = microwave_ms_gate_hamiltonian(
            self.ops,
            ions=ions,
            mode=mode,
            eta=eta_ions,
            gate_rabi_frequency=Omega,
            detuning=detuning,
            dressing_rabi_frequency=dressing_rabi_frequency,
        )
        tlist = np.linspace(0, tau, n_steps)
        return self._solve(H, tlist)

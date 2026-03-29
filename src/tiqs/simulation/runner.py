"""Top-level simulation runner: assembles Hamiltonians, noise, and solvers."""
import numpy as np
import qutip

from tiqs.simulation.config import SimulationConfig
from tiqs.chain.normal_modes import normal_modes
from tiqs.chain.lamb_dicke import lamb_dicke_parameters
from tiqs.hilbert_space.builder import HilbertSpace
from tiqs.hilbert_space.operators import OperatorFactory
from tiqs.hilbert_space.states import StateFactory
from tiqs.interaction.hamiltonian import carrier_hamiltonian
from tiqs.gates.molmer_sorensen import ms_gate_hamiltonian, ms_gate_duration
from tiqs.noise.motional import motional_heating_ops
from tiqs.noise.qubit import qubit_dephasing_op, spontaneous_emission_op
from tiqs.constants import TWO_PI


class SimulationRunner:
    """Orchestrates a full trapped-ion simulation from configuration.

    Computes normal modes, Lamb-Dicke parameters, builds the Hilbert space,
    constructs operators, and provides methods to run standard operations
    (carrier pulses, MS gates) with the configured noise model.
    """

    def __init__(self, config: SimulationConfig):
        self.config = config

        self.modes = normal_modes(config.n_ions, config.trap)

        self.hs = HilbertSpace(
            n_ions=config.n_ions,
            n_modes=config.n_modes,
            n_fock=config.n_fock,
        )
        self.ops = OperatorFactory(self.hs)
        self.sf = StateFactory(self.hs)

        k_eff = TWO_PI / (config.species.qubit_wavelength or 400e-9)
        self.eta = lamb_dicke_parameters(self.modes, config.species, k_eff, "axial")

        self._c_ops = self._build_collapse_operators()

    def _build_collapse_operators(self) -> list[qutip.Qobj]:
        c_ops = []
        cfg = self.config

        if cfg.heating_rate is not None and cfg.heating_rate > 0:
            for m in range(cfg.n_modes):
                c_ops.extend(motional_heating_ops(self.ops, m, cfg.heating_rate))

        for i in range(cfg.n_ions):
            if cfg.t2_qubit is not None:
                c_ops.append(qubit_dephasing_op(self.ops, i, cfg.t2_qubit))
            # Only include spontaneous emission when explicitly requested
            # via t1_qubit. The species default T1 (e.g., 1.168 s for Ca40
            # optical qubit) is not automatically included for sesolve runs.
            if cfg.t1_qubit is not None and cfg.t1_qubit < float('inf'):
                c_ops.append(spontaneous_emission_op(self.ops, i, cfg.t1_qubit))

        return c_ops

    def _initial_state(self) -> qutip.Qobj:
        n_bar = self.config.n_bar_initial
        if n_bar > 0 or self._c_ops:
            return self.sf.thermal_state(n_bar=[n_bar] * self.config.n_modes)
        return self.sf.ground_state()

    def _solve(self, H, tlist, psi0=None):
        if psi0 is None:
            psi0 = self._initial_state()

        opts = dict(self.config.solver_options)
        if opts.get("max_step", 0) <= 0 and len(tlist) > 1:
            opts["max_step"] = (tlist[-1] - tlist[0]) / (len(tlist) * 2)

        solver = self.config.solver
        if solver == "sesolve" and not self._c_ops:
            return qutip.sesolve(H, psi0, tlist, options=opts)
        elif solver == "mcsolve":
            return qutip.mcsolve(H, psi0, tlist, c_ops=self._c_ops, ntraj=100, options=opts)
        else:
            return qutip.mesolve(H, psi0, tlist, c_ops=self._c_ops, options=opts)

    def run_carrier_pulse(
        self,
        ion: int,
        theta: float,
        rabi_frequency: float = TWO_PI * 100e3,
        duration: float | None = None,
        n_steps: int = 200,
    ) -> qutip.Result:
        """Run a carrier rotation (single-qubit gate) on the specified ion."""
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
        """Run a Molmer-Sorensen entangling gate.

        Uses the time-dependent Hamiltonian from ms_gate_hamiltonian with
        Rabi frequency calibrated so that the geometric phase accumulates
        to pi/4 over the gate duration, producing a maximally entangling gate.

        For the MS gate: the geometric phase is
            phi = 2*pi * (eta*Omega/delta)^2 * K
        where K is the number of loops. For maximally entangling: phi = pi/4,
        giving eta*Omega = delta / (4 * sqrt(K)).
        Hence Omega = delta / (4 * eta_avg * sqrt(K)).
        """
        eta_ions = [float(self.eta[i, mode]) for i in ions]
        eta_avg = np.mean(eta_ions)

        if detuning is None:
            detuning = TWO_PI * 1e3

        Omega = detuning / (4 * eta_avg * np.sqrt(loops))
        tau = ms_gate_duration(detuning, loops)

        H = ms_gate_hamiltonian(
            self.ops, ions=ions, mode=mode, eta=eta_ions,
            rabi_frequency=Omega, detuning=detuning,
        )
        tlist = np.linspace(0, tau, n_steps)
        return self._solve(H, tlist)

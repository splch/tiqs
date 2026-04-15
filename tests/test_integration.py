"""End-to-end integration tests validating known trapped-ion physics."""

import numpy as np
import pytest
import qutip

from tiqs.analysis.fidelity import bell_state_fidelity
from tiqs.constants import TWO_PI
from tiqs.gates.molmer_sorensen import ms_gate_duration, ms_gate_hamiltonian
from tiqs.hilbert_space.builder import HilbertSpace
from tiqs.hilbert_space.operators import OperatorFactory
from tiqs.hilbert_space.states import StateFactory
from tiqs.interaction.hamiltonian import (
    carrier_hamiltonian,
    red_sideband_hamiltonian,
)
from tiqs.noise.motional import motional_heating_ops
from tiqs.noise.qubit import qubit_dephasing_op, spontaneous_emission_op
from tiqs.simulation.config import SimulationConfig
from tiqs.simulation.runner import SimulationRunner
from tiqs.species.ion import get_species
from tiqs.trap import PaulTrap


class TestRabiOscillations:
    """Validate carrier Rabi oscillations: population should oscillate
    sinusoidally."""

    def test_rabi_frequency_correct(self):
        hs = HilbertSpace(n_ions=1, n_modes=1, n_fock=5)
        ops = OperatorFactory(hs)
        sf = StateFactory(hs)

        Omega = TWO_PI * 200e3
        H = carrier_hamiltonian(ops, ion=0, rabi_frequency=Omega)
        psi0 = sf.ground_state()

        n_periods = 3
        t_total = n_periods * TWO_PI / Omega
        tlist = np.linspace(0, t_total, 500)

        sz = ops.sigma_z(0)
        result = qutip.sesolve(H, psi0, tlist, e_ops=[sz])

        # basis(2,0) has sigma_z = +1. With H = (Omega/2)*sigma_x,
        # sigma_z oscillates as cos(Omega*t), starting at +1.
        expected = np.cos(Omega * tlist)
        np.testing.assert_allclose(result.expect[0], expected, atol=0.05)


class TestSidebandSpectroscopy:
    """Validate that red/blue sideband transitions change phonon number
    correctly."""

    def test_red_sideband_rabi_frequency_scales_with_sqrt_n(self):
        """Simulate RSB dynamics and verify Rabi frequency scales as
        sqrt(n)."""
        hs = HilbertSpace(n_ions=1, n_modes=1, n_fock=15)
        ops = OperatorFactory(hs)

        eta = 0.1
        Omega = TWO_PI * 100e3
        H = red_sideband_hamiltonian(ops, 0, 0, Omega, eta)

        # For n_initial=1: RSB Rabi freq = eta*Omega*sqrt(1)
        # For n_initial=4: RSB Rabi freq = eta*Omega*sqrt(4) = 2x faster
        # After a pi-pulse time for n=1, the n=4 case should have
        # completed 2 pi-pulses (returned to initial state)
        rsb_rabi_n1 = eta * Omega * np.sqrt(1)
        t_pi_n1 = np.pi / rsb_rabi_n1

        # Simulate n=1: after t_pi, should be fully in |1,0>
        psi0_n1 = qutip.tensor(qutip.basis(2, 0), qutip.basis(15, 1))
        target_n1 = qutip.tensor(qutip.basis(2, 1), qutip.basis(15, 0))
        result_n1 = qutip.sesolve(H, psi0_n1, [0, t_pi_n1])
        fid_n1 = abs(result_n1.states[-1].overlap(target_n1)) ** 2
        assert fid_n1 > 0.95

        # Simulate n=4: after the same t_pi_n1, should have done 2 full
        # Rabi cycles (since Rabi freq is 2x faster), ending back near |0,4>
        psi0_n4 = qutip.tensor(qutip.basis(2, 0), qutip.basis(15, 4))
        result_n4 = qutip.sesolve(H, psi0_n4, [0, t_pi_n1])
        fid_return = abs(result_n4.states[-1].overlap(psi0_n4)) ** 2
        assert fid_return > 0.85  # back near start after 2 full cycles


class TestMSGatePhysics:
    """Validate Molmer-Sorensen gate against known analytical results."""

    def test_ms_bell_state_high_fidelity(self):
        """Full MS gate simulation should produce a Bell state with
        >95% fidelity."""
        hs = HilbertSpace(n_ions=2, n_modes=1, n_fock=20)
        ops = OperatorFactory(hs)
        sf = StateFactory(hs)

        eta = 0.05
        delta = TWO_PI * 15e3
        # For maximally entangling MS gate with two identically-coupled ions:
        # geometric phase = 2*eta^2*Omega^2*2pi/(delta^2) = pi/4
        # => eta*Omega = delta/4 => Omega = delta/(4*eta)
        Omega = delta / (4 * eta)
        tau = ms_gate_duration(delta, loops=1)

        H = ms_gate_hamiltonian(ops, [0, 1], 0, [eta, eta], Omega, delta)
        psi0 = sf.ground_state()
        tlist = np.linspace(0, tau, 800)
        result = qutip.sesolve(H, psi0, tlist, options={"max_step": tau / 200})

        rho_spin = result.states[-1].ptrace([0, 1])
        fid = bell_state_fidelity(rho_spin)
        assert fid > 0.95

    def test_ms_gate_motional_disentanglement(self):
        """After a complete MS gate, the motion should return to its
        initial state."""
        hs = HilbertSpace(n_ions=2, n_modes=1, n_fock=20)
        ops = OperatorFactory(hs)
        sf = StateFactory(hs)

        eta = 0.05
        delta = TWO_PI * 15e3
        Omega = delta / (4 * eta)
        tau = ms_gate_duration(delta, loops=1)

        H = ms_gate_hamiltonian(ops, [0, 1], 0, [eta, eta], Omega, delta)
        psi0 = sf.ground_state()
        tlist = np.linspace(0, tau, 800)
        result = qutip.sesolve(H, psi0, tlist, options={"max_step": tau / 200})

        n_op = ops.number(0)
        final_n = qutip.expect(n_op, result.states[-1])
        assert final_n == pytest.approx(0.0, abs=0.2)


class TestNoiseEffects:
    """Validate that noise degrades fidelity in the expected direction."""

    def test_heating_degrades_ms_gate(self):
        """Motional heating during an MS gate should reduce Bell state
        fidelity."""
        hs = HilbertSpace(n_ions=2, n_modes=1, n_fock=20)
        ops = OperatorFactory(hs)
        sf = StateFactory(hs)

        eta = 0.05
        delta = TWO_PI * 15e3
        Omega = delta / (4 * eta)
        tau = ms_gate_duration(delta, loops=1)
        H = ms_gate_hamiltonian(ops, [0, 1], 0, [eta, eta], Omega, delta)
        psi0 = sf.ground_state()
        tlist = np.linspace(0, tau, 500)

        # Without noise
        r_clean = qutip.sesolve(
            H, psi0, tlist, options={"max_step": tau / 100}
        )
        fid_clean = bell_state_fidelity(r_clean.states[-1].ptrace([0, 1]))

        # With heating
        c_ops = motional_heating_ops(ops, 0, heating_rate=1e5)
        r_noisy = qutip.mesolve(
            H, psi0, tlist, c_ops=c_ops, options={"max_step": tau / 100}
        )
        fid_noisy = bell_state_fidelity(r_noisy.states[-1].ptrace([0, 1]))

        assert fid_noisy < fid_clean

    def test_dephasing_degrades_rabi(self):
        """Qubit dephasing should cause Rabi oscillation contrast decay."""
        hs = HilbertSpace(n_ions=1, n_modes=1, n_fock=5)
        ops = OperatorFactory(hs)
        sf = StateFactory(hs)

        Omega = TWO_PI * 100e3
        H = carrier_hamiltonian(ops, 0, Omega)
        psi0 = sf.ground_state()
        t2 = 50e-6
        c_ops = [qubit_dephasing_op(ops, 0, t2)]

        tlist = np.linspace(0, 200e-6, 500)
        sz = ops.sigma_z(0)
        result = qutip.mesolve(H, psi0, tlist, c_ops=c_ops, e_ops=[sz])

        # Oscillation amplitude should decay over time
        early_amplitude = max(result.expect[0][:50]) - min(
            result.expect[0][:50]
        )
        late_amplitude = max(result.expect[0][-50:]) - min(
            result.expect[0][-50:]
        )
        assert late_amplitude < early_amplitude

    def test_spontaneous_decay_reduces_ms_fidelity(self):
        """[Benhelm2008] Ca-40 D5/2 decay (T1=1.168 s) should slightly
        reduce MS gate Bell fidelity. The gate is fast (~50 us) vs T1,
        so the reduction should be small."""
        hs = HilbertSpace(n_ions=2, n_modes=1, n_fock=20)
        ops = OperatorFactory(hs)
        sf = StateFactory(hs)

        eta = 0.05
        delta = TWO_PI * 20e3
        Omega = delta / (4 * eta)
        tau = ms_gate_duration(delta)
        H = ms_gate_hamiltonian(ops, [0, 1], 0, [eta, eta], Omega, delta)
        tlist = np.linspace(0, tau, 500)

        r_clean = qutip.sesolve(
            H, sf.ground_state(), tlist, options={"max_step": tau / 100}
        )
        fid_clean = bell_state_fidelity(r_clean.states[-1].ptrace([0, 1]))

        c_ops = [
            spontaneous_emission_op(ops, 0, t1=1.168),
            spontaneous_emission_op(ops, 1, t1=1.168),
        ]
        r_noisy = qutip.mesolve(
            H,
            sf.ground_state(),
            tlist,
            c_ops=c_ops,
            options={"max_step": tau / 100},
        )
        fid_noisy = bell_state_fidelity(r_noisy.states[-1].ptrace([0, 1]))
        assert fid_noisy < fid_clean
        assert fid_noisy > 0.98

    def test_combined_noise_reduces_ms_fidelity_further(self):
        """[Benhelm2008] Adding heating + dephasing + spontaneous decay
        should reduce MS gate fidelity more than any single source."""
        hs = HilbertSpace(n_ions=2, n_modes=1, n_fock=20)
        ops = OperatorFactory(hs)
        sf = StateFactory(hs)

        eta = 0.05
        delta = TWO_PI * 20e3
        Omega = delta / (4 * eta)
        tau = ms_gate_duration(delta)
        H = ms_gate_hamiltonian(ops, [0, 1], 0, [eta, eta], Omega, delta)
        tlist = np.linspace(0, tau, 500)

        r_clean = qutip.sesolve(
            H, sf.ground_state(), tlist, options={"max_step": tau / 100}
        )
        fid_clean = bell_state_fidelity(r_clean.states[-1].ptrace([0, 1]))

        c_ops = [
            spontaneous_emission_op(ops, 0, t1=1.168),
            spontaneous_emission_op(ops, 1, t1=1.168),
            *motional_heating_ops(ops, 0, heating_rate=100),
            qubit_dephasing_op(ops, 0, t2=10e-3),
            qubit_dephasing_op(ops, 1, t2=10e-3),
        ]
        r_noisy = qutip.mesolve(
            H,
            sf.ground_state(),
            tlist,
            c_ops=c_ops,
            options={"max_step": tau / 100},
        )
        fid_noisy = bell_state_fidelity(r_noisy.states[-1].ptrace([0, 1]))
        assert fid_noisy < fid_clean
        assert 0.95 < fid_noisy < 1.0


class TestFullSimulationRunner:
    """Test the high-level SimulationRunner with realistic parameters."""

    def test_ca40_ms_gate(self):
        trap = PaulTrap(
            v_rf=300.0,
            omega_rf=TWO_PI * 30e6,
            r0=0.5e-3,
            omega_axial=TWO_PI * 1.0e6,
            species=get_species("Ca40"),
        )
        config = SimulationConfig(
            species=get_species("Ca40"),
            trap=trap,
            n_ions=2,
            n_modes=1,
            n_fock=15,
            solver="sesolve",
        )
        runner = SimulationRunner(config)
        result = runner.run_ms_gate(ions=[0, 1], mode=0)
        rho_spin = result.states[-1].ptrace([0, 1])
        # The runner uses its own calibrated Rabi frequency;
        # check that the reduced state has purity < 0.9 (entangled)
        purity = (rho_spin * rho_spin).tr()
        assert purity < 0.9 or bell_state_fidelity(rho_spin) > 0.80

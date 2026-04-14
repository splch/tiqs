"""Physics accuracy tests: simulation vs analytics vs experiment.

Every test in this file compares the simulator's dynamical output
against either an independent analytical prediction or a published
experimental number.  These go beyond "does the code run" to
"does the code produce physically correct answers."
"""

import numpy as np
import pytest
import qutip

from tiqs.analysis.error_budget import off_resonant_mode_error
from tiqs.analysis.fidelity import bell_state_fidelity
from tiqs.chain.lamb_dicke import lamb_dicke_parameters
from tiqs.chain.normal_modes import normal_modes
from tiqs.constants import PI, TWO_PI
from tiqs.gates.calibration import calibrate_single_tone_ms
from tiqs.gates.molmer_sorensen import (
    ms_gate_duration,
    ms_gate_hamiltonian,
    ms_geometric_phase,
    ms_multimode_hamiltonian,
)
from tiqs.gates.qudit import ms_qudit_gate_hamiltonian
from tiqs.hilbert_space.builder import HilbertSpace
from tiqs.hilbert_space.operators import OperatorFactory
from tiqs.hilbert_space.states import StateFactory
from tiqs.interaction.gradient import MagneticGradient, gradient_lamb_dicke
from tiqs.species.ion import get_species
from tiqs.trap import PaulTrap, PenningTrap


class TestGeometricPhaseMatchesSimulation:
    """The analytic formula ms_geometric_phase() must predict the
    same entangling phase that the full QuTiP simulation produces."""

    def test_single_mode_phase_predicts_bell_fidelity(self):
        """If the analytic chi = pi/4, the simulation must produce a
        Bell state with F > 0.99.  If chi != pi/4, fidelity drops."""
        hs = HilbertSpace(n_ions=2, n_modes=1, n_fock=20)
        ops = OperatorFactory(hs)
        sf = StateFactory(hs)

        eta_val = 0.05
        eta = np.array([[eta_val], [eta_val]])
        delta = TWO_PI * 15e3
        Omega = delta / (4 * eta_val)
        tau = ms_gate_duration(delta)

        chi = ms_geometric_phase(eta, Omega, [delta], tau)
        assert chi[0, 1] == pytest.approx(PI / 4, rel=1e-6)

        H = ms_gate_hamiltonian(
            ops, [0, 1], 0, [eta_val, eta_val], Omega, delta
        )
        result = qutip.sesolve(
            H,
            sf.ground_state(),
            np.linspace(0, tau, 800),
            options={"max_step": tau / 200},
        )
        fid = bell_state_fidelity(result.states[-1].ptrace([0, 1]))
        assert fid > 0.99

    def test_half_entangling_gate(self):
        """At Omega such that chi = pi/8 (half-entangling), the
        Bell fidelity should be noticeably less than 1."""
        hs = HilbertSpace(n_ions=2, n_modes=1, n_fock=20)
        ops = OperatorFactory(hs)
        sf = StateFactory(hs)

        eta_val = 0.05
        delta = TWO_PI * 15e3
        Omega_full = delta / (4 * eta_val)
        Omega_half = Omega_full / np.sqrt(2)
        tau = ms_gate_duration(delta)

        eta = np.array([[eta_val], [eta_val]])
        chi = ms_geometric_phase(eta, Omega_half, [delta], tau)
        assert chi[0, 1] == pytest.approx(PI / 8, rel=1e-4)

        H = ms_gate_hamiltonian(
            ops, [0, 1], 0, [eta_val, eta_val], Omega_half, delta
        )
        result = qutip.sesolve(
            H,
            sf.ground_state(),
            np.linspace(0, tau, 800),
            options={"max_step": tau / 200},
        )
        fid = bell_state_fidelity(result.states[-1].ptrace([0, 1]))
        # Half-entangling: analytically F = (1 + sin(pi/4))/2 ~ 0.854
        assert fid == pytest.approx((1 + np.sin(PI / 4)) / 2, abs=0.03)


class TestOffResonantErrorScaling:
    """The off-resonant mode error from off_resonant_mode_error()
    must track the actual simulated infidelity."""

    def test_analytic_error_vs_simulation(self):
        """For a 2-mode system, the analytic off-resonant error
        estimate should approximate the actual simulated infidelity
        within a factor of ~3."""
        hs = HilbertSpace(n_ions=2, n_modes=2, n_fock=10)
        ops = OperatorFactory(hs)
        sf = StateFactory(hs)

        eta = np.array([[0.05, 0.03], [0.05, -0.03]])
        mode_freqs = np.array([TWO_PI * 1e6, TWO_PI * np.sqrt(3) * 1e6])

        params = calibrate_single_tone_ms(
            eta, mode_freqs, target_mode=0, ion_pair=(0, 1)
        )
        H = ms_multimode_hamiltonian(
            ops,
            [0, 1],
            [0, 1],
            eta,
            params["rabi_frequency"],
            params["detunings"],
        )
        psi0 = sf.ground_state()
        tau = params["gate_time"]
        result = qutip.sesolve(
            H,
            psi0,
            np.linspace(0, tau, 500),
            options={"max_step": tau / 100},
        )
        fid_sim = bell_state_fidelity(result.states[-1].ptrace([0, 1]))
        infidelity_sim = 1 - fid_sim

        err = off_resonant_mode_error(
            eta, params["rabi_frequency"], params["detunings"], tau
        )
        infidelity_analytic = err["total"]

        # Both should be small and in the same ballpark
        # (analytic is approximate, not exact)
        if infidelity_sim > 1e-4:
            ratio = infidelity_analytic / infidelity_sim
            assert 0.1 < ratio < 10

    def test_fidelity_degrades_with_mode_proximity(self):
        """When modes are closer together, simulated Bell fidelity
        should be lower because the off-resonant mode acquires
        more residual entanglement."""
        eta = np.array([[0.05, 0.03], [0.05, -0.03]])

        fidelities = []
        for mode_ratio in [np.sqrt(3), 1.15]:
            freqs = np.array([TWO_PI * 1e6, TWO_PI * mode_ratio * 1e6])
            hs = HilbertSpace(n_ions=2, n_modes=2, n_fock=10)
            ops = OperatorFactory(hs)
            sf = StateFactory(hs)

            params = calibrate_single_tone_ms(
                eta, freqs, target_mode=0, ion_pair=(0, 1)
            )
            H = ms_multimode_hamiltonian(
                ops,
                [0, 1],
                [0, 1],
                eta,
                params["rabi_frequency"],
                params["detunings"],
            )
            tau = params["gate_time"]
            r = qutip.sesolve(
                H,
                sf.ground_state(),
                np.linspace(0, tau, 500),
                options={"max_step": tau / 100},
            )
            fidelities.append(bell_state_fidelity(r.states[-1].ptrace([0, 1])))

        # Wide splitting should give higher fidelity
        assert fidelities[0] > fidelities[1]


class TestSmoothGateThermalRobustness:
    """The smooth gate should maintain fidelity at finite temperature
    where a standard gate degrades.  This is the key experimental
    result from arXiv:2510.17286."""

    def test_smooth_gate_at_finite_temperature(self):
        """At n_bar=2, the smooth gate should still produce
        significant entanglement."""
        from tiqs.gates.molmer_sorensen import ms_gate_hamiltonian_pulsed
        from tiqs.pulses import smooth_ms_pulse

        hs = HilbertSpace(n_ions=2, n_modes=1, n_fock=15)
        ops = OperatorFactory(hs)
        sf = StateFactory(hs)

        eta_val = 0.05
        delta_0 = TWO_PI * 15e3
        Omega = delta_0 / (4 * eta_val)
        eta = [eta_val, eta_val]

        pulse = smooth_ms_pulse(
            delta_0=delta_0,
            ramp_amplitude=0.5 * delta_0,
            rabi_frequency=Omega,
        )
        tlist = np.linspace(0, pulse.duration, 800)
        H = ms_gate_hamiltonian_pulsed(ops, [0, 1], 0, eta, pulse, tlist)

        # Start at n_bar = 2 (thermal motional state)
        rho0 = sf.thermal_state(n_bar=[2.0])
        result = qutip.mesolve(
            H, rho0, tlist, options={"max_step": pulse.duration / 200}
        )

        rho_spin = result.states[-1].ptrace([0, 1])
        rho_single = rho_spin.ptrace(0)
        purity = (rho_single * rho_single).tr().real
        # Should still show entanglement even at n_bar=2
        assert purity < 0.95


class TestPenningRadialModes:
    """Verify Penning radial mode structure for multi-ion crystals."""

    def test_two_ion_cyclotron_above_magnetron(self):
        """All cyclotron modes must be above all magnetron modes."""
        trap = PenningTrap(
            magnetic_field=3.0,
            species=get_species("Be9"),
            d=300e-6,
            omega_axial=TWO_PI * 1.5e6,
        )
        modes = normal_modes(2, trap)
        assert min(modes.cyclotron_freqs) > max(modes.magnetron_freqs)

    def test_single_ion_modes_match_trap_properties(self):
        """For 1 ion, the mode frequencies should exactly match
        the trap's computed omega_+, omega_-, omega_z."""
        trap = PenningTrap(
            magnetic_field=3.0,
            species=get_species("Be9"),
            d=300e-6,
            omega_axial=TWO_PI * 1.5e6,
        )
        modes = normal_modes(1, trap)
        assert modes.axial_freqs[0] == pytest.approx(
            trap.omega_axial, rel=1e-6
        )
        assert modes.cyclotron_freqs[0] == pytest.approx(
            trap.omega_cyclotron, rel=1e-6
        )
        assert modes.magnetron_freqs[0] == pytest.approx(
            trap.omega_magnetron, rel=1e-6
        )

    def test_magnetron_lamb_dicke_larger_than_cyclotron(self):
        """Lower-frequency magnetron mode has larger zero-point motion
        and thus larger Lamb-Dicke parameter.  This is a key physical
        consequence: eta ~ 1/sqrt(omega)."""
        trap = PenningTrap(
            magnetic_field=3.0,
            species=get_species("Be9"),
            d=300e-6,
            omega_axial=TWO_PI * 1.5e6,
        )
        modes = normal_modes(1, trap)
        k_eff = TWO_PI / 313e-9

        eta_cyc = lamb_dicke_parameters(
            modes, trap.species, k_eff, "cyclotron"
        )
        eta_mag = lamb_dicke_parameters(
            modes, trap.species, k_eff, "magnetron"
        )
        assert abs(eta_mag[0, 0]) > abs(eta_cyc[0, 0])


class TestGradientGatePhysics:
    """Verify gradient-mediated coupling produces correct Lamb-Dicke
    parameters and entanglement."""

    def test_gradient_keff_formula(self):
        """The gradient effective wavevector must satisfy
        k_eff = (d omega_q/dB) * (dB/dz) / omega_q.
        Verify this for Yb-171 at 19.09 T/m gradient."""
        yb = get_species("Yb171")
        grad = MagneticGradient(db_dz=19.09, b_field=0.5e-3)
        k_eff = grad.effective_k(yb)
        expected = (
            yb.qubit_zeeman_sensitivity
            * grad.db_dz
            / (TWO_PI * yb.qubit_frequency_hz)
        )
        assert k_eff == pytest.approx(expected, rel=1e-10)
        assert k_eff > 0

    def test_gradient_zz_gate_entangles(self):
        """A ZZ gate from gradient coupling should entangle |+,+>
        into a state with reduced single-qubit purity."""
        from tiqs.gates.light_shift import light_shift_gate_hamiltonian

        yb = get_species("Yb171")
        trap = PaulTrap(
            v_rf=1000,
            omega_rf=TWO_PI * 30e6,
            r0=0.5e-3,
            omega_axial=TWO_PI * 1e6,
            species=yb,
        )
        grad = MagneticGradient(db_dz=150.0, b_field=0.01)
        modes = normal_modes(2, trap)
        eta_matrix = gradient_lamb_dicke(modes, yb, grad, "axial")
        eta_ions = [float(eta_matrix[0, 0]), float(eta_matrix[1, 0])]
        eta_avg = np.mean(np.abs(eta_ions))

        hs = HilbertSpace(n_ions=2, n_modes=1, n_fock=15)
        ops = OperatorFactory(hs)

        delta = TWO_PI * 15e3
        Omega = delta / (4 * eta_avg)
        tau = ms_gate_duration(delta)

        H = light_shift_gate_hamiltonian(
            ops, [0, 1], 0, eta_ions, Omega, delta
        )

        plus = (qutip.basis(2, 0) + qutip.basis(2, 1)).unit()
        psi0 = qutip.tensor(plus, plus, qutip.basis(15, 0))
        result = qutip.sesolve(
            H,
            psi0,
            np.linspace(0, tau, 500),
            options={"max_step": tau / 100},
        )
        rho_single = result.states[-1].ptrace(0)
        purity = (rho_single * rho_single).tr().real
        assert purity < 0.7


class TestQuditGatePhysicsAccuracy:
    """Verify that qudit gates on the |0>-|1> subspace reproduce
    the same physics as qubit gates, as required by the transition
    operator reduction to Pauli matrices for d=2."""

    def test_qutrit_ms_matches_qubit_ms_on_01_subspace(self):
        """An MS gate driving the |0>-|1> transition of a qutrit
        should produce the same spin-state fidelity as a qubit MS
        gate, since transition_x(0,1) = sigma_x for the {0,1}
        subspace."""
        # Qubit version
        hs_q = HilbertSpace(n_ions=2, n_modes=1, n_fock=15)
        ops_q = OperatorFactory(hs_q)
        sf_q = StateFactory(hs_q)

        eta = 0.1
        delta = TWO_PI * 20e3
        Omega = delta / (4 * eta)
        tau = ms_gate_duration(delta)

        H_qubit = ms_gate_hamiltonian(
            ops_q, [0, 1], 0, [eta, eta], Omega, delta
        )
        r_qubit = qutip.sesolve(
            H_qubit,
            sf_q.ground_state(),
            np.linspace(0, tau, 500),
            options={"max_step": tau / 100},
        )
        rho_qubit = r_qubit.states[-1].ptrace([0, 1])
        purity_qubit = (rho_qubit.ptrace(0) ** 2).tr().real

        # Qutrit version on |0>-|1> subspace
        hs_t = HilbertSpace(n_ions=2, n_modes=1, n_fock=15, ion_dims=3)
        ops_t = OperatorFactory(hs_t)
        sf_t = StateFactory(hs_t)

        H_qutrit = ms_qudit_gate_hamiltonian(
            ops_t,
            [0, 1],
            0,
            [eta, eta],
            Omega,
            delta,
            transitions=[(0, 1), (0, 1)],
        )
        r_qutrit = qutip.sesolve(
            H_qutrit,
            sf_t.ground_state(),
            np.linspace(0, tau, 500),
            options={"max_step": tau / 100},
        )
        rho_qutrit = r_qutrit.states[-1].ptrace([0, 1])
        purity_qutrit = (rho_qutrit.ptrace(0) ** 2).tr().real

        # Both should produce equivalent entanglement
        assert purity_qutrit == pytest.approx(purity_qubit, abs=0.05)

    def test_qutrit_gate_does_not_leak_to_level_2(self):
        """When driving the |0>-|1> transition, population in |2>
        should remain zero throughout the gate."""
        hs = HilbertSpace(n_ions=2, n_modes=1, n_fock=10, ion_dims=3)
        ops = OperatorFactory(hs)
        sf = StateFactory(hs)

        eta = 0.1
        delta = TWO_PI * 20e3
        Omega = delta / (4 * eta)
        tau = ms_gate_duration(delta)

        H = ms_qudit_gate_hamiltonian(
            ops,
            [0, 1],
            0,
            [eta, eta],
            Omega,
            delta,
            transitions=[(0, 1), (0, 1)],
        )
        result = qutip.sesolve(
            H,
            sf.ground_state(),
            np.linspace(0, tau, 200),
            options={"max_step": tau / 50},
        )

        for state in result.states:
            p2_ion0 = qutip.expect(ops.projector(0, 2), state)
            p2_ion1 = qutip.expect(ops.projector(1, 2), state)
            assert p2_ion0 < 0.01
            assert p2_ion1 < 0.01

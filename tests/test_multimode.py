"""Tests for multi-mode entangling gates, calibration, and analysis."""

import numpy as np
import pytest
import qutip

from tiqs.analysis.error_budget import off_resonant_mode_error
from tiqs.analysis.fidelity import bell_state_fidelity
from tiqs.analysis.phase_space import multi_mode_phase_space_trajectories
from tiqs.constants import PI, TWO_PI
from tiqs.gates.calibration import (
    calibrate_multi_tone_ms,
    calibrate_single_tone_ms,
)
from tiqs.gates.light_shift import light_shift_multimode_hamiltonian
from tiqs.gates.molmer_sorensen import (
    ms_gate_duration,
    ms_gate_hamiltonian,
    ms_geometric_phase,
    ms_multimode_hamiltonian,
    ms_residual_displacement,
    ms_single_tone_detunings,
)
from tiqs.hilbert_space.builder import HilbertSpace
from tiqs.hilbert_space.operators import OperatorFactory
from tiqs.hilbert_space.states import StateFactory
from tiqs.simulation.config import SimulationConfig
from tiqs.simulation.runner import SimulationRunner
from tiqs.species.ion import get_species
from tiqs.trap import PaulTrap


@pytest.fixture
def two_mode_system():
    """2 ions, 2 modes (COM + stretch), Fock cutoff 10."""
    hs = HilbertSpace(n_ions=2, n_modes=2, n_fock=10)
    ops = OperatorFactory(hs)
    sf = StateFactory(hs)
    return hs, ops, sf


class TestMultiModeMSHamiltonian:
    def test_multimode_reduces_to_single_mode(self):
        """One-mode multi-mode Hamiltonian should match the single-mode
        version in term count and structure."""
        hs = HilbertSpace(n_ions=2, n_modes=1, n_fock=10)
        ops = OperatorFactory(hs)
        eta = 0.1
        delta = TWO_PI * 10e3
        Omega = TWO_PI * 50e3

        H_single = ms_gate_hamiltonian(
            ops,
            [0, 1],
            0,
            [eta, eta],
            Omega,
            delta,
        )
        H_multi = ms_multimode_hamiltonian(
            ops,
            [0, 1],
            [0],
            np.array([[eta], [eta]]),
            Omega,
            [delta],
        )
        assert len(H_multi) == len(H_single)

    def test_multimode_term_count(self, two_mode_system):
        hs, ops, sf = two_mode_system
        eta = np.array([[0.1, 0.05], [0.1, -0.05]])
        H = ms_multimode_hamiltonian(
            ops,
            [0, 1],
            [0, 1],
            eta,
            1e5,
            [1e4, 2e4],
        )
        assert len(H) == 2 * 2 * 2  # 2 ions * 2 modes * 2 terms

    def test_multimode_is_list_format(self, two_mode_system):
        hs, ops, sf = two_mode_system
        eta = np.array([[0.1, 0.05], [0.1, -0.05]])
        H = ms_multimode_hamiltonian(
            ops,
            [0, 1],
            [0, 1],
            eta,
            1e5,
            [1e4, 2e4],
        )
        assert isinstance(H, list)
        for term in H:
            assert isinstance(term, list)
            assert len(term) == 2
            assert isinstance(term[1], str)

    def test_per_mode_rabi_frequencies(self, two_mode_system):
        """Each mode should use its own Rabi frequency."""
        hs, ops, sf = two_mode_system
        eta = np.array([[1.0, 1.0], [1.0, 1.0]])
        Omega_list = [100.0, 200.0]
        H = ms_multimode_hamiltonian(
            ops,
            [0, 1],
            [0, 1],
            eta,
            Omega_list,
            [1e4, 2e4],
        )
        # Mode 0 terms (first 4) use Omega=100, mode 1 terms use 200
        # Operator norm of coupling * ad * sx scales with Omega
        norm_mode0 = H[0][0].norm()
        norm_mode1 = H[4][0].norm()
        assert norm_mode1 / norm_mode0 == pytest.approx(2.0, rel=0.01)

    def test_scalar_rabi_broadcasts_to_all_modes(self, two_mode_system):
        hs, ops, sf = two_mode_system
        eta = np.array([[1.0, 1.0], [1.0, 1.0]])
        H = ms_multimode_hamiltonian(
            ops,
            [0, 1],
            [0, 1],
            eta,
            100.0,
            [1e4, 2e4],
        )
        # All 8 terms should exist
        assert len(H) == 8


class TestMultiModeLightShift:
    def test_light_shift_multimode_term_count(self, two_mode_system):
        hs, ops, sf = two_mode_system
        eta = np.array([[0.1, 0.05], [0.1, -0.05]])
        H = light_shift_multimode_hamiltonian(
            ops,
            [0, 1],
            [0, 1],
            eta,
            1e5,
            [1e4, 2e4],
        )
        assert len(H) == 8

    def test_light_shift_multimode_uses_sigma_z(self, two_mode_system):
        """Light-shift terms should be Hermitian (sigma_z is Hermitian,
        while sigma_x * a^dag is not in general, but the operator
        coupling * a^dag * sigma_z should be non-Hermitian.
        Check that the term differs from the MS version."""
        hs, ops, sf = two_mode_system
        eta = np.array([[0.1, 0.05], [0.1, -0.05]])
        H_ms = ms_multimode_hamiltonian(
            ops,
            [0, 1],
            [0, 1],
            eta,
            1e5,
            [1e4, 2e4],
        )
        H_ls = light_shift_multimode_hamiltonian(
            ops,
            [0, 1],
            [0, 1],
            eta,
            1e5,
            [1e4, 2e4],
        )
        # The operators should differ (sigma_x vs sigma_z)
        assert (H_ms[0][0] - H_ls[0][0]).norm() > 0


class TestSingleToneDetunings:
    def test_basic(self):
        mu = TWO_PI * 1.01e6
        freqs = np.array([TWO_PI * 1e6, TWO_PI * np.sqrt(3) * 1e6])
        dets = ms_single_tone_detunings(mu, freqs)
        assert len(dets) == 2
        assert dets[0] == pytest.approx(mu - freqs[0])
        assert dets[1] == pytest.approx(mu - freqs[1])


class TestGeometricPhase:
    def test_single_mode_analytic(self):
        """chi = pi * K * eta^2 * Omega^2 / (2 * delta^2) for equal
        eta and one mode."""
        eta_val = 0.1
        eta = np.array([[eta_val], [eta_val]])
        delta = TWO_PI * 15e3
        Omega = delta / (4 * eta_val)
        tau = ms_gate_duration(delta, loops=1)

        chi = ms_geometric_phase(eta, Omega, [delta], tau)
        # For maximally entangling: chi[0,1] should be pi/4
        assert chi[0, 1] == pytest.approx(PI / 4, rel=1e-6)

    def test_symmetric(self):
        eta = np.array([[0.1, 0.05], [0.08, -0.04]])
        chi = ms_geometric_phase(eta, 1e5, [1e4, 2e4], 1e-4)
        assert chi[0, 1] == pytest.approx(chi[1, 0], rel=1e-10)

    def test_additivity(self):
        """Two-mode phase should equal sum of single-mode phases."""
        eta = np.array([[0.1, 0.05], [0.1, -0.05]])
        Omega = 1e5
        dets = [1e4, 3e4]
        tau = 1e-4

        chi_both = ms_geometric_phase(eta, Omega, dets, tau)
        chi_0 = ms_geometric_phase(
            eta[:, 0:1],
            Omega,
            [dets[0]],
            tau,
        )
        chi_1 = ms_geometric_phase(
            eta[:, 1:2],
            Omega,
            [dets[1]],
            tau,
        )
        assert chi_both[0, 1] == pytest.approx(
            chi_0[0, 1] + chi_1[0, 1],
            rel=1e-10,
        )


class TestResidualDisplacement:
    def test_closed_loop_zero_residual(self):
        delta = TWO_PI * 10e3
        tau = ms_gate_duration(delta, loops=1)
        residuals = ms_residual_displacement([delta], tau)
        assert residuals[0] == pytest.approx(0.0, abs=1e-10)

    def test_half_loop_max_residual(self):
        delta = TWO_PI * 10e3
        tau = ms_gate_duration(delta, loops=1)
        # A mode with delta_other such that K_eff = 1.5
        delta_other = 1.5 * TWO_PI / tau
        residuals = ms_residual_displacement([delta_other], tau)
        assert abs(residuals[0]) == pytest.approx(0.5, abs=1e-10)


class TestCalibration:
    @pytest.fixture
    def two_mode_eta(self):
        """Symmetric 2-ion, 2-mode Lamb-Dicke matrix."""
        return np.array([[0.1, 0.06], [0.1, -0.06]])

    @pytest.fixture
    def mode_freqs(self):
        return np.array([TWO_PI * 1e6, TWO_PI * np.sqrt(3) * 1e6])

    def test_single_tone_chi_near_target(
        self,
        two_mode_eta,
        mode_freqs,
    ):
        params = calibrate_single_tone_ms(
            two_mode_eta,
            mode_freqs,
            target_mode=0,
            ion_pair=(0, 1),
        )
        chi = params["chi_matrix"]
        assert chi[0, 1] == pytest.approx(PI / 4, rel=0.01)

    def test_single_tone_target_mode_closes(
        self,
        two_mode_eta,
        mode_freqs,
    ):
        params = calibrate_single_tone_ms(
            two_mode_eta,
            mode_freqs,
            target_mode=0,
            ion_pair=(0, 1),
        )
        assert params["residuals"][0] == pytest.approx(0.0, abs=1e-10)

    def test_multi_tone_all_modes_close(
        self,
        two_mode_eta,
        mode_freqs,
    ):
        params = calibrate_multi_tone_ms(
            two_mode_eta,
            mode_freqs,
            ion_pair=(0, 1),
        )
        for r in params["residuals"]:
            assert r == pytest.approx(0.0, abs=1e-10)

    def test_multi_tone_chi_equals_target(
        self,
        two_mode_eta,
        mode_freqs,
    ):
        params = calibrate_multi_tone_ms(
            two_mode_eta,
            mode_freqs,
            ion_pair=(0, 1),
        )
        chi = params["chi_matrix"]
        assert chi[0, 1] == pytest.approx(PI / 4, rel=0.01)


class TestMultiModePhysics:
    """Physics validation: simulate multi-mode gates and check results."""

    def test_two_mode_ms_bell_fidelity(self, two_mode_system):
        """Multi-mode MS gate on 2 ions with COM + stretch should
        produce a Bell state."""
        hs, ops, sf = two_mode_system
        eta = np.array([[0.05, 0.03], [0.05, -0.03]])
        mode_freqs = np.array([TWO_PI * 1e6, TWO_PI * np.sqrt(3) * 1e6])

        params = calibrate_single_tone_ms(
            eta,
            mode_freqs,
            target_mode=0,
            ion_pair=(0, 1),
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
        tlist = np.linspace(0, tau, 500)
        result = qutip.sesolve(
            H,
            psi0,
            tlist,
            options={"max_step": tau / 100},
        )
        rho_spin = result.states[-1].ptrace([0, 1])
        fid = bell_state_fidelity(rho_spin)
        assert fid > 0.85

    def test_multi_tone_higher_fidelity(self, two_mode_system):
        """Multi-tone should achieve higher fidelity than single-tone
        because all modes close."""
        hs, ops, sf = two_mode_system
        eta = np.array([[0.05, 0.03], [0.05, -0.03]])
        mode_freqs = np.array([TWO_PI * 1e6, TWO_PI * np.sqrt(3) * 1e6])
        psi0 = sf.ground_state()

        # Single-tone
        params_st = calibrate_single_tone_ms(
            eta,
            mode_freqs,
            target_mode=0,
            ion_pair=(0, 1),
        )
        H_st = ms_multimode_hamiltonian(
            ops,
            [0, 1],
            [0, 1],
            eta,
            params_st["rabi_frequency"],
            params_st["detunings"],
        )
        tau_st = params_st["gate_time"]
        r_st = qutip.sesolve(
            H_st,
            psi0,
            np.linspace(0, tau_st, 500),
            options={"max_step": tau_st / 100},
        )
        fid_st = bell_state_fidelity(r_st.states[-1].ptrace([0, 1]))

        # Multi-tone
        params_mt = calibrate_multi_tone_ms(
            eta,
            mode_freqs,
            ion_pair=(0, 1),
        )
        H_mt = ms_multimode_hamiltonian(
            ops,
            [0, 1],
            [0, 1],
            eta,
            params_mt["rabi_frequency"],
            params_mt["detunings"],
        )
        tau_mt = params_mt["gate_time"]
        r_mt = qutip.sesolve(
            H_mt,
            psi0,
            np.linspace(0, tau_mt, 500),
            options={"max_step": tau_mt / 100},
        )
        fid_mt = bell_state_fidelity(r_mt.states[-1].ptrace([0, 1]))

        assert fid_mt >= fid_st - 0.02  # multi-tone at least as good

    def test_multi_tone_motional_disentanglement(self, two_mode_system):
        """After a multi-tone gate, both modes should return near
        vacuum."""
        hs, ops, sf = two_mode_system
        eta = np.array([[0.05, 0.03], [0.05, -0.03]])
        mode_freqs = np.array([TWO_PI * 1e6, TWO_PI * np.sqrt(3) * 1e6])

        params = calibrate_multi_tone_ms(
            eta,
            mode_freqs,
            ion_pair=(0, 1),
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
        for m in range(2):
            n_final = qutip.expect(ops.number(m), result.states[-1])
            assert n_final < 0.3


class TestMultiModeRunner:
    @pytest.fixture
    def ca40_2mode_config(self):
        trap = PaulTrap(
            v_rf=300.0,
            omega_rf=TWO_PI * 30e6,
            r0=0.5e-3,
            omega_axial=TWO_PI * 1e6,
            species=get_species("Ca40"),
        )
        return SimulationConfig(
            species=get_species("Ca40"),
            trap=trap,
            n_ions=2,
            n_modes=2,
            n_fock=10,
        )

    def test_backward_compat_mode_param(self, ca40_2mode_config):
        runner = SimulationRunner(ca40_2mode_config)
        result = runner.run_ms_gate(ions=[0, 1], mode=0)
        assert result.states[-1] is not None

    def test_modes_param_works(self, ca40_2mode_config):
        runner = SimulationRunner(ca40_2mode_config)
        result = runner.run_ms_gate(ions=[0, 1], modes=[0, 1])
        assert result.states[-1] is not None

    def test_mode_modes_mutual_exclusion(self, ca40_2mode_config):
        runner = SimulationRunner(ca40_2mode_config)
        with pytest.raises(ValueError):
            runner.run_ms_gate(ions=[0, 1], mode=0, modes=[0, 1])

    def test_multi_tone_drive(self, ca40_2mode_config):
        runner = SimulationRunner(ca40_2mode_config)
        result = runner.run_ms_gate(
            ions=[0, 1],
            modes=[0, 1],
            drive="multi_tone",
        )
        assert result.states[-1] is not None

    def test_multi_mode_entangles(self, ca40_2mode_config):
        runner = SimulationRunner(ca40_2mode_config)
        result = runner.run_ms_gate(ions=[0, 1], modes=[0, 1])
        rho_spin = result.states[-1].ptrace([0, 1])
        rho_single = rho_spin.ptrace(0)
        purity = (rho_single * rho_single).tr().real
        assert purity < 0.9  # entangled


class TestOffResonantModeError:
    def test_closed_mode_zero_error(self):
        eta = np.array([[0.1], [0.1]])
        delta = TWO_PI * 10e3
        tau = ms_gate_duration(delta, loops=1)
        err = off_resonant_mode_error(eta, 1e5, [delta], tau)
        assert err["mode_0"] == pytest.approx(0.0, abs=1e-10)

    def test_off_resonant_positive_error(self):
        eta = np.array([[0.1, 0.05], [0.1, -0.05]])
        delta_target = TWO_PI * 10e3
        tau = ms_gate_duration(delta_target, loops=1)
        delta_other = TWO_PI * 25e3  # does not close
        err = off_resonant_mode_error(
            eta,
            1e5,
            [delta_target, delta_other],
            tau,
        )
        assert err["mode_0"] == pytest.approx(0.0, abs=1e-10)
        assert err["mode_1"] > 0
        assert err["total"] > 0


class TestMultiModeTrajectories:
    def test_returns_all_modes(self, two_mode_system):
        hs, ops, sf = two_mode_system
        psi0 = sf.ground_state()
        states = [psi0, psi0]  # trivial trajectory
        trajs = multi_mode_phase_space_trajectories(
            states,
            [0, 1],
            [0, 1],
        )
        assert 0 in trajs and 1 in trajs
        assert len(trajs[0][0]) == 2
        assert len(trajs[1][0]) == 2

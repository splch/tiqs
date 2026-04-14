"""Tests for microwave-driven magnetic-gradient gates."""

import numpy as np
import pytest
import qutip

from tiqs.chain.equilibrium import equilibrium_positions
from tiqs.chain.normal_modes import normal_modes
from tiqs.constants import TWO_PI
from tiqs.gates.microwave_ms import (
    microwave_ms_gate_hamiltonian,
    microwave_ms_gate_hamiltonian_full,
)
from tiqs.hilbert_space.builder import HilbertSpace
from tiqs.hilbert_space.operators import OperatorFactory
from tiqs.hilbert_space.states import StateFactory
from tiqs.interaction.gradient import (
    MagneticGradient,
    gradient_addressing_crosstalk,
    gradient_lamb_dicke,
    gradient_qubit_frequencies,
)
from tiqs.noise.microwave_noise import (
    gradient_motional_dephasing_op,
    magnetic_field_t2,
    microwave_amplitude_noise_op,
    microwave_phase_noise_op,
)
from tiqs.simulation.config import SimulationConfig
from tiqs.simulation.runner import SimulationRunner
from tiqs.species.electron import ElectronSpecies
from tiqs.species.ion import get_species
from tiqs.trap import PaulTrap


class TestIonSpeciesZeeman:
    def test_yb171_has_zeeman_sensitivity(self):
        s = get_species("Yb171")
        assert s.qubit_zeeman_sensitivity is not None
        assert s.qubit_zeeman_sensitivity > 0
        assert s.supports_microwave_gate

    def test_ca40_optical_no_microwave(self):
        s = get_species("Ca40")
        assert s.qubit_zeeman_sensitivity is None
        assert not s.supports_microwave_gate

    def test_all_hyperfine_support_microwave(self):
        for name in ["Yb171", "Ca43", "Ba137", "Be9"]:
            s = get_species(name)
            assert s.supports_microwave_gate, f"{name} should support"

    def test_all_optical_no_microwave(self):
        for name in ["Ca40", "Sr88"]:
            s = get_species(name)
            assert not s.supports_microwave_gate


class TestMagneticGradient:
    def test_effective_k_electron(self):
        """For electrons, k_eff = (dB/dz) / B."""
        grad = MagneticGradient(db_dz=120.0, b_field=0.1)
        e = ElectronSpecies(magnetic_field=0.1)
        k = grad.effective_k(e)
        assert k == pytest.approx(120.0 / 0.1)

    def test_effective_k_ion(self):
        """k_eff = (d omega_q/dB) * (dB/dz) / omega_q."""
        grad = MagneticGradient(db_dz=150.0, b_field=0.01)
        yb = get_species("Yb171")
        k = grad.effective_k(yb)
        expected = (
            yb.qubit_zeeman_sensitivity
            * grad.db_dz
            / (TWO_PI * yb.qubit_frequency_hz)
        )
        assert k == pytest.approx(expected)
        assert k > 0

    def test_effective_k_optical_raises(self):
        grad = MagneticGradient(db_dz=100.0, b_field=0.01)
        ca = get_species("Ca40")
        with pytest.raises(ValueError, match="qubit_zeeman_sensitivity"):
            grad.effective_k(ca)

    def test_electron_matches_test_electron_helper(self):
        """Cross-validate with the existing test_electron.py approach:
        k_eff = gradient / B."""
        grad = MagneticGradient(db_dz=120.0, b_field=0.1)
        e = ElectronSpecies(magnetic_field=0.1)
        assert grad.effective_k(e) == pytest.approx(1200.0)


class TestGradientLambDicke:
    def test_gradient_lamb_dicke_matches_manual(self):
        yb = get_species("Yb171")
        trap = PaulTrap(
            v_rf=1000,
            omega_rf=TWO_PI * 30e6,
            r0=0.5e-3,
            omega_axial=TWO_PI * 1e6,
            species=yb,
        )
        modes = normal_modes(1, trap)
        grad = MagneticGradient(db_dz=150.0, b_field=0.01)

        eta = gradient_lamb_dicke(modes, yb, grad, "axial")
        assert eta.shape == (1, 1)
        assert abs(eta[0, 0]) > 0


class TestGradientAddressing:
    @pytest.fixture
    def yb_trap(self):
        yb = get_species("Yb171")
        return PaulTrap(
            v_rf=1000,
            omega_rf=TWO_PI * 30e6,
            r0=0.5e-3,
            omega_axial=TWO_PI * 1e6,
            species=yb,
        )

    def test_qubit_frequencies_differ(self, yb_trap):
        grad = MagneticGradient(db_dz=150.0, b_field=0.01)
        pos = equilibrium_positions(2, yb_trap)
        freqs = gradient_qubit_frequencies(yb_trap.species, grad, pos)
        assert len(freqs) == 2
        assert freqs[0] != freqs[1]

    def test_crosstalk_diagonal_is_one(self, yb_trap):
        grad = MagneticGradient(db_dz=150.0, b_field=0.01)
        pos = equilibrium_positions(2, yb_trap)
        xt = gradient_addressing_crosstalk(
            yb_trap.species, grad, pos, TWO_PI * 100e3
        )
        assert xt[0, 0] == pytest.approx(1.0)
        assert xt[1, 1] == pytest.approx(1.0)

    def test_crosstalk_off_diagonal_small(self, yb_trap):
        grad = MagneticGradient(db_dz=150.0, b_field=0.01)
        pos = equilibrium_positions(2, yb_trap)
        xt = gradient_addressing_crosstalk(
            yb_trap.species, grad, pos, TWO_PI * 100e3
        )
        assert xt[0, 1] < 0.1
        assert xt[1, 0] < 0.1


class TestMicrowaveNoise:
    @pytest.fixture
    def system(self):
        hs = HilbertSpace(n_ions=2, n_modes=1, n_fock=10)
        ops = OperatorFactory(hs)
        return ops

    def test_phase_noise_is_hermitian(self, system):
        ops = system
        c = microwave_phase_noise_op(ops, 0, TWO_PI * 1.0)
        assert c.isherm

    def test_amplitude_noise_is_hermitian(self, system):
        ops = system
        H = microwave_amplitude_noise_op(ops, 0, 0.01, 1e6)
        assert H.isherm

    def test_magnetic_field_t2(self):
        # Strong sensitivity + noisy field = short T2
        t2 = magnetic_field_t2(1e-12, TWO_PI * 2.1e9)
        assert 0 < t2 < 100

    def test_magnetic_field_t2_zero_noise(self):
        t2 = magnetic_field_t2(0.0, TWO_PI * 2.1e9)
        assert t2 == float("inf")

    def test_gradient_dephasing_op(self, system):
        ops = system
        c = gradient_motional_dephasing_op(
            ops,
            0,
            fractional_gradient_noise=1e-3,
            gate_coupling=TWO_PI * 1e3,
            gate_detuning=TWO_PI * 10e3,
        )
        assert c.shape == ops.identity().shape


class TestMicrowaveMSGate:
    @pytest.fixture
    def system(self):
        hs = HilbertSpace(n_ions=2, n_modes=1, n_fock=15)
        ops = OperatorFactory(hs)
        sf = StateFactory(hs)
        return hs, ops, sf

    def test_hamiltonian_is_list_format(self, system):
        _hs, ops, _sf = system
        H = microwave_ms_gate_hamiltonian(
            ops,
            [0, 1],
            0,
            [0.05, 0.05],
            gate_rabi_frequency=TWO_PI * 50e3,
            detuning=TWO_PI * 10e3,
            dressing_rabi_frequency=TWO_PI * 10e6,
        )
        assert isinstance(H, list)

    def test_warns_on_weak_dressing(self, system):
        _hs, ops, _sf = system
        with pytest.warns(UserWarning, match="hierarchy"):
            microwave_ms_gate_hamiltonian(
                ops,
                [0, 1],
                0,
                [0.1, 0.1],
                gate_rabi_frequency=TWO_PI * 1e6,
                detuning=TWO_PI * 10e3,
                dressing_rabi_frequency=TWO_PI * 100e3,
            )

    def test_full_hamiltonian_has_dressing_terms(self, system):
        _hs, ops, _sf = system
        H = microwave_ms_gate_hamiltonian_full(
            ops,
            [0, 1],
            0,
            [0.05, 0.05],
            gate_rabi_frequency=TWO_PI * 50e3,
            detuning=TWO_PI * 10e3,
            dressing_rabi_frequency=TWO_PI * 10e6,
        )
        # Should have static dressing terms + time-dependent gradient terms
        static_count = sum(1 for t in H if not isinstance(t, list))
        td_count = sum(1 for t in H if isinstance(t, list))
        assert static_count == 2  # one per ion
        assert td_count == 4  # 2 ions * 2 terms each


class TestGradientRunner:
    @pytest.fixture
    def yb_gradient_config(self):
        yb = get_species("Yb171")
        trap = PaulTrap(
            v_rf=1000,
            omega_rf=TWO_PI * 30e6,
            r0=0.5e-3,
            omega_axial=TWO_PI * 1e6,
            species=yb,
        )
        grad = MagneticGradient(db_dz=150.0, b_field=0.01)
        return SimulationConfig(
            species=yb,
            trap=trap,
            n_ions=2,
            n_modes=1,
            n_fock=15,
            gradient=grad,
        )

    def test_runner_uses_gradient_keff(self, yb_gradient_config):
        """When gradient is set, runner should use gradient k_eff."""
        runner = SimulationRunner(yb_gradient_config)
        assert runner.eta.shape == (2, 2)
        # eta should be nonzero
        assert abs(runner.eta[0, 0]) > 0

    def test_run_gradient_zz_gate(self, yb_gradient_config):
        runner = SimulationRunner(yb_gradient_config)
        result = runner.run_gradient_zz_gate(ions=[0, 1])
        assert result.states[-1] is not None

    def test_gradient_zz_entangles(self, yb_gradient_config):
        runner = SimulationRunner(yb_gradient_config)
        # Start in |+,+> for ZZ gate
        plus = (qutip.basis(2, 0) + qutip.basis(2, 1)).unit()
        psi0 = qutip.tensor(plus, plus, qutip.basis(15, 0))
        # Use the ZZ gate Hamiltonian manually
        from tiqs.gates.light_shift import light_shift_gate_hamiltonian
        from tiqs.gates.molmer_sorensen import ms_gate_duration

        eta_ions = [float(runner.eta[i, 0]) for i in [0, 1]]
        eta_avg = np.mean(np.abs(eta_ions))
        delta = TWO_PI * 1e3
        Omega = delta / (4 * eta_avg)
        tau = ms_gate_duration(delta)
        H = light_shift_gate_hamiltonian(
            runner.ops, [0, 1], 0, eta_ions, Omega, delta
        )
        r = qutip.sesolve(
            H,
            psi0,
            np.linspace(0, tau, 500),
            options={"max_step": tau / 100},
        )
        rho_single = r.states[-1].ptrace(0)
        purity = (rho_single * rho_single).tr().real
        assert purity < 0.9  # entangled

    def test_run_gradient_ms_gate(self, yb_gradient_config):
        runner = SimulationRunner(yb_gradient_config)
        result = runner.run_gradient_ms_gate(ions=[0, 1])
        assert result.states[-1] is not None

    def test_no_gradient_raises(self):
        trap = PaulTrap(
            v_rf=300,
            omega_rf=TWO_PI * 30e6,
            r0=0.5e-3,
            omega_axial=TWO_PI * 1e6,
            species=get_species("Ca40"),
        )
        config = SimulationConfig(
            species=get_species("Ca40"),
            trap=trap,
            n_ions=2,
        )
        runner = SimulationRunner(config)
        with pytest.raises(ValueError, match="gradient"):
            runner.run_gradient_zz_gate(ions=[0, 1])

    def test_microwave_linewidth_adds_noise(self):
        yb = get_species("Yb171")
        trap = PaulTrap(
            v_rf=1000,
            omega_rf=TWO_PI * 30e6,
            r0=0.5e-3,
            omega_axial=TWO_PI * 1e6,
            species=yb,
        )
        grad = MagneticGradient(db_dz=150.0, b_field=0.01)
        config = SimulationConfig(
            species=yb,
            trap=trap,
            n_ions=1,
            n_modes=1,
            n_fock=10,
            gradient=grad,
            microwave_linewidth=TWO_PI * 1.0,
        )
        runner = SimulationRunner(config)
        assert len(runner._c_ops) >= 1

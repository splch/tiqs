import numpy as np
import pytest
import qutip

from tiqs.analysis.fidelity import gate_fidelity, state_fidelity, bell_state_fidelity
from tiqs.analysis.phase_space import motional_wigner, phase_space_trajectory
from tiqs.analysis.error_budget import compute_error_budget
from tiqs.hilbert_space.builder import HilbertSpace
from tiqs.hilbert_space.operators import OperatorFactory
from tiqs.hilbert_space.states import StateFactory


@pytest.fixture
def two_qubit_system():
    hs = HilbertSpace(n_ions=2, n_modes=1, n_fock=10)
    ops = OperatorFactory(hs)
    sf = StateFactory(hs)
    return hs, ops, sf


class TestFidelity:
    def test_perfect_state_fidelity(self, two_qubit_system):
        hs, ops, sf = two_qubit_system
        psi = sf.ground_state()
        fid = state_fidelity(psi, psi)
        assert fid == pytest.approx(1.0, abs=1e-10)

    def test_orthogonal_state_zero_fidelity(self, two_qubit_system):
        hs, ops, sf = two_qubit_system
        psi0 = sf.product_state([0, 0], [0])
        psi1 = sf.product_state([1, 1], [0])
        fid = state_fidelity(psi0, psi1)
        assert fid == pytest.approx(0.0, abs=1e-10)

    def test_bell_state_fidelity(self, two_qubit_system):
        hs, ops, sf = two_qubit_system
        ket_00 = qutip.tensor(qutip.basis(2, 0), qutip.basis(2, 0))
        ket_11 = qutip.tensor(qutip.basis(2, 1), qutip.basis(2, 1))
        bell = (ket_00 + 1j * ket_11).unit()
        rho_spin = qutip.ket2dm(bell)
        fid = bell_state_fidelity(rho_spin)
        assert fid == pytest.approx(1.0, abs=0.01)

    def test_gate_fidelity_with_motion(self, two_qubit_system):
        """Gate fidelity should trace out motional modes before comparing."""
        hs, ops, sf = two_qubit_system
        psi = sf.ground_state()
        rho_full = qutip.ket2dm(psi)
        target = qutip.ket2dm(qutip.tensor(qutip.basis(2, 0), qutip.basis(2, 0)))
        fid = gate_fidelity(rho_full, target, qubit_indices=[0, 1])
        assert fid == pytest.approx(1.0, abs=0.01)


class TestPhaseSpace:
    def test_wigner_shape(self, two_qubit_system):
        hs, ops, sf = two_qubit_system
        psi = sf.ground_state()
        xvec = np.linspace(-3, 3, 50)
        W = motional_wigner(psi, mode_index=0, qubit_indices=[0, 1], xvec=xvec)
        assert W.shape == (50, 50)

    def test_vacuum_wigner_positive(self, two_qubit_system):
        """Vacuum state Wigner function should be a positive Gaussian."""
        hs, ops, sf = two_qubit_system
        psi = sf.ground_state()
        xvec = np.linspace(-3, 3, 50)
        W = motional_wigner(psi, mode_index=0, qubit_indices=[0, 1], xvec=xvec)
        assert W.min() >= -0.01  # vacuum Wigner is positive


class TestErrorBudget:
    def test_error_budget_returns_dict(self):
        budget = compute_error_budget(
            ideal_fidelity=0.999,
            heating_error=1e-4,
            dephasing_error=5e-5,
            scattering_error=2e-4,
            spam_error=5e-4,
        )
        assert isinstance(budget, dict)
        assert "total_error" in budget
        assert budget["total_error"] > 0

    def test_error_budget_sums(self):
        budget = compute_error_budget(
            ideal_fidelity=1.0,
            heating_error=1e-3,
            dephasing_error=2e-3,
            scattering_error=3e-3,
        )
        expected_total = 1e-3 + 2e-3 + 3e-3
        assert budget["total_error"] == pytest.approx(expected_total, rel=0.1)

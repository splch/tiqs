import numpy as np
import pytest
import qutip

from tiqs.analysis.error_budget import compute_error_budget
from tiqs.analysis.fidelity import (
    bell_state_fidelity,
    gate_fidelity,
    state_fidelity,
)
from tiqs.analysis.phase_space import motional_wigner, phase_space_trajectory
from tiqs.hilbert_space.builder import HilbertSpace
from tiqs.hilbert_space.operators import OperatorFactory
from tiqs.hilbert_space.states import StateFactory

# Two generic two-qubit pure states, written out rather than sampled so
# the floating-point behaviour of the fidelity path is reproducible.
# The first maximizes the error of the density-matrix (sqrtm) fidelity
# path against the exact overlap; the second is a state for which that
# path returns F(rho, rho) > 1.
_GENERIC_AMPLITUDES = (
    0.28369303481978836 - 0.3581318515276112j,
    -0.10837262744138018 + 0.1659103186484139j,
    0.3322191429542366 - 0.13524457243612684j,
    0.3357328978830815 + 0.7145710228269316j,
)
_SELF_FIDELITY_AMPLITUDES = (
    0.5692283825795541 + 0.2962374300148772j,
    -0.3783737297375999 + 0.07237106185723802j,
    0.06316606814598438 - 0.2977921969717854j,
    0.49202418737808806 - 0.32412997063953436j,
)


def _two_qubit_ket(amplitudes) -> qutip.Qobj:
    """Normalized two-qubit ket from four amplitudes."""
    data = np.array(amplitudes, dtype=complex).reshape(4, 1)
    return qutip.Qobj(data, dims=[[2, 2], [1, 1]]).unit()


def _bell_ket(sign: int) -> qutip.Qobj:
    """(|00> + sign*i|11>)/sqrt(2)."""
    ket_00 = qutip.tensor(qutip.basis(2, 0), qutip.basis(2, 0))
    ket_11 = qutip.tensor(qutip.basis(2, 1), qutip.basis(2, 1))
    return (ket_00 + sign * 1j * ket_11).unit()


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

    def test_mixed_pure_fidelity_is_exact_overlap(self):
        """F(rho, |psi>) must equal |<psi|phi>|^2 to machine precision.

        Born's rule is exact, so the only tolerance here is
        floating-point. Routing a pure state through ``sqrtm`` of a
        rank-deficient density matrix leaves an absolute error of
        order 1e-8, which this bound rejects.
        """
        psi = _two_qubit_ket(_GENERIC_AMPLITUDES)
        phi = _bell_ket(+1)
        exact = abs(phi.overlap(psi)) ** 2
        assert state_fidelity(qutip.ket2dm(psi), phi) == pytest.approx(
            exact, abs=1e-14
        )
        assert state_fidelity(phi, qutip.ket2dm(psi)) == pytest.approx(
            exact, abs=1e-14
        )

    def test_fidelity_never_exceeds_one(self):
        """Fidelity is a probability: F <= 1 for every input.

        ``qutip.fidelity`` on two rank-deficient density matrices can
        return 1 + 5e-8, which makes ``1 - F`` negative and poisons
        any error budget built from it.
        """
        rho = qutip.ket2dm(_two_qubit_ket(_SELF_FIDELITY_AMPLITUDES))
        assert qutip.fidelity(rho, rho) ** 2 > 1.0  # the raw hazard
        assert state_fidelity(rho, rho) <= 1.0
        assert state_fidelity(rho, rho) == pytest.approx(1.0, abs=1e-7)

    def test_bell_state_fidelity(self, two_qubit_system):
        hs, ops, sf = two_qubit_system
        rho_spin = qutip.ket2dm(_bell_ket(+1))
        fid = bell_state_fidelity(rho_spin)
        assert fid == pytest.approx(1.0, abs=1e-12)

    def test_bell_state_fidelity_sign_selects_conjugate_state(self):
        """The two Bell phases are orthogonal, so the sign matters.

        <B_+|B_-> = 0 exactly for
        |B_s> = (|00> + s i|11>)/sqrt(2), so a perfect gate that
        produces one of them scores 1 against its own sign and 0
        against the other.
        """
        for sign in (+1, -1):
            rho = qutip.ket2dm(_bell_ket(sign))
            assert bell_state_fidelity(rho, sign) == pytest.approx(
                1.0, abs=1e-12
            )
            assert bell_state_fidelity(rho, -sign) == pytest.approx(
                0.0, abs=1e-12
            )

    def test_bell_state_fidelity_accepts_ket(self):
        """A ket argument gives the same number as its density matrix."""
        ket = _bell_ket(-1)
        assert bell_state_fidelity(ket, -1) == pytest.approx(
            bell_state_fidelity(qutip.ket2dm(ket), -1), abs=1e-14
        )

    def test_bell_state_fidelity_rejects_bad_sign(self):
        rho = qutip.ket2dm(_bell_ket(+1))
        with pytest.raises(ValueError, match="sign must be"):
            bell_state_fidelity(rho, sign=0)

    def test_gate_fidelity_with_motion(self, two_qubit_system):
        """Gate fidelity should trace out motional modes before comparing."""
        hs, ops, sf = two_qubit_system
        psi = sf.ground_state()
        rho_full = qutip.ket2dm(psi)
        target = qutip.ket2dm(
            qutip.tensor(qutip.basis(2, 0), qutip.basis(2, 0))
        )
        fid = gate_fidelity(rho_full, target, qubit_indices=[0, 1])
        assert fid == pytest.approx(1.0, abs=1e-12)

    def test_gate_fidelity_ket_target_exact(self, two_qubit_system):
        """A ket target uses <psi|rho|psi>, which has an exact answer.

        For rho = p|00><00| + (1-p)|11><11| and the Bell target
        (|00> + i|11>)/sqrt(2), <B|rho|B> = p/2 + (1-p)/2 = 1/2 for
        every p.
        """
        hs, ops, sf = two_qubit_system
        psi_00 = sf.product_state([0, 0], [0])
        psi_11 = sf.product_state([1, 1], [0])
        p = 0.3
        rho_full = p * qutip.ket2dm(psi_00) + (1 - p) * qutip.ket2dm(psi_11)
        fid = gate_fidelity(rho_full, _bell_ket(+1), qubit_indices=[0, 1])
        assert fid == pytest.approx(0.5, abs=1e-12)

    def test_gate_fidelity_honors_index_values(self, two_qubit_system):
        """A subset of qubit indices selects that subsystem, not a count."""
        hs, ops, sf = two_qubit_system
        psi = sf.product_state([0, 1], [0])
        rho = qutip.ket2dm(psi)
        assert gate_fidelity(
            rho, qutip.ket2dm(qutip.basis(2, 0)), qubit_indices=[0]
        ) == pytest.approx(1.0, abs=1e-12)
        assert gate_fidelity(
            rho, qutip.ket2dm(qutip.basis(2, 0)), qubit_indices=[1]
        ) == pytest.approx(0.0, abs=1e-12)


class TestPhaseSpace:
    def test_wigner_shape(self, two_qubit_system):
        hs, ops, sf = two_qubit_system
        psi = sf.ground_state()
        xvec = np.linspace(-3, 3, 50)
        W = motional_wigner(psi, mode_index=0, n_qubits=2, xvec=xvec)
        assert W.shape == (50, 50)

    def test_vacuum_wigner_matches_analytic_gaussian(self, two_qubit_system):
        """Vacuum Wigner function is W(x, p) = exp(-x^2-p^2)/pi.

        Exact for the harmonic-oscillator ground state in QuTiP's
        default (g = sqrt(2)) convention, so the peak is 1/pi and the
        function is everywhere positive.
        """
        hs, ops, sf = two_qubit_system
        psi = sf.ground_state()
        xvec = np.linspace(-3, 3, 61)
        W = motional_wigner(psi, mode_index=0, n_qubits=2, xvec=xvec)
        X, P = np.meshgrid(xvec, xvec)
        analytic = np.exp(-(X**2 + P**2)) / np.pi
        assert np.abs(W - analytic).max() < 1e-10
        assert W.min() >= 0.0

    def test_trajectory_matches_coherent_state_quadratures(self):
        """<x>, <p> of |alpha> are sqrt(2)Re(alpha), sqrt(2)Im(alpha).

        Reading the wrong subsystem returns the vacuum's (0, 0)
        instead, so this pins the mode-to-subsystem mapping.
        """
        alpha = 1 + 0.5j
        psi = qutip.tensor(
            qutip.basis(2, 0),
            qutip.basis(2, 0),
            qutip.basis(20, 0),
            qutip.coherent(20, alpha),
        )
        x, p = phase_space_trajectory([psi], mode_index=1, n_qubits=2)
        assert x[0] == pytest.approx(np.sqrt(2) * alpha.real, abs=1e-8)
        assert p[0] == pytest.approx(np.sqrt(2) * alpha.imag, abs=1e-8)
        x0, p0 = phase_space_trajectory([psi], mode_index=0, n_qubits=2)
        assert x0[0] == pytest.approx(0.0, abs=1e-12)
        assert p0[0] == pytest.approx(0.0, abs=1e-12)

    def test_qubit_index_list_rejected(self):
        """A qubit-index list is rejected instead of read for its length.

        A short list (``[0]`` for a two-qubit state) used to select a
        lower-numbered subsystem and return a plausible, normalized,
        wrong answer, and a list of nonsense indices of the right
        length returned the right answer.
        """
        psi = qutip.tensor(
            qutip.basis(2, 0),
            qutip.basis(2, 0),
            qutip.basis(20, 0),
            qutip.coherent(20, 1 + 0.5j),
        )
        for bad in ([0], [0, 1], [3, 7]):
            with pytest.raises(TypeError, match="n_qubits must be"):
                phase_space_trajectory([psi], mode_index=1, n_qubits=bad)
            with pytest.raises(TypeError, match="n_qubits must be"):
                motional_wigner(psi, mode_index=1, n_qubits=bad)

    def test_mode_index_out_of_range_rejected(self, two_qubit_system):
        hs, ops, sf = two_qubit_system
        psi = sf.ground_state()
        with pytest.raises(ValueError, match="mode_index 1 out of range"):
            motional_wigner(psi, mode_index=1, n_qubits=2)
        with pytest.raises(ValueError, match="mode_index -1 out of range"):
            phase_space_trajectory([psi], mode_index=-1, n_qubits=2)


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

    def test_error_budget_sums_exactly(self):
        """The total is a plain sum, so it must match bit-for-bit.

        Additive first-order error budgets are the convention in the
        literature (Ballance et al., PRL 117, 060504 (2016), Table 1
        sums 0.4 + 0.2 + 0.2 + 0.06 + 0.04 + 0.01 = 0.9, in 1e-3), so
        any tolerance here would only hide a coefficient error.
        """
        budget = compute_error_budget(
            ideal_fidelity=0.995,
            heating_error=1e-3,
            dephasing_error=2e-3,
            scattering_error=3e-3,
        )
        assert budget["ideal_infidelity"] == pytest.approx(5e-3, rel=1e-12)
        expected = 5e-3 + 1e-3 + 2e-3 + 3e-3
        assert budget["total_error"] == pytest.approx(expected, rel=1e-12)
        components = {k: v for k, v in budget.items() if k != "total_error"}
        assert sum(components.values()) == pytest.approx(
            budget["total_error"], rel=1e-12
        )

    def test_ballance_2016_budget_reproduced(self):
        """Reproduce the published two-qubit budget of Ballance 2016.

        Phys. Rev. Lett. 117, 060504 (2016), Table 1: photon
        scattering 0.4e-3, motional heating 0.2e-3, laser noise
        0.2e-3, motional dephasing 0.06e-3, crosstalk 0.04e-3,
        spin dephasing 0.01e-3, total 0.9e-3.
        """
        budget = compute_error_budget(
            scattering_error=0.4e-3,
            heating_error=0.2e-3,
            laser_noise_error=0.2e-3,
            motional_dephasing_error=0.06e-3,
            crosstalk_error=0.04e-3,
            dephasing_error=0.01e-3,
        )
        assert budget["total_error"] == pytest.approx(0.91e-3, rel=1e-12)

    def test_unphysical_fidelity_rejected(self):
        with pytest.raises(ValueError, match="ideal_fidelity"):
            compute_error_budget(ideal_fidelity=1.5)
        with pytest.raises(ValueError, match="ideal_fidelity"):
            compute_error_budget(ideal_fidelity=-0.1)

    def test_negative_error_contribution_rejected(self):
        with pytest.raises(ValueError, match="heating_error"):
            compute_error_budget(heating_error=-0.5)
        with pytest.raises(ValueError, match="spam_error"):
            compute_error_budget(spam_error=-1e-9)

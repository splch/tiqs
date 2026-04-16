"""Tests for motional potential types and energy level utilities."""

import numpy as np
import pytest

from tiqs.potential import (
    ArbitraryPotential,
    DuffingPotential,
    HarmonicPotential,
    Potential,
    check_convergence,
    energy_levels,
    transition_frequencies,
)


class TestPotentialProtocol:
    """Smoke tests for protocol attribute access.

    Full structural conformance is verified by running mypy --strict.
    """

    def test_harmonic_satisfies_protocol(self):
        pot = HarmonicPotential(omega=2 * np.pi * 1e6)

        def accepts_potential(p: Potential) -> float:
            return p.omega

        assert accepts_potential(pot) > 0


class TestHarmonicPotential:
    def test_equally_spaced_levels(self):
        """Harmonic oscillator has equally-spaced energy levels."""
        omega = 2 * np.pi * 1e6
        pot = HarmonicPotential(omega=omega)
        E = energy_levels(pot, n_fock=10)
        diffs = np.diff(E)
        np.testing.assert_allclose(diffs, omega, rtol=1e-10)

    def test_ground_state_energy_zero(self):
        """H = omega * n, so E_0 = 0 (no zero-point energy in
        this convention)."""
        pot = HarmonicPotential(omega=2 * np.pi * 1e6)
        E = energy_levels(pot, n_fock=10)
        assert E[0] == pytest.approx(0.0, abs=1e-20)

    def test_transition_frequencies_constant(self):
        """All transitions at omega for harmonic potential."""
        omega = 2 * np.pi * 5e6
        pot = HarmonicPotential(omega=omega)
        freqs = transition_frequencies(pot, n_fock=10)
        np.testing.assert_allclose(freqs, omega, rtol=1e-10)

    def test_hamiltonian_is_diagonal(self):
        """H = omega * n is diagonal in the Fock basis."""
        omega = 2 * np.pi * 1e6
        pot = HarmonicPotential(omega=omega)
        H = pot.single_mode_hamiltonian(5)
        H_dense = H.full()
        np.testing.assert_allclose(
            H_dense, np.diag(np.diag(H_dense)), atol=1e-20
        )


class TestDuffingPotential:
    def test_transmon_spectrum(self):
        """E(n->n+1) = omega + anharmonicity * n.

        |0>->|1> at omega, |1>->|2> at omega + anharmonicity."""
        omega = 2 * np.pi * 5e9
        alpha = -2 * np.pi * 300e6
        pot = DuffingPotential(omega=omega, anharmonicity=alpha)
        freqs = transition_frequencies(pot, n_fock=10)
        for n in range(5):
            expected = omega + alpha * n
            assert freqs[n] == pytest.approx(expected, rel=1e-10)

    def test_zero_anharmonicity_matches_harmonic(self):
        """DuffingPotential(alpha=0) == HarmonicPotential."""
        omega = 2 * np.pi * 1e6
        duffing = DuffingPotential(omega=omega, anharmonicity=0.0)
        harmonic = HarmonicPotential(omega=omega)
        E_d = energy_levels(duffing, n_fock=10)
        E_h = energy_levels(harmonic, n_fock=10)
        np.testing.assert_allclose(E_d, E_h, atol=1e-20)

    def test_negative_anharmonicity_subharmonic(self):
        """Negative alpha means higher transitions have LOWER
        frequency (transmon-like)."""
        omega = 2 * np.pi * 5e9
        alpha = -2 * np.pi * 300e6
        pot = DuffingPotential(omega=omega, anharmonicity=alpha)
        freqs = transition_frequencies(pot, n_fock=10)
        for i in range(len(freqs) - 1):
            assert freqs[i] > freqs[i + 1]

    def test_positive_anharmonicity_stiffening(self):
        """Positive alpha means higher transitions have HIGHER
        frequency (stiffening)."""
        omega = 2 * np.pi * 1e6
        alpha = 2 * np.pi * 50e3
        pot = DuffingPotential(omega=omega, anharmonicity=alpha)
        freqs = transition_frequencies(pot, n_fock=10)
        for i in range(len(freqs) - 1):
            assert freqs[i] < freqs[i + 1]

    def test_satisfies_protocol(self):
        pot = DuffingPotential(
            omega=2 * np.pi * 1e6, anharmonicity=-2 * np.pi * 50e3
        )

        def accepts_potential(p: Potential) -> float:
            return p.omega

        assert accepts_potential(pot) > 0

    def test_hamiltonian_is_diagonal(self):
        """Duffing Hamiltonian is diagonal in Fock basis."""
        pot = DuffingPotential(
            omega=2 * np.pi * 1e6, anharmonicity=-2 * np.pi * 50e3
        )
        H = pot.single_mode_hamiltonian(5)
        H_dense = H.full()
        np.testing.assert_allclose(
            H_dense, np.diag(np.diag(H_dense)), atol=1e-20
        )


class TestArbitraryPotential:
    def test_harmonic_v_matches_harmonic_potential(self):
        """ArbitraryPotential with V(q) = omega/4 * q^2 should
        produce the same spectrum as HarmonicPotential.

        The harmonic potential in dimensionless units is
        V(q) = omega/4 * q^2, which combined with the kinetic
        energy T = omega/4 * (2n+1-q^2) gives H = omega*(n+1/2)."""
        omega = 2 * np.pi * 1e6

        def v_harmonic(q_op):
            return omega / 4 * q_op * q_op

        arb = ArbitraryPotential(v_func=v_harmonic, omega=omega)
        harm = HarmonicPotential(omega=omega)
        E_arb = energy_levels(arb, n_fock=15)
        E_harm = energy_levels(harm, n_fock=15)
        E_arb_shifted = E_arb - E_arb[0]
        np.testing.assert_allclose(E_arb_shifted, E_harm, rtol=1e-8)

    def test_quartic_perturbation_theory(self):
        """First-order perturbation theory for V(q) = omega/4*q^2 + lam*q^4:

        The quartic perturbation is lam*q^4 and
        <n|q^4|n> = <n|(a+a_dag)^4|n> = 6n^2 + 6n + 3.

        So E_n^(1) = lam * (6n^2 + 6n + 3).

        Use small lam (lam << omega) so perturbation theory is
        accurate to ~1%."""
        omega = 2 * np.pi * 1e6
        lam = omega * 1e-4

        def v_quartic(q_op):
            return omega / 4 * q_op * q_op + lam * q_op**4

        arb = ArbitraryPotential(v_func=v_quartic, omega=omega)
        E = energy_levels(arb, n_fock=30)
        E_shifted = E - E[0]
        for n_level in range(1, 4):
            E_harmonic = n_level * omega
            correction = lam * 6 * n_level * (n_level + 1)
            expected = E_harmonic + correction
            assert E_shifted[n_level] == pytest.approx(expected, rel=0.01)

    def test_satisfies_protocol(self):
        omega = 2 * np.pi * 1e6

        def v_simple(q_op):
            return omega / 4 * q_op * q_op

        pot = ArbitraryPotential(v_func=v_simple, omega=omega)

        def accepts_potential(p: Potential) -> float:
            return p.omega

        assert accepts_potential(pot) > 0


class TestCheckConvergence:
    def test_harmonic_always_converged(self):
        pot = HarmonicPotential(omega=2 * np.pi * 1e6)
        assert check_convergence(pot, n_fock=10)

    def test_duffing_converged_at_reasonable_truncation(self):
        pot = DuffingPotential(
            omega=2 * np.pi * 5e9,
            anharmonicity=-2 * np.pi * 300e6,
        )
        assert check_convergence(pot, n_fock=10)

    def test_warns_on_insufficient_truncation(self):
        """A very strongly anharmonic potential should warn at
        low n_fock."""
        omega = 2 * np.pi * 1e6

        def v_strong_quartic(q_op):
            return omega / 4 * q_op * q_op + omega * 100 * q_op**4

        pot = ArbitraryPotential(v_func=v_strong_quartic, omega=omega)
        with pytest.warns(UserWarning, match="not converged"):
            check_convergence(pot, n_fock=5)


class TestModeHamiltonian:
    def test_lifts_to_full_space(self):
        """mode_hamiltonian produces an operator in the full
        tensor-product space."""
        from tiqs.hilbert_space.builder import HilbertSpace
        from tiqs.hilbert_space.operators import OperatorFactory
        from tiqs.potential import mode_hamiltonian

        hs = HilbertSpace(n_ions=1, n_modes=1, n_fock=10)
        ops = OperatorFactory(hs)
        pot = HarmonicPotential(omega=2 * np.pi * 1e6)
        H = mode_hamiltonian(pot, ops, mode=0)
        assert H.shape == (hs.total_dim, hs.total_dim)

    def test_harmonic_matches_omega_times_number(self):
        """mode_hamiltonian with HarmonicPotential should equal
        omega * ops.number(mode)."""
        from tiqs.hilbert_space.builder import HilbertSpace
        from tiqs.hilbert_space.operators import OperatorFactory
        from tiqs.potential import mode_hamiltonian

        omega = 2 * np.pi * 1e6
        hs = HilbertSpace(n_ions=1, n_modes=1, n_fock=10)
        ops = OperatorFactory(hs)
        pot = HarmonicPotential(omega=omega)
        H_pot = mode_hamiltonian(pot, ops, mode=0)
        H_expected = omega * ops.number(0)
        np.testing.assert_allclose(H_pot.full(), H_expected.full(), atol=1e-20)

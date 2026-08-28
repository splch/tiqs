"""Tests for motional potential types and energy level utilities."""

import numpy as np
import pytest
import qutip

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

    @pytest.mark.parametrize("n_fock", [10, 19, 25, 40])
    def test_ladder_exact_across_spectrum_turnover(self, n_fock):
        """The Kerr ladder is exactly omega + alpha*n at every n.

        E_n = omega*n + (alpha/2)*n*(n-1) is the closed-form Kerr
        spectrum, exact because H is diagonal in Fock, so the
        |n> -> |n+1> gap is omega + alpha*n for all n. For alpha < 0
        the ladder turns over at n = omega/|alpha| (16.67 here) and
        the gaps go negative; energy-ascending eigenvalue order then
        stops matching Fock order.

        Differencing sorted eigenvalues cannot express this: np.diff
        of an ascending array is non-negative by construction, so the
        negative gaps are unrepresentable there.
        """
        omega = 2 * np.pi * 5e9
        alpha = -2 * np.pi * 300e6
        pot = DuffingPotential(omega=omega, anharmonicity=alpha)
        freqs = transition_frequencies(pot, n_fock=n_fock)
        expected = omega + alpha * np.arange(n_fock - 1)
        np.testing.assert_allclose(freqs, expected, rtol=1e-10)
        # Above the turnover the ladder must actually be inverted.
        if n_fock > omega / abs(alpha) + 2:
            assert freqs[-1] < 0


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
        """First-order perturbation theory for V(q) = omega/4*q^2 + lam*q^4.

        The quartic perturbation is lam*q^4 and
        <n|q^4|n> = <n|(a+a_dag)^4|n> = 6n^2 + 6n + 3, so measured
        from the ground state E_n - E_0 = n*omega + lam*6n(n+1).

        The assertion is on the CORRECTION, not on the total level
        energy: at lam = 1e-4*omega the correction is only ~1e-3 of
        n*omega, so a 1% tolerance on the total is satisfied by the
        harmonic spectrum alone (see the negative control below) and
        tests nothing about the quartic term. First-order PT is
        accurate to 0.24-0.43% on the correction itself at these
        parameters, so 1% there is a real 2.3x-headroom constraint.
        """
        omega = 2 * np.pi * 1e6
        lam = omega * 1e-4

        def v_quartic(q_op):
            return omega / 4 * q_op * q_op + lam * q_op**4

        arb = ArbitraryPotential(v_func=v_quartic, omega=omega)
        E = energy_levels(arb, n_fock=30)
        E_shifted = E - E[0]
        for n_level in range(1, 4):
            correction = lam * 6 * n_level * (n_level + 1)
            measured = E_shifted[n_level] - n_level * omega
            assert measured == pytest.approx(correction, rel=0.01)

    def test_quartic_negative_control_rejects_harmonic(self):
        """Dropping the quartic term must fail the PT assertion.

        Guards test_quartic_perturbation_theory against re-loosening:
        the harmonic-only spectrum gives zero correction, a 100%
        deviation. It also catches a mis-normalized coordinate - if q
        were (a + a_dag)/sqrt(2) instead of the documented
        a + a_dag, the quartic shift would come out 4x too small,
        which the total-energy form of this assertion would miss.
        """
        omega = 2 * np.pi * 1e6
        lam = omega * 1e-4

        def v_harmonic_only(q_op):
            return omega / 4 * q_op * q_op

        arb = ArbitraryPotential(v_func=v_harmonic_only, omega=omega)
        E = energy_levels(arb, n_fock=30)
        E_shifted = E - E[0]
        for n_level in range(1, 4):
            correction = lam * 6 * n_level * (n_level + 1)
            measured = E_shifted[n_level] - n_level * omega
            assert measured != pytest.approx(correction, rel=0.01)

    def test_quartic_residual_scales_as_lambda_squared(self):
        """The PT-1 residual is second order in lam.

        Rayleigh-Schrodinger theory: E_n = E_n^(0) + lam*E_n^(1)
        + lam^2*E_n^(2) + ..., so the leftover after subtracting the
        first-order shift must fall by 4x when lam is halved. This
        law is independent of the implementation and fails for any
        wrong power of the coordinate in q^4.
        """
        omega = 2 * np.pi * 1e6
        lam = omega * 1e-4

        def residual(scale, n_level):
            def v_quartic(q_op):
                return omega / 4 * q_op * q_op + scale * q_op**4

            arb = ArbitraryPotential(v_func=v_quartic, omega=omega)
            E = energy_levels(arb, n_fock=40)
            first_order = scale * 6 * n_level * (n_level + 1)
            return (E[n_level] - E[0]) - n_level * omega - first_order

        for n_level in range(1, 4):
            ratio = residual(lam, n_level) / residual(lam / 2, n_level)
            assert ratio == pytest.approx(4.0, rel=0.02)

    def test_satisfies_protocol(self):
        omega = 2 * np.pi * 1e6

        def v_simple(q_op):
            return omega / 4 * q_op * q_op

        pot = ArbitraryPotential(v_func=v_simple, omega=omega)

        def accepts_potential(p: Potential) -> float:
            return p.omega

        assert accepts_potential(pot) > 0

    def test_rejects_non_hermitian_v(self):
        """A complex V(q) has no real spectrum and must be rejected.

        Previously T + V was returned unchecked and energy_levels
        discarded the imaginary parts, reporting a plausible-looking
        real spectrum for V(q) = i*omega*q.
        """
        omega = 2 * np.pi * 1e6
        pot = ArbitraryPotential(
            v_func=lambda q_op: 1j * omega * q_op, omega=omega
        )
        with pytest.raises(ValueError, match="non-Hermitian"):
            pot.single_mode_hamiltonian(6)

    @pytest.mark.parametrize("n_fock", [20, 40])
    def test_accepts_hermitian_double_well(self, n_fock):
        """A real V(q) is Hermitian up to round-off and must pass.

        The Hermiticity check has to be relative: matrix entries are
        ~1e9 rad/s, so float round-off leaves max|H - H^dag| ~ 1e-7
        absolute - well above QuTiP's absolute ``isherm`` tolerance
        of 1e-12, but ~1e-17 in relative terms.

        Physics anchor: V(-q) = V(q), so H commutes exactly with the
        parity operator (-1)^n. That holds only if both T and V are
        assembled correctly.
        """
        omega = 2 * np.pi * 1e6

        def v_double_well(q_op):
            return -omega / 2 * q_op * q_op + 0.05 * omega * q_op**4

        pot = ArbitraryPotential(v_func=v_double_well, omega=omega)
        H = pot.single_mode_hamiltonian(n_fock)
        dense = H.full()
        scale = np.max(np.abs(dense))
        assert np.max(np.abs(dense - dense.conj().T)) / scale < 1e-14

        parity = qutip.Qobj(np.diag((-1.0) ** np.arange(n_fock)))
        commutator = (H * parity - parity * H).full()
        assert np.max(np.abs(commutator)) / scale < 1e-14

    def test_transition_frequencies_warns_when_not_diagonal(self):
        """Fock states are not eigenstates of a quartic potential.

        No |n> -> |n+1> ladder exists, so the caller must be told
        that the returned gaps are between sorted eigenvalues.
        """
        omega = 2 * np.pi * 1e6

        def v_quartic(q_op):
            return omega / 4 * q_op * q_op + omega * 1e-4 * q_op**4

        pot = ArbitraryPotential(v_func=v_quartic, omega=omega)
        with pytest.warns(UserWarning, match="not diagonal in the Fock"):
            freqs = transition_frequencies(pot, n_fock=20)
        assert len(freqs) == 19
        assert np.all(freqs >= 0.0)


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

    def test_doubling_step_catches_unbounded_potential(self):
        """A quartic with the wrong sign has no ground state.

        V = omega/4*q^2 - 1e-3*omega*q^4 is unbounded below, so no
        truncation is converged. The escape into the runaway region
        is slower than five extra Fock levels, so the old additive
        n_fock + 5 comparison declared convergence at n_fock = 40
        (asserted below to keep the regression honest); comparing
        n_fock against 2*n_fock exposes it.
        """
        omega = 2 * np.pi * 1e6

        def v_unbounded(q_op):
            return omega / 4 * q_op * q_op - 1e-3 * omega * q_op**4

        pot = ArbitraryPotential(v_func=v_unbounded, omega=omega)
        additive = energy_levels(pot, 45)[:5]
        np.testing.assert_allclose(
            energy_levels(pot, 40)[:5], additive, rtol=1e-6
        )
        with pytest.warns(UserWarning, match="not converged"):
            assert not check_convergence(pot, n_fock=40)


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

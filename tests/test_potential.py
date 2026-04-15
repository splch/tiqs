"""Tests for motional potential types and energy level utilities."""

import numpy as np
import pytest
import qutip

from tiqs.potential import (
    DuffingPotential,
    HarmonicPotential,
    Potential,
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

"""Tests for motional potential types and energy level utilities."""

import numpy as np
import pytest
import qutip

from tiqs.potential import (
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

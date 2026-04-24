"""Tests for the Verdu frequency-shifts matrix (elliptical Penning trap).

Validates the orbit_params and frequency_shifts_matrix functions
against analytical properties and the Verdu (2011) reference trap.
"""

import numpy as np
import pytest

from tiqs.constants import ELECTRON_CHARGE, ELECTRON_MASS, TWO_PI
from tiqs.elliptical import (
    AnharmonicCoeffs,
    frequency_shifts_matrix,
    orbit_params,
)
from tiqs.species.electron import ElectronSpecies
from tiqs.trap import PenningTrap


@pytest.fixture
def verdu_trap():
    """Verdu's reference CPW trap: B=0.5T, nu_z=28 MHz, epsilon=0.41."""
    species = ElectronSpecies(magnetic_field=0.5)
    return PenningTrap(
        magnetic_field=0.5,
        species=species,
        d=3.5e-3,
        omega_axial=TWO_PI * 28e6,
        epsilon=0.41,
    )


class TestOrbitParams:
    def test_circular_xi_eta_unity(self):
        """At epsilon=0, xi_p = eta_p = 1 (circular orbits)."""
        species = ElectronSpecies(magnetic_field=5.0)
        trap = PenningTrap(
            magnetic_field=5.0,
            species=species,
            d=3.5e-3,
            omega_axial=TWO_PI * 200e6,
        )
        orb = orbit_params(
            trap.omega_cyclotron,
            trap.omega_axial,
            trap.omega_modified_cyclotron,
            0.0,
        )
        assert orb.xi_p == pytest.approx(1.0, rel=1e-6)
        assert orb.eta_p == pytest.approx(1.0, rel=1e-6)
        assert orb.xi_m == pytest.approx(orb.eta_m, rel=1e-6)

    def test_gamma_p_near_unity(self):
        """gamma_p = 1 - omega_z^2/(2*omega_+^2) is close to 1
        for high-field traps."""
        species = ElectronSpecies(magnetic_field=5.0)
        trap = PenningTrap(
            magnetic_field=5.0,
            species=species,
            d=3.5e-3,
            omega_axial=TWO_PI * 200e6,
        )
        orb = orbit_params(
            trap.omega_cyclotron,
            trap.omega_axial,
            trap.omega_modified_cyclotron,
            0.0,
        )
        assert orb.gamma_p == pytest.approx(1.0, abs=1e-4)

    def test_ellipticity_breaks_xi_eta_symmetry(self, verdu_trap):
        """At epsilon != 0, xi_m != eta_m (elliptical orbits)."""
        orb = orbit_params(
            verdu_trap.omega_cyclotron,
            verdu_trap.omega_axial,
            verdu_trap.omega_modified_cyclotron,
            0.41,
        )
        assert orb.xi_p == pytest.approx(1.0, rel=1e-4)
        assert orb.eta_p == pytest.approx(1.0, rel=1e-4)
        assert orb.xi_m != pytest.approx(orb.eta_m, rel=0.1)


class TestFrequencyShiftsMatrix:
    def test_zero_coeffs_gives_zero_matrix(self, verdu_trap):
        """No anharmonicities -> zero shifts."""
        orb = orbit_params(
            verdu_trap.omega_cyclotron,
            verdu_trap.omega_axial,
            verdu_trap.omega_modified_cyclotron,
            0.41,
        )
        coeffs = AnharmonicCoeffs(c002=1.0)
        M = frequency_shifts_matrix(
            verdu_trap.omega_modified_cyclotron / TWO_PI,
            verdu_trap.omega_axial / TWO_PI,
            verdu_trap.omega_magnetron / TWO_PI,
            orb,
            coeffs,
            ELECTRON_MASS,
        )
        np.testing.assert_allclose(M, 0.0, atol=1e-30)

    def test_c004_only_diagonal(self, verdu_trap):
        """C_004 only affects M[1,1] (axial self-shift)."""
        orb = orbit_params(
            verdu_trap.omega_cyclotron,
            verdu_trap.omega_axial,
            verdu_trap.omega_modified_cyclotron,
            0.41,
        )
        coeffs = AnharmonicCoeffs(c002=1.0, c004=1e10)
        nu_p = verdu_trap.omega_modified_cyclotron / TWO_PI
        nu_z = verdu_trap.omega_axial / TWO_PI
        nu_m = verdu_trap.omega_magnetron / TWO_PI
        M = frequency_shifts_matrix(
            nu_p, nu_z, nu_m, orb, coeffs, ELECTRON_MASS
        )
        # Only M[1,1] should be nonzero
        expected = (
            3
            * ELECTRON_CHARGE
            * 1e10
            / (16 * np.pi**4 * ELECTRON_MASS**2 * nu_z**3)
        )
        assert M[1, 1] == pytest.approx(expected, rel=1e-10)
        M_off = M.copy()
        M_off[1, 1] = 0
        np.testing.assert_allclose(M_off, 0.0, atol=1e-10)

    def test_m400_sparsity(self, verdu_trap):
        """M^400 has nonzero elements only at [0,0], [0,2], [2,0], [2,2]."""
        orb = orbit_params(
            verdu_trap.omega_cyclotron,
            verdu_trap.omega_axial,
            verdu_trap.omega_modified_cyclotron,
            0.41,
        )
        coeffs = AnharmonicCoeffs(c002=1.0, c400=1e15)
        nu_p = verdu_trap.omega_modified_cyclotron / TWO_PI
        nu_z = verdu_trap.omega_axial / TWO_PI
        nu_m = verdu_trap.omega_magnetron / TWO_PI
        M = frequency_shifts_matrix(
            nu_p, nu_z, nu_m, orb, coeffs, ELECTRON_MASS
        )
        assert M[0, 0] != 0
        assert M[0, 2] != 0
        assert M[2, 0] != 0
        assert M[2, 2] != 0
        assert M[1, 0] == pytest.approx(0.0, abs=1e-10)
        assert M[1, 1] == pytest.approx(0.0, abs=1e-10)
        assert M[1, 2] == pytest.approx(0.0, abs=1e-10)

    def test_m012_sparsity(self, verdu_trap):
        """M^012 has nonzero elements at [0,1], [1,0], [1,1],
        [1,2], [2,1]."""
        orb = orbit_params(
            verdu_trap.omega_cyclotron,
            verdu_trap.omega_axial,
            verdu_trap.omega_modified_cyclotron,
            0.41,
        )
        coeffs = AnharmonicCoeffs(c002=1.0, c012=1e8)
        nu_p = verdu_trap.omega_modified_cyclotron / TWO_PI
        nu_z = verdu_trap.omega_axial / TWO_PI
        nu_m = verdu_trap.omega_magnetron / TWO_PI
        M = frequency_shifts_matrix(
            nu_p, nu_z, nu_m, orb, coeffs, ELECTRON_MASS
        )
        assert M[0, 1] != 0
        assert M[1, 0] != 0
        assert M[1, 1] != 0
        assert M[1, 2] != 0
        assert M[2, 1] != 0
        assert M[0, 0] == pytest.approx(0.0, abs=1e-10)
        assert M[0, 2] == pytest.approx(0.0, abs=1e-10)

    def test_c012_scales_as_square(self, verdu_trap):
        """M^012 depends on C_012^2, so doubling C_012 quadruples M."""
        orb = orbit_params(
            verdu_trap.omega_cyclotron,
            verdu_trap.omega_axial,
            verdu_trap.omega_modified_cyclotron,
            0.41,
        )
        nu_p = verdu_trap.omega_modified_cyclotron / TWO_PI
        nu_z = verdu_trap.omega_axial / TWO_PI
        nu_m = verdu_trap.omega_magnetron / TWO_PI
        c1 = AnharmonicCoeffs(c002=1.0, c012=1e8)
        c2 = AnharmonicCoeffs(c002=1.0, c012=2e8)
        M1 = frequency_shifts_matrix(nu_p, nu_z, nu_m, orb, c1, ELECTRON_MASS)
        M2 = frequency_shifts_matrix(nu_p, nu_z, nu_m, orb, c2, ELECTRON_MASS)
        ratio = M2[1, 1] / M1[1, 1]
        assert ratio == pytest.approx(4.0, rel=1e-10)

    def test_superposition(self, verdu_trap):
        """M(c004 + c400) = M(c004) + M(c400) (linearity in sub-matrices)."""
        orb = orbit_params(
            verdu_trap.omega_cyclotron,
            verdu_trap.omega_axial,
            verdu_trap.omega_modified_cyclotron,
            0.41,
        )
        nu_p = verdu_trap.omega_modified_cyclotron / TWO_PI
        nu_z = verdu_trap.omega_axial / TWO_PI
        nu_m = verdu_trap.omega_magnetron / TWO_PI
        c_both = AnharmonicCoeffs(c002=1.0, c004=1e10, c400=1e15)
        c_004 = AnharmonicCoeffs(c002=1.0, c004=1e10)
        c_400 = AnharmonicCoeffs(c002=1.0, c400=1e15)
        M_both = frequency_shifts_matrix(
            nu_p, nu_z, nu_m, orb, c_both, ELECTRON_MASS
        )
        M_004 = frequency_shifts_matrix(
            nu_p, nu_z, nu_m, orb, c_004, ELECTRON_MASS
        )
        M_400 = frequency_shifts_matrix(
            nu_p, nu_z, nu_m, orb, c_400, ELECTRON_MASS
        )
        np.testing.assert_allclose(M_both, M_004 + M_400, rtol=1e-10)

"""Tests for ProtonSpecies and proton Penning trap physics.

Validates against the BASE experiment at CERN (Schneider et al.
Science 2017, Borchert et al. Nature 2022) and cross-checks
mass/frequency scaling against electrons.
"""

import numpy as np
import pytest

from tiqs.chain.equilibrium import equilibrium_positions
from tiqs.constants import (
    COULOMB_CONSTANT,
    ELECTRON_MASS,
    NUCLEAR_MAGNETON,
    PROTON_G_FACTOR,
    PROTON_MASS,
    TWO_PI,
)
from tiqs.species.electron import ElectronSpecies
from tiqs.species.proton import ProtonSpecies
from tiqs.trap import PenningTrap


class TestProtonSpecies:
    def test_mass(self):
        p = ProtonSpecies(magnetic_field=1.945)
        assert p.mass_kg == pytest.approx(1.6726e-27, rel=1e-3)

    def test_mass_ratio_to_electron(self):
        """m_p / m_e ~ 1836."""
        p = ProtonSpecies(magnetic_field=1.0)
        assert p.mass_kg / ELECTRON_MASS == pytest.approx(1836.2, rel=1e-3)

    def test_g_factor(self):
        p = ProtonSpecies(magnetic_field=1.0)
        assert p.g_factor == pytest.approx(5.5856946893, rel=1e-10)

    def test_larmor_frequency(self):
        """Larmor frequency at 1.945 T: ~83 MHz."""
        p = ProtonSpecies(magnetic_field=1.945)
        nu_L = p.qubit_frequency_hz
        expected = (
            PROTON_G_FACTOR * NUCLEAR_MAGNETON * 1.945
            / (1.054571817e-34 * TWO_PI)
        )
        assert nu_L == pytest.approx(expected, rel=1e-10)
        assert nu_L == pytest.approx(83e6, rel=0.01)

    def test_larmor_scales_with_field(self):
        p1 = ProtonSpecies(1.0)
        p2 = ProtonSpecies(2.0)
        assert p2.qubit_frequency_hz / p1.qubit_frequency_hz == (
            pytest.approx(2.0, rel=1e-10)
        )


class TestProtonPenningTrap:
    @pytest.fixture
    def base_trap(self):
        """BASE experiment: B = 1.945 T, nu_z ~ 630 kHz."""
        return PenningTrap(
            magnetic_field=1.945,
            species=ProtonSpecies(magnetic_field=1.945),
            d=1.8e-3,
            omega_axial=TWO_PI * 630e3,
            b2=300000.0,
        )

    def test_stability(self, base_trap):
        assert base_trap.is_stable()

    def test_cyclotron_frequency(self, base_trap):
        """nu_c ~ 29.65 MHz for protons at 1.945 T."""
        nu_c = base_trap.omega_cyclotron / TWO_PI
        assert nu_c == pytest.approx(29.65e6, rel=0.001)

    def test_modified_cyclotron(self, base_trap):
        """BASE reports nu_+ ~ 29.6 MHz."""
        nu_p = base_trap.omega_modified_cyclotron / TWO_PI
        assert nu_p == pytest.approx(29.6e6, rel=0.01)

    def test_magnetron(self, base_trap):
        """BASE reports nu_- ~ 7 kHz."""
        nu_m = base_trap.omega_magnetron / TWO_PI
        assert nu_m == pytest.approx(6.7e3, rel=0.1)

    def test_brown_gabrielse_invariant(self, base_trap):
        wc = base_trap.omega_cyclotron
        wp = base_trap.omega_modified_cyclotron
        wm = base_trap.omega_magnetron
        wz = base_trap.omega_axial
        assert wp**2 + wm**2 + wz**2 == pytest.approx(wc**2, rel=1e-10)

    def test_spin_flip_shift(self, base_trap):
        """Spin-flip shift uses proton g-factor, not electron.
        BASE reports ~170 mHz."""
        shift_up = base_trap.axial_frequency_shift(0, +0.5)
        shift_dn = base_trap.axial_frequency_shift(0, -0.5)
        spin_shift_hz = (shift_up - shift_dn) / TWO_PI
        assert 100e-3 < spin_shift_hz < 300e-3

    def test_g_over_2_from_shift_ratio(self, base_trap):
        """Ratio of spin-flip to cyclotron shift gives g_p/2."""
        spin = (
            base_trap.axial_frequency_shift(0, +0.5)
            - base_trap.axial_frequency_shift(0, -0.5)
        )
        cyc = (
            base_trap.axial_frequency_shift(1, -0.5)
            - base_trap.axial_frequency_shift(0, -0.5)
        )
        assert spin / cyc == pytest.approx(
            PROTON_G_FACTOR / 2, rel=1e-12
        )


class TestProtonVsElectronScaling:
    """Cross-species validation: same B field, same physics,
    different mass."""

    def test_cyclotron_scales_as_inverse_mass(self):
        """nu_c(e) / nu_c(p) = m_p / m_e."""
        B = 1.945
        trap_e = PenningTrap(
            magnetic_field=B,
            species=ElectronSpecies(magnetic_field=B),
            d=1.8e-3,
            omega_axial=TWO_PI * 630e3,
        )
        trap_p = PenningTrap(
            magnetic_field=B,
            species=ProtonSpecies(magnetic_field=B),
            d=1.8e-3,
            omega_axial=TWO_PI * 630e3,
        )
        ratio = trap_e.omega_cyclotron / trap_p.omega_cyclotron
        expected = PROTON_MASS / ELECTRON_MASS
        assert ratio == pytest.approx(expected, rel=1e-6)

    def test_spacing_scales_as_mass_third(self):
        """At same omega_z, spacing ~ m^(-1/3)."""
        omega_z = TWO_PI * 630e3
        trap_e = PenningTrap(
            magnetic_field=5.0,
            species=ElectronSpecies(magnetic_field=5.0),
            d=1.8e-3,
            omega_axial=omega_z,
        )
        trap_p = PenningTrap(
            magnetic_field=5.0,
            species=ProtonSpecies(magnetic_field=5.0),
            d=1.8e-3,
            omega_axial=omega_z,
        )
        d_e = equilibrium_positions(2, trap_e)
        d_p = equilibrium_positions(2, trap_p)
        ratio = (d_e[1] - d_e[0]) / (d_p[1] - d_p[0])
        expected = (PROTON_MASS / ELECTRON_MASS) ** (1 / 3)
        assert ratio == pytest.approx(expected, rel=0.001)

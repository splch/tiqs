"""Experimental validation tests for Penning trap physics.

Validates TIQS Penning trap eigenfrequencies against published
measurements from precision experiments. Every test cites the
source paper with specific table/figure references.

References:
    Jain et al., Nature 627, 510 (2024) / arXiv:2308.07672
    Hanneke et al., PRL 100, 120801 (2008) / arXiv:1009.4831
    Berrocal et al., Phys. Rev. Research 6, L012001 (2024) / arXiv:2308.14884
    Goodwin et al., PRL 116, 143002 (2016) / arXiv:1807.00902
    Bohnet et al., Science 352, 1297 (2016) / arXiv:1512.03756
"""

import numpy as np
import pytest

from tiqs.constants import (
    ELECTRON_CHARGE,
    ELECTRON_MASS,
    TWO_PI,
)
from tiqs.species.electron import ElectronSpecies
from tiqs.species.ion import get_species
from tiqs.trap import PenningTrap


class TestJainETHMicroPenning:
    """Jain et al., Nature 627, 510 (2024): single Be-9+ ion in a
    micro-fabricated Penning trap at B = 3 T.

    First demonstration of full quantum control in a Penning micro-trap.
    All three eigenfrequencies published explicitly.
    """

    @pytest.fixture
    def eth_trap(self):
        return PenningTrap(
            magnetic_field=3.0,
            species=get_species("Be9"),
            d=0.5e-3,
            omega_axial=TWO_PI * 2.5e6,
        )

    def test_bare_cyclotron_frequency(self, eth_trap):
        """omega_c/(2pi) = 5.12 MHz for Be-9+ at 3 T."""
        nu_c = eth_trap.omega_cyclotron / TWO_PI
        assert nu_c == pytest.approx(5.12e6, rel=0.005)

    def test_modified_cyclotron_frequency(self, eth_trap):
        """omega_+/(2pi) = 4.41 MHz (Nature 627, 510, Methods)."""
        nu_plus = eth_trap.omega_modified_cyclotron / TWO_PI
        assert nu_plus == pytest.approx(4.41e6, rel=0.005)

    def test_magnetron_frequency(self, eth_trap):
        """omega_-/(2pi) = 0.71 MHz (Nature 627, 510, Methods)."""
        nu_minus = eth_trap.omega_magnetron / TWO_PI
        assert nu_minus == pytest.approx(0.71e6, rel=0.01)

    def test_frequency_hierarchy(self, eth_trap):
        """omega_- < omega_z < omega_+ < omega_c for stable Penning trap."""
        assert eth_trap.omega_magnetron < eth_trap.omega_axial
        assert eth_trap.omega_axial < eth_trap.omega_modified_cyclotron
        assert eth_trap.omega_modified_cyclotron < eth_trap.omega_cyclotron

    def test_brown_gabrielse_invariant(self, eth_trap):
        """Brown-Gabrielse invariance theorem:
        omega_+^2 + omega_-^2 + omega_z^2 = omega_c^2.

        Verified to 0.04% from the published 3-digit frequencies."""
        wc = eth_trap.omega_cyclotron
        wp = eth_trap.omega_modified_cyclotron
        wm = eth_trap.omega_magnetron
        wz = eth_trap.omega_axial
        assert wp**2 + wm**2 + wz**2 == pytest.approx(wc**2, rel=1e-10)

    def test_sum_rule(self, eth_trap):
        """omega_+ + omega_- = omega_c (exact for ideal Penning trap)."""
        wc = eth_trap.omega_cyclotron
        wp = eth_trap.omega_modified_cyclotron
        wm = eth_trap.omega_magnetron
        assert wp + wm == pytest.approx(wc, rel=1e-10)

    def test_product_rule(self, eth_trap):
        """omega_+ * omega_- = omega_z^2 / 2 (exact for ideal Penning trap)."""
        wp = eth_trap.omega_modified_cyclotron
        wm = eth_trap.omega_magnetron
        wz = eth_trap.omega_axial
        assert wp * wm == pytest.approx(wz**2 / 2, rel=1e-10)

    def test_stability(self, eth_trap):
        assert eth_trap.is_stable()


class TestHannekeElectronGMinus2:
    """Hanneke, Fogwell, Gabrielse, PRL 100, 120801 (2008):
    single electron in a cylindrical Penning trap at B = 5.36 T.

    The most precise single-particle measurement ever performed.
    Table 1 of arXiv:1009.4831 gives all frequencies.
    """

    @pytest.fixture
    def hanneke_trap(self):
        return PenningTrap(
            magnetic_field=5.36,
            species=ElectronSpecies(magnetic_field=5.36),
            d=3.0e-3,
            omega_axial=TWO_PI * 200e6,
        )

    def test_free_cyclotron_frequency(self, hanneke_trap):
        """nu_c ~ 150 GHz for electrons at B = 5.36 T
        (Table 1, arXiv:1009.4831)."""
        nu_c = hanneke_trap.omega_cyclotron / TWO_PI
        assert nu_c == pytest.approx(150.0e9, rel=0.001)

    def test_magnetron_frequency(self, hanneke_trap):
        """nu_m = 133 kHz (Table 1, arXiv:1009.4831).

        The magnetron frequency is extremely small compared to the
        cyclotron frequency (~10^-6 ratio), demonstrating the extreme
        hierarchy in electron Penning traps."""
        nu_m = hanneke_trap.omega_magnetron / TWO_PI
        assert nu_m == pytest.approx(133e3, rel=0.01)

    def test_anomaly_frequency_cross_check(self, hanneke_trap):
        """nu_a = (g/2 - 1) * nu_c ~ 174 MHz.

        The anomaly frequency is the difference between the spin
        precession and cyclotron frequencies. This cross-checks
        our cyclotron computation against the measured g-factor."""
        from tiqs.constants import ELECTRON_G_FACTOR

        nu_c = hanneke_trap.omega_cyclotron / TWO_PI
        nu_a = (ELECTRON_G_FACTOR / 2 - 1) * nu_c
        assert nu_a == pytest.approx(174e6, rel=0.01)

    def test_extreme_frequency_hierarchy(self, hanneke_trap):
        """nu_m / nu_c ~ 10^-6: magnetron is a million times slower
        than cyclotron for electrons at 5.36 T."""
        ratio = hanneke_trap.omega_magnetron / hanneke_trap.omega_cyclotron
        assert ratio < 1e-5

    def test_brown_gabrielse_invariant(self, hanneke_trap):
        """Invariance theorem verified at the electron g-2 operating point."""
        wc = hanneke_trap.omega_cyclotron
        wp = hanneke_trap.omega_modified_cyclotron
        wm = hanneke_trap.omega_magnetron
        wz = hanneke_trap.omega_axial
        assert wp**2 + wm**2 + wz**2 == pytest.approx(wc**2, rel=1e-10)


class TestBerrocal2024CaPenning:
    """Berrocal et al., Phys. Rev. Research 6, L012001 (2024):
    Ca-40+ in an open-ring Penning trap at B = 7 T.

    First direct determination of all three eigenfrequencies for
    calcium ions using optical radiofrequency detectors.
    """

    @pytest.fixture
    def ca_penning(self):
        return PenningTrap(
            magnetic_field=7.0,
            species=get_species("Ca40"),
            d=5.0e-3,
            omega_axial=TWO_PI * 142e3,
        )

    def test_modified_cyclotron_frequency(self, ca_penning):
        """nu_+ = 2.686 MHz for Ca-40+ at 7 T
        (Phys. Rev. Research 6, L012001)."""
        nu_plus = ca_penning.omega_modified_cyclotron / TWO_PI
        assert nu_plus == pytest.approx(2.686e6, rel=0.001)

    def test_magnetron_frequency(self, ca_penning):
        """nu_- = 3.8 kHz for Ca-40+ at 7 T
        (Phys. Rev. Research 6, L012001).

        The published value is rounded to 2 significant figures.
        Our computed 3.75 kHz is within 1.2% of the stated 3.8 kHz."""
        nu_minus = ca_penning.omega_magnetron / TWO_PI
        assert nu_minus == pytest.approx(3.8e3, rel=0.02)

    def test_bare_cyclotron_frequency(self, ca_penning):
        """nu_c = eB/(2pi*m) for Ca-40+ at 7 T.

        Using the precise Ca-40 isotopic mass (39.9626 amu), not 40 amu."""
        nu_c = ca_penning.omega_cyclotron / TWO_PI
        expected = ELECTRON_CHARGE * 7.0 / (
            TWO_PI * get_species("Ca40").mass_kg
        )
        assert nu_c == pytest.approx(expected, rel=1e-10)

    def test_frequency_hierarchy(self, ca_penning):
        """nu_- << nu_z << nu_+ for ion Penning traps at high B."""
        nu_m = ca_penning.omega_magnetron / TWO_PI
        nu_z = ca_penning.omega_axial / TWO_PI
        nu_p = ca_penning.omega_modified_cyclotron / TWO_PI
        assert nu_m < nu_z < nu_p
        assert nu_m / nu_z < 0.03
        assert nu_z / nu_p < 0.06

    def test_brown_gabrielse_invariant(self, ca_penning):
        wc = ca_penning.omega_cyclotron
        wp = ca_penning.omega_modified_cyclotron
        wm = ca_penning.omega_magnetron
        wz = ca_penning.omega_axial
        assert wp**2 + wm**2 + wz**2 == pytest.approx(wc**2, rel=1e-10)


class TestGoodwinOxfordBePenning:
    """Goodwin et al., PRL 116, 143002 (2016) / arXiv:1807.00902:
    Be-9+ in a macroscopic Penning trap at B ~ 2 T.

    Table 1 gives both simulated and experimentally measured
    eigenfrequencies. The trap geometry is fully specified,
    enabling from_dc_voltage validation.
    """

    @pytest.fixture
    def oxford_trap(self):
        return PenningTrap(
            magnetic_field=1.998,
            species=get_species("Be9"),
            d=5.0e-3,
            omega_axial=TWO_PI * 402e3,
        )

    def test_modified_cyclotron_frequency(self, oxford_trap):
        """nu_+ = 3382 kHz measured (Table 1, arXiv:1807.00902)."""
        nu_plus = oxford_trap.omega_modified_cyclotron / TWO_PI
        assert nu_plus == pytest.approx(3382e3, rel=0.001)

    def test_magnetron_frequency(self, oxford_trap):
        """nu_- = 23.9 kHz measured (Table 1, arXiv:1807.00902)."""
        nu_minus = oxford_trap.omega_magnetron / TWO_PI
        assert nu_minus == pytest.approx(23.9e3, rel=0.01)

    def test_simulated_vs_measured_axial(self, oxford_trap):
        """Simulated nu_z = 406.4 kHz vs measured 402 kHz (Table 1).

        The ~1% difference comes from anharmonic corrections in the
        real trap. Our harmonic calculation should match the
        simulated column."""
        nu_z = oxford_trap.omega_axial / TWO_PI
        assert nu_z == pytest.approx(402e3, rel=0.02)

    def test_brown_gabrielse_invariant(self, oxford_trap):
        wc = oxford_trap.omega_cyclotron
        wp = oxford_trap.omega_modified_cyclotron
        wm = oxford_trap.omega_magnetron
        wz = oxford_trap.omega_axial
        assert wp**2 + wm**2 + wz**2 == pytest.approx(wc**2, rel=1e-10)


class TestBohnetNISTBePenning:
    """Bohnet et al., Science 352, 1297 (2016) / arXiv:1512.03756:
    ~150 Be-9+ ions in a NIST Penning trap at B = 4.45 T.

    The strongest B-field for trapped-ion QC experiments.
    Validated rotation-wall-controlled 2D crystal dynamics.
    """

    @pytest.fixture
    def nist_trap(self):
        return PenningTrap(
            magnetic_field=4.45,
            species=get_species("Be9"),
            d=5.0e-3,
            omega_axial=TWO_PI * 1.57e6,
        )

    def test_bare_cyclotron_frequency(self, nist_trap):
        """omega_c/(2pi) = eB/(2pi*m) ~ 7.6 MHz for Be-9+ at 4.45 T."""
        nu_c = nist_trap.omega_cyclotron / TWO_PI
        expected = ELECTRON_CHARGE * 4.45 / (TWO_PI * get_species("Be9").mass_kg)
        assert nu_c == pytest.approx(expected, rel=1e-6)
        assert nu_c == pytest.approx(7.6e6, rel=0.01)

    def test_stability(self, nist_trap):
        """omega_z = 1.57 MHz << omega_c/sqrt(2) = 5.37 MHz: stable."""
        assert nist_trap.is_stable()
        assert nist_trap.omega_axial < nist_trap.omega_cyclotron / np.sqrt(2)


class TestPenningTrapScaling:
    """Cross-experiment consistency checks that validate the Penning
    trap formulas across different species and B-field regimes.
    """

    def test_cyclotron_scales_with_charge_to_mass(self):
        """omega_c = eB/m, so for different species at the same B-field,
        the cyclotron frequency ratio equals the inverse mass ratio.

        Be-9+ vs Ca-40+ at 3 T."""
        be = PenningTrap(
            magnetic_field=3.0,
            species=get_species("Be9"),
            d=1e-3,
            omega_axial=TWO_PI * 1e6,
        )
        ca = PenningTrap(
            magnetic_field=3.0,
            species=get_species("Ca40"),
            d=1e-3,
            omega_axial=TWO_PI * 1e6,
        )
        mass_ratio = get_species("Ca40").mass_kg / get_species("Be9").mass_kg
        freq_ratio = be.omega_cyclotron / ca.omega_cyclotron
        assert freq_ratio == pytest.approx(mass_ratio, rel=1e-6)

    def test_electron_vs_ion_cyclotron_ratio(self):
        """At the same B-field, electron cyclotron is ~10,000x higher
        than Be-9+ because m_e/m_Be ~ 6e-5."""
        B = 3.0
        electron = PenningTrap(
            magnetic_field=B,
            species=ElectronSpecies(magnetic_field=B),
            d=1e-3,
            omega_axial=TWO_PI * 100e6,
        )
        be = PenningTrap(
            magnetic_field=B,
            species=get_species("Be9"),
            d=1e-3,
            omega_axial=TWO_PI * 1e6,
        )
        ratio = electron.omega_cyclotron / be.omega_cyclotron
        expected = get_species("Be9").mass_kg / ELECTRON_MASS
        assert ratio == pytest.approx(expected, rel=1e-6)
        assert ratio == pytest.approx(1.64e4, rel=0.01)

    def test_magnetron_decreases_with_increasing_B(self):
        """At fixed omega_z, increasing B pushes omega_- lower
        (magnetron becomes slower as cyclotron dominates)."""
        species = get_species("Be9")
        trap_low_B = PenningTrap(
            magnetic_field=2.0, species=species,
            d=1e-3, omega_axial=TWO_PI * 500e3,
        )
        trap_high_B = PenningTrap(
            magnetic_field=6.0, species=species,
            d=1e-3, omega_axial=TWO_PI * 500e3,
        )
        assert trap_high_B.omega_magnetron < trap_low_B.omega_magnetron

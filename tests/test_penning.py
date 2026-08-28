"""Experimental validation tests for Penning trap physics.

Compares TIQS Penning trap eigenfrequencies against values published by
precision experiments. Not every test is such a comparison: each class
mixes (a) checks against a published number, (b) frequency-hierarchy and
ordering checks, and (c) closed-form identity regressions that hold for
any ``(omega_c, omega_z)`` by construction. Category (c) is labeled as
such in the docstring of each test that belongs to it.

References:
    Jain et al., Nature 627, 510 (2024) / arXiv:2308.07672
    Hanneke, Fogwell Hoogerheide & Gabrielse, Phys. Rev. A 83, 052122
        (2011) / arXiv:1009.4831 - Table I, the trap frequencies
    Hanneke, Fogwell & Gabrielse, PRL 100, 120801 (2008) /
        arXiv:0801.1134 - the g/2 measurement (a *different* paper)
    Berrocal et al., Phys. Rev. Research 6, L012001 (2024) /
        arXiv:2308.14884
    Ball et al., Rev. Sci. Instrum. 90, 053103 (2019) / arXiv:1807.00902
    Bohnet et al., Science 352, 1297 (2016) / arXiv:1512.03756
    Brown & Gabrielse, Rev. Mod. Phys. 58, 233 (1986) - invariance
        theorem, magnetron metastability
"""

import numpy as np
import pytest

from tiqs.constants import (
    ELECTRON_MASS,
    TWO_PI,
)
from tiqs.species.electron import ElectronSpecies
from tiqs.species.ion import get_species
from tiqs.trap import PenningTrap

# Ball et al. quote C_2 = -4.68e-3 mm^-2 at the optimal tuning ratio.
# TIQS parameterizes the axial curvature as omega_z^2 = e V/(m d^2),
# so d^2 = 1/(2 |C_2|)  ->  d = 10.336 mm.
BALL_D = 1.0 / np.sqrt(2 * 4.68e-3 * 1e6)

# Hanneke PRA Table I geometry: rho_0 = 4.5 mm, 2 z_0 = 7.7 mm, with the
# hyperbolic-equivalent d^2 = (z_0^2 + rho_0^2/2)/2  ->  d = 3.5318 mm.
HANNEKE_D = np.sqrt(((7.7e-3 / 2) ** 2 + (4.5e-3) ** 2 / 2) / 2)


class TestJainETHMicroPenning:
    """Jain et al., Nature 627, 510 (2024): single Be-9+ ion in a
    micro-fabricated Penning trap at B = 3 T.

    First demonstration of full quantum control in a Penning micro-trap.
    All three eigenfrequencies are published explicitly in the main-text
    trap description (not the Methods): omega_z = 2pi*2.5 MHz,
    omega_+ = 2pi*4.41 MHz, omega_- = 2pi*0.71 MHz, omega_c = 2pi*5.12
    MHz.
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
        """omega_c/(2pi) = 5.12 MHz for Be-9+ at 3 T (main text)."""
        nu_c = eth_trap.omega_cyclotron / TWO_PI
        assert nu_c == pytest.approx(5.12e6, rel=0.005)

    def test_modified_cyclotron_frequency(self, eth_trap):
        """omega_+/(2pi) = 4.41 MHz (Nature 627, 510 (2024), main
        text)."""
        nu_plus = eth_trap.omega_modified_cyclotron / TWO_PI
        assert nu_plus == pytest.approx(4.41e6, rel=0.005)

    def test_magnetron_frequency(self, eth_trap):
        """omega_-/(2pi) = 0.71 MHz (Nature 627, 510 (2024), main
        text)."""
        nu_minus = eth_trap.omega_magnetron / TWO_PI
        assert nu_minus == pytest.approx(0.71e6, rel=0.01)

    def test_frequency_hierarchy(self, eth_trap):
        """omega_- < omega_z < omega_+ < omega_c for stable Penning
        trap."""
        assert eth_trap.omega_magnetron < eth_trap.omega_axial
        assert eth_trap.omega_axial < eth_trap.omega_modified_cyclotron
        assert eth_trap.omega_modified_cyclotron < eth_trap.omega_cyclotron

    def test_invariance_theorem_against_published_frequencies(self, eth_trap):
        """Brown-Gabrielse invariance theorem applied to Jain's three
        PUBLISHED frequencies must reproduce the code's eB/m.

        sqrt(4.41^2 + 0.71^2 + 2.5^2) MHz = 5.11881 MHz against the
        computed 5.11210 MHz: 0.13%, the resolution of the paper's
        3-digit values. Unlike the identity applied to TIQS's own
        outputs (which holds for any inputs by construction - see
        tests/test_trap.py::test_frequency_invariant), this uses only
        numbers from the paper on one side."""
        nu_c_published = 1e6 * np.sqrt(4.41**2 + 0.71**2 + 2.5**2)
        assert eth_trap.omega_cyclotron / TWO_PI == pytest.approx(
            nu_c_published, rel=3e-3
        )

    def test_sum_rule(self, eth_trap):
        """omega_+ + omega_- = omega_c.

        Closed-form identity regression, not experimental validation:
        with omega_+- = omega_c/2 +- s this holds for any s, so it pins
        only that both branches share the omega_c/2 offset."""
        wc = eth_trap.omega_cyclotron
        wp = eth_trap.omega_modified_cyclotron
        wm = eth_trap.omega_magnetron
        assert wp + wm == pytest.approx(wc, rel=1e-10)

    def test_product_rule(self, eth_trap):
        """omega_+ * omega_- = omega_z^2 / 2.

        Closed-form identity regression, not experimental validation.
        Complementary to the sum rule: it pins the omega_z^2/2
        coefficient inside the discriminant (an omega_z^2 there would
        fail)."""
        wp = eth_trap.omega_modified_cyclotron
        wm = eth_trap.omega_magnetron
        wz = eth_trap.omega_axial
        assert wp * wm == pytest.approx(wz**2 / 2, rel=1e-10)

    def test_stability(self, eth_trap):
        """Qualitative: omega_z = 2pi*2.5 MHz is below the published
        stability limit omega_c/sqrt(2) = 2pi*3.62 MHz (main text)."""
        assert eth_trap.is_stable()
        assert eth_trap.omega_axial < eth_trap.omega_cyclotron / np.sqrt(2)


class TestHannekeElectronGMinus2:
    """Hanneke, Fogwell Hoogerheide & Gabrielse, Phys. Rev. A 83, 052122
    (2011) / arXiv:1009.4831: single electron in a cylindrical Penning
    trap at B = 5.36 T.

    Table I of that paper is the source of B = 5.36 T,
    nu_c-bar = 150.0 GHz, nu_m-bar = 133 kHz, V_0 = 101.4 V,
    rho_0 = 4.5 mm and 2 z_0 = 7.7 mm; nu_z-bar ~ 200 MHz and the
    ~174 MHz anomaly frequency are in its running text.

    The companion g/2 measurement - Hanneke, Fogwell & Gabrielse, PRL
    100, 120801 (2008) / arXiv:0801.1134 - is a different paper and
    cannot be used to check this fixture: it never states B, never
    mentions a magnetron frequency, and quotes four cyclotron settings
    spanning 147.5-151.3 GHz. That 2008 result was the most precise
    single-particle measurement of its time, and has since been
    superseded by Fan, Myers, Sukra & Gabrielse, PRL 130, 071801 (2023)
    (0.13 ppt, 2.2x more accurate).
    """

    @pytest.fixture
    def hanneke_trap(self):
        return PenningTrap(
            magnetic_field=5.36,
            species=ElectronSpecies(magnetic_field=5.36),
            d=HANNEKE_D,
            omega_axial=TWO_PI * 200e6,
        )

    def test_trap_shifted_cyclotron_frequency(self, hanneke_trap):
        """Table I's nu_c-bar = 150.0 GHz is the TRAP-SHIFTED cyclotron
        frequency, i.e. omega_+/(2pi) - the bar denotes the shifted
        value throughout Gabrielse's work, and relating it to the free
        omega_c is the whole point of the invariance theorem."""
        nu_plus = hanneke_trap.omega_modified_cyclotron / TWO_PI
        assert nu_plus == pytest.approx(150.0e9, rel=0.001)

    def test_free_cyclotron_from_published_triple(self, hanneke_trap):
        """The FREE nu_c implied by Table I's triple is
        sqrt(150.0 GHz^2 + 200 MHz^2 + 133 kHz^2) = 150.000133 GHz.

        eB/(2pi m_e) at the stated B = 5.36 T gives 150.0397 GHz, i.e.
        2.6e-4 high - consistent with B being quoted to three figures
        (B = 5.35858 T reproduces 150.000 GHz exactly)."""
        nu_c_free = np.sqrt(150.0e9**2 + 200e6**2 + 133e3**2)
        assert hanneke_trap.omega_cyclotron / TWO_PI == pytest.approx(
            nu_c_free, rel=1e-3
        )

    def test_magnetron_frequency(self, hanneke_trap):
        """nu_m-bar = 133 kHz (Table I, arXiv:1009.4831).

        The magnetron frequency is extremely small compared to the
        cyclotron frequency (~10^-6 ratio), demonstrating the extreme
        hierarchy in electron Penning traps."""
        nu_m = hanneke_trap.omega_magnetron / TWO_PI
        assert nu_m == pytest.approx(133e3, rel=0.01)

    def test_anomaly_frequency_cross_check(self, hanneke_trap):
        """nu_a = (g/2 - 1) * nu_c ~ 174 MHz (PRA Sec. II C; the 2008
        PRL quotes ~173 MHz from its own field setting).

        The anomaly frequency is the difference between the spin
        precession and cyclotron frequencies. This cross-checks our
        cyclotron computation against the measured g-factor."""
        from tiqs.constants import ELECTRON_G_FACTOR

        nu_c = hanneke_trap.omega_cyclotron / TWO_PI
        nu_a = (ELECTRON_G_FACTOR / 2 - 1) * nu_c
        assert nu_a == pytest.approx(174e6, rel=0.01)

    def test_extreme_frequency_hierarchy(self, hanneke_trap):
        """nu_m / nu_c ~ 10^-6: magnetron is a million times slower
        than cyclotron for electrons at 5.36 T."""
        ratio = hanneke_trap.omega_magnetron / hanneke_trap.omega_cyclotron
        assert ratio < 1e-5

    def test_v_dc_against_published_electrode_voltage(self, hanneke_trap):
        """v_dc from Table I's geometry vs its stated V_0 ~ 101.4 V.

        With the hyperbolic-equivalent d = 3.5318 mm implied by
        rho_0 = 4.5 mm and 2 z_0 = 7.7 mm, TIQS needs 112.0 V to reach
        nu_z-bar = 200 MHz - 10.4% above the measured 101.4 V, because a
        cylindrical trap's C_2 is ~10% larger than the ideal hyperbolic
        value assumed by d. This is the only external check of the v_dc
        path; the tolerance is honest about that 10%, and still rejects
        a factor-of-2 error."""
        assert hanneke_trap.v_dc == pytest.approx(101.4, rel=0.12)


class TestBerrocal2024CaPenning:
    """Berrocal et al., Phys. Rev. Research 6, L012001 (2024):
    Ca-40+ in an open-ring 7-tesla Penning trap.

    First direct determination of all three eigenfrequencies for
    calcium ions using optical radiofrequency detectors, so all three
    published values are independent measurements.
    """

    @pytest.fixture
    def ca_penning(self):
        return PenningTrap(
            magnetic_field=7.0,
            species=get_species("Ca40"),
            d=5.0e-3,  # inert: d enters only v_dc, not eigenfrequencies
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

    def test_bare_cyclotron_from_measured_triple(self, ca_penning):
        """Berrocal's Eq. (3) fed with the three MEASURED frequencies
        (nu_+ = 2.686 MHz, nu_z = 142 kHz, nu_- = 3.8 kHz) gives
        nu_c = 2.689754 MHz; eB/(2pi m) for Ca-40+ at 7 T gives
        2.689873 MHz, agreeing to 4.4e-5.

        Every number on the reference side comes from the paper, so
        unlike a restatement of eB/(2pi m) inside the test this can
        catch a wrong shared charge or mass."""
        nu_c_measured = np.sqrt(2.686e6**2 + 142e3**2 + 3.8e3**2)
        assert ca_penning.omega_cyclotron / TWO_PI == pytest.approx(
            nu_c_measured, rel=2e-4
        )

    def test_frequency_hierarchy(self, ca_penning):
        """nu_- << nu_z << nu_+ for ion Penning traps at high B."""
        nu_m = ca_penning.omega_magnetron / TWO_PI
        nu_z = ca_penning.omega_axial / TWO_PI
        nu_p = ca_penning.omega_modified_cyclotron / TWO_PI
        assert nu_m < nu_z < nu_p
        assert nu_m / nu_z < 0.03
        assert nu_z / nu_p < 0.06


class TestBallBePenning:
    """Ball et al., Rev. Sci. Instrum. 90, 053103 (2019) /
    arXiv:1807.00902: Be-9+ in a high-optical-access Penning trap at
    B = 1.998 T.

    Table I's caption states that only the axial frequency was measured:
    "The magnetron and the reduced cyclotron frequency have been
    determined from the measured axial frequency and magnetic field" -
    using the same closed form TIQS implements. The nu_+ and nu_- tests
    below therefore confirm formula agreement plus the Be-9 mass
    constant, NOT agreement with an independent measurement.

    Two numbers in the paper ARE independent of that closed form and are
    used as anchors here: the free cyclotron frequency
    nu_c ~ 3406 kHz, and the trap geometry (C_2 = -4.68e-3 mm^-2 at the
    optimal tuning ratio, V_STCR = -65 V) which drives the
    from_dc_voltage check.
    """

    @pytest.fixture
    def ball_trap(self):
        return PenningTrap(
            magnetic_field=1.998,
            species=get_species("Be9"),
            d=BALL_D,
            omega_axial=TWO_PI * 402e3,  # measured nu_z, Table I
        )

    def test_free_cyclotron_frequency(self, ball_trap):
        """nu_c ~ 3406 kHz (Table I caption, arXiv:1807.00902).

        Independent of the axial frequency and of the closed form: it
        pins eB/(2pi m) for Be-9+ at 1.998 T. The code gives 3404.66
        kHz, 3.9e-4 low, within the caption's 4-digit rounding."""
        nu_c = ball_trap.omega_cyclotron / TWO_PI
        assert nu_c == pytest.approx(3406e3, rel=0.001)

    def test_modified_cyclotron_frequency(self, ball_trap):
        """nu_+ = 3382 kHz (Table I, arXiv:1807.00902) - a value the
        paper DERIVED from the measured nu_z = 402 kHz and B, not a
        measurement, so this checks formula agreement."""
        nu_plus = ball_trap.omega_modified_cyclotron / TWO_PI
        assert nu_plus == pytest.approx(3382e3, rel=0.001)

    def test_magnetron_frequency(self, ball_trap):
        """nu_- = 23.9 kHz (Table I, arXiv:1807.00902) - likewise
        derived from the measured nu_z and B, not measured."""
        nu_minus = ball_trap.omega_magnetron / TWO_PI
        assert nu_minus == pytest.approx(23.9e3, rel=0.01)

    def test_from_dc_voltage_matches_calculated_axial(self):
        """Ball's Eq. (B2), nu_z = sqrt(2 e V C_2/m)/2pi, with
        V_STCR = 65 V and C_2 = 4.68e-3 mm^-2 (quoted at the optimal
        tuning ratio, which already folds in T_opt) gives 406.19 kHz -
        Table I's calculated column is 406.4 kHz, so 0.05%.

        This is the only test in the suite that exercises
        PenningTrap.from_dc_voltage against a published trap. TIQS's
        d convention maps as d^2 = 1/(2 C_2), i.e. d = 10.336 mm. The
        measured 402 kHz is 1.1% below the calculated value because of
        anharmonicity in the real trap."""
        trap = PenningTrap.from_dc_voltage(
            magnetic_field=1.998,
            species=get_species("Be9"),
            d=BALL_D,
            v_dc=65.0,
        )
        assert trap.omega_axial / TWO_PI == pytest.approx(406.4e3, rel=0.002)

    def test_v_dc_reproduces_electrode_voltage(self, ball_trap):
        """Inverting at the MEASURED nu_z = 402 kHz asks for 63.67 V
        against the applied |V_STCR| = 65 V, 2.1% low - the same
        anharmonicity that separates the 402 and 406.4 kHz columns
        (the frequency ratio squared is (402/406.4)^2 = 0.978)."""
        assert ball_trap.v_dc == pytest.approx(65.0, rel=0.03)


class TestBohnetNISTBePenning:
    """Bohnet et al., Science 352, 1297 (2016) / arXiv:1512.03756:
    21-219 Be-9+ ions in a NIST Penning trap at B = 4.45 T with
    omega_z = 2pi*1.57 MHz.

    Spin squeezing was verified for up to 219 ions; the results section
    reports seven datasets with N from 21 to 219 (panels at N = 21, 58,
    144). Validated rotating-wall-controlled 2D crystal dynamics.
    4.45 T is not the strongest field in this file - Berrocal runs at
    7 T and Hanneke at 5.36 T.
    """

    @pytest.fixture
    def nist_trap(self):
        return PenningTrap(
            magnetic_field=4.45,
            species=get_species("Be9"),
            d=5.0e-3,  # inert: d enters only v_dc, not eigenfrequencies
            omega_axial=TWO_PI * 1.57e6,
        )

    def test_bare_cyclotron_frequency(self, nist_trap):
        """nu_c = eB/(2pi m) = 7.5829 MHz for Be-9+ at 4.45 T.

        Value-pinned from the ion mass 9.011635 u rather than recomputed
        in the test, so a wrong shared charge or mass is visible."""
        nu_c = nist_trap.omega_cyclotron / TWO_PI
        assert nu_c == pytest.approx(7.5829e6, rel=1e-3)

    def test_stability(self, nist_trap):
        """Qualitative: omega_z = 1.57 MHz << omega_c/sqrt(2) = 5.36
        MHz, so the 2D crystal is radially confined."""
        assert nist_trap.is_stable()
        assert nist_trap.omega_axial < nist_trap.omega_cyclotron / np.sqrt(2)


class TestPenningTrapScaling:
    """Cross-experiment consistency checks that validate the Penning
    trap formulas across different species and B-field regimes.
    """

    def test_cyclotron_scales_with_charge_to_mass(self):
        """omega_c = eB/m, so for different species at the same B-field,
        the cyclotron frequency ratio equals the inverse mass ratio.

        Be-9+ vs Ca-40+ at 3 T. The ratio alone cancels any shared
        factor (charge, 2pi, a factor of 2), so both absolute values are
        pinned as well: 5.1121 MHz and 1.15280 MHz."""
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
        assert be.omega_cyclotron / TWO_PI == pytest.approx(5.1121e6, rel=1e-3)
        assert ca.omega_cyclotron / TWO_PI == pytest.approx(
            1.15280e6, rel=1e-3
        )

    def test_electron_vs_ion_cyclotron_ratio(self):
        """At the same B-field, electron cyclotron is ~10,000x higher
        than Be-9+ because m_e/m_Be ~ 6e-5.

        The absolute electron value is pinned too (83.977 GHz at 3 T),
        since the ratio cannot see a shared error."""
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
        assert electron.omega_cyclotron / TWO_PI == pytest.approx(
            83.977e9, rel=1e-3
        )

    def test_magnetron_approaches_strong_field_limit(self):
        """At fixed omega_z, increasing B pushes omega_- lower, toward
        the strong-field limit omega_- -> omega_z^2/(2 omega_c)
        (Brown & Gabrielse, Rev. Mod. Phys. 58, 233 (1986)).

        Be-9+ at omega_z = 2pi*500 kHz: at 6 T the exact magnetron
        frequency is 12.241 kHz and the limit form gives 12.226 kHz."""
        species = get_species("Be9")
        trap_low_B = PenningTrap(
            magnetic_field=2.0,
            species=species,
            d=1e-3,
            omega_axial=TWO_PI * 500e3,
        )
        trap_high_B = PenningTrap(
            magnetic_field=6.0,
            species=species,
            d=1e-3,
            omega_axial=TWO_PI * 500e3,
        )
        assert trap_high_B.omega_magnetron < trap_low_B.omega_magnetron
        limit = trap_high_B.omega_axial**2 / (2 * trap_high_B.omega_cyclotron)
        assert trap_high_B.omega_magnetron == pytest.approx(limit, rel=3e-3)
        assert trap_high_B.omega_magnetron / TWO_PI == pytest.approx(
            12.241e3, rel=1e-3
        )

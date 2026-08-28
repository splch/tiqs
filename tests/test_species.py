"""Species-database tests anchored to external atomic data.

Every numeric assertion here is pinned to a value published outside
this repository - NIST ASD transition probabilities and vacuum
wavelengths, CIPM secondary representations of the second, CODATA 2022,
or a cited measurement - rather than to a literal copied out of
``src/tiqs/species``. Tolerances are set just tight enough to reject
the previously stored value.
"""

import math

import numpy as np
import pytest

from tiqs.constants import (
    AMU,
    BOLTZMANN,
    ELECTRON_MASS,
    HBAR,
    SPEED_OF_LIGHT,
    TWO_PI,
)
from tiqs.species.electron import ElectronSpecies
from tiqs.species.ion import get_species
from tiqs.species.protocol import Species
from tiqs.species.transitions import Transition

_ALL_SPECIES = ["Yb171", "Ca40", "Ca43", "Ba137", "Be9", "Sr88"]

# (species, transition name, NIST ASD vacuum wavelength in nm)
_WAVELENGTH_CASES = [
    ("Yb171", "2S1/2 -> 2P1/2", 369.5262),
    ("Yb171", "2D3/2 -> 3D[3/2]1/2", 935.187),
    ("Ca40", "4S1/2 -> 4P1/2", 396.959),
    ("Ca40", "3D3/2 -> 4P1/2", 866.452),
    ("Ca40", "3D5/2 -> 4P3/2", 854.444),
    ("Ca43", "4S1/2 -> 4P1/2", 396.959),
    ("Ca43", "3D3/2 -> 4P1/2", 866.452),
    ("Ba137", "6S1/2 -> 6P1/2", 493.5454),
    ("Ba137", "5D3/2 -> 6P1/2", 649.8693),
    ("Ba137", "5D5/2 -> 6P3/2", 614.3413),
    ("Be9", "2S1/2 -> 2P3/2", 313.13292),
    ("Sr88", "5S1/2 -> 5P1/2", 421.6711),
    ("Sr88", "4D3/2 -> 5P1/2", 1091.7864),
    ("Sr88", "4D5/2 -> 5P3/2", 1033.0139),
]


class TestTransition:
    def test_create_transition(self):
        t = Transition(
            name="S1/2 -> P1/2",
            wavelength=369.5e-9,
            linewidth=2 * np.pi * 23e6,
            branching_ratio=1.0,
        )
        assert t.name == "S1/2 -> P1/2"
        assert t.wavelength == pytest.approx(369.5e-9)
        assert t.branching_ratio == pytest.approx(1.0)

    def test_transition_frequency(self):
        t = Transition(name="test", wavelength=369.5e-9, linewidth=1e6)
        assert t.frequency == pytest.approx(
            SPEED_OF_LIGHT / 369.5e-9, rel=1e-12
        )

    def test_transition_wavevector(self):
        t = Transition(name="test", wavelength=369.5e-9, linewidth=1e6)
        assert t.wavevector == pytest.approx(2 * np.pi / 369.5e-9)


class TestIonMass:
    """`mass_kg` is the ion mass: neutral mass minus one electron."""

    @pytest.mark.parametrize("name", _ALL_SPECIES)
    def test_one_electron_is_removed(self, name):
        """The mass deficit is m_e = 5.485799090441e-4 u (CODATA 2022).

        Fails on the pre-fix ``mass_amu * AMU``, whose deficit is zero.
        """
        s = get_species(name)
        deficit_u = (s.mass_amu * AMU - s.mass_kg) / AMU
        assert deficit_u == pytest.approx(5.485799090441e-4, rel=1e-9, abs=0)

    @pytest.mark.parametrize(
        ("name", "neutral_amu"),
        [
            # NIST Atomic Weights and Isotopic Compositions (AME2020)
            # relative atomic masses of the neutral atoms.
            ("Yb171", 170.9363302),
            ("Ca40", 39.962590863),
            ("Ca43", 42.95876644),
            ("Ba137", 136.90582714),
            ("Be9", 9.012183065),
            ("Sr88", 87.9056125),
        ],
    )
    def test_neutral_mass_matches_ame2020(self, name, neutral_amu):
        """``mass_amu`` is the tabulated neutral relative atomic mass."""
        assert get_species(name).mass_amu == pytest.approx(
            neutral_amu, rel=1e-9
        )

    def test_be9_ion_mass_absolute(self):
        """9Be+ mass: 9.012183065 u - m_e = 9.011634485 u.

        The lightest species carries the largest electron correction
        (6.1e-5 relative), so it is the sharpest single check.
        """
        m_ion_u = 9.012183065 - 5.485799090441e-4
        assert get_species("Be9").mass_kg == pytest.approx(
            m_ion_u * AMU, rel=1e-9, abs=0
        )
        # And the old behaviour is now excluded outright.
        assert get_species("Be9").mass_kg < 9.012183065 * AMU

    def test_mass_ordering(self):
        masses = [get_species(n).mass_kg for n in _ALL_SPECIES]
        assert all(m > 0 for m in masses)
        assert get_species("Be9").mass_kg == min(masses)
        assert get_species("Yb171").mass_kg == max(masses)


class TestQubitFrequencies:
    """Qubit splittings against their published measurements."""

    def test_ba137_hyperfine_splitting(self):
        """8 037 741 667.7 Hz, Blatt and Werth, PRA 25, 1476 (1982).

        The pre-fix 8.038e9 was 2.583e5 Hz (3.2e-5) high.
        """
        f = get_species("Ba137").qubit_frequency_hz
        assert f == pytest.approx(8037741667.7, rel=1e-8)

    def test_yb171_hyperfine_splitting(self):
        """12 642 812 118.4690(8) Hz, Appl. Phys. Lett. 125, 084002
        (2024). The pre-fix 12.6428e9 was 1.2e4 Hz (9.6e-7) low."""
        f = get_species("Yb171").qubit_frequency_hz
        assert f == pytest.approx(12642812118.469, rel=1e-8)

    def test_be9_hyperfine_splitting_is_twice_a(self):
        """Splitting = 2|A| for I = 3/2, J = 1/2.

        A = -625.008840(35) MHz (arXiv:2601.14811) gives
        1.25001768(7) GHz. The pre-fix 1.25e9 was 1.8e4 Hz (1.4e-5) low.
        """
        f = get_species("Be9").qubit_frequency_hz
        assert f == pytest.approx(2 * 625.008840e6, rel=1e-7)

    def test_ca43_hyperfine_splitting(self):
        """3 225 608 286.4(3) Hz zero-field 4S1/2 splitting."""
        f = get_species("Ca43").qubit_frequency_hz
        assert f == pytest.approx(3225608286.4, rel=1e-8)

    def test_ca40_optical_qubit_is_the_clock_transition(self):
        """CIPM secondary representation: 411 042 129 776 400 Hz.

        ``qubit_wavelength`` feeds k = 2 pi / lambda and hence eta, so
        its 4.8e-4 rounding error (729 nm vs 729.3473 nm) propagated
        straight into every Lamb-Dicke parameter.
        """
        s = get_species("Ca40")
        assert s.qubit_type == "optical"
        assert s.qubit_frequency_hz == pytest.approx(
            411042129776400.0, rel=1e-6
        )
        assert s.qubit_frequency_hz == pytest.approx(
            SPEED_OF_LIGHT / s.qubit_wavelength, rel=1e-12
        )

    def test_sr88_optical_qubit_is_the_clock_transition(self):
        """CIPM secondary representation: 444 779 044 095 485 Hz."""
        s = get_species("Sr88")
        assert s.qubit_type == "optical"
        assert s.qubit_frequency_hz == pytest.approx(
            444779044095485.0, rel=1e-6
        )

    def test_hyperfine_species_have_no_optical_qubit(self):
        for name in ["Yb171", "Ca43", "Ba137", "Be9"]:
            s = get_species(name)
            assert s.qubit_type == "hyperfine"
            assert s.qubit_wavelength is None
            assert s.qubit_t1 == math.inf


class TestVacuumWavelengths:
    """Stored wavelengths are NIST ASD vacuum values, 6-7 figures.

    All twelve entries were previously rounded to 3-4 figures, up to
    1.1e-3 relative (Ba-137 493 nm), which is the largest systematic
    error in the species data because k = 2 pi / lambda sets eta.
    """

    @pytest.mark.parametrize(
        ("name", "line", "nm_vacuum"),
        _WAVELENGTH_CASES,
        ids=[f"{c[0]}-{c[2]}" for c in _WAVELENGTH_CASES],
    )
    def test_wavelength_matches_nist_asd(self, name, line, nm_vacuum):
        s = get_species(name)
        lines = (s.cooling_transition, *s.repump_transitions)
        matches = [t for t in lines if t.name == line]
        assert len(matches) == 1, f"{name}: no unique line {line!r}"
        assert matches[0].wavelength == pytest.approx(
            nm_vacuum * 1e-9, rel=1e-5, abs=0
        )

    def test_raman_wavelengths_track_their_transitions(self):
        """Raman beams sit within a few tens of GHz of the P line."""
        for name in ["Ca43", "Be9"]:
            s = get_species(name)
            assert s.raman_wavelength == pytest.approx(
                s.cooling_transition.wavelength, rel=1e-5, abs=0
            )
        # Yb-171 and Ba-137 drive far-detuned Raman beams from
        # solid-state lasers, which are not atomic lines.
        assert get_species("Yb171").raman_wavelength == 355e-9
        assert get_species("Ba137").raman_wavelength == 515e-9


class TestLinewidths:
    """Decay rates against NIST ASD transition probabilities."""

    def test_be9_313nm_from_nist_a_ki(self):
        """A_ki = 1.1292e8 s^-1 and 2P3/2 has no other decay channel.

        Gamma/2pi = 17.97 MHz (tau = 8.86 ns). The pre-fix 19.4 MHz -
        the value the trapped-ion literature propagates - is 8% high
        and is consumed by the Doppler limit and the sympathetic
        cooling rates.
        """
        gamma = get_species("Be9").cooling_transition.linewidth
        assert gamma == pytest.approx(1.1292e8, rel=1e-3)
        assert 1.0 / gamma == pytest.approx(8.856e-9, rel=1e-3, abs=0)

    def test_sr88_1092nm_from_nist_a_ki(self):
        """A_ki(1091.7864 nm) = 7.46e6 s^-1.

        The pre-fix 2 pi * 1.4e6 = 8.80e6 s^-1 was 18% high and
        matched no source. The 5e-3 tolerance is the three-significant-
        figure storage of A_ki / 2 pi that this file uses throughout.
        """
        (repump_d32, _) = get_species("Sr88").repump_transitions
        assert repump_d32.name == "4D3/2 -> 5P1/2"
        assert repump_d32.linewidth == pytest.approx(7.46e6, rel=5e-3)

    def test_ba137_614nm_from_nist_a_ki(self):
        """A_ki(614.3413 nm) = 4.12e7 s^-1 for the D5/2 clear-out."""
        lines = {t.name: t for t in get_species("Ba137").repump_transitions}
        assert lines["5D5/2 -> 6P3/2"].linewidth == pytest.approx(
            4.12e7, rel=5e-3
        )

    def test_sr88_1033nm_from_nist_a_ki(self):
        """A_ki(1033.0139 nm) = 8.7e6 s^-1 for the D5/2 clear-out."""
        lines = {t.name: t for t in get_species("Sr88").repump_transitions}
        assert lines["4D5/2 -> 5P3/2"].linewidth == pytest.approx(
            8.7e6, rel=5e-3
        )

    @pytest.mark.parametrize(
        ("name", "tau_ns"),
        [
            # Upper-state lifetimes: cooling `linewidth` is the TOTAL
            # natural linewidth, so 1/Gamma must reproduce tau.
            ("Ca40", 7.098),  # Hettrich, PRL 115, 013004 (2015)
            ("Ca43", 7.098),
            ("Yb171", 8.12),
            ("Ba137", 7.84),
            ("Sr88", 7.40),
            ("Be9", 8.856),  # NIST A_ki = 1.1292e8 s^-1
        ],
    )
    def test_cooling_linewidth_is_total_upper_state_width(self, name, tau_ns):
        gamma = get_species(name).cooling_transition.linewidth
        assert 1.0 / gamma == pytest.approx(tau_ns * 1e-9, rel=5e-3, abs=0)

    def test_repump_linewidth_is_a_partial_rate(self):
        """Repump `linewidth` is A_ki of that line, not Gamma_upper.

        Documented in `Transition`; the Ca-40 866 nm line is the
        canonical example - 1.06e7 s^-1 against a 4P1/2 upper state
        1.41e8 s^-1 wide, a factor 13.
        """
        ca = get_species("Ca40")
        d32 = ca.repump_transitions[0]
        assert d32.name == "3D3/2 -> 4P1/2"
        assert d32.linewidth == pytest.approx(1.06e7, rel=5e-3)
        assert ca.cooling_transition.linewidth / d32.linewidth > 10.0


class TestBranchingAndLifetimes:
    def test_ba137_cooling_branching_ratio(self):
        """1 - p(6P1/2 -> 5D3/2), p = 0.268177(37)(20).

        Arnold et al., PRA 100, 032503 (2019). The pre-fix 0.75 came
        from the NIST A_ki ratio and is 2.4% (about 500 sigma) high.
        """
        br = get_species("Ba137").cooling_transition.branching_ratio
        assert br == pytest.approx(1.0 - 0.268177, rel=1e-3)

    @pytest.mark.parametrize(
        ("name", "measured"),
        [
            ("Ca40", 0.93565),  # Ramm, PRL 111, 023004 (2013)
            ("Ca43", 0.93565),
            ("Sr88", 0.9453),  # Likforman, PRA 93, 052507 (2016)
            ("Yb171", 0.995),
            ("Be9", 1.0),  # 2P3/2 -> 2S1/2 is the only channel
        ],
    )
    def test_cooling_branching_ratios(self, name, measured):
        br = get_species(name).cooling_transition.branching_ratio
        assert br == pytest.approx(measured, rel=1e-3)
        assert 0.0 < br <= 1.0

    def test_ba137_metastable_lifetime(self):
        """5D5/2: 31.2(9) s, Auchter et al., PRA 90, 060501(R) (2014).

        The pre-fix 30.14 s traced to no measurement. Dijck et al.,
        PRA 97, 032508 (2018) report 25.6(5) s, about 5 sigma lower, so
        the second assertion only pins the literature span.
        """
        tau = get_species("Ba137").metastable_lifetime
        assert tau == pytest.approx(31.2, rel=0.01)
        assert 25.6 <= tau <= 32.1

    def test_ca40_metastable_lifetime(self):
        """Ca-40 3D5/2: 1.168(7) s, Barton et al., PRA 62, 032503."""
        assert get_species("Ca40").metastable_lifetime == pytest.approx(
            1.168, rel=0.01
        )

    def test_sr88_metastable_lifetime_matches_nist_a_ki(self):
        """4D5/2: NIST A_ki(674 nm E2) = 2.559 s^-1 -> tau = 0.391 s."""
        tau = get_species("Sr88").metastable_lifetime
        assert tau == pytest.approx(1.0 / 2.559, rel=0.01)
        assert get_species("Sr88").qubit_t1 == pytest.approx(tau)


class TestRepumpCoverage:
    """Every species with a populated D5/2 needs a clear-out laser."""

    @pytest.mark.parametrize(
        ("name", "nm"),
        [
            ("Ca40", 854.444),
            ("Ba137", 614.3413),  # was missing before the fix
            ("Sr88", 1033.0139),  # was missing before the fix
        ],
    )
    def test_d52_clearout_present(self, name, nm):
        s = get_species(name)
        assert any(
            t.wavelength == pytest.approx(nm * 1e-9, rel=1e-5, abs=0)
            for t in s.repump_transitions
        ), f"{name} has no D5/2 clear-out repumper near {nm} nm"

    def test_d32_repump_present_for_every_dark_d_state_species(self):
        expected = {
            "Yb171": 935.187,
            "Ca40": 866.452,
            "Ca43": 866.452,
            "Ba137": 649.8693,
            "Sr88": 1091.7864,
        }
        for name, nm in expected.items():
            s = get_species(name)
            assert any(
                t.wavelength == pytest.approx(nm * 1e-9, rel=1e-5, abs=0)
                for t in s.repump_transitions
            ), f"{name} is missing its D3/2 repumper"
        # Be-9 has no low-lying D state, hence no repumper.
        assert get_species("Be9").repump_transitions == ()


class TestDopplerLimits:
    def test_ca40_doppler_nbar_from_upper_state_lifetime(self):
        r"""[Leibfried2003] Eq. 6: $\bar n_D = \Gamma/2\omega$.

        Anchored to tau(4P1/2) = 7.098(20) ns rather than to the stored
        linewidth: nbar = 1/(2 omega tau) = 11.2 at 1 MHz.
        """
        nbar = get_species("Ca40").doppler_limit_nbar(1e6)
        expected = 1.0 / (7.098e-9 * 2 * TWO_PI * 1e6)
        assert nbar == pytest.approx(expected, rel=2e-3)

    def test_be9_doppler_limits_from_nist_a_ki(self):
        """T_D = hbar Gamma / 2 k_B with Gamma = A_ki = 1.1292e8 s^-1.

        431 uK and nbar = 8.99 at 1 MHz. The pre-fix 19.4 MHz linewidth
        gave 466 uK and 9.70, both 8% high.
        """
        be = get_species("Be9")
        t_d = HBAR * 1.1292e8 / (2 * BOLTZMANN)
        assert be.doppler_limit_temperature() == pytest.approx(
            t_d, rel=1e-3, abs=0
        )
        assert t_d == pytest.approx(431e-6, rel=1e-2, abs=0)
        assert be.doppler_limit_nbar(1e6) == pytest.approx(
            1.1292e8 / (2 * TWO_PI * 1e6), rel=1e-3
        )

    def test_doppler_nbar_scales_inversely_with_trap_frequency(self):
        be = get_species("Be9")
        assert be.doppler_limit_nbar(1e6) / be.doppler_limit_nbar(
            4e6
        ) == pytest.approx(4.0, rel=1e-12)

    @pytest.mark.parametrize("name", _ALL_SPECIES)
    def test_doppler_temperature_is_sub_millikelvin(self, name):
        """All six cooling transitions are 18-22 MHz wide."""
        t_d = get_species(name).doppler_limit_temperature()
        assert 0.4e-3 < t_d < 0.6e-3


class TestSpeciesLookup:
    def test_unknown_species_raises(self):
        with pytest.raises(KeyError):
            get_species("Unobtanium")

    def test_all_species_available(self):
        for name in _ALL_SPECIES:
            s = get_species(name)
            assert s.symbol == name
            assert s.mass_amu > 0
            assert s.cooling_transition is not None
            assert s.qubit_type in {"hyperfine", "optical", "zeeman"}
            assert s.nuclear_spin >= 0.0


class TestElectronSpecies:
    def test_mass_is_the_codata_electron_mass(self):
        e = ElectronSpecies(magnetic_field=0.1)
        assert e.mass_kg == ELECTRON_MASS

    def test_qubit_frequency_matches_codata_gyromagnetic_ratio(self):
        r"""$f = \gamma_e B/2\pi$ with CODATA 2022
        $\gamma_e = 1.76085962784(55)\times10^{11}$ s$^{-1}$T$^{-1}$,
        i.e. 28 024.951386 MHz/T. A CODATA 2018 constant set with a
        truncated ``HBAR`` misses this by 2.0e-9.
        """
        e = ElectronSpecies(magnetic_field=0.1)
        expected = 1.76085962784e11 / TWO_PI * 0.1
        assert e.qubit_frequency_hz == pytest.approx(expected, rel=1e-10)

    def test_frequency_scales_with_field(self):
        e1 = ElectronSpecies(0.1)
        e5 = ElectronSpecies(0.5)
        assert e5.qubit_frequency_hz / e1.qubit_frequency_hz == pytest.approx(
            5.0, rel=1e-12
        )


class TestSpeciesProtocol:
    """Smoke tests for protocol attribute access.

    Full structural conformance is verified by running mypy --strict.
    """

    def test_ion_satisfies_protocol(self):
        """IonSpecies structurally satisfies Species protocol."""
        ion = get_species("Yb171")

        def accepts_species(s: Species) -> tuple[float, float]:
            return s.mass_kg, s.qubit_frequency_hz

        mass, freq = accepts_species(ion)
        assert mass > 0
        assert freq > 0

    def test_electron_satisfies_protocol(self):
        """ElectronSpecies structurally satisfies Species protocol."""
        electron = ElectronSpecies(magnetic_field=0.1)

        def accepts_species(s: Species) -> tuple[float, float]:
            return s.mass_kg, s.qubit_frequency_hz

        mass, freq = accepts_species(electron)
        assert mass > 0
        assert freq > 0

import numpy as np
import pytest

from tiqs.species.ion import get_species
from tiqs.species.electron import ElectronSpecies
from tiqs.species.transitions import Transition


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
        assert t.frequency == pytest.approx(8.114e14, rel=1e-3)

    def test_transition_wavevector(self):
        t = Transition(name="test", wavelength=369.5e-9, linewidth=1e6)
        assert t.wavevector == pytest.approx(2 * np.pi / 369.5e-9)


class TestIonSpecies:
    def test_create_species(self):
        s = get_species("Yb171")
        assert s.symbol == "Yb171"
        assert s.mass_amu == pytest.approx(170.936, rel=1e-3)
        assert s.nuclear_spin == pytest.approx(0.5)

    def test_yb171_qubit_frequency(self):
        s = get_species("Yb171")
        assert s.qubit_frequency_hz == pytest.approx(12.6428e9, rel=1e-3)

    def test_yb171_cooling_wavelength(self):
        s = get_species("Yb171")
        assert s.cooling_transition.wavelength == pytest.approx(
            369.5e-9, rel=1e-2
        )

    def test_ca40_optical_qubit(self):
        s = get_species("Ca40")
        assert s.qubit_type == "optical"
        assert s.qubit_wavelength == pytest.approx(729e-9, rel=1e-2)

    def test_ca43_hyperfine(self):
        s = get_species("Ca43")
        assert s.qubit_type == "hyperfine"
        assert s.qubit_frequency_hz == pytest.approx(3.2256e9, rel=1e-3)
        assert s.nuclear_spin == pytest.approx(3.5)

    def test_ba137_visible_wavelengths(self):
        s = get_species("Ba137")
        cooling_wl = s.cooling_transition.wavelength
        assert 490e-9 < cooling_wl < 500e-9

    def test_be9_lightest(self):
        s = get_species("Be9")
        assert s.mass_amu == pytest.approx(9.012, rel=1e-3)

    def test_sr88_optical(self):
        s = get_species("Sr88")
        assert s.qubit_type == "optical"

    def test_mass_kg(self):
        s = get_species("Yb171")
        assert s.mass_kg == pytest.approx(
            170.936 * 1.66053906660e-27, rel=1e-3
        )

    def test_doppler_limit(self):
        s = get_species("Ca40")
        T_D = s.doppler_limit_temperature()
        assert 0.1e-3 < T_D < 2e-3

    def test_unknown_species_raises(self):
        with pytest.raises(KeyError):
            get_species("Unobtanium")

    def test_all_species_available(self):
        for name in ["Yb171", "Ca40", "Ca43", "Ba137", "Be9", "Sr88"]:
            s = get_species(name)
            assert s.mass_amu > 0
            assert s.cooling_transition is not None


class TestElectronSpecies:
    def test_mass(self):
        e = ElectronSpecies(magnetic_field=0.1)
        assert e.mass_kg == pytest.approx(9.1094e-31, rel=1e-3)

    def test_qubit_frequency(self):
        """At B = 0.1 T, qubit frequency should be ~2.803 GHz."""
        e = ElectronSpecies(magnetic_field=0.1)
        assert e.qubit_frequency_hz == pytest.approx(2.803e9, rel=1e-2)

    def test_frequency_scales_with_field(self):
        e1 = ElectronSpecies(0.1)
        e5 = ElectronSpecies(0.5)
        assert e5.qubit_frequency_hz / e1.qubit_frequency_hz == pytest.approx(
            5.0, rel=1e-6
        )

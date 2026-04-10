from tiqs.constants import TWO_PI
from tiqs.cooling.doppler import doppler_cooled_nbar
from tiqs.cooling.eit_cooling import eit_cooling_nbar
from tiqs.cooling.sideband_cooling import (
    sideband_cooling_nbar,
    sideband_cooling_simulate,
)
from tiqs.hilbert_space.builder import HilbertSpace
from tiqs.hilbert_space.operators import OperatorFactory
from tiqs.species.ion import get_species


class TestDopplerCooling:
    def test_ca40_doppler_limit(self):
        species = get_species("Ca40")
        n_bar = doppler_cooled_nbar(species, trap_frequency_hz=1e6)
        assert 1 < n_bar < 50

    def test_higher_trap_freq_gives_lower_nbar(self):
        species = get_species("Ca40")
        n1 = doppler_cooled_nbar(species, trap_frequency_hz=1e6)
        n2 = doppler_cooled_nbar(species, trap_frequency_hz=3e6)
        assert n2 < n1

    def test_yb171_doppler(self):
        species = get_species("Yb171")
        n_bar = doppler_cooled_nbar(species, trap_frequency_hz=1e6)
        assert 1 < n_bar < 50


class TestSidebandCooling:
    def test_sbc_analytical_nbar(self):
        n_bar = sideband_cooling_nbar(
            gamma_eff=TWO_PI * 1e3,
            trap_frequency=TWO_PI * 1e6,
        )
        assert n_bar < 0.1

    def test_sbc_simulation_cools(self):
        """Resolved sideband cooling should bring a thermal state
        close to ground."""
        hs = HilbertSpace(n_ions=1, n_modes=1, n_fock=20)
        ops = OperatorFactory(hs)
        n_bar_initial = 5.0
        n_bar_final = sideband_cooling_simulate(
            ops,
            ion=0,
            mode=0,
            n_bar_initial=n_bar_initial,
            eta=0.1,
            rabi_frequency=TWO_PI * 100e3,
            optical_pumping_rate=TWO_PI * 10e3,
            n_cycles=30,
        )
        assert n_bar_final < n_bar_initial
        assert n_bar_final < 0.5


class TestEITCooling:
    def test_eit_cooling_nbar(self):
        n_bar = eit_cooling_nbar(
            gamma_eit=TWO_PI * 100e3,
            trap_frequency=TWO_PI * 1e6,
            carrier_suppression=0.01,
        )
        assert n_bar < 1.0

    def test_eit_cools_multiple_modes(self):
        """EIT cooling has broader bandwidth than sideband cooling."""
        n_bar_1MHz = eit_cooling_nbar(TWO_PI * 200e3, TWO_PI * 1e6, 0.01)
        n_bar_2MHz = eit_cooling_nbar(TWO_PI * 200e3, TWO_PI * 2e6, 0.01)
        # Both modes should be cooled, though less efficiently at higher freq
        assert n_bar_1MHz < 0.5
        assert n_bar_2MHz < 2.0

# tests/test_chain.py
import numpy as np
import pytest

from tiqs.chain.equilibrium import equilibrium_positions
from tiqs.chain.normal_modes import normal_modes, NormalModeResult
from tiqs.chain.lamb_dicke import lamb_dicke_parameters
from tiqs.species.data import get_species
from tiqs.trap.paul_trap import PaulTrap
from tiqs.constants import TWO_PI


@pytest.fixture
def ca40_trap():
    return PaulTrap(
        v_rf=300.0,
        omega_rf=TWO_PI * 30e6,
        r0=0.5e-3,
        omega_axial=TWO_PI * 1.0e6,
        species=get_species("Ca40"),
    )


class TestEquilibriumPositions:
    def test_single_ion_at_origin(self, ca40_trap):
        pos = equilibrium_positions(1, ca40_trap)
        assert len(pos) == 1
        assert pos[0] == pytest.approx(0.0)

    def test_two_ions_symmetric(self, ca40_trap):
        pos = equilibrium_positions(2, ca40_trap)
        assert len(pos) == 2
        assert pos[0] == pytest.approx(-pos[1])
        assert pos[1] > pos[0]

    def test_two_ion_spacing(self, ca40_trap):
        """Spacing d0 = (e^2 / (4*pi*eps0 * m * omega_ax^2))^(1/3) ~ 5 um for Ca40 @ 1 MHz."""
        pos = equilibrium_positions(2, ca40_trap)
        spacing = pos[1] - pos[0]
        assert 1e-6 < spacing < 20e-6

    def test_three_ions_symmetric(self, ca40_trap):
        pos = equilibrium_positions(3, ca40_trap)
        assert len(pos) == 3
        assert pos[0] == pytest.approx(-pos[2], abs=1e-12)
        assert pos[1] == pytest.approx(0.0, abs=1e-12)

    def test_five_ions_ordered(self, ca40_trap):
        pos = equilibrium_positions(5, ca40_trap)
        assert len(pos) == 5
        for i in range(4):
            assert pos[i] < pos[i + 1]

    def test_monotonic_spacing_decrease_from_center(self, ca40_trap):
        """Ions are closer together at the center of the chain."""
        pos = equilibrium_positions(5, ca40_trap)
        spacings = np.diff(pos)
        center_spacing = spacings[2]
        edge_spacing = spacings[0]
        assert center_spacing < edge_spacing


class TestNormalModes:
    def test_single_ion_one_axial_mode(self, ca40_trap):
        result = normal_modes(1, ca40_trap)
        assert len(result.axial_freqs) == 1
        assert result.axial_freqs[0] == pytest.approx(ca40_trap.omega_axial, rel=1e-6)

    def test_two_ion_com_mode(self, ca40_trap):
        result = normal_modes(2, ca40_trap)
        omega_com = result.axial_freqs[0]
        assert omega_com == pytest.approx(ca40_trap.omega_axial, rel=1e-6)

    def test_two_ion_stretch_mode(self, ca40_trap):
        """Stretch mode at sqrt(3) * omega_axial for two ions."""
        result = normal_modes(2, ca40_trap)
        omega_stretch = result.axial_freqs[1]
        expected = np.sqrt(3) * ca40_trap.omega_axial
        assert omega_stretch == pytest.approx(expected, rel=1e-4)

    def test_two_ion_com_eigenvector(self, ca40_trap):
        """COM mode: both ions oscillate in phase with equal amplitude."""
        result = normal_modes(2, ca40_trap)
        v_com = result.axial_vectors[:, 0]
        assert abs(v_com[0]) == pytest.approx(abs(v_com[1]), rel=1e-6)
        assert np.sign(v_com[0]) == np.sign(v_com[1])

    def test_two_ion_stretch_eigenvector(self, ca40_trap):
        """Stretch mode: ions oscillate out of phase."""
        result = normal_modes(2, ca40_trap)
        v_str = result.axial_vectors[:, 1]
        assert abs(v_str[0]) == pytest.approx(abs(v_str[1]), rel=1e-6)
        assert np.sign(v_str[0]) != np.sign(v_str[1])

    def test_three_ion_mode_count(self, ca40_trap):
        result = normal_modes(3, ca40_trap)
        assert len(result.axial_freqs) == 3

    def test_mode_frequencies_increasing(self, ca40_trap):
        result = normal_modes(5, ca40_trap)
        for i in range(len(result.axial_freqs) - 1):
            assert result.axial_freqs[i] < result.axial_freqs[i + 1]

    def test_eigenvectors_orthonormal(self, ca40_trap):
        result = normal_modes(4, ca40_trap)
        V = result.axial_vectors
        product = V.T @ V
        np.testing.assert_allclose(product, np.eye(4), atol=1e-10)

    def test_radial_modes_exist(self, ca40_trap):
        result = normal_modes(3, ca40_trap)
        assert len(result.radial_x_freqs) == 3
        assert len(result.radial_y_freqs) == 3

    def test_radial_frequencies_near_trap_radial(self, ca40_trap):
        result = normal_modes(2, ca40_trap)
        omega_r = ca40_trap.omega_radial
        for f in result.radial_x_freqs:
            assert abs(f - omega_r) / omega_r < 0.5


class TestLambDicke:
    def test_single_ion_single_mode(self, ca40_trap):
        modes = normal_modes(1, ca40_trap)
        laser_wavevector = TWO_PI / 729e-9
        eta = lamb_dicke_parameters(
            modes=modes,
            species=ca40_trap.species,
            k_eff=laser_wavevector,
            direction="axial",
        )
        assert eta.shape == (1, 1)
        assert 0.01 < eta[0, 0] < 0.5

    def test_two_ions_eta_matrix_shape(self, ca40_trap):
        modes = normal_modes(2, ca40_trap)
        laser_k = TWO_PI / 729e-9
        eta = lamb_dicke_parameters(modes, ca40_trap.species, laser_k, "axial")
        assert eta.shape == (2, 2)

    def test_com_mode_equal_coupling(self, ca40_trap):
        """Both ions couple equally to COM mode."""
        modes = normal_modes(2, ca40_trap)
        laser_k = TWO_PI / 729e-9
        eta = lamb_dicke_parameters(modes, ca40_trap.species, laser_k, "axial")
        assert eta[0, 0] == pytest.approx(eta[1, 0], rel=1e-6)

    def test_lighter_ion_larger_eta(self):
        """Be9 should have larger Lamb-Dicke parameter than Yb171 (eta ~ 1/sqrt(m))."""
        be_trap = PaulTrap(v_rf=300, omega_rf=TWO_PI * 30e6, r0=0.5e-3,
                           omega_axial=TWO_PI * 1e6, species=get_species("Be9"))
        yb_trap = PaulTrap(v_rf=1000, omega_rf=TWO_PI * 30e6, r0=0.5e-3,
                           omega_axial=TWO_PI * 1e6, species=get_species("Yb171"))
        be_modes = normal_modes(1, be_trap)
        yb_modes = normal_modes(1, yb_trap)
        k = TWO_PI / 400e-9
        eta_be = lamb_dicke_parameters(be_modes, be_trap.species, k, "axial")[0, 0]
        eta_yb = lamb_dicke_parameters(yb_modes, yb_trap.species, k, "axial")[0, 0]
        assert eta_be > eta_yb

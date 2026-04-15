# tests/test_chain.py
import numpy as np
import pytest

from tiqs.chain.equilibrium import equilibrium_positions
from tiqs.chain.lamb_dicke import lamb_dicke_parameters
from tiqs.chain.normal_modes import ModeGroup, normal_modes
from tiqs.constants import TWO_PI
from tiqs.species.ion import get_species
from tiqs.trap import PaulTrap


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
        """Spacing d0 = (e^2 / (4*pi*eps0 * m * omega_ax^2))^(1/3)
        ~ 5 um for Ca40 @ 1 MHz."""
        pos = equilibrium_positions(2, ca40_trap)
        spacing = pos[1] - pos[0]
        assert 1e-6 < spacing < 20e-6

    def test_three_ion_center_at_origin(self, ca40_trap):
        """[James1998]: center ion at z=0 by symmetry."""
        pos = equilibrium_positions(3, ca40_trap)
        assert len(pos) == 3
        assert pos[0] == pytest.approx(-pos[2], abs=1e-12)
        assert pos[1] == pytest.approx(0.0, abs=1e-12)

    def test_three_ion_dimensionless_offset(self, ca40_trap):
        """[James1998] Table I: outer ions at u = +/-(5/4)^(1/3)
        in dimensionless units."""
        from tiqs.constants import ELECTRON_CHARGE, EPSILON_0, PI

        pos = equilibrium_positions(3, ca40_trap)
        length_scale = (
            ELECTRON_CHARGE**2
            / (
                4
                * PI
                * EPSILON_0
                * ca40_trap.species.mass_kg
                * ca40_trap.omega_axial**2
            )
        ) ** (1 / 3)
        u_outer = pos[2] / length_scale
        assert u_outer == pytest.approx((5 / 4) ** (1 / 3), rel=1e-3)

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
        axial = result.modes["axial"]
        assert len(axial.freqs) == 1
        assert axial.freqs[0] == pytest.approx(ca40_trap.omega_axial, rel=1e-6)

    def test_two_ion_com_mode(self, ca40_trap):
        result = normal_modes(2, ca40_trap)
        omega_com = result.modes["axial"].freqs[0]
        assert omega_com == pytest.approx(ca40_trap.omega_axial, rel=1e-6)

    def test_two_ion_stretch_mode(self, ca40_trap):
        """Stretch mode at sqrt(3) * omega_axial for two ions."""
        result = normal_modes(2, ca40_trap)
        omega_stretch = result.modes["axial"].freqs[1]
        expected = np.sqrt(3) * ca40_trap.omega_axial
        assert omega_stretch == pytest.approx(expected, rel=1e-4)

    def test_two_ion_com_eigenvector(self, ca40_trap):
        """COM mode: both ions oscillate in phase with equal amplitude."""
        result = normal_modes(2, ca40_trap)
        v_com = result.modes["axial"].vectors[:, 0]
        assert abs(v_com[0]) == pytest.approx(abs(v_com[1]), rel=1e-6)
        assert np.sign(v_com[0]) == np.sign(v_com[1])

    def test_two_ion_stretch_eigenvector(self, ca40_trap):
        """Stretch mode: ions oscillate out of phase."""
        result = normal_modes(2, ca40_trap)
        v_str = result.modes["axial"].vectors[:, 1]
        assert abs(v_str[0]) == pytest.approx(abs(v_str[1]), rel=1e-6)
        assert np.sign(v_str[0]) != np.sign(v_str[1])

    def test_three_ion_tilt_mode_ratio(self, ca40_trap):
        """[James1998] Table I: tilt mode at sqrt(3) * omega_z."""
        result = normal_modes(3, ca40_trap)
        axial = result.modes["axial"]
        ratio = axial.freqs[1] / axial.freqs[0]
        assert ratio == pytest.approx(np.sqrt(3), rel=1e-4)

    def test_three_ion_breathing_mode_ratio(self, ca40_trap):
        """[James1998] Table I: breathing mode at sqrt(29/5) * omega_z."""
        result = normal_modes(3, ca40_trap)
        axial = result.modes["axial"]
        ratio = axial.freqs[2] / axial.freqs[0]
        assert ratio == pytest.approx(np.sqrt(29 / 5), rel=1e-4)

    def test_three_ion_mode_count(self, ca40_trap):
        result = normal_modes(3, ca40_trap)
        assert len(result.modes["axial"].freqs) == 3

    def test_mode_frequencies_increasing(self, ca40_trap):
        result = normal_modes(5, ca40_trap)
        freqs = result.modes["axial"].freqs
        for i in range(len(freqs) - 1):
            assert freqs[i] < freqs[i + 1]

    def test_eigenvectors_orthonormal(self, ca40_trap):
        result = normal_modes(4, ca40_trap)
        V = result.modes["axial"].vectors
        product = V.T @ V
        np.testing.assert_allclose(product, np.eye(4), atol=1e-10)

    def test_radial_modes_exist(self, ca40_trap):
        result = normal_modes(3, ca40_trap)
        assert len(result.modes["radial_x"].freqs) == 3
        assert len(result.modes["radial_y"].freqs) == 3

    def test_radial_frequencies_near_trap_radial(self, ca40_trap):
        result = normal_modes(2, ca40_trap)
        omega_r = ca40_trap.omega_radial
        for f in result.modes["radial_x"].freqs:
            assert abs(f - omega_r) / omega_r < 0.5

    def test_paul_trap_mode_labels(self, ca40_trap):
        """Paul trap produces axial, radial_x, radial_y mode groups."""
        result = normal_modes(2, ca40_trap)
        assert set(result.modes.keys()) == {"axial", "radial_x", "radial_y"}

    def test_mode_group_structure(self, ca40_trap):
        result = normal_modes(2, ca40_trap)
        for _label, group in result.modes.items():
            assert isinstance(group, ModeGroup)
            assert group.freqs.shape == (2,)
            assert group.vectors.shape == (2, 2)


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
        """Be9 should have larger Lamb-Dicke parameter than Yb171
        (eta ~ 1/sqrt(m))."""
        be_trap = PaulTrap(
            v_rf=300,
            omega_rf=TWO_PI * 30e6,
            r0=0.5e-3,
            omega_axial=TWO_PI * 1e6,
            species=get_species("Be9"),
        )
        yb_trap = PaulTrap(
            v_rf=1000,
            omega_rf=TWO_PI * 30e6,
            r0=0.5e-3,
            omega_axial=TWO_PI * 1e6,
            species=get_species("Yb171"),
        )
        be_modes = normal_modes(1, be_trap)
        yb_modes = normal_modes(1, yb_trap)
        k = TWO_PI / 400e-9
        eta_be = lamb_dicke_parameters(be_modes, be_trap.species, k, "axial")[
            0, 0
        ]
        eta_yb = lamb_dicke_parameters(yb_modes, yb_trap.species, k, "axial")[
            0, 0
        ]
        assert eta_be > eta_yb

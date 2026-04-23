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
        from tiqs.constants import COULOMB_CONSTANT

        pos = equilibrium_positions(3, ca40_trap)
        length_scale = (
            COULOMB_CONSTANT
            / (ca40_trap.species.mass_kg * ca40_trap.omega_axial**2)
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

    def test_paul_trap_positive_energy(self, ca40_trap):
        """All Paul trap modes have positive energy."""
        result = normal_modes(2, ca40_trap)
        for group in result.modes.values():
            assert group.negative_energy is False


class TestPenningModeEnergy:
    """Magnetron mode has negative energy in a Penning trap."""

    def test_magnetron_negative_energy(self, penning_trap):
        result = normal_modes(1, penning_trap)
        assert result.modes["magnetron"].negative_energy is True

    def test_axial_positive_energy(self, penning_trap):
        result = normal_modes(1, penning_trap)
        assert result.modes["axial"].negative_energy is False

    def test_cyclotron_positive_energy(self, penning_trap):
        result = normal_modes(1, penning_trap)
        assert result.modes["modified_cyclotron"].negative_energy is False


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


@pytest.fixture
def be9():
    return get_species("Be9")


@pytest.fixture
def ca40():
    return get_species("Ca40")


@pytest.fixture
def mixed_trap(ca40):
    """Paul trap configured for Ca40 (reference species)."""
    return PaulTrap(
        v_rf=300.0,
        omega_rf=TWO_PI * 30e6,
        r0=0.5e-3,
        omega_axial=TWO_PI * 1.0e6,
        species=ca40,
    )


@pytest.fixture
def penning_trap(ca40):
    from tiqs.trap import PenningTrap

    return PenningTrap(
        magnetic_field=7.0,
        species=ca40,
        d=5e-3,
        omega_axial=TWO_PI * 0.5e6,
    )


class TestMixedSpeciesNormalModes:
    """Mixed-species chain tests using the mass-weighted dynamical matrix."""

    def test_uniform_masses_matches_single_species(self, ca40, mixed_trap):
        """Passing explicit uniform masses must reproduce the default."""
        result_default = normal_modes(2, mixed_trap)
        masses = np.array([ca40.mass_kg, ca40.mass_kg])
        result_explicit = normal_modes(2, mixed_trap, masses=masses)

        np.testing.assert_allclose(
            result_explicit.modes["axial"].freqs,
            result_default.modes["axial"].freqs,
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            np.abs(result_explicit.modes["axial"].vectors),
            np.abs(result_default.modes["axial"].vectors),
            atol=1e-12,
        )

    def test_two_ion_analytical_frequencies(self, be9, ca40, mixed_trap):
        r"""Verify against the analytical two-ion mixed-species formula.

        For ions with masses $m_1$, $m_2$ sharing axial spring constant
        $K = m_\mathrm{ref}\,\omega_{z,\mathrm{ref}}^2$, the mode
        frequencies are (Home 2013, Sosnova 2021):

        $$
        \omega_\pm^2 = \omega_{z,1}^2 \,
        \frac{1 + \mu \pm \sqrt{1 - \mu + \mu^2}}{\mu}
        $$

        where $\mu = m_2 / m_1$.
        """
        masses = np.array([be9.mass_kg, ca40.mass_kg])
        result = normal_modes(2, mixed_trap, masses=masses)
        axial_freqs = result.modes["axial"].freqs

        K = ca40.mass_kg * mixed_trap.omega_axial**2
        omega_z1 = np.sqrt(K / be9.mass_kg)
        mu = ca40.mass_kg / be9.mass_kg

        discriminant = np.sqrt(1 - mu + mu**2)
        omega_ip_sq = omega_z1**2 * (1 + mu - discriminant) / mu
        omega_op_sq = omega_z1**2 * (1 + mu + discriminant) / mu

        assert axial_freqs[0] == pytest.approx(np.sqrt(omega_ip_sq), rel=1e-4)
        assert axial_freqs[1] == pytest.approx(np.sqrt(omega_op_sq), rel=1e-4)

    def test_eigenvectors_orthonormal_mixed(self, be9, ca40, mixed_trap):
        """Mass-weighted eigenvectors satisfy e^T e = I."""
        masses = np.array([be9.mass_kg, ca40.mass_kg])
        result = normal_modes(2, mixed_trap, masses=masses)
        V = result.modes["axial"].vectors
        product = V.T @ V
        np.testing.assert_allclose(product, np.eye(2), atol=1e-10)

    def test_lighter_ion_larger_participation(self, be9, ca40, mixed_trap):
        """In the out-of-phase mode, the lighter ion has the larger
        mass-weighted eigenvector component."""
        masses = np.array([be9.mass_kg, ca40.mass_kg])
        result = normal_modes(2, mixed_trap, masses=masses)
        v_op = result.modes["axial"].vectors[:, 1]
        assert abs(v_op[0]) > abs(v_op[1])

    def test_mixed_frequencies_differ_from_single_species(
        self, be9, ca40, mixed_trap
    ):
        """Mixed-species mode frequencies must differ from single-species."""
        result_single = normal_modes(2, mixed_trap)
        masses = np.array([be9.mass_kg, ca40.mass_kg])
        result_mixed = normal_modes(2, mixed_trap, masses=masses)

        single_freqs = result_single.modes["axial"].freqs
        mixed_freqs = result_mixed.modes["axial"].freqs
        assert not np.allclose(single_freqs, mixed_freqs, rtol=1e-3)

    def test_radial_modes_mixed_species(self, be9, ca40, mixed_trap):
        """Per-ion Mathieu parameters produce radial modes that differ
        from the single-species case."""
        masses = np.array([be9.mass_kg, ca40.mass_kg])
        result = normal_modes(2, mixed_trap, masses=masses)
        radial = result.modes["radial_x"]

        assert radial.freqs.shape == (2,)
        assert radial.vectors.shape == (2, 2)
        assert radial.freqs[0] > 0
        assert radial.freqs[1] > radial.freqs[0]

    def test_masses_wrong_length_raises(self, mixed_trap):
        with pytest.raises(ValueError, match="masses must have shape"):
            normal_modes(2, mixed_trap, masses=np.array([1.0, 2.0, 3.0]))

    def test_penning_mixed_species_axial(self, be9, ca40, penning_trap):
        """Mixed-species Penning axial modes use the dynamical matrix."""
        masses = np.array([be9.mass_kg, ca40.mass_kg])
        result = normal_modes(2, penning_trap, masses=masses)

        axial = result.modes["axial"]
        assert axial.freqs.shape == (2,)
        assert axial.freqs[0] > 0
        assert axial.freqs[1] > axial.freqs[0]

    def test_penning_mixed_species_transverse(self, be9, ca40, penning_trap):
        """Mixed-species Penning transverse modes have per-ion frequencies."""
        masses = np.array([be9.mass_kg, ca40.mass_kg])
        result = normal_modes(2, penning_trap, masses=masses)

        cyc = result.modes["modified_cyclotron"]
        mag = result.modes["magnetron"]
        assert cyc.freqs.shape == (2,)
        assert mag.freqs.shape == (2,)
        # Lighter Be9 has higher cyclotron frequency
        assert cyc.freqs[1] > cyc.freqs[0]
        # Both magnetron frequencies are positive
        assert mag.freqs[0] > 0

    def test_paul_radial_instability_raises(self, ca40, mixed_trap):
        """Very heavy species can be radially unstable."""
        from tiqs.constants import AMU

        masses = np.array([ca40.mass_kg, 1000 * AMU])
        with pytest.raises(ValueError, match="Radially unstable"):
            normal_modes(2, mixed_trap, masses=masses)

    def test_penning_instability_raises(self, ca40):
        """Heavy species can be Penning-unstable at low B field."""
        from tiqs.trap import PenningTrap

        weak_trap = PenningTrap(
            magnetic_field=0.01,
            species=ca40,
            d=5e-3,
            omega_axial=TWO_PI * 0.5e6,
        )
        with pytest.raises(ValueError, match="Penning-unstable"):
            normal_modes(2, weak_trap, masses=np.array([ca40.mass_kg] * 2))

    def test_three_ion_mixed_chain(self, be9, ca40, mixed_trap):
        """Three-ion mixed chain exercises the general N-body dynamical
        matrix beyond the symmetric 2-ion case."""
        masses = np.array([be9.mass_kg, ca40.mass_kg, be9.mass_kg])
        result = normal_modes(3, mixed_trap, masses=masses)

        axial = result.modes["axial"]
        assert axial.freqs.shape == (3,)
        assert axial.vectors.shape == (3, 3)
        # Frequencies are positive and ascending
        for i in range(2):
            assert axial.freqs[i + 1] > axial.freqs[i] > 0
        # Eigenvectors are orthonormal
        np.testing.assert_allclose(
            axial.vectors.T @ axial.vectors, np.eye(3), atol=1e-10
        )


class TestMixedSpeciesLambDicke:
    """Lamb-Dicke parameters for mixed-species chains."""

    def test_per_ion_mass_lighter_ion_larger_eta(self, be9, ca40, mixed_trap):
        """Be9 (lighter) gets a larger Lamb-Dicke parameter than Ca40
        on the out-of-phase mode, where the lighter ion dominates."""
        masses = np.array([be9.mass_kg, ca40.mass_kg])
        modes = normal_modes(2, mixed_trap, masses=masses)
        k = TWO_PI / 400e-9
        eta = lamb_dicke_parameters(modes, [be9, ca40], k, "axial")

        assert eta.shape == (2, 2)
        # On the out-of-phase (higher) mode, Be9 has larger eta
        assert abs(eta[0, 1]) > abs(eta[1, 1])

    def test_per_ion_k_eff(self, ca40, mixed_trap):
        """Per-ion k_eff values produce different eta even for equal masses."""
        modes = normal_modes(2, mixed_trap)
        k1 = TWO_PI / 729e-9
        k2 = TWO_PI / 400e-9
        eta = lamb_dicke_parameters(modes, ca40, [k1, k2], "axial")

        assert eta.shape == (2, 2)
        assert not np.allclose(eta[0, :], eta[1, :])

    def test_species_list_wrong_length_raises(self, be9, ca40, mixed_trap):
        modes = normal_modes(2, mixed_trap)
        with pytest.raises(ValueError, match="species list length"):
            lamb_dicke_parameters(modes, [be9, ca40, be9], TWO_PI / 400e-9)

    def test_k_eff_list_wrong_length_raises(self, ca40, mixed_trap):
        modes = normal_modes(2, mixed_trap)
        with pytest.raises(ValueError, match="k_eff list length"):
            lamb_dicke_parameters(modes, ca40, [1.0, 2.0, 3.0])

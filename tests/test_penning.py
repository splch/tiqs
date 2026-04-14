"""Tests for Penning trap confinement support."""

import numpy as np
import pytest

from tiqs.chain.equilibrium import equilibrium_positions
from tiqs.chain.lamb_dicke import lamb_dicke_parameters
from tiqs.chain.normal_modes import (
    NormalModeResult,
    PenningNormalModeResult,
    normal_modes,
)
from tiqs.constants import TWO_PI
from tiqs.species.ion import get_species
from tiqs.trap import PaulTrap, PenningTrap


@pytest.fixture
def be9_penning():
    """Be-9+ in a 3T Penning trap (ETH-style parameters)."""
    return PenningTrap(
        magnetic_field=3.0,
        species=get_species("Be9"),
        d=300e-6,
        omega_axial=TWO_PI * 1.5e6,
    )


class TestPenningTrap:
    def test_stability(self, be9_penning):
        assert be9_penning.is_stable()

    def test_unstable_trap(self):
        """omega_z > omega_c/sqrt(2) should be unstable."""
        trap = PenningTrap(
            magnetic_field=0.001,
            species=get_species("Be9"),
            d=300e-6,
            omega_axial=TWO_PI * 100e6,
        )
        assert not trap.is_stable()

    def test_frequency_hierarchy(self, be9_penning):
        """omega_- < omega_z < omega_+ < omega_c."""
        wm, wz, wp, wc = be9_penning.frequency_hierarchy
        assert wm < wz < wp < wc

    def test_cyclotron_sum_rule(self, be9_penning):
        """omega_+ + omega_- = omega_c."""
        wp = be9_penning.omega_cyclotron
        wm = be9_penning.omega_magnetron
        wc = be9_penning.omega_cyclotron_free
        assert wp + wm == pytest.approx(wc, rel=1e-10)

    def test_brown_gabrielse_invariant(self, be9_penning):
        """omega_+^2 + omega_-^2 + omega_z^2 = omega_c^2."""
        wp = be9_penning.omega_cyclotron
        wm = be9_penning.omega_magnetron
        wz = be9_penning.omega_axial
        wc = be9_penning.omega_cyclotron_free
        assert wp**2 + wm**2 + wz**2 == pytest.approx(wc**2, rel=1e-10)

    def test_product_relation(self, be9_penning):
        """omega_+ * omega_- = omega_z^2 / 2."""
        wp = be9_penning.omega_cyclotron
        wm = be9_penning.omega_magnetron
        wz = be9_penning.omega_axial
        assert wp * wm == pytest.approx(wz**2 / 2, rel=1e-10)

    def test_v_dc_derived_from_omega_axial(self, be9_penning):
        assert be9_penning.v_dc is not None
        assert be9_penning.v_dc > 0

    def test_omega_axial_derived_from_v_dc(self):
        be = get_species("Be9")
        trap = PenningTrap(magnetic_field=3.0, species=be, d=300e-6, v_dc=10.0)
        assert trap.omega_axial > 0

    def test_neither_raises(self):
        with pytest.raises(ValueError, match="Must specify"):
            PenningTrap(
                magnetic_field=3.0,
                species=get_species("Be9"),
                d=300e-6,
            )

    def test_trap_depth_positive(self, be9_penning):
        assert be9_penning.trap_depth_axial_eV > 0

    def test_magnetron_less_than_axial(self, be9_penning):
        assert be9_penning.omega_magnetron < be9_penning.omega_axial

    def test_cyclotron_greater_than_axial(self, be9_penning):
        assert be9_penning.omega_cyclotron > be9_penning.omega_axial


class TestPenningEquilibrium:
    def test_single_ion_at_origin(self, be9_penning):
        pos = equilibrium_positions(1, be9_penning)
        assert len(pos) == 1
        assert pos[0] == pytest.approx(0.0)

    def test_two_ions_symmetric(self, be9_penning):
        pos = equilibrium_positions(2, be9_penning)
        assert len(pos) == 2
        assert pos[0] == pytest.approx(-pos[1])

    def test_reasonable_spacing(self, be9_penning):
        pos = equilibrium_positions(2, be9_penning)
        spacing = pos[1] - pos[0]
        assert 1e-6 < spacing < 100e-6


class TestPenningNormalModes:
    def test_single_ion_returns_penning_result(self, be9_penning):
        result = normal_modes(1, be9_penning)
        assert isinstance(result, PenningNormalModeResult)

    def test_single_ion_axial_frequency(self, be9_penning):
        result = normal_modes(1, be9_penning)
        assert result.axial_freqs[0] == pytest.approx(
            be9_penning.omega_axial, rel=1e-6
        )

    def test_single_ion_cyclotron_frequency(self, be9_penning):
        result = normal_modes(1, be9_penning)
        assert result.cyclotron_freqs[0] == pytest.approx(
            be9_penning.omega_cyclotron, rel=1e-6
        )

    def test_single_ion_magnetron_frequency(self, be9_penning):
        result = normal_modes(1, be9_penning)
        assert result.magnetron_freqs[0] == pytest.approx(
            be9_penning.omega_magnetron, rel=1e-6
        )

    def test_two_ion_mode_counts(self, be9_penning):
        result = normal_modes(2, be9_penning)
        assert len(result.axial_freqs) == 2
        assert len(result.cyclotron_freqs) == 2
        assert len(result.magnetron_freqs) == 2

    def test_two_ion_axial_com(self, be9_penning):
        """Axial COM mode should equal omega_z."""
        result = normal_modes(2, be9_penning)
        assert result.axial_freqs[0] == pytest.approx(
            be9_penning.omega_axial, rel=1e-4
        )

    def test_two_ion_axial_stretch(self, be9_penning):
        """Axial stretch mode at sqrt(3) * omega_z."""
        result = normal_modes(2, be9_penning)
        expected = np.sqrt(3) * be9_penning.omega_axial
        assert result.axial_freqs[1] == pytest.approx(expected, rel=1e-4)

    def test_paul_trap_still_returns_normal_result(self):
        """Paul trap should still return NormalModeResult."""
        trap = PaulTrap(
            v_rf=300,
            omega_rf=TWO_PI * 30e6,
            r0=0.5e-3,
            omega_axial=TWO_PI * 1e6,
            species=get_species("Ca40"),
        )
        result = normal_modes(2, trap)
        assert isinstance(result, NormalModeResult)


class TestPenningLambDicke:
    def test_axial_lamb_dicke(self, be9_penning):
        modes = normal_modes(1, be9_penning)
        k_eff = TWO_PI / 313e-9
        eta = lamb_dicke_parameters(modes, be9_penning.species, k_eff, "axial")
        assert eta.shape == (1, 1)
        assert 0.01 < abs(eta[0, 0]) < 1.0

    def test_cyclotron_direction(self, be9_penning):
        modes = normal_modes(1, be9_penning)
        k_eff = TWO_PI / 313e-9
        eta = lamb_dicke_parameters(
            modes, be9_penning.species, k_eff, "cyclotron"
        )
        assert eta.shape == (1, 1)
        assert abs(eta[0, 0]) > 0

    def test_magnetron_direction(self, be9_penning):
        modes = normal_modes(1, be9_penning)
        k_eff = TWO_PI / 313e-9
        eta = lamb_dicke_parameters(
            modes, be9_penning.species, k_eff, "magnetron"
        )
        assert eta.shape == (1, 1)
        # Magnetron eta should be larger than cyclotron (lower frequency)
        eta_cyc = lamb_dicke_parameters(
            modes, be9_penning.species, k_eff, "cyclotron"
        )
        assert abs(eta[0, 0]) > abs(eta_cyc[0, 0])

    def test_invalid_direction_raises(self, be9_penning):
        modes = normal_modes(1, be9_penning)
        with pytest.raises(ValueError, match="Unknown direction"):
            lamb_dicke_parameters(modes, be9_penning.species, 1e6, "radial_x")

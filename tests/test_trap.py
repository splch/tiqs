# tests/test_trap.py
import numpy as np
import pytest

from tiqs.species.ion import get_species
from tiqs.trap import PaulTrap


class TestPaulTrap:
    @pytest.fixture
    def yb_trap(self):
        """Standard Yb171 trap: V_rf=1000V, Omega_rf=2pi*30MHz,
        r0=0.5mm, U_dc for 1MHz axial."""
        return PaulTrap(
            v_rf=1000.0,
            omega_rf=2 * np.pi * 30e6,
            r0=0.5e-3,
            u_dc_axial=None,
            omega_axial=2 * np.pi * 1.0e6,
            species=get_species("Yb171"),
        )

    def test_mathieu_q(self, yb_trap):
        q = yb_trap.mathieu_q
        assert 0.01 < q < 0.9

    def test_mathieu_a(self, yb_trap):
        a = yb_trap.mathieu_a
        assert abs(a) < 0.5

    def test_stability(self, yb_trap):
        assert yb_trap.is_stable()

    def test_secular_frequency_radial(self, yb_trap):
        omega_r = yb_trap.omega_radial
        assert 0.5e6 < omega_r / (2 * np.pi) < 10e6

    def test_secular_frequency_axial(self, yb_trap):
        omega_a = yb_trap.omega_axial
        assert omega_a == pytest.approx(2 * np.pi * 1.0e6)

    def test_pseudopotential_depth(self, yb_trap):
        depth_eV = yb_trap.pseudopotential_depth_eV
        assert 0.01 < depth_eV < 100

    def test_ion_electrode_distance(self, yb_trap):
        assert yb_trap.r0 == pytest.approx(0.5e-3)

    def test_unstable_trap_detected(self):
        trap = PaulTrap(
            v_rf=5000.0,
            omega_rf=2 * np.pi * 1e6,
            r0=0.1e-3,
            omega_axial=2 * np.pi * 0.5e6,
            species=get_species("Yb171"),
        )
        assert not trap.is_stable()

    def test_radial_exceeds_axial(self, yb_trap):
        """Radial frequency should exceed axial for linear chain stability."""
        assert yb_trap.omega_radial > yb_trap.omega_axial

    def test_micromotion_amplitude(self, yb_trap):
        """Intrinsic micromotion amplitude at a displacement of 1 um
        from RF null."""
        displacement = 1e-6
        amp = yb_trap.micromotion_amplitude(displacement)
        assert amp > 0
        assert (
            amp < displacement
        )  # micromotion amplitude is smaller than displacement for q < 1

    def test_excess_micromotion_from_stray_field(self, yb_trap):
        """Stray field of 1 V/m displaces ion from RF null."""
        stray_E = 1.0  # V/m
        displacement = yb_trap.stray_field_displacement(stray_E)
        assert displacement > 0
        assert displacement < 1e-3  # less than trap size

# tests/test_trap.py
import numpy as np
import pytest

from tiqs.species.electron import ElectronSpecies
from tiqs.species.ion import get_species
from tiqs.trap import PaulTrap, PenningTrap, Trap


class TestPaulTrap:
    @pytest.fixture
    def yb_trap(self):
        """Standard Yb171 trap: V_rf=1000V, Omega_rf=2pi*30MHz,
        r0=0.5mm, 1MHz axial."""
        return PaulTrap(
            v_rf=1000.0,
            omega_rf=2 * np.pi * 30e6,
            r0=0.5e-3,
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

    def test_unstable_trap_omega_radial_raises(self):
        """Accessing omega_radial when beta^2 <= 0 raises ValueError.
        Strong axial (25 MHz) with weak RF (10 V) makes radial
        confinement impossible."""
        trap = PaulTrap(
            v_rf=10.0,
            omega_rf=2 * np.pi * 30e6,
            r0=0.5e-3,
            omega_axial=2 * np.pi * 25e6,
            species=get_species("Yb171"),
        )
        with pytest.raises(ValueError, match="unstable"):
            _ = trap.omega_radial

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


class TestPaulTrapFactory:
    def test_from_dc_voltage(self):
        """Construct PaulTrap from voltage; omega_axial is derived."""
        trap = PaulTrap.from_dc_voltage(
            v_rf=300.0,
            omega_rf=2 * np.pi * 30e6,
            r0=0.5e-3,
            species=get_species("Ca40"),
            u_dc_axial=10.0,
        )
        assert trap.omega_axial > 0

    def test_u_dc_axial_property_roundtrip(self):
        """omega_axial -> u_dc_axial -> omega_axial roundtrips."""
        trap = PaulTrap(
            v_rf=300.0,
            omega_rf=2 * np.pi * 30e6,
            r0=0.5e-3,
            species=get_species("Ca40"),
            omega_axial=2 * np.pi * 1e6,
        )
        trap2 = PaulTrap.from_dc_voltage(
            v_rf=300.0,
            omega_rf=2 * np.pi * 30e6,
            r0=0.5e-3,
            species=get_species("Ca40"),
            u_dc_axial=trap.u_dc_axial,
        )
        assert trap2.omega_axial == pytest.approx(trap.omega_axial, rel=1e-10)


class TestTrapProtocol:
    def test_paul_trap_satisfies_protocol(self):
        """PaulTrap structurally satisfies Trap protocol."""
        trap = PaulTrap(
            v_rf=300.0,
            omega_rf=2 * np.pi * 30e6,
            r0=0.5e-3,
            species=get_species("Ca40"),
            omega_axial=2 * np.pi * 1e6,
        )

        def accepts_trap(t: Trap) -> tuple[float, bool]:
            return t.omega_axial, t.is_stable()

        freq, stable = accepts_trap(trap)
        assert freq > 0
        assert stable

    def test_penning_trap_satisfies_protocol(self):
        """PenningTrap structurally satisfies Trap protocol."""
        trap = PenningTrap(
            magnetic_field=5.0,
            species=ElectronSpecies(magnetic_field=5.0),
            d=3.5e-3,
            omega_axial=2 * np.pi * 64e6,
        )

        def accepts_trap(t: Trap) -> float:
            return t.omega_axial

        assert accepts_trap(trap) > 0


class TestPenningTrap:
    @pytest.fixture
    def electron_penning(self):
        """Electron in a 5T Penning trap, d=3.5mm, omega_z=2pi*64MHz."""
        return PenningTrap(
            magnetic_field=5.0,
            species=ElectronSpecies(magnetic_field=5.0),
            d=3.5e-3,
            omega_axial=2 * np.pi * 64e6,
        )

    def test_cyclotron_frequency(self, electron_penning):
        """omega_c = eB/m ~ 2pi*140 GHz for electrons at 5T."""
        omega_c = electron_penning.omega_cyclotron
        assert omega_c / (2 * np.pi) == pytest.approx(140e9, rel=0.01)

    def test_modified_cyclotron_near_cyclotron(self, electron_penning):
        """omega_+ is slightly less than omega_c."""
        wp = electron_penning.omega_modified_cyclotron
        assert wp < electron_penning.omega_cyclotron
        assert wp > electron_penning.omega_axial

    def test_magnetron_frequency_positive(self, electron_penning):
        """Magnetron frequency is positive and much smaller than cyclotron."""
        omega_m = electron_penning.omega_magnetron
        assert omega_m > 0
        assert omega_m < electron_penning.omega_axial

    def test_frequency_hierarchy(self, electron_penning):
        """omega_- << omega_z << omega_+ ~ omega_c."""
        wm = electron_penning.omega_magnetron
        wz = electron_penning.omega_axial
        wp = electron_penning.omega_modified_cyclotron
        assert wm < wz < wp

    def test_frequency_invariant(self, electron_penning):
        """omega_+^2 + omega_-^2 + omega_z^2 = omega_c^2."""
        wc = electron_penning.omega_cyclotron
        wp = electron_penning.omega_modified_cyclotron
        wm = electron_penning.omega_magnetron
        wz = electron_penning.omega_axial
        assert wp**2 + wm**2 + wz**2 == pytest.approx(wc**2, rel=1e-10)

    def test_stability(self, electron_penning):
        assert electron_penning.is_stable()

    def test_unstable_penning(self):
        """omega_c < sqrt(2)*omega_z is unstable."""
        trap = PenningTrap(
            magnetic_field=0.001,
            species=ElectronSpecies(magnetic_field=0.001),
            d=3.5e-3,
            omega_axial=2 * np.pi * 64e6,
        )
        assert not trap.is_stable()

    def test_unstable_penning_raises_on_transverse_freq(self):
        """Accessing transverse frequencies on an unstable trap raises."""
        trap = PenningTrap(
            magnetic_field=0.001,
            species=ElectronSpecies(magnetic_field=0.001),
            d=3.5e-3,
            omega_axial=2 * np.pi * 64e6,
        )
        with pytest.raises(ValueError, match="unstable"):
            _ = trap.omega_modified_cyclotron
        with pytest.raises(ValueError, match="unstable"):
            _ = trap.omega_magnetron

    def test_negative_voltage_raises(self):
        """Negative voltage in from_dc_voltage raises ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            PaulTrap.from_dc_voltage(
                v_rf=300.0,
                omega_rf=2 * np.pi * 30e6,
                r0=0.5e-3,
                species=get_species("Ca40"),
                u_dc_axial=-10.0,
            )
        with pytest.raises(ValueError, match="non-negative"):
            PenningTrap.from_dc_voltage(
                magnetic_field=5.0,
                species=ElectronSpecies(magnetic_field=5.0),
                d=3.5e-3,
                v_dc=-100.0,
            )

    def test_mismatched_magnetic_field_raises(self):
        """PenningTrap rejects inconsistent species magnetic field."""
        with pytest.raises(ValueError, match="must match"):
            PenningTrap(
                magnetic_field=5.0,
                species=ElectronSpecies(magnetic_field=3.0),
                d=3.5e-3,
                omega_axial=2 * np.pi * 64e6,
            )


class TestPenningTrapFactory:
    def test_from_dc_voltage(self):
        species = ElectronSpecies(magnetic_field=5.0)
        trap = PenningTrap.from_dc_voltage(
            magnetic_field=5.0,
            species=species,
            d=3.5e-3,
            v_dc=100.0,
        )
        assert trap.omega_axial > 0

    def test_v_dc_property_roundtrip(self):
        species = ElectronSpecies(magnetic_field=5.0)
        trap = PenningTrap(
            magnetic_field=5.0,
            species=species,
            d=3.5e-3,
            omega_axial=2 * np.pi * 64e6,
        )
        trap2 = PenningTrap.from_dc_voltage(
            magnetic_field=5.0,
            species=species,
            d=3.5e-3,
            v_dc=trap.v_dc,
        )
        assert trap2.omega_axial == pytest.approx(trap.omega_axial, rel=1e-10)


class TestMagneticBottle:
    """Tests for the magnetic bottle (B2) feature on PenningTrap."""

    @pytest.fixture
    def bottle_trap(self):
        """Electron Penning trap with a magnetic bottle."""
        return PenningTrap(
            magnetic_field=6.0,
            species=ElectronSpecies(magnetic_field=6.0),
            d=3.0e-3,
            omega_axial=2 * np.pi * 200e6,
            b2=9000.0,
        )

    def test_bottle_shift_fan_2025(self, bottle_trap):
        """Fan et al. PRA 111 (2025): B2=9000 T/m^2 at 200 MHz
        gives delta/(2pi) ~ 23 Hz."""
        delta_nu = bottle_trap.bottle_shift / (2 * np.pi)
        assert delta_nu == pytest.approx(23, rel=0.05)

    def test_bottle_shift_zero_without_b2(self):
        """No bottle (b2=0) gives zero shift."""
        trap = PenningTrap(
            magnetic_field=5.0,
            species=ElectronSpecies(magnetic_field=5.0),
            d=3.0e-3,
            omega_axial=2 * np.pi * 200e6,
        )
        assert trap.bottle_shift == 0.0

    def test_bottle_shift_scales_linearly_with_b2(self):
        """Bottle shift is proportional to B2."""
        species = ElectronSpecies(magnetic_field=5.0)
        trap1 = PenningTrap(
            magnetic_field=5.0,
            species=species,
            d=3.0e-3,
            omega_axial=2 * np.pi * 200e6,
            b2=1000,
        )
        trap2 = PenningTrap(
            magnetic_field=5.0,
            species=species,
            d=3.0e-3,
            omega_axial=2 * np.pi * 200e6,
            b2=3000,
        )
        assert trap2.bottle_shift / trap1.bottle_shift == pytest.approx(
            3.0, rel=1e-10
        )

    def test_axial_frequency_shift_spin_flip(self, bottle_trap):
        """Spin flip shift = delta * g/2 ~ delta * 1.001."""
        shift_up = bottle_trap.axial_frequency_shift(0, +0.5)
        shift_dn = bottle_trap.axial_frequency_shift(0, -0.5)
        spin_flip = shift_up - shift_dn
        from tiqs.constants import ELECTRON_G_FACTOR

        assert spin_flip == pytest.approx(
            bottle_trap.bottle_shift * ELECTRON_G_FACTOR / 2,
            rel=1e-10,
        )

    def test_axial_frequency_shift_cyclotron_jump(self, bottle_trap):
        """Cyclotron jump shift = delta."""
        shift_n0 = bottle_trap.axial_frequency_shift(0, -0.5)
        shift_n1 = bottle_trap.axial_frequency_shift(1, -0.5)
        assert shift_n1 - shift_n0 == pytest.approx(
            bottle_trap.bottle_shift, rel=1e-10
        )

    def test_g_over_2_from_shift_ratio(self, bottle_trap):
        """g/2 = spin_flip_shift / cyclotron_shift exactly."""
        from tiqs.constants import ELECTRON_G_FACTOR

        spin_shift = bottle_trap.axial_frequency_shift(
            0, +0.5
        ) - bottle_trap.axial_frequency_shift(0, -0.5)
        cyc_shift = bottle_trap.axial_frequency_shift(
            1, -0.5
        ) - bottle_trap.axial_frequency_shift(0, -0.5)
        assert spin_shift / cyc_shift == pytest.approx(
            ELECTRON_G_FACTOR / 2, rel=1e-12
        )

    def test_from_dc_voltage_preserves_b1_b2(self):
        """from_dc_voltage passes b1 and b2 through."""
        species = ElectronSpecies(magnetic_field=5.0)
        trap = PenningTrap.from_dc_voltage(
            magnetic_field=5.0,
            species=species,
            d=3.0e-3,
            v_dc=100.0,
            b1=50.0,
            b2=500.0,
        )
        assert trap.b1 == 50.0
        assert trap.b2 == 500.0
        assert trap.bottle_shift > 0

    def test_cyclotron_shift_proportional_to_n_axial(self, bottle_trap):
        """Cyclotron shift increases with axial quantum number."""
        s0 = bottle_trap.cyclotron_frequency_shift(n_axial=0)
        s5 = bottle_trap.cyclotron_frequency_shift(n_axial=5)
        assert s5 > s0

    def test_magnetron_shift_opposite_sign(self, bottle_trap):
        """Magnetron axial term has opposite sign to cyclotron's.
        Test with m_spin=0 to isolate the axial contribution."""
        cyc = bottle_trap.cyclotron_frequency_shift(n_axial=1, m_spin=0.0)
        mag = bottle_trap.magnetron_frequency_shift(n_axial=1, m_spin=0.0)
        assert np.sign(cyc) != np.sign(mag)

    def test_axial_shift_includes_magnetron(self, bottle_trap):
        """Axial shift with magnetron quantum number differs from
        the shift without it."""
        s_no_mag = bottle_trap.axial_frequency_shift(
            n_cyclotron=0, m_spin=-0.5, n_magnetron=0
        )
        s_with_mag = bottle_trap.axial_frequency_shift(
            n_cyclotron=0, m_spin=-0.5, n_magnetron=5
        )
        assert s_with_mag != pytest.approx(s_no_mag)

    def test_b1_equilibrium_shift_positive(self, bottle_trap):
        """B1 gradient displaces the equilibrium position."""
        trap_b1 = PenningTrap(
            magnetic_field=6.0,
            species=ElectronSpecies(magnetic_field=6.0),
            d=3.0e-3,
            omega_axial=2 * np.pi * 200e6,
            b1=100.0,
            b2=9000.0,
        )
        assert trap_b1.b1_equilibrium_shift > 0

    def test_b1_zero_gives_zero_shift(self):
        """No B1 means no equilibrium displacement."""
        trap = PenningTrap(
            magnetic_field=5.0,
            species=ElectronSpecies(magnetic_field=5.0),
            d=3.0e-3,
            omega_axial=2 * np.pi * 200e6,
        )
        assert trap.b1_equilibrium_shift == 0.0

    def test_three_mode_shifts_zero_without_b2(self):
        """All mode shifts are zero without a bottle."""
        trap = PenningTrap(
            magnetic_field=5.0,
            species=ElectronSpecies(magnetic_field=5.0),
            d=3.0e-3,
            omega_axial=2 * np.pi * 200e6,
        )
        assert trap.cyclotron_frequency_shift(n_axial=5) == 0.0
        assert trap.magnetron_frequency_shift(n_axial=5) == 0.0
        assert trap.axial_frequency_shift(n_cyclotron=5) == 0.0

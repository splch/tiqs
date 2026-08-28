# tests/test_trap.py
import numpy as np
import pytest

from tiqs.constants import ELECTRON_CHARGE
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
        """q = 2 e V_rf / (m Omega_rf^2 r0^2) = 0.1270910.

        Hand-computed with m = 170.9363 u - m_e for the standard
        Mathieu parameterization (Wineland et al., J. Res. NIST 103,
        259 (1998), Sec. 2; Berkeland et al., J. Appl. Phys. 83, 5025
        (1998), Eq. (2)). Pinned rather than bracketed so a factor of 2
        or 2*pi cannot hide inside an order-of-magnitude range."""
        assert yb_trap.mathieu_q == pytest.approx(0.1270910, rel=1e-4)

    def test_mathieu_a(self, yb_trap):
        """a = -2 omega_z^2 / Omega_rf^2 = -2/900 for this fixture.

        Berkeland et al., J. Appl. Phys. 83, 5025 (1998), Eq. (5); the
        mass cancels, so this pins the numeric factor alone."""
        assert yb_trap.mathieu_a == pytest.approx(-2.0 / 900.0, rel=1e-12)

    def test_stability(self, yb_trap):
        assert yb_trap.is_stable()

    def test_secular_frequency_radial(self, yb_trap):
        """omega_r/2pi = 1,147,655 Hz = (Omega_rf/2) sqrt(a + q^2/2).

        Value-pinned. The RF-only approximation q*Omega_rf/(2 sqrt 2)
        would give 1,348,003 Hz, so this also pins that the defocusing
        DC term is included: |a| << q is satisfied here yet dropping a
        is 17% wrong (the valid condition is |a| << q^2/2)."""
        nu_r = yb_trap.omega_radial / (2 * np.pi)
        assert nu_r == pytest.approx(1147655.3, rel=1e-4)
        q_only = yb_trap.mathieu_q * yb_trap.omega_rf / (2 * np.sqrt(2))
        assert q_only / (2 * np.pi) == pytest.approx(1348003.2, rel=1e-4)

    def test_secular_frequency_axial(self, yb_trap):
        omega_a = yb_trap.omega_axial
        assert omega_a == pytest.approx(2 * np.pi * 1.0e6)

    def test_pseudopotential_depth(self, yb_trap):
        """Psi_0 = e V_rf^2 / (4 m Omega_rf^2 r0^2) = 15.8864 eV.

        RF-only depth at r = r0 (Wineland 1998 Eq. 6). Value-pinned;
        the a-inclusive identity (1/2) m omega_radial^2 r0^2 / e would
        give 11.515 eV, 38% lower, which is why the docstring names
        r = r0 and the RF-only secular frequency."""
        assert yb_trap.pseudopotential_depth_eV == pytest.approx(
            15.886370, rel=1e-4
        )

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
        with pytest.raises(ValueError, match="beta"):
            _ = trap.omega_radial

    def test_omega_radial_raises_above_first_stability_region(self):
        """q > 0.908 must raise even though beta^2 = a + q^2/2 > 0.

        beta^2 > 0 is satisfied for arbitrarily large q (a is bounded
        while q^2/2 is not), so it is not a stability test on its own.
        The exact first Mathieu region at a ~ 0 closes at q = 0.9080:
        integrating u'' + (a - 2q cos 2xi) u = 0 over one period gives
        |tr M / 2| = 0.9998 at q = 0.908 (stable) and 1.186 at q = 0.95
        (unstable) - so the pseudopotential value the old code returned
        here, 10.05 MHz, described unbounded motion."""
        trap = PaulTrap(
            v_rf=7475.0,
            omega_rf=2 * np.pi * 30e6,
            r0=0.5e-3,
            omega_axial=2 * np.pi * 1e6,
            species=get_species("Yb171"),
        )
        assert trap.mathieu_q == pytest.approx(0.95, rel=1e-3)
        assert trap.mathieu_a + trap.mathieu_q**2 / 2 > 0
        assert not trap.is_stable()
        with pytest.raises(ValueError, match=r"0\.908"):
            _ = trap.omega_radial

    def test_grossly_unstable_trap_omega_radial_raises(self):
        """The repo's own unstable fixture (q = 14298) must raise.

        Unguarded, the pseudopotential formula returned
        omega_r = 5055 * Omega_rf - a secular frequency 5000x the RF
        drive, which refutes the approximation it came from."""
        trap = PaulTrap(
            v_rf=5000.0,
            omega_rf=2 * np.pi * 1e6,
            r0=0.1e-3,
            omega_axial=2 * np.pi * 0.5e6,
            species=get_species("Yb171"),
        )
        assert trap.mathieu_q > 1e4
        with pytest.raises(ValueError, match="unstable"):
            _ = trap.omega_radial
        with pytest.raises(ValueError, match="unstable"):
            _ = trap.stray_field_displacement(1.0)

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

    @pytest.mark.parametrize(
        ("field", "bad_value"),
        [
            ("omega_rf", -2 * np.pi * 30e6),
            ("omega_rf", 0.0),
            ("r0", 0.0),
            ("r0", -0.5e-3),
            ("z0", 0.0),
            ("kappa", 0.0),
            ("v_rf", -1000.0),
            ("omega_axial", -2 * np.pi * 1e6),
        ],
    )
    def test_unphysical_constructor_args_raise(self, field, bad_value):
        """Unphysical inputs must raise, not produce unphysical outputs.

        Without validation, v_rf = -1000 V gave a negative micromotion
        amplitude while omega_radial stayed positive and is_stable()
        returned False - three mutually inconsistent answers - and
        kappa = 0 made u_dc_axial return NaN."""
        kwargs = {
            "v_rf": 1000.0,
            "omega_rf": 2 * np.pi * 30e6,
            "r0": 0.5e-3,
            "omega_axial": 2 * np.pi * 1e6,
            "species": get_species("Yb171"),
            field: bad_value,
        }
        with pytest.raises(ValueError, match=field):
            PaulTrap(**kwargs)


class TestPaulTrapFactory:
    """The DC-voltage <-> axial-frequency map, pinned to literature.

    omega_z^2 = 2 kappa e U_dc / (m z0^2), with kappa dimensionless and
    z0^2 explicit as in TIQS. The factor 2 follows from Laplace's
    equation for the only harmonic potential with the stated axial
    curvature, and appears in this dimensionless-kappa form in
    Wineland et al., J. Res. NIST 103, 259 (1998) (their a_x =
    (4q/m Omega^2)(U_r/R^2 - kappa U_0/z_0^2)), Berkeland et al.,
    J. Appl. Phys. 83, 5025 (1998), Eqs. (2), (5) and (9), and
    arXiv:2012.12766 Eqs. (1), (3) and (6). Wineland's Eq. (2) states
    omega_z = (2 kappa q U_0/m)^(1/2) with z0^2 absorbed into kappa,
    so it agrees on the factor 2 but not on the dimensions.
    """

    def test_from_dc_voltage_ca40_literature_value(self):
        """Ca-40+, U_dc = 10 V, kappa = 0.4, z0 = 2.5 mm -> 279.788 kHz.

        Independently computed from omega_z = sqrt(2 kappa e U/(m z0^2));
        the factor-1 form gives 197.84 kHz, 1/sqrt(2) too low."""
        trap = PaulTrap.from_dc_voltage(
            v_rf=300.0,
            omega_rf=2 * np.pi * 30e6,
            r0=0.5e-3,
            species=get_species("Ca40"),
            u_dc_axial=10.0,
        )
        assert trap.omega_axial / (2 * np.pi) == pytest.approx(
            279787.9, rel=1e-4
        )

    def test_from_dc_voltage_yb171_literature_value(self):
        """Yb-171+, U_dc = 10 V, kappa = 0.4, z0 = 2.5 mm -> 135.282 kHz.

        The factor-1 form gives 95.659 kHz."""
        trap = PaulTrap.from_dc_voltage(
            v_rf=1000.0,
            omega_rf=2 * np.pi * 30e6,
            r0=0.5e-3,
            species=get_species("Yb171"),
            u_dc_axial=10.0,
        )
        assert trap.omega_axial / (2 * np.pi) == pytest.approx(
            135281.6, rel=1e-4
        )

    @pytest.mark.parametrize(
        ("symbol", "volts"),
        [("Yb171", 546.415), ("Ca40", 127.744)],
    )
    def test_u_dc_axial_literature_value(self, symbol, volts):
        """U_dc required for omega_z = 2pi*1 MHz at kappa = 0.4,
        z0 = 2.5 mm: 546.4 V for Yb-171+, 127.7 V for Ca-40+.

        Computed from U_dc = m omega_z^2 z0^2/(2 kappa e), not from the
        property under test - the earlier round-trip test could not see
        that the code demanded exactly twice these voltages."""
        trap = PaulTrap(
            v_rf=300.0,
            omega_rf=2 * np.pi * 30e6,
            r0=0.5e-3,
            species=get_species(symbol),
            omega_axial=2 * np.pi * 1e6,
        )
        assert trap.u_dc_axial == pytest.approx(volts, rel=1e-4)

    def test_mathieu_a_matches_berkeland_voltage_form(self):
        """a from omega_axial equals Berkeland Eq. (5)'s voltage form.

        a_x = a_y = -a_z/2 = -4 e kappa U_dc/(m Omega_rf^2 z0^2) is an
        independent expression: it never touches omega_axial, so it
        agrees with the returned -2 omega_z^2/Omega_rf^2 only if
        from_dc_voltage carries the factor 2."""
        kappa, z0, u_dc = 0.4, 2.5e-3, 10.0
        omega_rf = 2 * np.pi * 30e6
        species = get_species("Yb171")
        trap = PaulTrap.from_dc_voltage(
            v_rf=1000.0,
            omega_rf=omega_rf,
            r0=0.5e-3,
            species=species,
            u_dc_axial=u_dc,
            z0=z0,
            kappa=kappa,
        )
        a_berkeland = (
            -4
            * ELECTRON_CHARGE
            * kappa
            * u_dc
            / (species.mass_kg * omega_rf**2 * z0**2)
        )
        assert trap.mathieu_a == pytest.approx(a_berkeland, rel=1e-12)

    def test_u_dc_axial_property_roundtrip(self):
        """omega_axial -> u_dc_axial -> omega_axial roundtrips.

        Exact algebraic inverse, so this is blind to a shared factor in
        both directions; the absolute scale is pinned by
        test_u_dc_axial_literature_value."""
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
        """omega_+^2 + omega_-^2 + omega_z^2 = omega_c^2.

        For the ideal closed form omega_+- = omega_c/2 +- s this is an
        algebraic identity for any (omega_c, s), so it is kept once as a
        numerical-stability regression, not as physics validation. The
        Brown-Gabrielse theorem has content for imperfect traps (Rev.
        Mod. Phys. 58, 233 (1986), Sec. II C, where it survives
        electrode tilt and ellipticity); tests/test_penning.py checks
        it against published eigenfrequencies instead."""
        wc = electron_penning.omega_cyclotron
        wp = electron_penning.omega_modified_cyclotron
        wm = electron_penning.omega_magnetron
        wz = electron_penning.omega_axial
        assert wp**2 + wm**2 + wz**2 == pytest.approx(wc**2, rel=1e-10)

    def test_magnetron_docstring_records_negative_energy(self):
        """The magnetron mode carries NEGATIVE total energy.

        omega_magnetron returns a positive float, so the sign lives only
        in the docstring; a caller who feeds it to HarmonicPotential,
        mode_hamiltonian or a red-sideband detuning gets inverted
        physics (Brown & Gabrielse RMP 58, 233 (1986); Dehmelt, Nobel
        lecture 1989, Fig. 4). Guard the record so it cannot be dropped
        silently."""
        doc = PenningTrap.omega_magnetron.__doc__
        assert "NEGATIVE" in doc
        assert "metastable" in doc
        assert "sideband" in doc

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

    def test_critically_unstable_penning_raises(self):
        """omega_c = sqrt(2)*omega_z (discriminant exactly 0) raises.

        At the critical point omega_+ = omega_- and the radial motion is
        not bounded, so is_stable() and the transverse frequencies must
        agree. Previously is_stable() said False while both frequencies
        returned the same finite number (Brown & Gabrielse RMP 58, 233
        (1986), Sec. II)."""
        species = get_species("Be9")
        omega_axial = 2 * np.pi * 1e6
        b_crit = np.sqrt(2) * omega_axial * species.mass_kg / ELECTRON_CHARGE
        trap = PenningTrap(
            magnetic_field=b_crit,
            species=species,
            d=1e-3,
            omega_axial=omega_axial,
        )
        assert trap.omega_cyclotron == pytest.approx(
            np.sqrt(2) * omega_axial, rel=1e-12
        )
        assert not trap.is_stable()
        with pytest.raises(ValueError, match="unstable"):
            _ = trap.omega_modified_cyclotron
        with pytest.raises(ValueError, match="unstable"):
            _ = trap.omega_magnetron

    @pytest.mark.parametrize(
        ("field", "bad_value"),
        [
            ("magnetic_field", -3.0),
            ("magnetic_field", 0.0),
            ("d", 0.0),
            ("d", -1e-3),
            ("omega_axial", 0.0),
            ("omega_axial", -2 * np.pi * 1e6),
        ],
    )
    def test_unphysical_constructor_args_raise(self, field, bad_value):
        """Unphysical inputs must raise, not propagate silently.

        B = -3 T previously returned negative omega_c, omega_+ and
        omega_-, which are meaningless as mode frequencies and would
        put a sqrt of a negative number into a Lamb-Dicke factor;
        omega_axial = 0 left a particle with no axial well reported as
        stable."""
        kwargs = {
            "magnetic_field": 3.0,
            "species": get_species("Be9"),
            "d": 1e-3,
            "omega_axial": 2 * np.pi * 1e6,
            field: bad_value,
        }
        with pytest.raises(ValueError, match=field):
            PenningTrap(**kwargs)

    def test_zero_voltage_penning_raises(self):
        """from_dc_voltage(v_dc=0) leaves no axial well: must raise."""
        with pytest.raises(ValueError, match="omega_axial"):
            PenningTrap.from_dc_voltage(
                magnetic_field=3.0,
                species=get_species("Be9"),
                d=1e-3,
                v_dc=0.0,
            )

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

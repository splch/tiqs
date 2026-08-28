# tests/test_chain.py
import numpy as np
import pytest

from tiqs.chain.equilibrium import _force_jacobian, equilibrium_positions
from tiqs.chain.lamb_dicke import (
    gradient_lamb_dicke_parameters,
    lamb_dicke_parameters,
)
from tiqs.chain.normal_modes import (
    ModeGroup,
    NormalModeResult,
    normal_modes,
)
from tiqs.constants import (
    BOHR_MAGNETON,
    COULOMB_CONSTANT,
    ELECTRON_CHARGE,
    ELECTRON_G_FACTOR,
    HBAR,
    TWO_PI,
)
from tiqs.species.ion import get_species
from tiqs.trap import PaulTrap

# Exact linear-to-zigzag thresholds c_N = sqrt((mu_max - 1) / 2), where
# mu_max is the largest axial eigenvalue in units of omega_z^2 (James,
# Appl. Phys. B 66, 181 (1998), Table 2). Independently reproduced by
# direct 2D (x, z) energy minimisation with bisection on the anisotropy.
ZIGZAG_THRESHOLDS = {
    2: 1.000000,
    3: 1.549193,
    4: 2.038179,
    5: 2.497482,
    6: 2.938172,
    7: 3.365697,
}

# [James1998] Table I: dimensionless equilibrium positions u_m = z_m / l.
JAMES_TABLE_I = {
    2: [-((1 / 2) ** (2 / 3)), (1 / 2) ** (2 / 3)],
    3: [-((5 / 4) ** (1 / 3)), 0.0, (5 / 4) ** (1 / 3)],
    4: [-1.43680, -0.45438, 0.45438, 1.43680],
    5: [-1.742900, -0.822101, 0.0, 0.822101, 1.742900],
}


def _james_force_residual(u: np.ndarray) -> float:
    r"""Max |force| of the [James1998] Eq. (2.5) balance, recomputed here.

    $u_i - \sum_{j \neq i} \mathrm{sign}(u_i - u_j) / (u_i - u_j)^2 = 0$,
    written from the paper rather than reusing library internals.
    """
    force = np.array([
        u[i]
        - sum(
            np.sign(u[i] - u[j]) / (u[i] - u[j]) ** 2
            for j in range(len(u))
            if j != i
        )
        for i in range(len(u))
    ])
    return float(np.max(np.abs(force)))


def _length_scale(trap: PaulTrap) -> float:
    """[James1998] Eq. (2.4) length scale for the reference species."""
    return (
        COULOMB_CONSTANT / (trap.species.mass_kg * trap.omega_axial**2)
    ) ** (1 / 3)


def _trap_with_radial_ratio(
    species,
    ratio: float,
    omega_axial: float = TWO_PI * 1.0e6,
    omega_rf: float = TWO_PI * 30e6,
    r0: float = 0.5e-3,
) -> PaulTrap:
    """PaulTrap whose omega_radial/omega_axial equals ``ratio`` exactly.

    Inverts omega_r = (Omega/2) sqrt(a + q^2/2) with
    a = -2 omega_z^2/Omega^2 for the RF amplitude.
    """
    omega_r = ratio * omega_axial
    a = -2 * omega_axial**2 / omega_rf**2
    q = np.sqrt(2 * ((2 * omega_r / omega_rf) ** 2 - a))
    v_rf = q * species.mass_kg * omega_rf**2 * r0**2 / (2 * ELECTRON_CHARGE)
    return PaulTrap(
        v_rf=v_rf,
        omega_rf=omega_rf,
        r0=r0,
        omega_axial=omega_axial,
        species=species,
    )


@pytest.fixture
def ca40_trap():
    return PaulTrap(
        v_rf=300.0,
        omega_rf=TWO_PI * 30e6,
        r0=0.5e-3,
        omega_axial=TWO_PI * 1.0e6,
        species=get_species("Ca40"),
    )


@pytest.fixture
def long_chain_trap():
    """Ca40 trap stiff enough radially to stay linear up to N = 20.

    omega_radial/omega_axial = 8.6201 clears the exact zigzag threshold
    for every N <= 20 (c_20 = 8.4046, c_21 = 8.7709), while Mathieu
    q = 0.1631 keeps the pseudopotential approximation valid.
    """
    return PaulTrap(
        v_rf=300.0,
        omega_rf=TWO_PI * 30e6,
        r0=0.5e-3,
        omega_axial=TWO_PI * 0.2e6,
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

    @pytest.mark.parametrize("n_ions", sorted(JAMES_TABLE_I))
    def test_james_table_i_positions(self, ca40_trap, n_ions):
        """[James1998] Table I dimensionless positions, N = 2..5.

        N = 2 and N = 3 have the closed forms (1/2)^(2/3) and
        (5/4)^(1/3); N = 4 and N = 5 are the tabulated values.
        """
        u = equilibrium_positions(n_ions, ca40_trap) / _length_scale(ca40_trap)
        np.testing.assert_allclose(
            u, JAMES_TABLE_I[n_ions], rtol=1e-5, atol=1e-9
        )

    @pytest.mark.parametrize("n_ions", list(range(2, 41)))
    def test_converges_for_every_chain_length(self, ca40_trap, n_ions):
        """Every chain length in 2..40 solves the [James1998] Eq. (2.5)
        balance, is strictly ordered, and is symmetric about the trap
        center.

        The pre-fix initial guess (uniform spacing 1.5, half-width
        0.75(N-1)) diverged for N = 16, 17, 23, 25-29 and every
        N >= 31, raising RuntimeError.
        """
        u = equilibrium_positions(n_ions, ca40_trap) / _length_scale(ca40_trap)
        assert len(u) == n_ions
        assert np.all(np.diff(u) > 0)
        np.testing.assert_allclose(u, -u[::-1], atol=1e-9)
        scale = max(1.0, float(np.max(np.abs(u))))
        assert _james_force_residual(u) < 1e-10 * scale

    @pytest.mark.parametrize("n_ions", [5, 10, 20, 40, 60])
    def test_minimum_spacing_matches_james_scaling(self, ca40_trap, n_ions):
        """[James1998] Eq. (2.8): u_min = 2.018 * N^(-0.559)."""
        u = equilibrium_positions(n_ions, ca40_trap) / _length_scale(ca40_trap)
        u_min = float(np.min(np.diff(u)))
        assert u_min == pytest.approx(2.018 * n_ions**-0.559, rel=0.05)

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

    @pytest.mark.parametrize("n_ions", [3, 7, 12])
    def test_analytic_jacobian_matches_central_differences(self, n_ions):
        """The supplied Jacobian must be the true derivative of the
        [James1998] Eq. (2.5) force, checked by central differences."""
        u = np.linspace(-1.0, 1.0, n_ions) * 0.79 * n_ions**0.56
        analytic = _force_jacobian(u)

        h = 1e-6
        numeric = np.zeros((n_ions, n_ions))
        for k in range(n_ions):
            up, um = u.copy(), u.copy()
            up[k] += h
            um[k] -= h
            forward = np.array([
                up[i]
                - sum(
                    np.sign(up[i] - up[j]) / (up[i] - up[j]) ** 2
                    for j in range(n_ions)
                    if j != i
                )
                for i in range(n_ions)
            ])
            backward = np.array([
                um[i]
                - sum(
                    np.sign(um[i] - um[j]) / (um[i] - um[j]) ** 2
                    for j in range(n_ions)
                    if j != i
                )
                for i in range(n_ions)
            ])
            numeric[:, k] = (forward - backward) / (2 * h)

        np.testing.assert_allclose(analytic, numeric, rtol=1e-6, atol=1e-6)

    def test_unconverged_solution_is_rejected(self, ca40_trap, monkeypatch):
        """The residual is an independent acceptance criterion. Pre-fix
        the guard was ``not sol.success and residual > 1e-10``, so a
        solver reporting success on its step-size criterion was accepted
        no matter how bad the force residual was."""

        class _FakeSolution:
            x = np.array([-9.0, -3.0, 3.0, 9.0])
            success = True
            message = "fake xtol success"

        monkeypatch.setattr(
            "tiqs.chain.equilibrium.root",
            lambda *args, **kwargs: _FakeSolution(),
        )
        with pytest.raises(RuntimeError, match="force residual"):
            equilibrium_positions(4, ca40_trap)

    def test_collapsed_solution_is_rejected(self, ca40_trap, monkeypatch):
        """Two ions on the same site is not an equilibrium."""

        class _FakeSolution:
            x = np.array([0.0, 0.0, 1.0])
            success = True
            message = "fake xtol success"

        monkeypatch.setattr(
            "tiqs.chain.equilibrium.root",
            lambda *args, **kwargs: _FakeSolution(),
        )
        with pytest.raises(RuntimeError, match="not strictly ordered"):
            equilibrium_positions(3, ca40_trap)


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

    def test_mode_frequencies_increasing(self, long_chain_trap):
        result = normal_modes(5, long_chain_trap)
        freqs = result.modes["axial"].freqs
        for i in range(len(freqs) - 1):
            assert freqs[i] < freqs[i + 1]

    def test_eigenvectors_orthonormal(self, long_chain_trap):
        result = normal_modes(4, long_chain_trap)
        V = result.modes["axial"].vectors
        product = V.T @ V
        np.testing.assert_allclose(product, np.eye(4), atol=1e-10)

    @pytest.mark.parametrize("n_ions", [4, 8, 12])
    def test_axial_mode_ratios_match_james_table_ii(
        self, long_chain_trap, n_ions
    ):
        """[James1998] Table II: the two lowest axial modes are the COM
        at omega_z and the tilt mode at sqrt(3) omega_z for every N."""
        freqs = normal_modes(n_ions, long_chain_trap).modes["axial"].freqs
        omega_z = long_chain_trap.omega_axial
        assert freqs[0] == pytest.approx(omega_z, rel=1e-9)
        assert freqs[1] == pytest.approx(np.sqrt(3) * omega_z, rel=1e-9)

    def test_radial_modes_exist(self, ca40_trap):
        result = normal_modes(3, ca40_trap)
        assert len(result.modes["radial_x"].freqs) == 3
        assert len(result.modes["radial_y"].freqs) == 3

    def test_two_ion_radial_frequencies_exact(self, ca40_trap):
        r"""Exact two-ion transverse spectrum.

        The transverse Coulomb term cancels for common-mode motion, so
        omega_COM = omega_r exactly; the transverse softening equals the
        axial spring constant (2C/d^3 = m omega_z^2 at equilibrium), so
        omega_rock = sqrt(omega_r^2 - omega_z^2) exactly. Both are
        independent of the equilibrium spacing.
        """
        result = normal_modes(2, ca40_trap)
        omega_r = ca40_trap.omega_radial
        omega_z = ca40_trap.omega_axial
        np.testing.assert_allclose(
            result.modes["radial_x"].freqs,
            [np.sqrt(omega_r**2 - omega_z**2), omega_r],
            rtol=1e-12,
        )

    @pytest.mark.parametrize("n_ions", [2, 3, 5, 8, 12, 20])
    def test_axial_radial_sum_rule(self, long_chain_trap, n_ions):
        r"""Exact sum rule pinning the radial Coulomb term.

        The axial Coulomb stiffening is exactly $-2\times$ the radial
        softening, mode by mode in the trace, so

        $$
        \sum_m \omega_{\mathrm{ax},m}^2 + 2\sum_m \omega_{\mathrm{rad},m}^2
        = N(\omega_z^2 + 2\omega_r^2)
        $$

        independently of the equilibrium positions. A sign flip or a
        wrong factor in the radial Coulomb term breaks it.
        """
        result = normal_modes(n_ions, long_chain_trap)
        total = np.sum(result.modes["axial"].freqs ** 2) + 2 * np.sum(
            result.modes["radial_x"].freqs ** 2
        )
        expected = n_ions * (
            long_chain_trap.omega_axial**2
            + 2 * long_chain_trap.omega_radial**2
        )
        assert total == pytest.approx(expected, rel=1e-12)

    def test_paul_trap_mode_labels(self, ca40_trap):
        """Paul trap produces axial, radial_x, radial_y mode groups."""
        result = normal_modes(2, ca40_trap)
        assert set(result.modes.keys()) == {"axial", "radial_x", "radial_y"}

    def test_radial_y_exactly_degenerate_with_radial_x(self, ca40_trap):
        """radial_y is degenerate by construction: the model hard-codes
        the symmetric DC split a = -2 omega_z^2/Omega^2 (Wuebbena 2012
        Eq. (6) with alpha = 1/2) and PaulTrap has no asymmetry knob."""
        result = normal_modes(3, ca40_trap)
        np.testing.assert_array_equal(
            result.modes["radial_x"].freqs, result.modes["radial_y"].freqs
        )
        np.testing.assert_array_equal(
            result.modes["radial_x"].vectors, result.modes["radial_y"].vectors
        )

    def test_mode_group_structure(self, ca40_trap):
        result = normal_modes(2, ca40_trap)
        for _label, group in result.modes.items():
            assert isinstance(group, ModeGroup)
            assert group.freqs.shape == (2,)
            assert group.vectors.shape == (2, 2)


class TestZigzagStability:
    """The linear chain must be rejected once it buckles."""

    @pytest.mark.parametrize("n_ions, c_n", sorted(ZIGZAG_THRESHOLDS.items()))
    def test_threshold_bracket(self, n_ions, c_n):
        """1% above the exact c_N the linear chain exists; 1% below it
        does not and normal_modes must raise rather than clamping the
        imaginary radial frequencies to zero."""
        ca40 = get_species("Ca40")
        stable = _trap_with_radial_ratio(ca40, c_n * 1.01)
        result = normal_modes(n_ions, stable)
        assert np.all(result.modes["radial_x"].freqs > 0)

        buckled = _trap_with_radial_ratio(ca40, c_n * 0.99)
        with pytest.raises(ValueError, match="zigzag"):
            normal_modes(n_ions, buckled)

    @pytest.mark.parametrize("n_ions", [4, 5, 7])
    def test_readme_trap_buckles_above_three_ions(self, ca40_trap, n_ions):
        """The quick-start trap has omega_r/omega_z = 1.5787, below
        c_4 = 2.0382, so chains of 4+ Ca40 ions are physically zigzags.
        Pre-fix these returned zero-frequency modes with no warning."""
        with pytest.raises(ValueError, match="negative eigenvalue"):
            normal_modes(n_ions, ca40_trap)

    def test_error_message_reports_ratio_and_threshold(self, ca40_trap):
        with pytest.raises(ValueError) as excinfo:
            normal_modes(4, ca40_trap)
        message = str(excinfo.value)
        assert "1.5787" in message
        assert "2.0382" in message

    def test_lamb_dicke_never_sees_a_clamped_mode(self, ca40_trap):
        """The pre-fix path produced eta = 0 columns for buckled chains;
        now the mode computation fails first."""
        with pytest.raises(ValueError):
            modes = normal_modes(5, ca40_trap)
            lamb_dicke_parameters(
                modes, ca40_trap.species, TWO_PI / 729e-9, "radial_x"
            )


class TestMathieuStabilityRegion:
    def test_ion_outside_first_region_raises(self, ca40_trap):
        """q >= 0.908 is outside the first Mathieu stability region, so
        the ion is not trapped at all. beta^2 = a + q^2/2 > 0 alone stays
        satisfied there, and pre-fix an 11 MHz radial frequency was
        returned for such an ion with only an accuracy warning."""
        ca40 = get_species("Ca40")
        masses = np.array([ca40.mass_kg, 0.15 * ca40.mass_kg])
        with pytest.raises(ValueError, match="first Mathieu stability"):
            normal_modes(2, ca40_trap, masses=masses)

    def test_pseudopotential_accuracy_warning(self, ca40_trap):
        """0.4 < q < 0.908 is trapped but outside the pseudopotential's
        accurate range, so it warns instead of raising."""
        ca40 = get_species("Ca40")
        be9 = get_species("Be9")
        masses = np.array([be9.mass_kg, ca40.mass_kg])
        with pytest.warns(UserWarning, match="Mathieu q > 0.4"):
            normal_modes(2, ca40_trap, masses=masses)


@pytest.fixture
def be9():
    return get_species("Be9")


@pytest.fixture
def ca40():
    return get_species("Ca40")


@pytest.fixture
def mixed_trap(ca40):
    """Paul trap configured for Ca40 (reference species).

    omega_rf is doubled and v_rf raised relative to ``ca40_trap`` so
    that Be9 sits at Mathieu q = 0.362, inside the pseudopotential's
    validity range (q < 0.4), while omega_radial for the reference
    species is unchanged at 2*pi*1.5787 MHz.
    """
    return PaulTrap(
        v_rf=600.0,
        omega_rf=TWO_PI * 60e6,
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

    def test_laser_eta_scales_as_inverse_sqrt_omega(self, ca40_trap):
        """A fixed wavevector gives eta ~ omega_m^(-1/2), so the
        two-ion stretch/COM ratio is exactly 3^(-1/4)."""
        modes = normal_modes(2, ca40_trap)
        eta = lamb_dicke_parameters(
            modes, ca40_trap.species, TWO_PI / 729e-9, "axial"
        )
        ratio = abs(eta[0, 1]) / abs(eta[0, 0])
        assert ratio == pytest.approx(3.0**-0.25, rel=1e-9)

    def test_lighter_ion_larger_eta(self):
        """Be9 should have larger Lamb-Dicke parameter than Yb171
        (eta ~ 1/sqrt(m))."""
        be_trap = PaulTrap(
            v_rf=600,
            omega_rf=TWO_PI * 60e6,
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

    def test_non_positive_frequency_raises(self, ca40_trap, ca40):
        """A zero-frequency mode has a divergent zero-point spread, so
        eta is undefined. Pre-fix it was silently set to exactly 0, the
        opposite of the physical limit."""
        modes = NormalModeResult(
            positions=np.array([-1e-6, 1e-6]),
            modes={
                "axial": ModeGroup(
                    freqs=np.array([0.0, TWO_PI * 1e6]),
                    vectors=np.eye(2),
                )
            },
        )
        with pytest.raises(ValueError, match="non-positive frequencies"):
            lamb_dicke_parameters(modes, ca40, TWO_PI / 729e-9, "axial")

    def test_unknown_direction_raises(self, ca40_trap, ca40):
        modes = normal_modes(2, ca40_trap)
        with pytest.raises(ValueError, match="Unknown direction"):
            lamb_dicke_parameters(modes, ca40, 1.0, "radial_z")

    def test_radial_eta_drives_a_red_sideband(self, ca40_trap):
        r"""The radial eta must work in a solver, not just in a shape
        check.

        In the Lamb-Dicke regime the first red sideband couples
        $|g, n\rangle \leftrightarrow |e, n-1\rangle$ at
        $\Omega_{n,n-1} = \eta\,\Omega\,\sqrt{n}$, so a pulse of
        duration $\pi/(\eta\Omega)$ transfers $|g, 1\rangle$ to
        $|e, 0\rangle$ completely. Built here from raw qutip operators
        so the check is independent of the library's Hamiltonians.
        """
        import qutip

        modes = normal_modes(2, ca40_trap)
        eta = lamb_dicke_parameters(
            modes, ca40_trap.species, TWO_PI / 729e-9, "radial_x"
        )
        eta_com = abs(eta[0, 1])
        assert 0.01 < eta_com < 0.2

        n_fock = 6
        rabi = TWO_PI * 20e3
        # qutip.sigmam() maps basis(2, 0) -> basis(2, 1), i.e. it is the
        # excitation operator under the repo convention |0> = ground.
        excite = qutip.tensor(qutip.sigmam(), qutip.qeye(n_fock))
        a = qutip.tensor(qutip.qeye(2), qutip.destroy(n_fock))
        h_rsb = (eta_com * rabi / 2) * (excite * a + excite.dag() * a.dag())

        psi0 = qutip.tensor(qutip.basis(2, 0), qutip.basis(n_fock, 1))
        target = qutip.tensor(qutip.basis(2, 1), qutip.basis(n_fock, 0))
        t_pi = np.pi / (eta_com * rabi)
        psi = (-1j * h_rsb * t_pi).expm() * psi0
        assert abs(target.overlap(psi)) ** 2 == pytest.approx(1.0, abs=1e-6)


class TestGradientLambDicke:
    """Magnetic-gradient (MAGIC) coupling: eta ~ omega_m^(-3/2)."""

    def test_mode_scaling_is_inverse_omega_three_halves(self, ca40_trap):
        """Per-mode k_eff = g mu_B G / (hbar omega_m) makes eta scale as
        omega_m^(-3/2), so for two ions the stretch/COM ratio is exactly
        3^(-3/4) rather than the laser case's 3^(-1/4)."""
        modes = normal_modes(2, ca40_trap)
        eta = gradient_lamb_dicke_parameters(
            modes, ca40_trap.species, 24.0, "axial"
        )
        ratio = abs(eta[0, 1]) / abs(eta[0, 0])
        assert ratio == pytest.approx(3.0**-0.75, rel=1e-9)

    def test_scaling_across_trap_frequencies(self, ca40):
        """Halving the trap frequency multiplies eta by 2^(3/2)."""
        etas = []
        for omega_axial in (TWO_PI * 1.0e6, TWO_PI * 0.25e6):
            trap = PaulTrap(
                v_rf=300.0,
                omega_rf=TWO_PI * 30e6,
                r0=0.5e-3,
                omega_axial=omega_axial,
                species=ca40,
            )
            modes = normal_modes(1, trap)
            etas.append(
                gradient_lamb_dicke_parameters(modes, ca40, 24.0)[0, 0]
            )
        assert etas[1] / etas[0] == pytest.approx(4.0**1.5, rel=1e-9)

    def test_linear_in_gradient(self, ca40_trap):
        modes = normal_modes(2, ca40_trap)
        eta_1 = gradient_lamb_dicke_parameters(modes, ca40_trap.species, 10.0)
        eta_3 = gradient_lamb_dicke_parameters(modes, ca40_trap.species, 30.0)
        np.testing.assert_allclose(eta_3, 3.0 * eta_1, rtol=1e-12)

    def test_scalar_k_eff_overstates_the_stretch_mode(self, ca40_trap):
        """A COM-derived scalar k_eff passed to lamb_dicke_parameters
        overstates the stretch-mode gradient eta by exactly
        omega_stretch/omega_COM = sqrt(3): the whole reason the per-mode
        helper exists."""
        modes = normal_modes(2, ca40_trap)
        omega_com = modes.modes["axial"].freqs[0]
        k_com = ELECTRON_G_FACTOR * BOHR_MAGNETON * 24.0 / (HBAR * omega_com)

        eta_scalar = lamb_dicke_parameters(
            modes, ca40_trap.species, k_com, "axial"
        )
        eta_mode = gradient_lamb_dicke_parameters(
            modes, ca40_trap.species, 24.0, "axial"
        )
        np.testing.assert_allclose(
            eta_scalar[:, 0], eta_mode[:, 0], rtol=1e-12
        )
        assert abs(eta_scalar[0, 1]) / abs(eta_mode[0, 1]) == pytest.approx(
            np.sqrt(3), rel=1e-9
        )

    def test_negative_gradient_raises(self, ca40_trap):
        modes = normal_modes(1, ca40_trap)
        with pytest.raises(ValueError, match="gradient must be non-negative"):
            gradient_lamb_dicke_parameters(modes, ca40_trap.species, -1.0)

    def test_mixed_species_uses_per_ion_mass(self, be9, ca40, mixed_trap):
        """Only the mass factor is per-ion; k_eff is per mode."""
        masses = np.array([be9.mass_kg, ca40.mass_kg])
        modes = normal_modes(2, mixed_trap, masses=masses)
        eta = gradient_lamb_dicke_parameters(modes, [be9, ca40], 24.0)
        vectors = modes.modes["axial"].vectors
        freqs = modes.modes["axial"].freqs
        for m in range(2):
            k_m = ELECTRON_G_FACTOR * BOHR_MAGNETON * 24.0 / (HBAR * freqs[m])
            for i, mass in enumerate(masses):
                expected = (
                    k_m * vectors[i, m] * np.sqrt(HBAR / (2 * mass * freqs[m]))
                )
                assert eta[i, m] == pytest.approx(expected, rel=1e-12)


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
        frequencies are (Wuebbena, Amairi, Mandel and Schmidt, PRA 85,
        043412 (2012), Eqs. (12)-(13); general framework: Home,
        Adv. At. Mol. Opt. Phys. 62, 231 (2013), Eq. (8))

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

        assert axial_freqs[0] == pytest.approx(np.sqrt(omega_ip_sq), rel=1e-9)
        assert axial_freqs[1] == pytest.approx(np.sqrt(omega_op_sq), rel=1e-9)

    def test_com_is_not_at_omega_z_for_mixed_species(
        self, be9, ca40, mixed_trap
    ):
        """The 'COM at omega_z with b_i = 1/sqrt(N)' result is
        single-species only: for Be9/Ca40 the lowest axial mode sits
        18.6% above omega_z and the eigenvector is mass-weighted."""
        masses = np.array([be9.mass_kg, ca40.mass_kg])
        result = normal_modes(2, mixed_trap, masses=masses)
        lowest = result.modes["axial"].freqs[0]
        assert lowest / mixed_trap.omega_axial == pytest.approx(
            1.18567, rel=1e-4
        )
        v = np.abs(result.modes["axial"].vectors[:, 0])
        assert not np.allclose(v, 1 / np.sqrt(2), atol=1e-3)

    def test_vectors_are_mass_weighted_not_displacements(
        self, be9, ca40, mixed_trap
    ):
        r"""``vectors`` are eigenvectors of $M^{-1/2} V M^{-1/2}$, so the
        physical displacement of ion $i$ is
        ``vectors[i, m] / sqrt(m_i)``. For the out-of-phase Be9/Ca40
        mode the two ratios differ by exactly
        $\sqrt{m_\mathrm{Ca}/m_\mathrm{Be}}$.
        """
        masses = np.array([be9.mass_kg, ca40.mass_kg])
        result = normal_modes(2, mixed_trap, masses=masses)
        v = result.modes["axial"].vectors[:, 1]
        weighted_ratio = abs(v[0]) / abs(v[1])
        displacement = v / np.sqrt(masses)
        physical_ratio = abs(displacement[0]) / abs(displacement[1])
        assert physical_ratio / weighted_ratio == pytest.approx(
            np.sqrt(ca40.mass_kg / be9.mass_kg), rel=1e-12
        )

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
        r"""The mixed-species sum rule pins the radial spectrum:

        $$
        \sum_m \omega_{\mathrm{ax},m}^2 + 2\sum_m \omega_{\mathrm{rad},m}^2
        = \sum_i \omega_{z,i}^2 + 2\sum_i \omega_{r,i}^2
        $$

        with per-ion $\omega_{z,i} = \sqrt{K/m_i}$ and $\omega_{r,i}$
        from the per-ion Mathieu parameters. Replaces a positivity-only
        assertion that a zero-clamped buckled chain also satisfied.
        """
        masses = np.array([be9.mass_kg, ca40.mass_kg])
        result = normal_modes(2, mixed_trap, masses=masses)
        radial = result.modes["radial_x"]

        assert radial.freqs.shape == (2,)
        assert radial.vectors.shape == (2, 2)
        assert np.all(radial.freqs > 0)

        K = ca40.mass_kg * mixed_trap.omega_axial**2
        omega_z = np.sqrt(K / masses)
        q = (
            2
            * ELECTRON_CHARGE
            * mixed_trap.v_rf
            / (masses * mixed_trap.omega_rf**2 * mixed_trap.r0**2)
        )
        a = -2 * omega_z**2 / mixed_trap.omega_rf**2
        omega_r = (mixed_trap.omega_rf / 2) * np.sqrt(a + q**2 / 2)

        total = np.sum(result.modes["axial"].freqs ** 2) + 2 * np.sum(
            radial.freqs**2
        )
        expected = np.sum(omega_z**2) + 2 * np.sum(omega_r**2)
        assert total == pytest.approx(expected, rel=1e-12)

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
        r"""Exact per-ion Penning transverse spectrum.

        $\omega_\pm = \omega_c/2 \pm \sqrt{(\omega_c/2)^2 -
        \omega_z^2/2}$ with $\omega_c = eB/m_i$ and
        $\omega_{z,i} = \sqrt{K/m_i}$, computed here from $B$ and the
        masses alone. ``freqs`` stores the unsigned magnetron frequency;
        the magnetron mode carries NEGATIVE total energy (Dehmelt 1989;
        Brown and Gabrielse, RMP 58, 233 (1986)), which nothing in
        ``ModeGroup`` encodes.
        """
        masses = np.array([be9.mass_kg, ca40.mass_kg])
        result = normal_modes(2, penning_trap, masses=masses)

        omega_c = ELECTRON_CHARGE * penning_trap.magnetic_field / masses
        K = ca40.mass_kg * penning_trap.omega_axial**2
        omega_z = np.sqrt(K / masses)
        root = np.sqrt((omega_c / 2) ** 2 - omega_z**2 / 2)
        expected_plus = omega_c / 2 + root
        expected_minus = omega_c / 2 - root

        np.testing.assert_allclose(
            result.modes["modified_cyclotron"].freqs,
            np.sort(expected_plus),
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            result.modes["magnetron"].freqs,
            np.sort(expected_minus),
            rtol=1e-12,
        )
        assert np.all(result.modes["magnetron"].freqs > 0)
        # Exact invariance relations (Brown and Gabrielse Sec. II).
        np.testing.assert_allclose(
            expected_plus + expected_minus, omega_c, rtol=1e-12
        )
        np.testing.assert_allclose(
            expected_plus * expected_minus, omega_z**2 / 2, rtol=1e-12
        )

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

    def test_species_tuple_accepted(self, be9, ca40, mixed_trap):
        """Sequence handling must not be gated on ``isinstance(list)``:
        a tuple previously raised AttributeError instead of working."""
        masses = np.array([be9.mass_kg, ca40.mass_kg])
        modes = normal_modes(2, mixed_trap, masses=masses)
        k = TWO_PI / 400e-9
        from_tuple = lamb_dicke_parameters(modes, (be9, ca40), k, "axial")
        from_list = lamb_dicke_parameters(modes, [be9, ca40], k, "axial")
        np.testing.assert_array_equal(from_tuple, from_list)

    def test_k_eff_array_length_validated(self, ca40, mixed_trap):
        """A numpy k_eff previously skipped the length check and was
        broadcast by np.full or failed with an opaque numpy error."""
        modes = normal_modes(2, mixed_trap)
        with pytest.raises(ValueError, match="k_eff must be a scalar"):
            lamb_dicke_parameters(
                modes, ca40, np.array([1.0, 2.0, 3.0]), "axial"
            )

    def test_k_eff_array_accepted(self, ca40, mixed_trap):
        modes = normal_modes(2, mixed_trap)
        k = np.array([TWO_PI / 729e-9, TWO_PI / 400e-9])
        from_array = lamb_dicke_parameters(modes, ca40, k, "axial")
        from_list = lamb_dicke_parameters(modes, ca40, list(k), "axial")
        np.testing.assert_array_equal(from_array, from_list)

    def test_species_sequence_wrong_length_raises(self, be9, ca40, mixed_trap):
        modes = normal_modes(2, mixed_trap)
        with pytest.raises(ValueError, match="species sequence length"):
            lamb_dicke_parameters(modes, [be9, ca40, be9], TWO_PI / 400e-9)

    def test_k_eff_list_wrong_length_raises(self, ca40, mixed_trap):
        modes = normal_modes(2, mixed_trap)
        with pytest.raises(ValueError, match="k_eff must be a scalar"):
            lamb_dicke_parameters(modes, ca40, [1.0, 2.0, 3.0])

"""Tests for the generalized Penning-trap multipole module.

Validates the new fully-general framework against three established
limits:

1. Brown--Gabrielse cylindrical limit (no ellipticity, no off-diagonal H).
2. Kretzschmar elliptical limit (single epsilon, no off-diagonal H).
3. Rotational invariance: rotating the trap in the xy-plane (still
   commuting with B = B0 z-hat) leaves the eigenfrequencies unchanged.

Plus genuinely new physics:

4. Off-diagonal H_xy from a non-axis-aligned in-plane elliptical
   trap.
5. Off-diagonal H_xz, H_yz from a tilted-axis trap.
6. The Brown-Gabrielse invariance theorem holds for arbitrary
   off-diagonal H, magnetic field, mass, and charge.
"""

import numpy as np
import pytest

from tiqs.constants import (
    ELECTRON_CHARGE,
    ELECTRON_MASS,
    PROTON_MASS,
    TWO_PI,
)
from tiqs.multipole import (
    ElectrostaticPotential,
    canonical_hessian,
    linear_modes,
)
from tiqs.species.electron import ElectronSpecies
from tiqs.trap import PenningTrap


class TestElectrostaticPotential:
    def test_empty_potential(self):
        pot = ElectrostaticPotential()
        assert pot.order == 0
        assert pot.laplace_residual() == 0.0
        np.testing.assert_array_equal(pot.hessian(), np.zeros((3, 3)))

    def test_invalid_indices_raise(self):
        with pytest.raises(ValueError, match="Invalid"):
            ElectrostaticPotential({(2, 0): 1.0})
        with pytest.raises(ValueError, match="Invalid"):
            ElectrostaticPotential({(2, -1, 0): 1.0})
        with pytest.raises(ValueError, match="Invalid"):
            ElectrostaticPotential({(2.0, 0, 0): 1.0})

    def test_hessian_diagonal(self):
        pot = ElectrostaticPotential({
            (2, 0, 0): 1.5,
            (0, 2, 0): -2.5,
            (0, 0, 2): 3.0,
        })
        H = pot.hessian()
        np.testing.assert_allclose(H, np.diag([3.0, -5.0, 6.0]))

    def test_hessian_off_diagonal(self):
        pot = ElectrostaticPotential({
            (1, 1, 0): 2.0,
            (1, 0, 1): 3.0,
            (0, 1, 1): 4.0,
        })
        H = pot.hessian()
        expected = np.array([[0, 2, 3], [2, 0, 4], [3, 4, 0]], dtype=float)
        np.testing.assert_allclose(H, expected)

    def test_laplace_residual_harmonic(self):
        pot = ElectrostaticPotential({
            (2, 0, 0): -0.5,
            (0, 2, 0): -0.5,
            (0, 0, 2): 1.0,
        })
        assert pot.laplace_residual() == pytest.approx(0.0, abs=1e-15)

    def test_laplace_residual_non_harmonic(self):
        pot = ElectrostaticPotential({(2, 0, 0): 1.0})
        assert pot.laplace_residual() == pytest.approx(2.0)

    def test_from_quadrupole_is_laplacian(self):
        for eps in [0.0, 0.3, 0.7, -0.5]:
            pot = ElectrostaticPotential.from_quadrupole(
                TWO_PI * 200e6, ELECTRON_MASS, -ELECTRON_CHARGE, epsilon=eps
            )
            assert pot.laplace_residual() < 1e-14, (
                f"epsilon={eps} not Laplacian"
            )

    def test_from_quadrupole_invalid_epsilon(self):
        with pytest.raises(ValueError, match="epsilon"):
            ElectrostaticPotential.from_quadrupole(
                TWO_PI * 200e6, ELECTRON_MASS, -ELECTRON_CHARGE, epsilon=1.0
            )

    def test_restrict_to_orders(self):
        pot = ElectrostaticPotential({
            (2, 0, 0): 1.0,
            (0, 0, 2): -2.0,
            (3, 0, 0): 3.0,
            (4, 0, 0): 4.0,
        })
        sub = pot.restrict_to_orders(min_order=2, max_order=2)
        assert set(sub.coeffs.keys()) == {(2, 0, 0), (0, 0, 2)}

    def test_coeffs_of_order(self):
        pot = ElectrostaticPotential({
            (2, 0, 0): 1.0,
            (1, 1, 0): 2.0,
            (3, 0, 0): 3.0,
        })
        assert pot.coeffs_of_order(2) == {(2, 0, 0): 1.0, (1, 1, 0): 2.0}
        assert pot.coeffs_of_order(3) == {(3, 0, 0): 3.0}

    def test_addition_and_scale(self):
        a = ElectrostaticPotential({(2, 0, 0): 1.0, (0, 2, 0): 2.0})
        b = ElectrostaticPotential({(2, 0, 0): 0.5, (0, 0, 2): 3.0})
        c = a + b
        assert c.get((2, 0, 0)) == 1.5
        assert c.get((0, 2, 0)) == 2.0
        assert c.get((0, 0, 2)) == 3.0

        d = a.scale(2.0)
        assert d.get((2, 0, 0)) == 2.0
        assert d.get((0, 2, 0)) == 4.0


class TestLinearModesBrownGabrielse:
    """Cylindrically symmetric Penning trap."""

    @pytest.fixture
    def cyl_setup(self):
        B = 5.0
        omega_z = TWO_PI * 200e6
        m = ELECTRON_MASS
        q = -ELECTRON_CHARGE
        pot = ElectrostaticPotential.from_quadrupole(omega_z, m, q)
        return pot, B, m, q, omega_z

    def test_matches_existing_PenningTrap(self, cyl_setup):
        """Reduces exactly to Brown-Gabrielse formulas."""
        pot, B, m, q, omega_z = cyl_setup
        result = linear_modes(pot, B, m, q)

        sp = ElectronSpecies(magnetic_field=B)
        trap = PenningTrap(
            magnetic_field=B, species=sp, d=3.5e-3, omega_axial=omega_z
        )
        assert result.omega_plus == pytest.approx(
            trap.omega_modified_cyclotron, rel=1e-10
        )
        assert result.omega_z == pytest.approx(omega_z, rel=1e-10)
        assert result.omega_minus == pytest.approx(
            trap.omega_magnetron, rel=1e-10
        )

    def test_brown_gabrielse_invariance(self, cyl_setup):
        pot, B, m, q, _ = cyl_setup
        result = linear_modes(pot, B, m, q)
        assert result.invariance_residual() < 1e-14

    def test_signatures(self, cyl_setup):
        pot, B, m, q, _ = cyl_setup
        result = linear_modes(pot, B, m, q)
        assert result.signatures == (1, 1, -1)

    def test_symplectic_transform(self, cyl_setup):
        """S^T J S = J to machine precision."""
        pot, B, m, q, _ = cyl_setup
        result = linear_modes(pot, B, m, q)
        S = result.transform
        J = np.zeros((6, 6))
        J[:3, 3:] = np.eye(3)
        J[3:, :3] = -np.eye(3)
        np.testing.assert_allclose(S.T @ J @ S, J, atol=1e-13)

    def test_canonical_form(self, cyl_setup):
        """S^T Sigma S = diag(omega_+, omega_z, -omega_-, ...)."""
        pot, B, m, q, _ = cyl_setup
        result = linear_modes(pot, B, m, q)
        H_e = q * pot.hessian()
        Sigma = canonical_hessian(H_e, B, m, q)
        canon = result.transform.T @ Sigma @ result.transform
        # Action coefficients (ω_+, ω_z, -ω_-) repeat in q- and p-blocks.
        # Floating-point on the magnetron-vs-cyclotron ratio caps rtol
        # at ~1e-9.
        diag = np.diag(canon)
        np.testing.assert_allclose(diag[:3], diag[3:], rtol=1e-9)
        np.testing.assert_allclose(
            diag[:3],
            [
                result.omega_plus,
                result.omega_z,
                -result.omega_minus,
            ],
            rtol=1e-9,
        )
        off = canon - np.diag(diag)
        assert np.max(np.abs(off)) / max(np.abs(diag)) < 1e-12


class TestLinearModesKretzschmar:
    """Kretzschmar elliptical Penning trap."""

    @pytest.fixture
    def v3p4_params(self):
        return dict(
            B=0.140,
            omega_z=TWO_PI * 2623.14e6,
            m=ELECTRON_MASS,
            q=-ELECTRON_CHARGE,
        )

    @pytest.mark.parametrize("epsilon", [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, -0.4])
    def test_matches_existing_kretzschmar(self, v3p4_params, epsilon):
        """Reproduces existing PenningTrap.epsilon formula."""
        pot = ElectrostaticPotential.from_quadrupole(
            v3p4_params["omega_z"],
            v3p4_params["m"],
            v3p4_params["q"],
            epsilon=epsilon,
        )
        result = linear_modes(
            pot, v3p4_params["B"], v3p4_params["m"], v3p4_params["q"]
        )
        sp = ElectronSpecies(magnetic_field=v3p4_params["B"])
        trap = PenningTrap(
            magnetic_field=v3p4_params["B"],
            species=sp,
            d=3.5e-3,
            omega_axial=v3p4_params["omega_z"],
            epsilon=epsilon,
        )
        assert result.omega_plus == pytest.approx(
            trap.omega_modified_cyclotron, rel=1e-10
        )
        assert result.omega_z == pytest.approx(
            v3p4_params["omega_z"], rel=1e-10
        )
        assert result.omega_minus == pytest.approx(
            trap.omega_magnetron, rel=1e-10
        )

    @pytest.mark.parametrize("epsilon", [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, -0.4])
    def test_brown_gabrielse_at_all_epsilons(self, v3p4_params, epsilon):
        pot = ElectrostaticPotential.from_quadrupole(
            v3p4_params["omega_z"],
            v3p4_params["m"],
            v3p4_params["q"],
            epsilon=epsilon,
        )
        result = linear_modes(
            pot, v3p4_params["B"], v3p4_params["m"], v3p4_params["q"]
        )
        assert result.invariance_residual() < 1e-13


class TestRotationalInvariance:
    """Rotating the trap in the xy-plane (still about B = B0 z-hat)
    must not change the eigenfrequencies."""

    @pytest.mark.parametrize("theta_deg", [0, 30, 45, 90, -60, 137])
    def test_xy_rotation_preserves_freqs(self, theta_deg):
        B = 0.140
        omega_z = TWO_PI * 2623.14e6
        m = ELECTRON_MASS
        q = -ELECTRON_CHARGE
        pot_unrot = ElectrostaticPotential.from_quadrupole(
            omega_z, m, q, epsilon=0.5
        )
        H_unrot = pot_unrot.hessian()

        theta = np.deg2rad(theta_deg)
        R = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ])
        H_rot = R.T @ H_unrot @ R
        pot_rot = ElectrostaticPotential({
            (2, 0, 0): H_rot[0, 0] / 2,
            (0, 2, 0): H_rot[1, 1] / 2,
            (0, 0, 2): H_rot[2, 2] / 2,
            (1, 1, 0): H_rot[0, 1],
            (1, 0, 1): H_rot[0, 2],
            (0, 1, 1): H_rot[1, 2],
        })
        assert pot_rot.laplace_residual() < 1e-13

        res_unrot = linear_modes(pot_unrot, B, m, q)
        res_rot = linear_modes(pot_rot, B, m, q)

        assert res_rot.omega_plus == pytest.approx(
            res_unrot.omega_plus, rel=1e-13
        )
        assert res_rot.omega_z == pytest.approx(res_unrot.omega_z, rel=1e-13)
        assert res_rot.omega_minus == pytest.approx(
            res_unrot.omega_minus, rel=1e-13
        )


class TestArbitraryOffDiagonal:
    """Genuinely new: full 3x3 off-diagonal Hessian."""

    @pytest.fixture
    def cross_coupled(self):
        """eps=0.4 base + 5% C_110/C_101/C_011 cross terms."""
        B = 0.140
        omega_z = TWO_PI * 2623.14e6
        m = ELECTRON_MASS
        q = -ELECTRON_CHARGE
        pot_kretz = ElectrostaticPotential.from_quadrupole(
            omega_z, m, q, epsilon=0.4
        )
        H_base = pot_kretz.hessian()
        scale = np.abs(H_base).max()
        delta = 0.05 * scale
        pot = ElectrostaticPotential({
            (2, 0, 0): H_base[0, 0] / 2,
            (0, 2, 0): H_base[1, 1] / 2,
            (0, 0, 2): H_base[2, 2] / 2,
            (1, 1, 0): 0.3 * delta,
            (1, 0, 1): delta,
            (0, 1, 1): -0.5 * delta,
        })
        return pot, B, m, q

    def test_brown_gabrielse_holds(self, cross_coupled):
        """BG invariance must hold even with a full off-diagonal H."""
        pot, B, m, q = cross_coupled
        result = linear_modes(pot, B, m, q)
        assert result.invariance_residual() < 1e-13

    def test_signatures_unchanged(self, cross_coupled):
        pot, B, m, q = cross_coupled
        result = linear_modes(pot, B, m, q)
        assert result.signatures == (1, 1, -1)

    def test_symplectic_transform(self, cross_coupled):
        pot, B, m, q = cross_coupled
        result = linear_modes(pot, B, m, q)
        S = result.transform
        J = np.zeros((6, 6))
        J[:3, 3:] = np.eye(3)
        J[3:, :3] = -np.eye(3)
        np.testing.assert_allclose(S.T @ J @ S, J, atol=1e-12)

    def test_frequencies_change_smoothly(self):
        """Sweeping cross-coupling fraction shifts frequencies smoothly."""
        B = 0.140
        omega_z = TWO_PI * 2623.14e6
        m = ELECTRON_MASS
        q = -ELECTRON_CHARGE
        pot_base = ElectrostaticPotential.from_quadrupole(
            omega_z, m, q, epsilon=0.4
        )
        H_base = pot_base.hessian()
        scale = np.abs(H_base).max()
        prev = linear_modes(pot_base, B, m, q)
        for frac in [0.001, 0.01, 0.05]:
            d = frac * scale
            pot = ElectrostaticPotential({
                (2, 0, 0): H_base[0, 0] / 2,
                (0, 2, 0): H_base[1, 1] / 2,
                (0, 0, 2): H_base[2, 2] / 2,
                (1, 0, 1): d,
            })
            cur = linear_modes(pot, B, m, q)
            assert cur.invariance_residual() < 1e-13
            # Small perturbations shift frequencies by less than 50%.
            for prev_w, cur_w in zip(
                prev.frequencies, cur.frequencies, strict=True
            ):
                assert abs(cur_w - prev_w) / prev_w < 0.5
            prev = cur


class TestUnstableTrap:
    """Configurations that should produce a clear instability error."""

    def test_too_weak_b_field(self):
        # Brown-Gabrielse instability: ω_c < √2 ω_z.
        omega_z = TWO_PI * 200e6
        m = ELECTRON_MASS
        q = -ELECTRON_CHARGE
        pot = ElectrostaticPotential.from_quadrupole(omega_z, m, q)
        with pytest.raises(ValueError, match="unstable"):
            linear_modes(pot, magnetic_field=0.0001, mass=m, charge=q)

    def test_strong_axial_radial_coupling(self):
        """Tilting the trap axis past instability."""
        B = 0.140
        omega_z = TWO_PI * 2623.14e6
        m = ELECTRON_MASS
        q = -ELECTRON_CHARGE
        pot_cyl = ElectrostaticPotential.from_quadrupole(omega_z, m, q)
        H_cyl = pot_cyl.hessian()
        # 30 degree tilt about the y-axis.
        theta = np.deg2rad(30)
        R = np.array([
            [np.cos(theta), 0, -np.sin(theta)],
            [0, 1, 0],
            [np.sin(theta), 0, np.cos(theta)],
        ])
        H = R.T @ H_cyl @ R
        pot = ElectrostaticPotential({
            (2, 0, 0): H[0, 0] / 2,
            (0, 2, 0): H[1, 1] / 2,
            (0, 0, 2): H[2, 2] / 2,
            (1, 1, 0): H[0, 1],
            (1, 0, 1): H[0, 2],
            (0, 1, 1): H[1, 2],
        })
        with pytest.raises(ValueError, match="unstable"):
            linear_modes(pot, B, m, q)


class TestProtonAndPositiveCharge:
    """Cross-species sanity check (BASE-class proton trap)."""

    def test_proton_BASE(self):
        B = 1.945
        omega_z = TWO_PI * 630e3
        m = PROTON_MASS
        q = +ELECTRON_CHARGE
        pot = ElectrostaticPotential.from_quadrupole(omega_z, m, q)
        result = linear_modes(pot, B, m, q)
        # Cyclotron at 1.945 T for a proton: ~29.65 MHz.
        assert result.cyclotron_frequency / TWO_PI == pytest.approx(
            29.65e6, rel=1e-3
        )
        assert result.invariance_residual() < 1e-12
        assert result.signatures == (1, 1, -1)


class TestPolynomial:
    def test_zero_and_one(self):
        from tiqs.multipole import Polynomial

        zero = Polynomial.zero()
        one = Polynomial.one()
        assert not zero  # falsy
        assert bool(one)
        assert one.terms == {(0, 0, 0, 0, 0, 0): 1 + 0j}

    def test_variable_factory(self):
        from tiqs.multipole import Polynomial

        a_plus = Polynomial.variable(0)
        assert a_plus.terms == {(1, 0, 0, 0, 0, 0): 1 + 0j}

    def test_invalid_variable_raises(self):
        from tiqs.multipole import Polynomial

        with pytest.raises(IndexError):
            Polynomial.variable(7)

    def test_addition(self):
        from tiqs.multipole import Polynomial

        a = Polynomial.variable(0)
        b = Polynomial.variable(1)
        c = a + b
        assert (1, 0, 0, 0, 0, 0) in c.terms
        assert (0, 1, 0, 0, 0, 0) in c.terms
        d = c + (-a)
        assert (1, 0, 0, 0, 0, 0) not in d.terms

    def test_scalar_multiply(self):
        from tiqs.multipole import Polynomial

        a = Polynomial.variable(0) * 3.0
        assert a.terms[(1, 0, 0, 0, 0, 0)] == 3.0 + 0j

    def test_multiplication(self):
        from tiqs.multipole import Polynomial

        a = Polynomial.variable(0)
        abar = Polynomial.variable(3)
        action = a * abar
        assert action.terms == {(1, 0, 0, 1, 0, 0): 1 + 0j}

    def test_power(self):
        from tiqs.multipole import Polynomial

        a = Polynomial.variable(0) + Polynomial.variable(3)
        a2 = a**2
        # (a + ā)^2 = a^2 + 2 a ā + ā^2
        assert a2.terms[(2, 0, 0, 0, 0, 0)] == 1 + 0j
        assert a2.terms[(1, 0, 0, 1, 0, 0)] == 2 + 0j
        assert a2.terms[(0, 0, 0, 2, 0, 0)] == 1 + 0j

    def test_total_degree(self):
        from tiqs.multipole import Polynomial

        p = Polynomial.variable(0) + Polynomial.variable(0) ** 3
        assert p.total_degree == 3

    def test_homogeneous_part(self):
        from tiqs.multipole import Polynomial

        p = Polynomial.variable(0) + Polynomial.variable(0) ** 3
        h1 = p.homogeneous_part(1)
        h3 = p.homogeneous_part(3)
        assert h1.total_degree == 1
        assert h3.total_degree == 3


class TestPoissonBracket:
    """Verify the Poisson bracket on (a, ā) ladder operators."""

    def test_basic_canonical_commutator(self):
        """{a_+, ā_+} = -i."""
        from tiqs.multipole import Polynomial

        a_plus = Polynomial.variable(0)
        abar_plus = Polynomial.variable(3)
        br = a_plus.poisson_bracket(abar_plus)
        assert br.terms == {(0, 0, 0, 0, 0, 0): -1j}

    def test_antisymmetry(self):
        """{f, g} = -{g, f}."""
        from tiqs.multipole import Polynomial

        a_plus = Polynomial.variable(0)
        abar_plus = Polynomial.variable(3)
        br1 = a_plus.poisson_bracket(abar_plus)
        br2 = abar_plus.poisson_bracket(a_plus)
        sum_p = br1 + br2
        assert not sum_p.terms

    def test_action_self_commutes(self):
        """I_+ = a_+ ā_+. {I_+, I_+} = 0."""
        from tiqs.multipole import Polynomial

        I_plus = Polynomial.variable(0) * Polynomial.variable(3)
        br = I_plus.poisson_bracket(I_plus)
        assert not br.terms

    def test_actions_commute(self):
        from tiqs.multipole import Polynomial

        I_plus = Polynomial.variable(0) * Polynomial.variable(3)
        I_minus = Polynomial.variable(2) * Polynomial.variable(5)
        br = I_plus.poisson_bracket(I_minus)
        assert not br.terms

    def test_h2_acting_on_a(self):
        """{H_2, a_α} = i ω_α a_α (eigenvalue equation)."""
        from tiqs.multipole import Polynomial, quadratic_normal_form

        omega_plus, omega_z, omega_minus = 5.0, 3.0, 1.0
        H2 = quadratic_normal_form(omega_plus, omega_z, omega_minus)

        a_plus = Polynomial.variable(0)
        result = H2.poisson_bracket(a_plus)
        expected = (1j * omega_plus) * a_plus
        diff = result + (-expected)
        for v in diff.terms.values():
            assert abs(v) < 1e-14

    def test_h2_acting_on_abar(self):
        """{H_2, ā_α} = -i ω_α ā_α."""
        from tiqs.multipole import Polynomial, quadratic_normal_form

        omega_plus, omega_z, omega_minus = 5.0, 3.0, 1.0
        H2 = quadratic_normal_form(omega_plus, omega_z, omega_minus)

        abar_plus = Polynomial.variable(3)
        result = H2.poisson_bracket(abar_plus)
        expected = (-1j * omega_plus) * abar_plus
        diff = result + (-expected)
        for v in diff.terms.values():
            assert abs(v) < 1e-14

    def test_h2_acting_on_magnetron_a(self):
        """{H_2, a_-} = -i ω_- a_- (because H_2 has -ω_- I_-)."""
        from tiqs.multipole import Polynomial, quadratic_normal_form

        omega_plus, omega_z, omega_minus = 5.0, 3.0, 1.0
        H2 = quadratic_normal_form(omega_plus, omega_z, omega_minus)

        a_minus = Polynomial.variable(2)
        result = H2.poisson_bracket(a_minus)
        # H_2 has the -ω_- I_- term, so {H_2, a_-} = -i ω_- a_-.
        expected = (-1j * omega_minus) * a_minus
        diff = result + (-expected)
        for v in diff.terms.values():
            assert abs(v) < 1e-14


class TestSplitKernelImage:
    def test_action_diagonal_in_kernel(self):
        """I_+^2 lives entirely in the action-diagonal kernel."""
        from tiqs.multipole import Polynomial, split_kernel_image

        a_plus = Polynomial.variable(0)
        abar_plus = Polynomial.variable(3)
        I_plus = a_plus * abar_plus
        I_plus_sq = I_plus * I_plus
        kernel, image = split_kernel_image(I_plus_sq, 5.0, 3.0, 1.0)
        assert image.terms == {}
        assert kernel.terms == I_plus_sq.terms

    def test_off_diagonal_in_image(self):
        """a_+² has spectral coefficient 2iω_+ ≠ 0, so it lives in the
        image of the homological operator."""
        from tiqs.multipole import Polynomial, split_kernel_image

        p = Polynomial.variable(0) ** 2
        kernel, image = split_kernel_image(p, 5.0, 3.0, 1.0)
        assert kernel.terms == {}
        assert image.terms == p.terms

    def test_resonance_detected(self):
        """At ω_+ = 2 ω_-, the monomial a_+ a_-² has spectral coefficient
        i(ω_+ - 2ω_-) = 0 and so falls into the resonant kernel."""
        from tiqs.multipole import Polynomial, split_kernel_image

        p = Polynomial({(1, 0, 2, 0, 0, 0): 1 + 0j})
        kernel, image = split_kernel_image(p, 2.0, 3.0, 1.0)
        assert kernel.terms == p.terms
        assert image.terms == {}


class TestQuadraticNormalForm:
    def test_h2_structure(self):
        from tiqs.multipole import quadratic_normal_form

        H2 = quadratic_normal_form(5.0, 3.0, 1.0)
        assert H2.terms[(1, 0, 0, 1, 0, 0)] == 5.0 + 0j  # ω_+ I_+
        assert H2.terms[(0, 1, 0, 0, 1, 0)] == 3.0 + 0j  # ω_z I_z
        assert H2.terms[(0, 0, 1, 0, 0, 1)] == -1.0 + 0j  # -ω_- I_-


class TestCartesianPolynomials:
    def test_round_trip_quadratic_h2(self):
        """Substituting cartesian polynomials into the quadratic H
        recovers H_2 in mode-coordinate diagonal form.
        """
        from tiqs.multipole import (
            ElectrostaticPotential,
            cartesian_polynomials,
            linear_modes,
        )

        B = 0.5
        omega_z = TWO_PI * 28e6
        m = ELECTRON_MASS
        q = +ELECTRON_CHARGE
        pot = ElectrostaticPotential.from_quadrupole(
            omega_z, m, q, epsilon=0.3
        )
        modes = linear_modes(pot, B, m, q)
        cart = cartesian_polynomials(modes.transform)

        # Substituting cart -> H_z(p_z, z) should give ω_z (a_z ā_z)
        # on the action diagonal. Axial decouples in the elliptical
        # case so this is the cleanest sector to test.
        z_op = cart[2]
        pz_op = cart[5]
        H_z = (pz_op * pz_op) * (1.0 / (2.0 * m)) + (
            (m * omega_z**2 / 2.0) * (z_op * z_op)
        )
        action_part = {k: v for k, v in H_z.terms.items() if k[:3] == k[3:]}
        expected_az = action_part.get((0, 1, 0, 0, 1, 0), 0)
        assert abs(expected_az - omega_z) / abs(omega_z) < 1e-10


class TestBirkhoffNormalFormDiagonal:
    """The diagonal entries of the M-matrix from BGNF must match
    Verdú's elliptical.py implementation to machine precision.

    This is the rigorous validation of the polynomial algebra +
    Lie-Deprit recursion.
    """

    @pytest.fixture
    def verdu_setup(self):
        B = 0.5
        omega_z = TWO_PI * 28e6
        m = ELECTRON_MASS
        q = +ELECTRON_CHARGE
        eps = 0.41
        return dict(B=B, omega_z=omega_z, m=m, q=q, eps=eps)

    def _verdu_match_diagonal(self, kw, coeff_kw, monomial):
        """Helper: build potential and compare the matching diagonal
        entry of M_new with M_verdu.
        """

        from tiqs.elliptical import (
            AnharmonicCoeffs,
            frequency_shifts_matrix,
            orbit_params,
        )
        from tiqs.multipole import (
            ElectrostaticPotential,
            shift_matrix_general,
        )
        from tiqs.species.electron import ElectronSpecies
        from tiqs.trap import PenningTrap

        sp = ElectronSpecies(magnetic_field=kw["B"])
        trap = PenningTrap(
            magnetic_field=kw["B"],
            species=sp,
            d=3.5e-3,
            omega_axial=kw["omega_z"],
            epsilon=kw["eps"],
        )
        nu_p = trap.omega_modified_cyclotron / TWO_PI
        nu_z = trap.omega_axial / TWO_PI
        nu_m = trap.omega_magnetron / TWO_PI
        orb = orbit_params(
            trap.omega_cyclotron,
            trap.omega_axial,
            trap.omega_modified_cyclotron,
            kw["eps"],
        )
        coeffs = AnharmonicCoeffs(c002=1.0, **coeff_kw)
        M_v = frequency_shifts_matrix(nu_p, nu_z, nu_m, orb, coeffs, kw["m"])
        pot = ElectrostaticPotential.from_quadrupole(
            kw["omega_z"], kw["m"], kw["q"], epsilon=kw["eps"]
        ) + ElectrostaticPotential(monomial)
        M_n = shift_matrix_general(pot, kw["B"], kw["m"], kw["q"], order=4)
        return M_v, M_n

    def test_M004_axial_diagonal_matches_verdu(self, verdu_setup):
        M_v, M_n = self._verdu_match_diagonal(
            verdu_setup, {"c004": 1e10}, {(0, 0, 4): 1e10}
        )
        assert M_v[1, 1] != 0
        denom = abs(M_v[1, 1])
        assert abs(M_n[1, 1] - M_v[1, 1]) / denom < 1e-12

    def test_M400_cyclotron_diagonal_matches_verdu(self, verdu_setup):
        M_v, M_n = self._verdu_match_diagonal(
            verdu_setup, {"c400": 1e15}, {(4, 0, 0): 1e15}
        )
        denom = abs(M_v[0, 0])
        assert abs(M_n[0, 0] - M_v[0, 0]) / denom < 1e-10


class TestCylindricalSymmetry:
    """For any cylindrically-symmetric perturbation V(ρ, z) (where
    ρ² = x² + y²), the angular momentum L_z = N_+ - N_- is conserved
    by the perturbation theory. This forces the structural constraint

        M^I[+, +] = M^I[-, -]                  (cylindrical symmetry)

    This is a convention-independent, rigorous structural test that
    any correct perturbation theory MUST satisfy at any trap parameters.
    """

    def _cylindrical_C4(self, omega_z, m, q, C4=1e10):
        """Laplacian cylindrically-symmetric quartic multipole
            V_4(x, y, z) = z^4 - 3 z² (x²+y²) + (3/8) (x²+y²)²
        which has zero Laplacian by construction.
        """
        from tiqs.multipole import ElectrostaticPotential

        return ElectrostaticPotential.from_quadrupole(
            omega_z, m, q, epsilon=0.0
        ) + ElectrostaticPotential({
            (0, 0, 4): C4 * 1.0,
            (2, 0, 2): C4 * -3.0,
            (0, 2, 2): C4 * -3.0,
            (4, 0, 0): C4 * 3.0 / 8.0,
            (0, 4, 0): C4 * 3.0 / 8.0,
            (2, 2, 0): C4 * 3.0 / 4.0,
        })

    @pytest.mark.parametrize("omega_z_GHz", [0.5, 1.0, 1.5, 2.0, 2.5])
    def test_cylindrical_C4_diagonals_match(self, omega_z_GHz):
        """M^I[+,+] = M^I[-,-] to machine precision for any
        cylindrically-symmetric V."""
        from tiqs.multipole import linear_modes, shift_matrix_general

        B = 0.140
        m = ELECTRON_MASS
        q = +ELECTRON_CHARGE
        omega_z = TWO_PI * omega_z_GHz * 1e9
        pot = self._cylindrical_C4(omega_z, m, q)
        modes = linear_modes(
            pot.restrict_to_orders(min_order=2, max_order=2), B, m, q
        )
        sign = np.array([1, 1, -1])
        omegas = np.array([modes.omega_plus, modes.omega_z, modes.omega_minus])
        M_V = shift_matrix_general(pot, B, m, q, order=4)
        M_I = M_V * (TWO_PI * sign * omegas)[np.newaxis, :]
        diff = M_I[0, 0] - M_I[2, 2]
        scale = max(abs(M_I[0, 0]), abs(M_I[2, 2]), 1e-30)
        assert abs(diff) / scale < 1e-12, (
            f"ω_z={omega_z_GHz}GHz: M^I[+,+]={M_I[0, 0]:.4e}, "
            f"M^I[-,-]={M_I[2, 2]:.4e}, rel diff={abs(diff) / scale:.3e}"
        )

    def test_ellipticity_breaks_cylindrical_symmetry(self):
        """At ε ≠ 0 the cylindrical symmetry is broken, and the
        diagonals SHOULD differ (else we're not detecting the
        symmetry breaking)."""
        from tiqs.multipole import (
            ElectrostaticPotential,
            linear_modes,
            shift_matrix_general,
        )

        B = 0.140
        m = ELECTRON_MASS
        q = +ELECTRON_CHARGE
        omega_z = TWO_PI * 2.0e9
        C4 = 1e10
        pot = ElectrostaticPotential.from_quadrupole(
            omega_z, m, q, epsilon=0.3
        ) + ElectrostaticPotential({
            (0, 0, 4): C4 * 1.0,
            (2, 0, 2): C4 * -3.0,
            (0, 2, 2): C4 * -3.0,
            (4, 0, 0): C4 * 3.0 / 8.0,
            (0, 4, 0): C4 * 3.0 / 8.0,
            (2, 2, 0): C4 * 3.0 / 4.0,
        })
        modes = linear_modes(
            pot.restrict_to_orders(min_order=2, max_order=2), B, m, q
        )
        sign = np.array([1, 1, -1])
        omegas = np.array([modes.omega_plus, modes.omega_z, modes.omega_minus])
        M_V = shift_matrix_general(pot, B, m, q, order=4)
        M_I = M_V * (TWO_PI * sign * omegas)[np.newaxis, :]
        diff = M_I[0, 0] - M_I[2, 2]
        scale = max(abs(M_I[0, 0]), abs(M_I[2, 2]), 1e-30)
        assert abs(diff) / scale > 0.01


class TestRadialCrossModeAgainstFockDiagonalization:
    """Validate the **cross-mode** entries of the M-matrix against
    direct numerical Fock-basis diagonalization of the radial Penning
    Hamiltonian.

    This is the load-bearing test that the generalized BGNF gives
    correct cross-mode shifts (where Verdú's published formulas
    disagree by 4x-1000x; see scripts/validate_radial_bgnf_sweep.py).

    The test diagonalizes
        H = ℏω_+ N_+ - ℏω_- N_- + q*C_400 x^4
    in a 2D Fock basis (cyclotron × magnetron) in dimensionless
    units (energy / ℏω_+) -- working in dimensionless coordinates
    is necessary to avoid a known QuTiP precision bug at the
    ℏω ≈ 1e-24 J scale where Dia-format subtraction silently
    produces zero (verified at qutip 5.2.3, numpy 2.4.4).
    """

    def _numerical_M_I(
        self,
        potential,
        B,
        m,
        q,
        n_fock=10,
        overlap_threshold=0.99,
    ):
        """Diagonalize H_2 + V in a 2D Fock basis and return
        numerical M^I[+,+], M^I[+,-], M^I[-,-] (action-derivative
        convention).
        """
        import qutip
        import scipy.optimize

        from tiqs.constants import HBAR
        from tiqs.multipole import linear_modes

        quad_pot = potential.restrict_to_orders(min_order=2, max_order=2)
        pert_pot = potential.restrict_to_orders(min_order=3)
        modes = linear_modes(quad_pot, B, m, q)
        omega_p = modes.omega_plus
        omega_m = modes.omega_minus
        S = modes.transform

        a_p = qutip.tensor(qutip.destroy(n_fock), qutip.qeye(n_fock))
        a_m = qutip.tensor(qutip.qeye(n_fock), qutip.destroy(n_fock))
        sqrt_h = np.sqrt(HBAR)

        def cart_op(row):
            op = (
                S[row, 0] * sqrt_h / np.sqrt(2) * (a_p + a_p.dag())
                + S[row, 2] * sqrt_h / np.sqrt(2) * (a_m + a_m.dag())
                + S[row, 3] * (-1j * sqrt_h / np.sqrt(2)) * (a_p - a_p.dag())
                + S[row, 5] * (-1j * sqrt_h / np.sqrt(2)) * (a_m - a_m.dag())
            )
            return (op + op.dag()) / 2

        x_op = cart_op(0)
        y_op = cart_op(1)
        Id_arr = qutip.tensor(qutip.qeye(n_fock), qutip.qeye(n_fock)).full()
        x_arr = x_op.full()
        y_arr = y_op.full()
        V_phys = np.zeros_like(Id_arr)
        for (i, j, k), c in pert_pot.coeffs.items():
            if k != 0 or c == 0:
                continue
            op = Id_arr.copy()
            for _ in range(i):
                op = op @ x_arr
            for _ in range(j):
                op = op @ y_arr
            V_phys = V_phys + (q * c) * op

        N_p = a_p.dag() * a_p
        N_m = a_m.dag() * a_m
        H = (
            N_p.full()
            - (omega_m / omega_p) * N_m.full()
            + V_phys / (HBAR * omega_p)
        )
        eig, vecs = np.linalg.eigh(H)

        shifts = []
        n_max = 4
        for np_v in range(n_max):
            for nm_v in range(n_max):
                un = (
                    qutip
                    .tensor(qutip.fock(n_fock, np_v), qutip.fock(n_fock, nm_v))
                    .full()
                    .flatten()
                )
                un_E = np_v - (omega_m / omega_p) * nm_v
                ovs = np.abs(vecs.conj().T @ un) ** 2
                bi = int(np.argmax(ovs))
                if ovs[bi] < overlap_threshold:
                    continue
                shifts.append((np_v, nm_v, eig[bi].real - un_E))

        if len(shifts) < 6:
            return None

        sh = np.array([s[2] for s in shifts])
        co = np.array([(s[0], s[1]) for s in shifts], dtype=float)

        def f(X, a, b, c, d, e, ff):
            np_a, nm_a = X[:, 0], X[:, 1]
            return (
                a
                + b * np_a
                + c * nm_a
                + d * np_a**2
                + e * np_a * nm_a
                + ff * nm_a**2
            )

        popt, _ = scipy.optimize.curve_fit(f, co, sh)
        _, _, _, d, e, fcoef = popt
        return np.array([
            2 * d * omega_p / HBAR,
            e * omega_p / HBAR,
            2 * fcoef * omega_p / HBAR,
        ])

    def _bgnf_M_I(self, potential, B, m, q):
        """Extract M^I (action-derivative) from BGNF M^V."""
        from tiqs.multipole import linear_modes, shift_matrix_general

        M_V = shift_matrix_general(potential, B, m, q, order=4)
        quad = potential.restrict_to_orders(min_order=2, max_order=2)
        modes = linear_modes(quad, B, m, q)
        sign = np.array([1, 1, -1])
        omegas = np.array([modes.omega_plus, modes.omega_z, modes.omega_minus])
        M_I = M_V * (TWO_PI * sign * omegas)[np.newaxis, :]
        return np.array([M_I[0, 0], M_I[0, 2], M_I[2, 2]])

    @pytest.mark.parametrize(
        "omega_z_GHz,gamma_tilde",
        [
            (2.20, 1e-6),
            (1.00, 1e-6),
        ],
    )
    def test_C400_against_fock(self, omega_z_GHz, gamma_tilde):
        """Pure C_400 quartic perturbation should give numerical
        M^I matching BGNF to better than 1%."""
        from tiqs.multipole import (
            ElectrostaticPotential,
            linear_modes,
        )

        B = 0.140
        omega_z = TWO_PI * omega_z_GHz * 1e9
        m = ELECTRON_MASS
        q = +ELECTRON_CHARGE
        # Calibrate C_400 to give specified perturbation strength.
        quad = ElectrostaticPotential.from_quadrupole(
            omega_z, m, q, epsilon=0.0
        )
        modes = linear_modes(quad, B, m, q)
        # γ = q C x_zpf⁴ / (ℏω_+) sets the perturbation strength.
        from tiqs.constants import HBAR

        x_zpf_p = np.sqrt(HBAR / (2 * m * modes.omega_plus))
        C = gamma_tilde * HBAR * modes.omega_plus / (q * x_zpf_p**4)
        pot = quad + ElectrostaticPotential({(4, 0, 0): C})
        M_I_num = self._numerical_M_I(pot, B, m, q, n_fock=10)
        assert M_I_num is not None, "insufficient overlap data"
        M_I_bgnf = self._bgnf_M_I(pot, B, m, q)
        for label, num, bgnf in zip(
            ("[+,+]", "[+,-]", "[-,-]"), M_I_num, M_I_bgnf, strict=True
        ):
            ratio = num / bgnf
            assert 0.98 < ratio < 1.02, (
                f"{label}: numerical={num:.3e}, BGNF={bgnf:.3e}, "
                f"ratio={ratio:.4f}"
            )

    def test_cubic_C300_against_fock(self):
        """Cubic C_300 perturbation tests the (1/2){W_3, H_3}
        Lie-Deprit triangle factor and the homological-equation
        sign convention.
        """
        from tiqs.constants import HBAR
        from tiqs.multipole import (
            ElectrostaticPotential,
            linear_modes,
        )

        B = 0.140
        omega_z = TWO_PI * 2.20e9
        m = ELECTRON_MASS
        q = +ELECTRON_CHARGE
        quad = ElectrostaticPotential.from_quadrupole(
            omega_z, m, q, epsilon=0.0
        )
        modes = linear_modes(quad, B, m, q)
        x_zpf_p = np.sqrt(HBAR / (2 * m * modes.omega_plus))
        # Cubic shifts are O(γ²), so use a larger γ than the quartic test.
        gamma_tilde = 1e-4
        C = gamma_tilde * HBAR * modes.omega_plus / (q * x_zpf_p**3)
        pot = quad + ElectrostaticPotential({(3, 0, 0): C})
        M_I_num = self._numerical_M_I(pot, B, m, q, n_fock=12)
        assert M_I_num is not None
        M_I_bgnf = self._bgnf_M_I(pot, B, m, q)
        for label, num, bgnf in zip(
            ("[+,+]", "[+,-]", "[-,-]"), M_I_num, M_I_bgnf, strict=True
        ):
            ratio = num / bgnf
            assert 0.98 < ratio < 1.02, (
                f"{label}: numerical={num:.3e}, BGNF={bgnf:.3e}, "
                f"ratio={ratio:.4f}"
            )


class TestArbitraryPotentialBGNF:
    """Genuinely-new tests: cross-mode shifts from off-diagonal
    Hessian or higher-order cross terms (no published reference).
    Verifies internal consistency."""

    @pytest.fixture
    def trap_params(self):
        return dict(
            B=0.140,
            omega_z=TWO_PI * 2623.14e6,
            m=ELECTRON_MASS,
            q=-ELECTRON_CHARGE,
        )

    def test_K_is_real_for_real_potential(self, trap_params):
        """For a real electrostatic potential, the normal form K
        must have real coefficients (after projection)."""
        from tiqs.multipole import (
            ElectrostaticPotential,
            birkhoff_normal_form,
            cartesian_polynomials,
            linear_modes,
            potential_polynomial,
        )

        pot = ElectrostaticPotential.from_quadrupole(
            trap_params["omega_z"],
            trap_params["m"],
            trap_params["q"],
            epsilon=0.3,
        ) + ElectrostaticPotential({(0, 0, 4): 1e10, (4, 0, 0): 1e15})

        modes = linear_modes(
            pot,
            trap_params["B"],
            trap_params["m"],
            trap_params["q"],
        )
        H_pert = potential_polynomial(
            pot,
            cartesian_polynomials(modes.transform)[:3],
            trap_params["q"],
            min_order=3,
            max_order=4,
        )
        bnf = birkhoff_normal_form(
            H_pert,
            modes.omega_plus,
            modes.omega_z,
            modes.omega_minus,
            order=4,
        )
        for k, c in bnf.K.terms.items():
            if k[:3] == k[3:]:
                assert abs(c.imag) < 1e-12 * (abs(c.real) + 1)


class TestResonanceDetection:
    def test_no_resonance_at_BG_cylindrical(self):
        """Brown-Gabrielse cylindrical (ω_+ ≫ ω_z ≫ ω_-): no integer
        combination vanishes at order ≤ 4."""
        from tiqs.multipole import detect_resonances

        omega_plus = 8.79e10
        omega_z = 1.76e8
        omega_minus = 1.76e5
        resonances = detect_resonances(
            omega_plus, omega_z, omega_minus, max_total_degree=4
        )
        truly_resonant = [r for r in resonances if r[1] < omega_minus * 1e-3]
        assert len(truly_resonant) == 0

    def test_v3p4_one_to_minus_two_resonance(self):
        """v3p4 chip trap: ω_+ ≈ 2 ω_- gives a (1, 0, 2) resonance."""
        from tiqs.multipole import detect_resonances

        omega_plus = 2.0
        omega_z = 1.5
        omega_minus = 1.0
        resonances = detect_resonances(
            omega_plus, omega_z, omega_minus, max_total_degree=4
        )
        assert any(r[0] == (1, 0, 2) for r in resonances)


class TestPolynomialEdgeCases:
    def test_from_dict_validation(self):
        from tiqs.multipole import Polynomial

        with pytest.raises(ValueError, match="length"):
            Polynomial.from_dict({(1, 2, 3): 1.0})
        with pytest.raises(ValueError, match="non-negative"):
            Polynomial.from_dict({(-1, 0, 0, 0, 0, 0): 1.0})
        p = Polynomial.from_dict({(1, 0, 0, 0, 0, 0): 0.0})
        assert not p.terms

    def test_constant(self):
        from tiqs.multipole import Polynomial

        assert not Polynomial.constant(0).terms
        c = Polynomial.constant(2.5)
        assert c.terms == {(0, 0, 0, 0, 0, 0): 2.5 + 0j}

    def test_chop(self):
        from tiqs.multipole import Polynomial

        p = Polynomial.from_dict({
            (1, 0, 0, 0, 0, 0): 1.0,
            (0, 1, 0, 0, 0, 0): 1e-15,
        })
        chopped = p.chop(tol=1e-12)
        assert (1, 0, 0, 0, 0, 0) in chopped.terms
        assert (0, 1, 0, 0, 0, 0) not in chopped.terms

    def test_real_part(self):
        from tiqs.multipole import Polynomial

        p = Polynomial.from_dict({
            (1, 0, 0, 0, 0, 0): 1.0 + 2j,
            (0, 1, 0, 0, 0, 0): 0.0 + 1.5j,
        })
        rp = p.real_part()
        assert rp.terms[(1, 0, 0, 0, 0, 0)] == 1.0 + 0j
        # Pure-imaginary term is dropped (zero real part).
        assert (0, 1, 0, 0, 0, 0) not in rp.terms

    def test_repr_formats_polynomial(self):
        from tiqs.multipole import Polynomial

        p = Polynomial.variable(0) + Polynomial.variable(3)
        assert isinstance(repr(p), str)
        assert "0" in repr(Polynomial.zero())

    def test_negative_power_raises(self):
        from tiqs.multipole import Polynomial

        with pytest.raises(ValueError, match="non-negative"):
            Polynomial.variable(0) ** (-1)


class TestLinearModesOptionalArgs:
    def test_label_overlap_basis(self):
        """The optional ``label_overlap_basis`` argument lets the
        caller override the default labeling heuristic."""
        from tiqs.multipole import (
            ElectrostaticPotential,
            linear_modes,
        )

        pot = ElectrostaticPotential.from_quadrupole(
            TWO_PI * 200e6, ELECTRON_MASS, -ELECTRON_CHARGE, epsilon=0.0
        )
        # Polarization-vector candidates: cyclotron-x, axial-z,
        # magnetron-y (opposite rotation in the x-y plane).
        e_plus = np.array([1.0, 0, 0, 0, 0, 0])
        e_z = np.array([0, 0, 1.0, 0, 0, 0])
        e_minus = np.array([0, 1.0, 0, 0, 0, 0])
        result = linear_modes(
            pot,
            5.0,
            ELECTRON_MASS,
            -ELECTRON_CHARGE,
            label_overlap_basis=(e_plus, e_z, e_minus),
        )
        assert result.signatures == (1, 1, -1)


class TestCanonicalHessianValidation:
    def test_non_square_raises(self):
        from tiqs.multipole import canonical_hessian

        bad = np.zeros((3, 4))
        with pytest.raises(ValueError, match="3x3"):
            canonical_hessian(bad, 1.0, 1e-30, 1.6e-19)

    def test_non_symmetric_raises(self):
        from tiqs.multipole import canonical_hessian

        bad = np.array([[1, 2, 0], [3, 4, 0], [0, 0, 0]], dtype=float)
        with pytest.raises(ValueError, match="symmetric"):
            canonical_hessian(bad, 1.0, 1e-30, 1.6e-19)


class TestBirkhoffNormalFormErrors:
    def test_invalid_order_raises(self):
        from tiqs.multipole import (
            Polynomial,
            birkhoff_normal_form,
        )

        with pytest.raises(ValueError, match="order"):
            birkhoff_normal_form(Polynomial.zero(), 1.0, 1.0, 0.5, order=5)

    def test_homological_solver_resonant_input_raises(self):
        """Pure I_+^2 lives in the kernel, so the homological solver
        cannot invert it."""
        from tiqs.multipole import Polynomial, homological_solver

        p = Polynomial({(1, 0, 0, 1, 0, 0): 1 + 0j})
        with pytest.raises(ValueError, match="Resonant"):
            homological_solver(p, 1.0, 1.0, 0.5)


class TestPotentialPolynomial:
    def test_min_order_filter(self):
        from tiqs.multipole import (
            ElectrostaticPotential,
            cartesian_polynomials,
            linear_modes,
            potential_polynomial,
        )

        pot = ElectrostaticPotential.from_quadrupole(
            TWO_PI * 200e6, ELECTRON_MASS, -ELECTRON_CHARGE
        ) + ElectrostaticPotential({(0, 0, 4): 1e10})
        modes = linear_modes(pot, 5.0, ELECTRON_MASS, -ELECTRON_CHARGE)
        cart = cartesian_polynomials(modes.transform)
        # Default min_order=3 excludes the quadratic part.
        H = potential_polynomial(pot, cart[:3], -ELECTRON_CHARGE, min_order=3)
        # All terms should have total degree >= 3.
        for k in H.terms:
            assert sum(k) >= 3


class TestShiftMatrixGeneralReturn:
    def test_return_normal_form(self):
        from tiqs.multipole import (
            BirkhoffNormalForm,
            ElectrostaticPotential,
            LinearModeResult,
            shift_matrix_general,
        )

        pot = ElectrostaticPotential.from_quadrupole(
            TWO_PI * 200e6, ELECTRON_MASS, -ELECTRON_CHARGE
        ) + ElectrostaticPotential({(0, 0, 4): 1e10})
        result = shift_matrix_general(
            pot,
            5.0,
            ELECTRON_MASS,
            -ELECTRON_CHARGE,
            return_normal_form=True,
        )
        M, bnf, modes = result
        assert M.shape == (3, 3)
        assert isinstance(bnf, BirkhoffNormalForm)
        assert isinstance(modes, LinearModeResult)


class TestLinearModeResultHelpers:
    def test_actions_in_h2(self):
        """The `actions_in_h2` property returns the signed
        coefficients (omega_+, omega_z, -omega_-)."""
        from tiqs.multipole import (
            ElectrostaticPotential,
            linear_modes,
        )

        pot = ElectrostaticPotential.from_quadrupole(
            TWO_PI * 200e6, ELECTRON_MASS, -ELECTRON_CHARGE
        )
        r = linear_modes(pot, 5.0, ELECTRON_MASS, -ELECTRON_CHARGE)
        a = r.actions_in_h2
        assert a[0] == r.omega_plus
        assert a[1] == r.omega_z
        assert a[2] == -r.omega_minus


class TestElectrostaticPotentialHelpers:
    def test_get_default_zero(self):
        from tiqs.multipole import ElectrostaticPotential

        pot = ElectrostaticPotential()
        assert pot.get((1, 1, 1)) == 0.0

    def test_addition_drops_zero_results(self):
        from tiqs.multipole import ElectrostaticPotential

        a = ElectrostaticPotential({(2, 0, 0): 1.0})
        b = ElectrostaticPotential({(2, 0, 0): -1.0})
        c = a + b
        assert (2, 0, 0) not in c.coeffs

    def test_scale_drops_zero_factor(self):
        from tiqs.multipole import ElectrostaticPotential

        a = ElectrostaticPotential({(2, 0, 0): 1.0})
        b = a.scale(0.0)
        assert b.coeffs == {}

    def test_laplacian_dictionary(self):
        """φ = z² has Laplacian = 2 (constant)."""
        from tiqs.multipole import ElectrostaticPotential

        pot = ElectrostaticPotential({(0, 0, 2): 1.0})
        lap = pot.laplacian()
        assert lap == {(0, 0, 0): 2.0}


class TestSplitKernelEdge:
    def test_zero_omega_uses_unit_scale(self):
        """If all omegas are zero, threshold uses unit scale (no
        division by zero)."""
        from tiqs.multipole import Polynomial, split_kernel_image

        p = Polynomial.variable(0) + Polynomial.variable(3)
        kernel, image = split_kernel_image(p, 0.0, 0.0, 0.0)
        # All spectral coefficients are zero so everything sits in K.
        assert image.terms == {}


class TestRestrictToOrders:
    def test_max_order_default(self):
        from tiqs.multipole import ElectrostaticPotential

        pot = ElectrostaticPotential({
            (2, 0, 0): 1.0,
            (4, 0, 0): 2.0,
            (6, 0, 0): 3.0,
        })
        sub = pot.restrict_to_orders(min_order=3)
        assert (2, 0, 0) not in sub.coeffs
        assert (4, 0, 0) in sub.coeffs
        assert (6, 0, 0) in sub.coeffs


class TestCoverageHelpers:
    def test_polynomial_add_scalar(self):
        """Scalar + Polynomial uses Polynomial.constant via __radd__."""
        from tiqs.multipole import Polynomial

        p = Polynomial.variable(0)
        result = 3.0 + p
        assert (0, 0, 0, 0, 0, 0) in result.terms
        assert (1, 0, 0, 0, 0, 0) in result.terms

    def test_polynomial_rmul_scalar(self):
        from tiqs.multipole import Polynomial

        p = Polynomial.variable(0)
        result = 2.5 * p
        assert result.terms == {(1, 0, 0, 0, 0, 0): 2.5 + 0j}

    def test_polynomial_mul_zero(self):
        from tiqs.multipole import Polynomial

        p = Polynomial.variable(0)
        z = p * 0
        assert not z.terms

    def test_potential_polynomial_cache_paths(self):
        """Build several cartesian monomials in sequence to exercise
        the polynomial cache (re-uses smaller monomials)."""
        from tiqs.multipole import (
            ElectrostaticPotential,
            cartesian_polynomials,
            linear_modes,
            potential_polynomial,
        )

        pot = ElectrostaticPotential.from_quadrupole(
            TWO_PI * 200e6, ELECTRON_MASS, -ELECTRON_CHARGE
        ) + ElectrostaticPotential({
            (4, 0, 0): 1e15,
            (2, 2, 0): 1e15,
            (0, 0, 4): 1e10,
        })
        modes = linear_modes(pot, 5.0, ELECTRON_MASS, -ELECTRON_CHARGE)
        cart = cartesian_polynomials(modes.transform)
        H = potential_polynomial(pot, cart[:3], -ELECTRON_CHARGE, min_order=3)
        assert H.terms

    def test_birkhoff_with_resonant_cubic(self):
        """Tune ω_z = 2 ω_- so that a_z a_-² has zero spectral
        coefficient. BGNF should park the cubic in K_3."""
        from tiqs.multipole import (
            Polynomial,
            birkhoff_normal_form,
        )

        omega_plus = 5.0
        omega_minus = 1.0
        omega_z = 2.0

        H_pert = Polynomial({
            (0, 1, 2, 0, 0, 0): 1.0 + 0j,
            (0, 0, 0, 0, 1, 2): 1.0 + 0j,
        })
        bnf = birkhoff_normal_form(
            H_pert,
            omega_plus,
            omega_z,
            omega_minus,
            order=4,
        )
        assert bnf.detected_resonances
        K3 = bnf.K.homogeneous_part(3)
        assert K3.terms

    def test_polynomial_repr_complex_coeffs(self):
        """__repr__ handles real, imaginary, and mixed coefficients."""
        from tiqs.multipole import Polynomial

        p = Polynomial.from_dict({
            (1, 0, 0, 0, 0, 0): 2.0 + 0j,
            (0, 1, 0, 0, 0, 0): 0.0 + 3.0j,
            (0, 0, 1, 0, 0, 0): 1.0 + 1.0j,
        })
        s = repr(p)
        assert "Polynomial" in s


class TestEdgeCaseCoverage:
    """A few small targeted tests for edge-case branches."""

    def test_invariance_residual_zero_b_field(self):
        """If magnetic_field is 0, omega_c is 0 and invariance
        residual returns 0 by convention."""
        from tiqs.multipole import LinearModeResult

        r = LinearModeResult(
            omega_plus=1.0,
            omega_z=2.0,
            omega_minus=0.0,
            transform=np.eye(6),
            signatures=(1, 1, -1),
            cyclotron_frequency=0.0,
        )
        assert r.invariance_residual() == 0.0

    def test_polynomial_subtract_partial_cancel(self):
        """Subtracting partially-overlapping polynomials cleans up
        zeros from the result dict (else branch in __add__)."""
        from tiqs.multipole import Polynomial

        a = Polynomial.from_dict({
            (1, 0, 0, 0, 0, 0): 1.0,
            (0, 1, 0, 0, 0, 0): 2.0,
        })
        b = Polynomial.from_dict({(0, 1, 0, 0, 0, 0): 1.0})
        c = a + b  # (0, 1, 0, ...) coeff becomes 3, kept (else branch)
        assert c.terms[(0, 1, 0, 0, 0, 0)] == 3.0 + 0j

    def test_resonant_M_matrix_raises(self):
        """A normal form with a non-diagonal monomial (a_+ ā_-) means
        the regime is resonant; reading M must raise."""
        from tiqs.multipole import (
            BirkhoffNormalForm,
            Polynomial,
            frequency_shift_matrix_actions,
        )

        K = Polynomial.from_dict({(1, 0, 0, 0, 0, 1): 1.0 + 0j})
        bnf = BirkhoffNormalForm(
            K=K,
            generators={},
            omegas=(1.0, 1.0, 0.5),
            order=4,
            resonance_tol=1e-9,
            detected_resonances=(),
        )
        with pytest.raises(ValueError, match="non-diagonal"):
            frequency_shift_matrix_actions(bnf)


class TestPotentialPolynomialCacheHits:
    """Ensure the monomial-cache fast paths in potential_polynomial
    are exercised."""

    def test_chained_x_powers(self):
        """A potential with (3, 0, 0), (4, 0, 0) sequentially lets the
        (4, 0, 0) lookup re-use the cached (3, 0, 0). Only orders ≥3 are
        used so the quadratic Hessian (and trap stability) is untouched.
        """
        from tiqs.multipole import (
            ElectrostaticPotential,
            cartesian_polynomials,
            linear_modes,
            potential_polynomial,
        )

        coeffs = {
            (3, 0, 0): 1e10,
            (4, 0, 0): 1e10,
            (0, 3, 0): 1e10,
            (0, 4, 0): 1e10,
            (0, 0, 3): 1e10,
            (0, 0, 4): 1e10,
            (2, 1, 0): 1e10,
            (1, 2, 1): 1e10,
        }
        pot = ElectrostaticPotential.from_quadrupole(
            TWO_PI * 200e6, ELECTRON_MASS, -ELECTRON_CHARGE
        ) + ElectrostaticPotential(coeffs)
        modes = linear_modes(pot, 5.0, ELECTRON_MASS, -ELECTRON_CHARGE)
        cart = cartesian_polynomials(modes.transform)
        H = potential_polynomial(
            pot, cart[:3], -ELECTRON_CHARGE, min_order=3, max_order=4
        )
        assert H.terms

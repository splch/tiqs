import warnings

import numpy as np
import pytest
import qutip

from tiqs.hilbert_space.builder import MIN_FOCK_DIM, HilbertSpace
from tiqs.hilbert_space.operators import OperatorFactory
from tiqs.hilbert_space.states import (
    THERMAL_MEAN_TOL,
    StateFactory,
    _suggested_fock_dim,
)


def _truncated_thermal_mean(dim: int, n_bar: float) -> float:
    """Exact mean of a Bose-Einstein distribution cut at ``dim`` levels.

    Independent reference for the truncation: the geometric weights
    $(\\bar{n}/(1+\\bar{n}))^n$ renormalized over $n < dim$.
    """
    ratio = n_bar / (1.0 + n_bar)
    n = np.arange(dim)
    weights = ratio**n
    return float((n * weights).sum() / weights.sum())


class TestHilbertSpace:
    def test_dimensions_single_ion_single_mode(self):
        hs = HilbertSpace(n_ions=1, n_modes=1, n_fock=10)
        assert hs.dims == [2, 10]
        assert hs.total_dim == 20

    def test_dimensions_two_ions_two_modes(self):
        hs = HilbertSpace(n_ions=2, n_modes=2, n_fock=5)
        assert hs.dims == [2, 2, 5, 5]
        assert hs.total_dim == 100

    def test_dimensions_three_ions_three_modes(self):
        hs = HilbertSpace(n_ions=3, n_modes=3, n_fock=4)
        assert hs.dims == [2, 2, 2, 4, 4, 4]
        assert hs.total_dim == 512

    def test_custom_fock_per_mode(self):
        hs = HilbertSpace(n_ions=2, n_modes=2, n_fock=[10, 5])
        assert hs.dims == [2, 2, 10, 5]

    def test_zero_ions_pure_bosonic(self):
        """n_ions=0 gives a purely bosonic Hilbert space."""
        hs = HilbertSpace(n_ions=0, n_modes=2, n_fock=10)
        assert hs.dims == [10, 10]
        assert hs.total_dim == 100

    @pytest.mark.parametrize("bad", [1, 0, -3])
    def test_n_fock_below_minimum_raises(self, bad):
        """Cutoffs < 2 used to yield dims like [2, 0] and total_dim 0."""
        with pytest.raises(ValueError, match="n_fock must be >= 2"):
            HilbertSpace(n_ions=1, n_modes=1, n_fock=bad)

    @pytest.mark.parametrize("bad", [True, 2.5, 10.0, None])
    def test_n_fock_non_int_raises(self, bad):
        """``True`` used to pass as a 1-level mode; floats hit len()."""
        with pytest.raises(TypeError, match="n_fock must be an int"):
            HilbertSpace(n_ions=1, n_modes=1, n_fock=bad)

    def test_n_fock_list_elements_are_validated(self):
        """Per-mode cutoffs get the same checks, naming the mode."""
        with pytest.raises(ValueError, match=r"n_fock\[1\] must be >= 2"):
            HilbertSpace(n_ions=1, n_modes=2, n_fock=[10, 0])
        with pytest.raises(TypeError, match=r"n_fock\[0\] must be an int"):
            HilbertSpace(n_ions=1, n_modes=2, n_fock=[True, 10])

    def test_minimum_fock_dim_still_has_a_ladder(self):
        """The lowest allowed cutoff obeys a|1> = |0> exactly.

        This is why 2 is the floor: at dim 1 the ladder vanishes
        (``qutip.destroy(1)`` cannot even be constructed, while
        ``qutip.num(1)`` silently freezes the motion).
        """
        hs = HilbertSpace(n_ions=0, n_modes=1, n_fock=MIN_FOCK_DIM)
        a = OperatorFactory(hs).annihilate(0)
        excited = qutip.basis(MIN_FOCK_DIM, 1)
        residual = a * excited - qutip.basis(MIN_FOCK_DIM, 0)
        assert residual.norm() == pytest.approx(0.0, abs=1e-12)


class TestOperatorFactory:
    @pytest.fixture
    def ops(self):
        hs = HilbertSpace(n_ions=2, n_modes=2, n_fock=5)
        return OperatorFactory(hs)

    def test_sigma_z_shape(self, ops):
        sz = ops.sigma_z(0)
        assert sz.shape == (100, 100)

    def test_sigma_z_is_hermitian(self, ops):
        sz = ops.sigma_z(0)
        assert sz.isherm

    def test_sigma_z_different_ions(self, ops):
        sz0 = ops.sigma_z(0)
        sz1 = ops.sigma_z(1)
        assert sz0 != sz1
        comm = sz0 * sz1 - sz1 * sz0
        assert comm.norm() == pytest.approx(0.0, abs=1e-12)

    def test_annihilation_shape(self, ops):
        a = ops.annihilate(0)
        assert a.shape == (100, 100)

    def test_annihilation_different_modes(self, ops):
        a0 = ops.annihilate(0)
        a1 = ops.annihilate(1)
        assert a0 != a1

    def test_number_operator(self, ops):
        n = ops.number(0)
        assert n.isherm

    def test_sigma_plus_minus(self, ops):
        sp = ops.sigma_plus(0)
        sm = ops.sigma_minus(0)
        assert (sp.dag() - sm).norm() == pytest.approx(0.0, abs=1e-12)

    def test_sigma_minus_excites_sigma_plus_de_excites(self, ops):
        """TIQS convention: basis(2, 0) is the ground state.

        So ``sigma_minus`` = |1><0| is the excitation operator and
        ``sigma_plus`` = |0><1| is the decay (de-excitation) operator,
        the opposite of the "raising"/"lowering" matrix names.
        """
        sf = StateFactory(ops.hs)
        ground = sf.product_state(qubit_states=[0, 0], fock_states=[0, 0])
        excited = sf.product_state(qubit_states=[1, 0], fock_states=[0, 0])
        assert (ops.sigma_minus(0) * ground - excited).norm() == (
            pytest.approx(0.0, abs=1e-12)
        )
        assert (ops.sigma_plus(0) * excited - ground).norm() == (
            pytest.approx(0.0, abs=1e-12)
        )
        # Neither operator can act twice in the same direction.
        assert (ops.sigma_minus(0) * excited).norm() == pytest.approx(
            0.0, abs=1e-12
        )
        assert (ops.sigma_plus(0) * ground).norm() == pytest.approx(
            0.0, abs=1e-12
        )

    def test_sigma_docstrings_state_the_physical_role(self):
        """Guards the label that misled noise.md into a sign error."""
        plus = OperatorFactory.sigma_plus.__doc__.splitlines()[0].lower()
        minus = OperatorFactory.sigma_minus.__doc__.splitlines()[0].lower()
        assert plus.startswith("de-excitation")
        assert minus.startswith("excitation")

    def test_quadratures_have_unit_commutator(self):
        """[x, p] = i fixes x = (a + adag)/sqrt(2), not a + adag."""
        dim = 40
        hs = HilbertSpace(n_ions=0, n_modes=1, n_fock=dim)
        ops = OperatorFactory(hs)
        x = ops.position(0)
        p = ops.momentum(0)
        comm = (x * p - p * x).full()
        # Truncation spoils only the top Fock level.
        np.testing.assert_allclose(np.diag(comm)[:-1], 1j, atol=1e-12)
        vacuum = StateFactory(hs).ground_state()
        assert qutip.expect(x * x, vacuum) == pytest.approx(0.5, abs=1e-12)

    def test_potential_coordinate_is_sqrt2_times_position(self):
        """tiqs.potential's q = a + adag equals sqrt(2) x.

        Vacuum variances differ accordingly: <x^2> = 1/2 but <q^2> = 1.
        A potential written in q must not be handed x values.
        """
        hs = HilbertSpace(n_ions=0, n_modes=1, n_fock=8)
        ops = OperatorFactory(hs)
        a = ops.annihilate(0)
        q = a + a.dag()
        assert (q - 2**0.5 * ops.position(0)).norm() == pytest.approx(
            0.0, abs=1e-12
        )
        vacuum = StateFactory(hs).ground_state()
        assert qutip.expect(q * q, vacuum) == pytest.approx(1.0, abs=1e-12)

    def test_identity(self, ops):
        eye = ops.identity()
        assert eye.shape == (100, 100)
        assert eye.tr() == pytest.approx(100.0)

    def test_invalid_ion_index(self, ops):
        with pytest.raises(IndexError):
            ops.sigma_z(5)

    def test_invalid_mode_index(self, ops):
        with pytest.raises(IndexError):
            ops.annihilate(5)

    def test_zero_ions_operators(self):
        """OperatorFactory works with n_ions=0 (pure bosonic)."""
        hs = HilbertSpace(n_ions=0, n_modes=2, n_fock=5)
        ops = OperatorFactory(hs)
        a = ops.annihilate(0)
        assert a.shape == (25, 25)
        assert ops.number(1).isherm
        with pytest.raises(IndexError):
            ops.sigma_z(0)


class TestStateFactory:
    @pytest.fixture
    def sf(self):
        hs = HilbertSpace(n_ions=2, n_modes=1, n_fock=10)
        return StateFactory(hs)

    def test_ground_state_is_ket(self, sf):
        psi = sf.ground_state()
        assert psi.type == "ket"
        assert psi.shape == (40, 1)

    def test_ground_state_norm(self, sf):
        psi = sf.ground_state()
        assert abs(psi.norm() - 1.0) < 1e-12

    def test_thermal_state_is_dm(self, sf):
        rho = sf.thermal_state(n_bar=[0.5])
        assert rho.type == "oper"
        assert rho.tr() == pytest.approx(1.0, abs=1e-10)

    def test_thermal_state_mean_phonon(self, sf):
        """Realized <n> matches the exact truncated Bose-Einstein sum.

        At n_fock = 10 the cut costs 0.98% of n_bar = 1, so a
        percent-level tolerance here would hide a real coefficient
        error; pin the exact law instead.
        """
        n_bar = 1.0
        rho = sf.thermal_state(n_bar=[n_bar])
        n_op = qutip.tensor(qutip.qeye(2), qutip.qeye(2), qutip.num(10))
        mean_n = qutip.expect(n_op, rho)
        assert mean_n == pytest.approx(
            _truncated_thermal_mean(10, n_bar), rel=1e-12
        )
        # Truncation only ever removes population from high n.
        assert 0.0 < n_bar - mean_n < 0.01 * n_bar

    def test_thermal_state_negative_n_bar_raises(self, sf):
        """A negative n_bar used to return a NaN-valued density matrix."""
        with pytest.raises(ValueError, match=r"n_bar\[0\] must be >= 0"):
            sf.thermal_state(n_bar=[-1.0])

    def test_thermal_state_warns_on_inadequate_cutoff(self, sf):
        """n_bar = 5 at n_fock = 10 silently started at <n> = 3.07."""
        with pytest.warns(UserWarning, match="requested n_bar=5"):
            rho = sf.thermal_state(n_bar=[5.0])
        n_op = qutip.tensor(qutip.qeye(2), qutip.qeye(2), qutip.num(10))
        assert qutip.expect(n_op, rho) == pytest.approx(
            _truncated_thermal_mean(10, 5.0), rel=1e-12
        )

    def test_thermal_state_warning_names_the_mode(self):
        """Only the under-resolved mode is reported."""
        hs = HilbertSpace(n_ions=0, n_modes=2, n_fock=[10, 60])
        sf = StateFactory(hs)
        with pytest.warns(UserWarning, match="mode 0: requested n_bar=8"):
            sf.thermal_state(n_bar=[8.0, 8.0])

    @pytest.mark.parametrize("n_bar", [0.1, 1.0, 3.0, 5.0, 10.0, 20.0])
    def test_suggested_cutoff_achieves_tolerance(self, n_bar):
        """The n_fock named in the warning really does deliver 2%."""
        dim = _suggested_fock_dim(n_bar)
        sf = StateFactory(HilbertSpace(n_ions=0, n_modes=1, n_fock=dim))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            rho = sf.thermal_state(n_bar=[n_bar])
        mean_n = qutip.expect(qutip.num(dim), rho)
        assert abs(mean_n - n_bar) <= THERMAL_MEAN_TOL * n_bar
        assert mean_n == pytest.approx(
            _truncated_thermal_mean(dim, n_bar), rel=1e-12
        )

    def test_thermal_state_zero_n_bar_is_the_vacuum(self, sf):
        """n_bar = 0 is legal and exact - no warning, no NaN."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            rho = sf.thermal_state(n_bar=[0.0])
        n_op = qutip.tensor(qutip.qeye(2), qutip.qeye(2), qutip.num(10))
        assert qutip.expect(n_op, rho) == pytest.approx(0.0, abs=1e-14)
        assert rho.tr() == pytest.approx(1.0, abs=1e-12)

    def test_custom_qubit_states(self, sf):
        psi = sf.product_state(qubit_states=[1, 0], fock_states=[0])
        assert psi.type == "ket"

    def test_all_ions_down_motional_ground(self, sf):
        psi = sf.ground_state()
        sz0 = qutip.tensor(qutip.sigmaz(), qutip.qeye(2), qutip.qeye(10))
        assert qutip.expect(sz0, psi) == pytest.approx(1.0)

    def test_zero_ions_states(self):
        """StateFactory works with n_ions=0 (pure bosonic)."""
        hs = HilbertSpace(n_ions=0, n_modes=2, n_fock=10)
        sf = StateFactory(hs)
        psi = sf.ground_state()
        assert psi.type == "ket"
        assert psi.shape == (100, 1)
        psi2 = sf.product_state(qubit_states=[], fock_states=[3, 0])
        assert psi2.type == "ket"
        rho = sf.thermal_state(n_bar=[1.0, 0.5])
        assert rho.type == "oper"
        assert rho.tr() == pytest.approx(1.0, abs=1e-10)

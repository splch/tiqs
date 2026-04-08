import pytest
import qutip

from tiqs.hilbert_space.builder import HilbertSpace
from tiqs.hilbert_space.operators import OperatorFactory
from tiqs.hilbert_space.states import StateFactory


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
        rho = sf.thermal_state(n_bar=[5.0])
        assert rho.type == "oper"
        assert rho.tr() == pytest.approx(1.0, abs=1e-10)

    def test_thermal_state_mean_phonon(self, sf):
        n_bar = 1.0
        rho = sf.thermal_state(n_bar=[n_bar])
        n_op = qutip.tensor(qutip.qeye(2), qutip.qeye(2), qutip.num(10))
        mean_n = qutip.expect(n_op, rho)
        assert mean_n == pytest.approx(n_bar, rel=0.1)

    def test_custom_qubit_states(self, sf):
        psi = sf.product_state(qubit_states=[1, 0], fock_states=[0])
        assert psi.type == "ket"

    def test_all_ions_down_motional_ground(self, sf):
        psi = sf.ground_state()
        sz0 = qutip.tensor(qutip.sigmaz(), qutip.qeye(2), qutip.qeye(10))
        assert qutip.expect(sz0, psi) == pytest.approx(1.0)

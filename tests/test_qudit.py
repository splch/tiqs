"""Tests for qudit support: d > 2 level ions."""

import numpy as np
import pytest
import qutip

from tiqs.constants import TWO_PI
from tiqs.gates.molmer_sorensen import ms_gate_duration
from tiqs.gates.qudit import (
    light_shift_qudit_gate_hamiltonian,
    ms_qudit_gate_hamiltonian,
    r_transition,
)
from tiqs.hilbert_space.builder import HilbertSpace
from tiqs.hilbert_space.operators import OperatorFactory
from tiqs.hilbert_space.states import StateFactory


class TestHilbertSpaceQudit:
    def test_default_is_qubit(self):
        hs = HilbertSpace(n_ions=2, n_modes=1, n_fock=5)
        assert hs.dims == [2, 2, 5]
        assert hs.ion_dim(0) == 2
        assert hs.ion_dim(1) == 2

    def test_uniform_qutrit(self):
        hs = HilbertSpace(n_ions=2, n_modes=1, n_fock=5, ion_dims=3)
        assert hs.dims == [3, 3, 5]
        assert hs.ion_dim(0) == 3
        assert hs.total_dim == 45

    def test_per_ion_dims(self):
        hs = HilbertSpace(n_ions=3, n_modes=1, n_fock=5, ion_dims=[2, 4, 3])
        assert hs.dims == [2, 4, 3, 5]
        assert hs.ion_dim(0) == 2
        assert hs.ion_dim(1) == 4
        assert hs.ion_dim(2) == 3
        assert hs.total_dim == 120

    def test_dim_less_than_2_raises(self):
        with pytest.raises(ValueError, match="must be >= 2"):
            HilbertSpace(n_ions=1, n_modes=1, ion_dims=1)

    def test_wrong_length_raises(self):
        with pytest.raises(ValueError, match="ion_dims length"):
            HilbertSpace(n_ions=2, n_modes=1, ion_dims=[3])


class TestOperatorFactoryQudit:
    @pytest.fixture
    def qutrit_ops(self):
        hs = HilbertSpace(n_ions=2, n_modes=1, n_fock=5, ion_dims=3)
        return OperatorFactory(hs)

    @pytest.fixture
    def mixed_ops(self):
        """One qubit (d=2) and one qutrit (d=3)."""
        hs = HilbertSpace(n_ions=2, n_modes=1, n_fock=5, ion_dims=[2, 3])
        return OperatorFactory(hs)

    def test_sigma_x_on_qubit_works(self):
        hs = HilbertSpace(n_ions=1, n_modes=1, n_fock=3)
        ops = OperatorFactory(hs)
        sx = ops.sigma_x(0)
        assert sx.isherm
        assert sx.shape == (6, 6)

    def test_sigma_x_on_qutrit_raises(self, qutrit_ops):
        with pytest.raises(ValueError, match="sigma_"):
            qutrit_ops.sigma_x(0)

    def test_sigma_z_on_qutrit_raises(self, qutrit_ops):
        with pytest.raises(ValueError, match="sigma_"):
            qutrit_ops.sigma_z(0)

    def test_sigma_on_mixed_qubit_ok(self, mixed_ops):
        """sigma_x on ion 0 (qubit) should work."""
        sx = mixed_ops.sigma_x(0)
        assert sx.isherm

    def test_sigma_on_mixed_qutrit_raises(self, mixed_ops):
        """sigma_x on ion 1 (qutrit) should raise."""
        with pytest.raises(ValueError, match="sigma_"):
            mixed_ops.sigma_x(1)

    def test_transition_operator(self, qutrit_ops):
        t01 = qutrit_ops.transition(0, 0, 1)
        assert t01.shape == (45, 45)
        # |0><1| is not Hermitian
        assert not t01.isherm

    def test_transition_out_of_range(self, qutrit_ops):
        with pytest.raises(IndexError):
            qutrit_ops.transition(0, 0, 3)  # d=3, so level 3 invalid

    def test_transition_x_is_hermitian(self, qutrit_ops):
        tx = qutrit_ops.transition_x(0, 0, 1)
        assert tx.isherm

    def test_transition_y_is_hermitian(self, qutrit_ops):
        ty = qutrit_ops.transition_y(0, 0, 1)
        assert ty.isherm

    def test_transition_z_is_hermitian(self, qutrit_ops):
        tz = qutrit_ops.transition_z(0, 0, 1)
        assert tz.isherm

    def test_transition_xz_match_pauli_for_qubit(self):
        """For d=2, transition_x(0,1) should equal sigma_x."""
        hs = HilbertSpace(n_ions=1, n_modes=1, n_fock=3)
        ops = OperatorFactory(hs)
        tx = ops.transition_x(0, 0, 1)
        sx = ops.sigma_x(0)
        assert (tx - sx).norm() < 1e-12

    def test_transition_z_matches_sigma_z_for_qubit(self):
        hs = HilbertSpace(n_ions=1, n_modes=1, n_fock=3)
        ops = OperatorFactory(hs)
        tz = ops.transition_z(0, 0, 1)
        sz = ops.sigma_z(0)
        assert (tz - sz).norm() < 1e-12

    def test_projector(self, qutrit_ops):
        p0 = qutrit_ops.projector(0, 0)
        p1 = qutrit_ops.projector(0, 1)
        # Projectors are orthogonal
        assert (p0 * p1).norm() < 1e-12
        assert p0.isherm

    def test_spin_j_z(self, qutrit_ops):
        """For d=3 (j=1), J_z eigenvalues are {-1, 0, +1}."""
        jz = qutrit_ops.spin_j(0, "z")
        assert jz.isherm
        # Trace should be zero (sum of eigenvalues)
        jz_local = jz.ptrace(0)
        assert jz_local.tr() == pytest.approx(0.0, abs=1e-10)

    def test_annihilate_still_works(self, qutrit_ops):
        """Motional operators are independent of ion dimension."""
        a = qutrit_ops.annihilate(0)
        assert a.shape == (45, 45)


class TestStateFactoryQudit:
    def test_ground_state_qutrit(self):
        hs = HilbertSpace(n_ions=2, n_modes=1, n_fock=5, ion_dims=3)
        sf = StateFactory(hs)
        gs = sf.ground_state()
        assert gs.type == "ket"
        assert gs.shape == (45, 1)
        assert abs(gs.norm() - 1.0) < 1e-12

    def test_product_state_qutrit(self):
        hs = HilbertSpace(n_ions=2, n_modes=1, n_fock=5, ion_dims=3)
        sf = StateFactory(hs)
        psi = sf.product_state([2, 1], [0])
        assert psi.type == "ket"

    def test_product_state_out_of_range(self):
        hs = HilbertSpace(n_ions=1, n_modes=1, n_fock=3, ion_dims=3)
        sf = StateFactory(hs)
        with pytest.raises(ValueError, match="out of range"):
            sf.product_state([3], [0])  # d=3, so level 3 invalid

    def test_thermal_state_qutrit(self):
        hs = HilbertSpace(n_ions=1, n_modes=1, n_fock=5, ion_dims=3)
        sf = StateFactory(hs)
        rho = sf.thermal_state(n_bar=[1.0])
        assert rho.type == "oper"
        assert rho.tr() == pytest.approx(1.0, abs=1e-10)

    def test_mixed_dims_ground_state(self):
        hs = HilbertSpace(n_ions=2, n_modes=1, n_fock=5, ion_dims=[2, 4])
        sf = StateFactory(hs)
        gs = sf.ground_state()
        assert gs.shape == (40, 1)


class TestQuditGates:
    @pytest.fixture
    def qutrit_system(self):
        hs = HilbertSpace(n_ions=2, n_modes=1, n_fock=10, ion_dims=3)
        ops = OperatorFactory(hs)
        sf = StateFactory(hs)
        return hs, ops, sf

    def test_r_transition_flips(self, qutrit_system):
        """Pi rotation in |0>-|1> subspace should flip |0> -> |1>."""
        _hs, ops, sf = qutrit_system
        psi0 = sf.ground_state()
        gate = r_transition(ops, 0, 0, 1, theta=np.pi)
        result = qutip.sesolve(gate.hamiltonian, psi0, [0, gate.duration])
        final = result.states[-1]
        target = sf.product_state([1, 0], [0])
        fid = abs(final.overlap(target)) ** 2
        assert fid > 0.99

    def test_r_transition_preserves_third_level(self, qutrit_system):
        """|0>-|1> rotation should not populate |2>."""
        _hs, ops, sf = qutrit_system
        psi0 = sf.ground_state()
        gate = r_transition(ops, 0, 0, 1, theta=np.pi / 2)
        result = qutip.sesolve(gate.hamiltonian, psi0, [0, gate.duration])
        final = result.states[-1]
        p2 = qutip.expect(ops.projector(0, 2), final)
        assert p2 < 0.01

    def test_ms_qudit_gate_list_format(self, qutrit_system):
        _hs, ops, _sf = qutrit_system
        H = ms_qudit_gate_hamiltonian(
            ops,
            [0, 1],
            0,
            [0.1, 0.1],
            TWO_PI * 50e3,
            TWO_PI * 10e3,
            transitions=[(0, 1), (0, 1)],
        )
        assert isinstance(H, list)
        assert len(H) == 4  # 2 ions * 2 terms

    def test_ms_qudit_entangles(self, qutrit_system):
        """MS gate on |0>-|1> subspace of qutrits should entangle."""
        _hs, ops, sf = qutrit_system
        eta = 0.1
        delta = TWO_PI * 20e3
        Omega = delta / (4 * eta)
        tau = ms_gate_duration(delta)

        H = ms_qudit_gate_hamiltonian(
            ops,
            [0, 1],
            0,
            [eta, eta],
            Omega,
            delta,
            transitions=[(0, 1), (0, 1)],
        )
        psi0 = sf.ground_state()
        tlist = np.linspace(0, tau, 500)
        result = qutip.sesolve(H, psi0, tlist, options={"max_step": tau / 100})
        rho_spin = result.states[-1].ptrace([0, 1])
        rho_single = rho_spin.ptrace(0)
        purity = (rho_single * rho_single).tr().real
        assert purity < 0.9  # entangled

    def test_light_shift_qudit_list_format(self, qutrit_system):
        _hs, ops, _sf = qutrit_system
        H = light_shift_qudit_gate_hamiltonian(
            ops,
            [0, 1],
            0,
            [0.1, 0.1],
            TWO_PI * 50e3,
            TWO_PI * 10e3,
            transitions=[(0, 1), (0, 1)],
        )
        assert isinstance(H, list)
        assert len(H) == 4


class TestBackwardCompatibility:
    """Verify that qubit code is completely unaffected."""

    def test_hilbert_space_default_is_qubit(self):
        hs = HilbertSpace(n_ions=2, n_modes=1, n_fock=10)
        assert hs.dims == [2, 2, 10]
        assert hs.total_dim == 40

    def test_operator_factory_qubit_unchanged(self):
        hs = HilbertSpace(n_ions=2, n_modes=1, n_fock=5)
        ops = OperatorFactory(hs)
        sx = ops.sigma_x(0)
        sz = ops.sigma_z(1)
        sp = ops.sigma_plus(0)
        sm = ops.sigma_minus(0)
        assert sx.isherm
        assert sz.isherm
        assert (sp.dag() - sm).norm() < 1e-12

    def test_state_factory_qubit_unchanged(self):
        hs = HilbertSpace(n_ions=2, n_modes=1, n_fock=10)
        sf = StateFactory(hs)
        gs = sf.ground_state()
        assert gs.type == "ket"
        assert gs.shape == (40, 1)
        ps = sf.product_state([1, 0], [0])
        assert ps.type == "ket"

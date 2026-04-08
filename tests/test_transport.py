import numpy as np
import pytest
import qutip

from tiqs.hilbert_space.builder import HilbertSpace
from tiqs.hilbert_space.operators import OperatorFactory
from tiqs.hilbert_space.states import StateFactory
from tiqs.transport import (
    apply_shuttling_noise,
    shuttle_motional_excitation,
    split_crystal_excitation,
)


@pytest.fixture
def system():
    hs = HilbertSpace(n_ions=1, n_modes=1, n_fock=20)
    ops = OperatorFactory(hs)
    sf = StateFactory(hs)
    return hs, ops, sf


class TestShuttling:
    def test_fast_shuttle_heats(self):
        delta_n = shuttle_motional_excitation(
            distance=200e-6,
            duration=50e-6,
            trap_frequency=2 * np.pi * 1e6,
        )
        assert delta_n > 0

    def test_slow_shuttle_less_heating(self):
        fast = shuttle_motional_excitation(200e-6, 10e-6, 2 * np.pi * 1e6)
        slow = shuttle_motional_excitation(200e-6, 300e-6, 2 * np.pi * 1e6)
        assert slow < fast

    def test_apply_shuttling_noise(self, system):
        hs, ops, sf = system
        rho0 = qutip.ket2dm(sf.ground_state())
        n_op = ops.number(0)
        n_before = qutip.expect(n_op, rho0)
        rho_after = apply_shuttling_noise(rho0, ops, mode=0, added_quanta=0.5)
        n_after = qutip.expect(n_op, rho_after)
        assert n_after > n_before


class TestCrystalSplitting:
    def test_split_excitation_positive(self):
        delta_n = split_crystal_excitation(
            trap_frequency=2 * np.pi * 1e6,
            split_duration=100e-6,
        )
        assert delta_n >= 0

    def test_slower_split_less_heating(self):
        fast = split_crystal_excitation(2 * np.pi * 1e6, 20e-6)
        slow = split_crystal_excitation(2 * np.pi * 1e6, 200e-6)
        assert slow <= fast

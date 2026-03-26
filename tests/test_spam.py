import numpy as np
import pytest
import qutip

from tiqs.spam.preparation import optical_pumping_ops, prepare_qubit
from tiqs.spam.measurement import (
    fluorescence_probabilities,
    sample_measurement,
    measurement_fidelity,
    mid_circuit_measurement,
)
from tiqs.hilbert_space.builder import HilbertSpace
from tiqs.hilbert_space.operators import OperatorFactory
from tiqs.hilbert_space.states import StateFactory


@pytest.fixture
def system():
    hs = HilbertSpace(n_ions=2, n_modes=1, n_fock=5)
    ops = OperatorFactory(hs)
    sf = StateFactory(hs)
    return hs, ops, sf


class TestPreparation:
    def test_optical_pumping_ops_list(self, system):
        hs, ops, sf = system
        c_ops = optical_pumping_ops(ops, ion=0, pumping_rate=1e6)
        assert len(c_ops) >= 1

    def test_prepare_qubit_returns_ground(self, system):
        hs, ops, sf = system
        rho0 = sf.thermal_state(n_bar=[0.0], qubit_states=[1, 0])
        rho_prepared = prepare_qubit(ops, ion=0, initial_state=rho0, pumping_rate=1e7, duration=10e-6)
        rho_qubit = rho_prepared.ptrace(0)
        p_ground = rho_qubit[0, 0].real
        assert p_ground > 0.95


class TestMeasurement:
    def test_fluorescence_ground_state_bright(self, system):
        """Ground state |0> should be 'bright' (scatters photons)."""
        hs, ops, sf = system
        psi = sf.ground_state()
        probs = fluorescence_probabilities(psi, ions=[0, 1])
        assert probs[0] > 0.99  # ion 0 in |0> = bright
        assert probs[1] > 0.99  # ion 1 in |0> = bright

    def test_fluorescence_excited_dark(self, system):
        """Excited state |1> should be 'dark'."""
        hs, ops, sf = system
        psi = sf.product_state([1, 0], [0])
        probs = fluorescence_probabilities(psi, ions=[0, 1])
        assert probs[0] < 0.01  # ion 0 in |1> = dark
        assert probs[1] > 0.99  # ion 1 in |0> = bright

    def test_sample_measurement_returns_bits(self, system):
        hs, ops, sf = system
        psi = sf.ground_state()
        bits = sample_measurement(psi, ions=[0, 1], rng=np.random.default_rng(42))
        assert len(bits) == 2
        assert all(b in [0, 1] for b in bits)

    def test_measurement_fidelity_perfect(self):
        fid = measurement_fidelity(
            bright_photon_rate=1e7,
            dark_photon_rate=100,
            detection_window=300e-6,
            collection_efficiency=0.03,
        )
        assert fid > 0.99

    def test_mid_circuit_projects(self, system):
        hs, ops, sf = system
        plus = (sf.product_state([0, 0], [0]) + sf.product_state([1, 0], [0])).unit()
        rho_out, outcome = mid_circuit_measurement(
            qutip.ket2dm(plus), ops, ion=0, rng=np.random.default_rng(42),
        )
        # After measurement, ion 0 should be in a definite state
        rho_q = rho_out.ptrace(0)
        eigenvalues = rho_q.eigenenergies()
        assert max(eigenvalues) > 0.99

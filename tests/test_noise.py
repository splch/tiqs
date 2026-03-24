import numpy as np
import pytest
import qutip

from tiqs.noise.motional import motional_heating_ops, motional_dephasing_op, heating_rate_from_noise
from tiqs.noise.qubit import qubit_dephasing_op, spontaneous_emission_op
from tiqs.noise.photon_scattering import rayleigh_scattering_op, raman_scattering_op
from tiqs.noise.laser_noise import laser_phase_noise_op, laser_intensity_noise_op
from tiqs.noise.crosstalk import crosstalk_hamiltonian
from tiqs.hilbert_space.builder import HilbertSpace
from tiqs.hilbert_space.operators import OperatorFactory
from tiqs.hilbert_space.states import StateFactory
from tiqs.constants import TWO_PI


@pytest.fixture
def system():
    hs = HilbertSpace(n_ions=2, n_modes=1, n_fock=20)
    ops = OperatorFactory(hs)
    sf = StateFactory(hs)
    return hs, ops, sf


class TestMotionalNoise:
    def test_heating_ops_are_list(self, system):
        hs, ops, sf = system
        c_ops = motional_heating_ops(ops, mode=0, heating_rate=10.0, n_bar_env=0.0)
        assert isinstance(c_ops, list)
        assert len(c_ops) >= 1

    def test_heating_increases_phonon_number(self, system):
        """Motional heating should increase <n> over time."""
        hs, ops, sf = system
        psi0 = sf.ground_state()
        n_op = ops.number(0)
        c_ops = motional_heating_ops(ops, mode=0, heating_rate=1e4, n_bar_env=0.0)
        tlist = np.linspace(0, 1e-3, 50)
        result = qutip.mesolve(0 * ops.identity(), psi0, tlist, c_ops=c_ops, e_ops=[n_op])
        assert result.expect[0][-1] > result.expect[0][0]

    def test_motional_dephasing_op(self, system):
        hs, ops, sf = system
        c_op = motional_dephasing_op(ops, mode=0, rate=1e3)
        assert c_op.shape == (ops.identity().shape)

    def test_heating_rate_from_noise_d4_scaling(self):
        """Heating rate should scale as d^-4 with ion-electrode distance."""
        rate_100um = heating_rate_from_noise(
            spectral_density=1e-11, distance=100e-6, frequency=1e6,
        )
        rate_50um = heating_rate_from_noise(
            spectral_density=1e-11, distance=50e-6, frequency=1e6,
        )
        ratio = rate_50um / rate_100um
        assert ratio == pytest.approx(16.0, rel=0.5)  # (100/50)^4 = 16


class TestQubitNoise:
    def test_dephasing_reduces_coherence(self, system):
        """Dephasing should reduce off-diagonal elements of qubit density matrix."""
        hs, ops, sf = system
        plus = (sf.product_state([0, 0], [0]) + sf.product_state([1, 0], [0])).unit()
        c_ops = [qubit_dephasing_op(ops, ion=0, t2=1e-4)]
        tlist = np.linspace(0, 5e-4, 50)
        result = qutip.mesolve(0 * ops.identity(), plus, tlist, c_ops=c_ops)
        rho_init = result.states[0].ptrace(0)
        rho_final = result.states[-1].ptrace(0)
        assert abs(rho_final[0, 1]) < abs(rho_init[0, 1])

    def test_spontaneous_emission_decays_excited(self, system):
        """Spontaneous emission should decay |1> to |0>."""
        hs, ops, sf = system
        psi0 = sf.product_state([1, 0], [0])
        c_ops = [spontaneous_emission_op(ops, ion=0, t1=1e-4)]
        tlist = np.linspace(0, 5e-4, 50)
        sz = ops.sigma_z(0)
        result = qutip.mesolve(0 * ops.identity(), psi0, tlist, c_ops=c_ops, e_ops=[sz])
        # Convention: |0> = ground = basis(2,0), sigma_z|0> = +|0>, sigma_z|1> = -|1>
        # We start in |1> -> <sz> = -1, should decay toward |0> -> <sz> = +1
        assert result.expect[0][-1] > result.expect[0][0]


class TestPhotonScattering:
    def test_rayleigh_is_dephasing_type(self, system):
        hs, ops, sf = system
        c_op = rayleigh_scattering_op(ops, ion=0, rate=1e3)
        # Rayleigh scattering -> proportional to sigma_z (dephasing)
        assert c_op.isherm

    def test_raman_scattering_causes_decay(self, system):
        hs, ops, sf = system
        c_op = raman_scattering_op(ops, ion=0, rate=1e3)
        assert not c_op.isherm  # sigma_plus is not Hermitian


class TestLaserNoise:
    def test_phase_noise_op(self, system):
        hs, ops, sf = system
        c_op = laser_phase_noise_op(ops, ion=0, rate=1e3)
        assert c_op.shape == ops.identity().shape

    def test_intensity_noise_as_hamiltonian(self, system):
        hs, ops, sf = system
        # Intensity noise can be modeled as fluctuation in Rabi frequency
        H_noise = laser_intensity_noise_op(ops, ion=0, fractional_rms=0.01, rabi_frequency=1e6)
        assert H_noise.isherm


class TestCrosstalk:
    def test_crosstalk_hamiltonian_shape(self, system):
        hs, ops, sf = system
        H_xt = crosstalk_hamiltonian(ops, target_ion=0, neighbor_ion=1, crosstalk_fraction=0.01,
                                     rabi_frequency=1e6)
        assert H_xt.shape == ops.identity().shape

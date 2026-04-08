# tests/test_interaction.py
import numpy as np
import pytest
import qutip

from tiqs.constants import TWO_PI
from tiqs.hilbert_space.builder import HilbertSpace
from tiqs.hilbert_space.operators import OperatorFactory
from tiqs.interaction.hamiltonian import (
    blue_sideband_hamiltonian,
    carrier_hamiltonian,
    full_interaction_hamiltonian,
    red_sideband_hamiltonian,
)
from tiqs.interaction.laser import LaserBeam
from tiqs.interaction.raman import RamanPair


@pytest.fixture
def simple_system():
    """One ion, one motional mode, Fock cutoff 15."""
    hs = HilbertSpace(n_ions=1, n_modes=1, n_fock=15)
    ops = OperatorFactory(hs)
    return hs, ops


class TestLaserBeam:
    def test_create_laser(self):
        laser = LaserBeam(
            wavelength=729e-9,
            rabi_frequency=TWO_PI * 100e3,
            detuning=0.0,
            phase=0.0,
        )
        assert laser.wavevector == pytest.approx(TWO_PI / 729e-9)

    def test_laser_rabi_frequency(self):
        laser = LaserBeam(wavelength=729e-9, rabi_frequency=TWO_PI * 50e3)
        assert laser.rabi_frequency == pytest.approx(TWO_PI * 50e3)


class TestCarrierHamiltonian:
    def test_carrier_is_hermitian(self, simple_system):
        hs, ops = simple_system
        H = carrier_hamiltonian(ops, ion=0, rabi_frequency=1.0, phase=0.0)
        assert H.isherm

    def test_carrier_drives_rabi_oscillations(self, simple_system):
        """A carrier pi-pulse should flip |0> -> |1>."""
        hs, ops = simple_system
        Omega = TWO_PI * 100e3
        H = carrier_hamiltonian(ops, ion=0, rabi_frequency=Omega, phase=0.0)
        psi0 = qutip.tensor(qutip.basis(2, 0), qutip.basis(15, 0))
        t_pi = np.pi / Omega
        result = qutip.sesolve(H, psi0, [0, t_pi])
        final = result.states[-1]
        p_excited = (
            abs(
                final.overlap(
                    qutip.tensor(qutip.basis(2, 1), qutip.basis(15, 0))
                )
            )
            ** 2
        )
        assert p_excited == pytest.approx(1.0, abs=0.01)


class TestSidebandHamiltonians:
    def test_red_sideband_removes_phonon(self, simple_system):
        """RSB on |0, n=1> should drive to |1, n=0>."""
        hs, ops = simple_system
        eta = 0.1
        Omega = TWO_PI * 100e3
        H = red_sideband_hamiltonian(
            ops, ion=0, mode=0, rabi_frequency=Omega, eta=eta, phase=0.0
        )
        psi0 = qutip.tensor(qutip.basis(2, 0), qutip.basis(15, 1))
        rsb_rabi = eta * Omega * np.sqrt(1)
        t_pi = np.pi / rsb_rabi
        result = qutip.sesolve(H, psi0, [0, t_pi])
        final = result.states[-1]
        target = qutip.tensor(qutip.basis(2, 1), qutip.basis(15, 0))
        fid = abs(final.overlap(target)) ** 2
        assert fid == pytest.approx(1.0, abs=0.05)

    def test_blue_sideband_adds_phonon(self, simple_system):
        """BSB on |0, n=0> should drive to |1, n=1>."""
        hs, ops = simple_system
        eta = 0.1
        Omega = TWO_PI * 100e3
        H = blue_sideband_hamiltonian(
            ops, ion=0, mode=0, rabi_frequency=Omega, eta=eta, phase=0.0
        )
        psi0 = qutip.tensor(qutip.basis(2, 0), qutip.basis(15, 0))
        bsb_rabi = eta * Omega * np.sqrt(1)
        t_pi = np.pi / bsb_rabi
        result = qutip.sesolve(H, psi0, [0, t_pi])
        final = result.states[-1]
        target = qutip.tensor(qutip.basis(2, 1), qutip.basis(15, 1))
        fid = abs(final.overlap(target)) ** 2
        assert fid == pytest.approx(1.0, abs=0.05)

    def test_rsb_hermitian(self, simple_system):
        hs, ops = simple_system
        H = red_sideband_hamiltonian(
            ops, ion=0, mode=0, rabi_frequency=1.0, eta=0.1
        )
        assert H.isherm

    def test_bsb_hermitian(self, simple_system):
        hs, ops = simple_system
        H = blue_sideband_hamiltonian(
            ops, ion=0, mode=0, rabi_frequency=1.0, eta=0.1
        )
        assert H.isherm


class TestFullInteractionHamiltonian:
    def test_returns_list_format(self, simple_system):
        hs, ops = simple_system
        H = full_interaction_hamiltonian(
            ops,
            ion=0,
            mode=0,
            rabi_frequency=1.0,
            eta=0.1,
            detuning=0.0,
            mode_frequency=TWO_PI * 1e6,
            phase=0.0,
        )
        assert isinstance(H, list)
        assert len(H) >= 1

    def test_resonant_drive_matches_carrier(self, simple_system):
        """On resonance with detuning=0, the full Hamiltonian should
        behave as a carrier.

        A carrier pi-pulse flips |0> -> |1>. In QuTiP's convention,
        |0> has <sigma_z> = +1 and |1> has <sigma_z> = -1, so after
        the pi-pulse sigma_z goes from +1 to -1.
        """
        hs, ops = simple_system
        Omega = TWO_PI * 100e3
        omega_mode = TWO_PI * 1e6
        H = full_interaction_hamiltonian(
            ops,
            ion=0,
            mode=0,
            rabi_frequency=Omega,
            eta=0.1,
            detuning=0.0,
            mode_frequency=omega_mode,
            phase=0.0,
        )
        psi0 = qutip.tensor(qutip.basis(2, 0), qutip.basis(15, 0))
        t_pi = np.pi / Omega
        tlist = np.linspace(0, t_pi, 200)
        result = qutip.sesolve(H, psi0, tlist, options={"max_step": t_pi / 50})
        sz = qutip.tensor(qutip.sigmaz(), qutip.qeye(15))
        final_sz = qutip.expect(sz, result.states[-1])
        assert final_sz == pytest.approx(-1.0, abs=0.15)


class TestRamanPair:
    def test_raman_effective_rabi(self):
        raman = RamanPair(
            omega_1=TWO_PI * 1e14,
            omega_2=TWO_PI * 1e14 - TWO_PI * 12.6e9,
            rabi_1=TWO_PI * 1e9,
            rabi_2=TWO_PI * 1e9,
            detuning_from_excited=TWO_PI * 100e9,
        )
        omega_eff = raman.effective_rabi_frequency
        expected = (TWO_PI * 1e9) ** 2 / (2 * TWO_PI * 100e9)
        assert omega_eff == pytest.approx(expected, rel=0.1)

    def test_raman_scattering_rate(self):
        raman = RamanPair(
            omega_1=TWO_PI * 1e14,
            omega_2=TWO_PI * 1e14,
            rabi_1=TWO_PI * 1e9,
            rabi_2=TWO_PI * 1e9,
            detuning_from_excited=TWO_PI * 100e9,
            excited_state_linewidth=TWO_PI * 20e6,
        )
        rate = raman.scattering_rate
        assert rate > 0

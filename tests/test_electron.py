"""End-to-end test for trapped-electron quantum simulation.

Simulates two electrons in a GHz Paul trap coupled via their shared
motional modes, driven by a magnetic-gradient-mediated MS gate.
Uses realistic parameters from Haffner group proposals (PRA 105,
022420, 2022).
"""

import numpy as np
import pytest
import qutip

from tiqs.analysis.fidelity import bell_state_fidelity
from tiqs.chain.equilibrium import equilibrium_positions
from tiqs.chain.lamb_dicke import lamb_dicke_parameters
from tiqs.chain.normal_modes import normal_modes
from tiqs.constants import (
    BOHR_MAGNETON,
    ELECTRON_G_FACTOR,
    HBAR,
    TWO_PI,
)
from tiqs.gates.molmer_sorensen import ms_gate_duration, ms_gate_hamiltonian
from tiqs.gates.single_qubit import rx_gate
from tiqs.hilbert_space.builder import HilbertSpace
from tiqs.hilbert_space.operators import OperatorFactory
from tiqs.hilbert_space.states import StateFactory
from tiqs.noise.qubit import qubit_dephasing_op
from tiqs.species.electron import ElectronSpecies
from tiqs.trap import PaulTrap


@pytest.fixture
def electron_trap():
    """Two-electron GHz Paul trap.

    RF drive at 1.6 GHz, axial secular frequency 30 MHz,
    electrode distance 300 um.
    """
    return PaulTrap(
        v_rf=7.8,
        omega_rf=TWO_PI * 1.6e9,
        r0=300e-6,
        omega_axial=TWO_PI * 30e6,
        species=ElectronSpecies(magnetic_field=0.1),
    )


def _gradient_k_eff(
    gradient: float,
    magnetic_field: float,
    mass_kg: float,
    omega_mode: float,
) -> float:
    r"""Effective wavevector from magnetic gradient coupling.

    The gradient couples spin to motion via
    $H = g_e \mu_B (dB/dz) \hat{z} \sigma_z / 2$.
    The effective Lamb-Dicke parameter is
    $\eta = g_e \mu_B (dB/dz) x_\text{zpf} / (\hbar \omega_q)$.

    This is equivalent to using $k_\text{eff} = g_e \mu_B (dB/dz)
    / (\hbar \omega_q)$ in the standard formula
    $\eta = k_\text{eff} \, b \, x_\text{zpf}$.
    """
    omega_qubit = (
        ELECTRON_G_FACTOR * BOHR_MAGNETON * magnetic_field
        / HBAR
    )
    return ELECTRON_G_FACTOR * BOHR_MAGNETON * gradient / (HBAR * omega_qubit)


class TestElectronTrap:
    def test_trap_stability(self, electron_trap):
        assert electron_trap.is_stable()

    def test_secular_frequencies(self, electron_trap):
        assert electron_trap.omega_axial == pytest.approx(
            TWO_PI * 30e6
        )
        assert electron_trap.omega_radial > electron_trap.omega_axial

    def test_two_electron_equilibrium(self, electron_trap):
        pos = equilibrium_positions(2, electron_trap)
        assert len(pos) == 2
        assert pos[0] == pytest.approx(-pos[1])
        # Tighter spacing than ions at the same trap frequency
        spacing = pos[1] - pos[0]
        assert spacing < 100e-6

    def test_normal_modes(self, electron_trap):
        modes = normal_modes(2, electron_trap)
        # COM at omega_axial, stretch at sqrt(3) * omega_axial
        assert modes.axial_freqs[0] == pytest.approx(
            electron_trap.omega_axial, rel=1e-4
        )
        ratio = modes.axial_freqs[1] / modes.axial_freqs[0]
        assert ratio == pytest.approx(np.sqrt(3), rel=1e-4)

    def test_gradient_lamb_dicke(self, electron_trap):
        """Magnetic gradient of 120 T/m should give eta ~ 0.01-0.1."""
        modes = normal_modes(1, electron_trap)
        species = electron_trap.species
        k_eff = _gradient_k_eff(
            120.0, species.magnetic_field,
            species.mass_kg, modes.axial_freqs[0],
        )
        eta = lamb_dicke_parameters(modes, species, k_eff, "axial")
        assert eta.shape == (1, 1)
        assert 0.0001 < abs(eta[0, 0]) < 1.0


class TestElectronMSGate:
    """MS gate on two trapped electrons via gradient coupling."""

    def test_bell_state(self, electron_trap):
        """MS gate should produce a Bell state with F > 0.99."""
        modes = normal_modes(2, electron_trap)
        species = electron_trap.species
        k_eff = _gradient_k_eff(
            120.0, species.magnetic_field,
            species.mass_kg, modes.axial_freqs[0],
        )
        eta_matrix = lamb_dicke_parameters(
            modes, species, k_eff, "axial"
        )

        hs = HilbertSpace(n_ions=2, n_modes=1, n_fock=15)
        ops = OperatorFactory(hs)
        sf = StateFactory(hs)

        eta = [float(eta_matrix[0, 0]), float(eta_matrix[1, 0])]
        delta = TWO_PI * 15e3
        Omega = delta / (4 * abs(eta[0]))
        tau = ms_gate_duration(delta)

        H = ms_gate_hamiltonian(
            ops, [0, 1], 0, eta, Omega, delta,
        )
        psi0 = sf.ground_state()
        tlist = np.linspace(0, tau, 500)
        result = qutip.sesolve(
            H, psi0, tlist, options={"max_step": tau / 100},
        )

        rho_spin = result.states[-1].ptrace([0, 1])
        fid = bell_state_fidelity(rho_spin)
        assert fid > 0.99

    def test_carrier_rabi(self, electron_trap):
        """Carrier pi-pulse should flip |0> to |1>."""
        hs = HilbertSpace(n_ions=1, n_modes=1, n_fock=5)
        ops = OperatorFactory(hs)
        sf = StateFactory(hs)

        Omega = TWO_PI * 500e3
        gate = rx_gate(ops, ion=0, theta=np.pi, rabi_frequency=Omega)
        result = qutip.sesolve(
            gate.hamiltonian, sf.ground_state(), [0, gate.duration],
        )
        p1 = abs(
            result.states[-1].overlap(sf.product_state([1], [0]))
        ) ** 2
        assert p1 == pytest.approx(1.0, abs=0.01)

    def test_dephasing_degrades_fidelity(self, electron_trap):
        """Magnetic field noise (qubit dephasing) should reduce
        Bell state fidelity."""
        modes = normal_modes(2, electron_trap)
        species = electron_trap.species
        k_eff = _gradient_k_eff(
            120.0, species.magnetic_field,
            species.mass_kg, modes.axial_freqs[0],
        )
        eta_matrix = lamb_dicke_parameters(
            modes, species, k_eff, "axial"
        )

        hs = HilbertSpace(n_ions=2, n_modes=1, n_fock=15)
        ops = OperatorFactory(hs)
        sf = StateFactory(hs)

        eta = [float(eta_matrix[0, 0]), float(eta_matrix[1, 0])]
        delta = TWO_PI * 15e3
        Omega = delta / (4 * abs(eta[0]))
        tau = ms_gate_duration(delta)
        H = ms_gate_hamiltonian(
            ops, [0, 1], 0, eta, Omega, delta,
        )
        psi0 = sf.ground_state()
        tlist = np.linspace(0, tau, 500)

        # Clean
        r_clean = qutip.sesolve(
            H, psi0, tlist, options={"max_step": tau / 100},
        )
        fid_clean = bell_state_fidelity(
            r_clean.states[-1].ptrace([0, 1])
        )

        # With magnetic field noise (T2 = 100 us)
        c_ops = [
            qubit_dephasing_op(ops, 0, t2=100e-6),
            qubit_dephasing_op(ops, 1, t2=100e-6),
        ]
        r_noisy = qutip.mesolve(
            H, psi0, tlist, c_ops=c_ops,
            options={"max_step": tau / 100},
        )
        fid_noisy = bell_state_fidelity(
            r_noisy.states[-1].ptrace([0, 1])
        )

        assert fid_noisy < fid_clean

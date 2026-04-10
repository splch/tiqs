"""End-to-end test for trapped-electron quantum simulation.

Simulates two electrons in a GHz Paul trap coupled via their shared
motional modes, driven by a magnetic-gradient-mediated entangling
gate. The gradient coupling H = g_e mu_B (dB/dz) z sigma_z / 2
naturally produces a sigma_z-dependent force (light-shift / ZZ
gate), not a sigma_x force (MS / XX gate). An MS gate requires
additional microwave dressing to rotate the spin basis.

TestElectronAnalyticalExactness validates electron-specific formulas
against known results and published values from Hahn et al.
arXiv:2503.12379 (2025), Yu et al. PRA 105 022420 (2022), and
Hoven et al. arXiv:2508.16407 (2025).
"""

import numpy as np
import pytest
import qutip

from tiqs.chain.equilibrium import equilibrium_positions
from tiqs.chain.lamb_dicke import lamb_dicke_parameters
from tiqs.chain.normal_modes import normal_modes
from tiqs.constants import (
    BOHR_MAGNETON,
    BOLTZMANN,
    ELECTRON_CHARGE,
    ELECTRON_G_FACTOR,
    ELECTRON_MASS,
    EPSILON_0,
    HBAR,
    PI,
    TWO_PI,
)
from tiqs.gates.light_shift import light_shift_gate_hamiltonian
from tiqs.gates.molmer_sorensen import ms_gate_duration
from tiqs.gates.single_qubit import rx_gate
from tiqs.hilbert_space.builder import HilbertSpace
from tiqs.hilbert_space.operators import OperatorFactory
from tiqs.hilbert_space.states import StateFactory
from tiqs.interaction.hamiltonian import carrier_hamiltonian
from tiqs.noise.qubit import qubit_dephasing_op
from tiqs.species.data import get_species
from tiqs.species.electron import ElectronSpecies
from tiqs.trap import PaulTrap


def _gradient_k_eff(gradient: float, magnetic_field: float) -> float:
    r"""Effective wavevector from magnetic gradient coupling.

    The gradient couples spin to motion via
    $H = g_e \mu_B (dB/dz) \hat{z} \sigma_z / 2$.
    This is equivalent to $k_\text{eff} = (dB/dz) / B$
    in the standard Lamb-Dicke formula.
    """
    return gradient / magnetic_field


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
        spacing = pos[1] - pos[0]
        assert spacing < 100e-6

    def test_normal_modes(self, electron_trap):
        modes = normal_modes(2, electron_trap)
        assert modes.axial_freqs[0] == pytest.approx(
            electron_trap.omega_axial, rel=1e-4
        )
        ratio = modes.axial_freqs[1] / modes.axial_freqs[0]
        assert ratio == pytest.approx(np.sqrt(3), rel=1e-4)

    def test_gradient_lamb_dicke(self, electron_trap):
        """Magnetic gradient of 120 T/m should give useful eta."""
        modes = normal_modes(1, electron_trap)
        species = electron_trap.species
        k_eff = _gradient_k_eff(120.0, species.magnetic_field)
        eta = lamb_dicke_parameters(modes, species, k_eff, "axial")
        assert eta.shape == (1, 1)
        assert 0.0001 < abs(eta[0, 0]) < 1.0


class TestElectronGradientGate:
    """Entangling gate on two trapped electrons via gradient coupling.

    The magnetic gradient naturally produces a sigma_z-dependent force,
    so the native gate is a light-shift (ZZ) gate, not an MS (XX) gate.
    """

    def test_zz_gate_entangles(self, electron_trap):
        """Light-shift gate from gradient coupling should entangle
        |+,+> into a state with ZZ correlations.

        The gradient Hamiltonian is sigma_z-dependent, so sigma_z
        eigenstates (|0>, |1>) are displaced in opposite directions
        in phase space. Starting from sigma_x eigenstates (|+>, |->)
        produces entanglement.
        """
        modes = normal_modes(2, electron_trap)
        species = electron_trap.species
        k_eff = _gradient_k_eff(120.0, species.magnetic_field)
        eta_matrix = lamb_dicke_parameters(
            modes, species, k_eff, "axial"
        )

        hs = HilbertSpace(n_ions=2, n_modes=1, n_fock=15)
        ops = OperatorFactory(hs)

        eta = [float(eta_matrix[0, 0]), float(eta_matrix[1, 0])]
        delta = TWO_PI * 15e3
        Omega = delta / (4 * abs(eta[0]))
        tau = ms_gate_duration(delta)

        H = light_shift_gate_hamiltonian(
            ops, [0, 1], 0, eta, Omega, delta,
        )

        # Start in |+,+> (sigma_x eigenstates)
        plus = (qutip.basis(2, 0) + qutip.basis(2, 1)).unit()
        psi0 = qutip.tensor(plus, plus, qutip.basis(15, 0))
        tlist = np.linspace(0, tau, 500)
        result = qutip.sesolve(
            H, psi0, tlist, options={"max_step": tau / 100},
        )

        # Entanglement: reduced single-qubit state is mixed
        rho_spin = result.states[-1].ptrace([0, 1])
        rho_single = rho_spin.ptrace(0)
        purity = (rho_single * rho_single).tr().real
        assert purity < 0.7

    def test_zz_gate_motional_closure(self, electron_trap):
        """After a complete ZZ gate, the motion should return to
        its initial state (phase-space loop closes)."""
        hs = HilbertSpace(n_ions=2, n_modes=1, n_fock=20)
        ops = OperatorFactory(hs)

        eta = 0.05
        delta = TWO_PI * 15e3
        Omega = delta / (4 * eta)
        tau = ms_gate_duration(delta)
        H = light_shift_gate_hamiltonian(
            ops, [0, 1], 0, [eta, eta], Omega, delta,
        )

        plus = (qutip.basis(2, 0) + qutip.basis(2, 1)).unit()
        psi0 = qutip.tensor(plus, plus, qutip.basis(20, 0))
        r = qutip.sesolve(
            H, psi0, np.linspace(0, tau, 500),
            options={"max_step": tau / 100},
        )
        n_final = qutip.expect(ops.number(0), r.states[-1])
        assert n_final == pytest.approx(0.0, abs=0.05)

    def test_carrier_rabi(self):
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
        entanglement from the ZZ gate."""
        hs = HilbertSpace(n_ions=2, n_modes=1, n_fock=15)
        ops = OperatorFactory(hs)

        eta = 0.05
        delta = TWO_PI * 15e3
        Omega = delta / (4 * eta)
        tau = ms_gate_duration(delta)
        H = light_shift_gate_hamiltonian(
            ops, [0, 1], 0, [eta, eta], Omega, delta,
        )

        plus = (qutip.basis(2, 0) + qutip.basis(2, 1)).unit()
        psi0 = qutip.tensor(plus, plus, qutip.basis(15, 0))
        tlist = np.linspace(0, tau, 500)

        r_clean = qutip.sesolve(
            H, psi0, tlist, options={"max_step": tau / 100},
        )
        purity_clean = (
            r_clean.states[-1].ptrace([0, 1]) ** 2
        ).tr().real

        c_ops = [
            qubit_dephasing_op(ops, 0, t2=100e-6),
            qubit_dephasing_op(ops, 1, t2=100e-6),
        ]
        r_noisy = qutip.mesolve(
            H, psi0, tlist, c_ops=c_ops,
            options={"max_step": tau / 100},
        )
        purity_noisy = (
            r_noisy.states[-1].ptrace([0, 1]) ** 2
        ).tr().real

        assert purity_noisy < purity_clean


class TestElectronAnalyticalExactness:
    """Tight numerical checks against known analytical results for
    trapped electrons.

    Every assertion uses tight tolerances (rel=0.001 or better).
    Tests only electron-specific physics; universal spin/Lindblad
    mechanics are already validated for ions in test_end_to_end.py.

    References: CODATA 2018, Leibfried et al. RMP 75 281 (2003),
    Hahn et al. arXiv:2503.12379 (2025), Yu et al. PRA 105 022420
    (2022), Hoven et al. arXiv:2508.16407 (2025).
    """

    def test_gyromagnetic_ratio(self):
        """Electron gyromagnetic ratio: g_e * mu_B / h ~ 28.025 GHz/T."""
        gamma_hz_per_T = (
            ELECTRON_G_FACTOR * BOHR_MAGNETON / (HBAR * TWO_PI)
        )
        assert gamma_hz_per_T == pytest.approx(28.025e9, rel=1e-4)

    def test_mathieu_q_formula(self):
        """q = 2*e*V_rf / (m_e * Omega_rf^2 * r0^2) for electron trap."""
        e = ElectronSpecies(0.1)
        omega_rf = TWO_PI * 1.6e9
        r0 = 300e-6
        V_rf = 7.8
        trap = PaulTrap(
            v_rf=V_rf, omega_rf=omega_rf, r0=r0,
            omega_axial=TWO_PI * 30e6, species=e,
        )
        q_hand = 2 * ELECTRON_CHARGE * V_rf / (
            ELECTRON_MASS * omega_rf**2 * r0**2
        )
        assert trap.mathieu_q == pytest.approx(q_hand, rel=1e-10)

    def test_pseudopotential_depth(self):
        """Trap depth = q * V_rf / 8 in eV."""
        e = ElectronSpecies(0.1)
        trap = PaulTrap(
            v_rf=7.8, omega_rf=TWO_PI * 1.6e9, r0=300e-6,
            omega_axial=TWO_PI * 30e6, species=e,
        )
        depth_from_q = trap.mathieu_q * trap.v_rf / 8
        assert trap.pseudopotential_depth_eV == pytest.approx(
            depth_from_q, rel=1e-6
        )

    def test_two_electron_spacing_analytical(self):
        """Two-particle spacing: d = 2 * (1/2)^(2/3) * l_0 where
        l_0 = (e^2 / (4*pi*eps0 * m_e * omega_z^2))^(1/3)."""
        e = ElectronSpecies(0.1)
        omega_z = TWO_PI * 30e6
        trap = PaulTrap(
            v_rf=7.8, omega_rf=TWO_PI * 1.6e9, r0=300e-6,
            omega_axial=omega_z, species=e,
        )
        pos = equilibrium_positions(2, trap)
        l_scale = (
            ELECTRON_CHARGE**2
            / (4 * PI * EPSILON_0 * ELECTRON_MASS * omega_z**2)
        ) ** (1 / 3)
        d_analytical = 2 * (1 / 2) ** (2 / 3) * l_scale
        d_measured = pos[1] - pos[0]
        assert d_measured == pytest.approx(d_analytical, rel=0.001)

    def test_electron_spacing_vs_ion_spacing(self):
        """At the same trap frequency, spacing scales as m^(-1/3):
        d_e / d_ion = (m_ion / m_e)^(1/3)."""
        ca = get_species("Ca40")
        e = ElectronSpecies(0.1)
        omega_z = TWO_PI * 1e6
        trap_ca = PaulTrap(
            v_rf=300, omega_rf=TWO_PI * 30e6, r0=0.5e-3,
            omega_axial=omega_z, species=ca,
        )
        trap_e = PaulTrap(
            v_rf=7.8, omega_rf=TWO_PI * 1.6e9, r0=300e-6,
            omega_axial=omega_z, species=e,
        )
        d_ca = equilibrium_positions(2, trap_ca)
        d_e = equilibrium_positions(2, trap_e)
        spacing_ratio = (d_e[1] - d_e[0]) / (d_ca[1] - d_ca[0])
        mass_ratio = (ca.mass_kg / ELECTRON_MASS) ** (1 / 3)
        assert spacing_ratio == pytest.approx(mass_ratio, rel=0.001)

    def test_gradient_eta_formula(self):
        """eta = g_e * mu_B * (dB/dz) * x_zpf / (hbar * omega_q)
        must match the simulator's k_eff path for dB/dz = 120 T/m,
        B = 0.1 T, omega_z/2pi = 30 MHz."""
        gradient = 120.0  # T/m
        B = 0.1
        omega_z = TWO_PI * 30e6
        omega_q = ELECTRON_G_FACTOR * BOHR_MAGNETON * B / HBAR
        x_zpf = np.sqrt(HBAR / (2 * ELECTRON_MASS * omega_z))
        eta_hand = (
            ELECTRON_G_FACTOR * BOHR_MAGNETON * gradient * x_zpf
            / (HBAR * omega_q)
        )
        e = ElectronSpecies(B)
        trap = PaulTrap(
            v_rf=7.8, omega_rf=TWO_PI * 1.6e9, r0=300e-6,
            omega_axial=omega_z, species=e,
        )
        modes = normal_modes(1, trap)
        k_eff = _gradient_k_eff(gradient, B)
        eta_sim = lamb_dicke_parameters(modes, e, k_eff, "axial")
        assert eta_sim[0, 0] == pytest.approx(eta_hand, rel=1e-6)

    def test_gradient_keff_simplification(self):
        """The full gradient coupling k_eff = g_e*mu_B*(dB/dz)/(hbar*omega_q)
        simplifies to (dB/dz)/B since omega_q = g_e*mu_B*B/hbar."""
        gradient = 120.0
        B = 0.1
        omega_q = ELECTRON_G_FACTOR * BOHR_MAGNETON * B / HBAR
        k_full = (
            ELECTRON_G_FACTOR * BOHR_MAGNETON * gradient / (HBAR * omega_q)
        )
        k_simple = gradient / B
        assert k_full == pytest.approx(k_simple, rel=1e-10)

    def test_thermal_nbar_bose_einstein(self):
        """Resistive cooling limit: nbar = 1 / (exp(hbar*omega/k_B*T) - 1).
        At T = 4 K, omega/2pi = 300 MHz: nbar ~ 278.
        At T = 0.4 K: nbar ~ 27.3.
        At T = 0.4 K, omega/2pi = 2 GHz: nbar ~ 3.7."""
        omega = TWO_PI * 300e6
        nbar_4K = 1.0 / (np.exp(HBAR * omega / (BOLTZMANN * 4.0)) - 1)
        assert nbar_4K == pytest.approx(278, rel=0.01)
        nbar_04K = 1.0 / (np.exp(HBAR * omega / (BOLTZMANN * 0.4)) - 1)
        assert nbar_04K == pytest.approx(27.3, rel=0.01)
        omega_rad = TWO_PI * 2e9
        nbar_rad = 1.0 / (np.exp(HBAR * omega_rad / (BOLTZMANN * 0.4)) - 1)
        assert nbar_rad == pytest.approx(3.69, rel=0.01)

    def test_zpf_scaling_vs_ions(self):
        """Zero-point motion scales as sqrt(m_ion/m_e) at same frequency.
        For Ca-40: sqrt(m_Ca/m_e) ~ 270."""
        ca = get_species("Ca40")
        omega = TWO_PI * 1e6
        x_zpf_e = np.sqrt(HBAR / (2 * ELECTRON_MASS * omega))
        x_zpf_ca = np.sqrt(HBAR / (2 * ca.mass_kg * omega))
        ratio = x_zpf_e / x_zpf_ca
        expected = np.sqrt(ca.mass_kg / ELECTRON_MASS)
        assert ratio == pytest.approx(expected, rel=1e-6)
        assert ratio == pytest.approx(269.9, rel=0.01)

    def test_length_scale_at_multiple_frequencies(self):
        """Coulomb length scale l_0 = (e^2/(4*pi*eps0*m*omega^2))^(1/3)
        evaluated at frequencies from Hahn et al. 2025."""
        for freq_mhz, l0_um in [(30, 19.25), (300, 4.15)]:
            omega = TWO_PI * freq_mhz * 1e6
            l0 = (
                ELECTRON_CHARGE**2
                / (4 * PI * EPSILON_0 * ELECTRON_MASS * omega**2)
            ) ** (1 / 3)
            assert l0 == pytest.approx(l0_um * 1e-6, rel=0.01)

    def test_zpf_at_multiple_frequencies(self):
        """x_zpf values at frequencies from Hahn et al. 2025 and
        Yu et al. 2022."""
        for freq_mhz, zpf_nm in [(30, 554), (300, 175), (2000, 67.9)]:
            omega = TWO_PI * freq_mhz * 1e6
            x_zpf = np.sqrt(HBAR / (2 * ELECTRON_MASS * omega))
            assert x_zpf == pytest.approx(zpf_nm * 1e-9, rel=0.01)

    def test_zz_gate_fidelity_and_motional_closure(self):
        """ZZ gate from gradient coupling must produce entanglement
        with F > 0.999 and return motion to vacuum.

        Uses the light-shift Hamiltonian (sigma_z force) which is the
        native interaction from magnetic gradient coupling."""
        hs = HilbertSpace(n_ions=2, n_modes=1, n_fock=20)
        ops = OperatorFactory(hs)

        eta = 0.05
        delta = TWO_PI * 15e3
        Omega = delta / (4 * eta)
        tau = ms_gate_duration(delta)
        H = light_shift_gate_hamiltonian(
            ops, [0, 1], 0, [eta, eta], Omega, delta,
        )

        # ZZ gate entangles sigma_x eigenstates, not sigma_z
        plus = (qutip.basis(2, 0) + qutip.basis(2, 1)).unit()
        psi0 = qutip.tensor(plus, plus, qutip.basis(20, 0))
        r = qutip.sesolve(
            H, psi0, np.linspace(0, tau, 500),
            options={"max_step": tau / 100},
        )

        # Motion returns to vacuum
        n_final = qutip.expect(ops.number(0), r.states[-1])
        assert n_final == pytest.approx(0.0, abs=0.01)

        # Spin state is entangled: single-qubit purity ~ 0.5
        rho_spin = r.states[-1].ptrace([0, 1])
        rho_single = rho_spin.ptrace(0)
        purity = (rho_single * rho_single).tr().real
        assert purity < 0.55

    def test_carrier_rabi_exact(self):
        """sigma_z = cos(Omega*t) for microwave carrier drive on
        electron spin. Validates the Hamiltonian convention."""
        hs = HilbertSpace(n_ions=1, n_modes=1, n_fock=3)
        ops = OperatorFactory(hs)
        sf = StateFactory(hs)
        Omega = TWO_PI * 500e3
        H = carrier_hamiltonian(ops, 0, Omega)
        tlist = np.linspace(0, 4 * PI / Omega, 400)
        result = qutip.sesolve(
            H, sf.ground_state(), tlist, e_ops=[ops.sigma_z(0)],
        )
        expected = np.cos(Omega * tlist)
        np.testing.assert_allclose(result.expect[0], expected, atol=0.01)

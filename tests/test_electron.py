"""End-to-end test for trapped-electron quantum simulation.

Simulates two electrons in a GHz Paul trap coupled via their shared
motional modes, driven by a magnetic-gradient-mediated entangling
gate. The gradient coupling H = g_e mu_B (dB/dz) z sigma_z / 2
naturally produces a sigma_z-dependent force (light-shift / ZZ
gate), not a sigma_x force (MS / XX gate). An MS gate requires
additional microwave dressing to rotate the spin basis.

The gradient Lamb-Dicke parameter comes from
`tiqs.chain.lamb_dicke.gradient_lamb_dicke_parameters`, whose per-mode
wavevector k_eff = g mu_B (dB/dz) / (hbar omega_m) is the position
gradient of the qubit frequency divided by the MODE frequency (Mintert
and Wunderlich, PRL 87, 257904 (2001)). The bias field B_0 does not
appear in the coupling Hamiltonian and therefore cannot appear in eta.

TestElectronAnalyticalExactness validates electron-specific physics
against published values from Huang et al. arXiv:2503.12379 (2025),
Yu et al. PRA 105 022420 (2022), Mikhailovskii et al.
arXiv:2508.16407 (2025) and Weidt et al. PRL 117 220501 (2016).
"""

import numpy as np
import pytest
import qutip

from tiqs.chain.equilibrium import equilibrium_positions
from tiqs.chain.lamb_dicke import (
    gradient_lamb_dicke_parameters,
    lamb_dicke_parameters,
)
from tiqs.chain.normal_modes import normal_modes
from tiqs.constants import (
    BOHR_MAGNETON,
    COULOMB_CONSTANT,
    ELECTRON_G_FACTOR,
    ELECTRON_MASS,
    HBAR,
    TWO_PI,
)
from tiqs.gates.light_shift import light_shift_gate_hamiltonian
from tiqs.gates.molmer_sorensen import ms_gate_duration
from tiqs.hilbert_space.builder import HilbertSpace
from tiqs.hilbert_space.operators import OperatorFactory
from tiqs.hilbert_space.states import StateFactory
from tiqs.interaction.hamiltonian import carrier_hamiltonian
from tiqs.noise.qubit import qubit_dephasing_op
from tiqs.species.electron import ElectronSpecies
from tiqs.species.ion import get_species
from tiqs.trap import PaulTrap

GRADIENT = 120.0
"""Magnetic field gradient in T/m used by the gradient gate tests."""


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
        assert electron_trap.omega_axial == pytest.approx(TWO_PI * 30e6)
        assert electron_trap.omega_radial > electron_trap.omega_axial

    def test_two_electron_equilibrium(self, electron_trap):
        pos = equilibrium_positions(2, electron_trap)
        assert len(pos) == 2
        assert pos[0] == pytest.approx(-pos[1])
        spacing = pos[1] - pos[0]
        assert spacing < 100e-6

    def test_normal_modes(self, electron_trap):
        modes = normal_modes(2, electron_trap)
        axial = modes.modes["axial"]
        assert axial.freqs[0] == pytest.approx(
            electron_trap.omega_axial, rel=1e-4
        )
        ratio = axial.freqs[1] / axial.freqs[0]
        assert ratio == pytest.approx(np.sqrt(3), rel=1e-4)

    def test_gradient_lamb_dicke(self, electron_trap):
        """A 120 T/m gradient gives eta ~ 0.062 on a 30 MHz mode.

        eta is the frequency-modulation index of the qubit: the Zeeman
        frequency swings by g_e mu_B (dB/dz) x_zpf / hbar as the
        electron moves over its zero-point spread, measured in units
        of the mode frequency. Values of order 0.01-0.1 are what makes
        a MAGIC gate practical; the (dB/dz)/B_0 parametrisation this
        file used previously returns 6.6e-4, 93x too small.
        """
        modes = normal_modes(1, electron_trap)
        species = electron_trap.species
        eta = gradient_lamb_dicke_parameters(modes, species, GRADIENT, "axial")
        assert eta.shape == (1, 1)
        assert eta[0, 0] == pytest.approx(0.0621, rel=0.01)


class TestGradientCoupling:
    """The magnetic-gradient Lamb-Dicke parameter, anchored two ways.

    Both anchors are independent of the implementation: one is the
    exact dynamics of the spin-motion Hamiltonian written directly in
    SI units, the other a published experimental value.
    """

    def test_eta_equals_max_displacement_of_spin_dependent_force(
        self, electron_trap
    ):
        r"""eta is fixed by the dynamics of H = (g mu_B/2)(dB/dz) z sz.

        For H/hbar = omega a^dag a + (eta omega / 2)(a + a^dag) sigma_z
        the branch with spin eigenvalue s is a displaced oscillator
        whose ground state sits at alpha = -s eta / 2. Vacuum therefore
        circles that centre at radius eta/2 and reaches
        |alpha| = eta after half a trap period, so

            <n>(pi / omega) = eta^2   exactly.

        The Hamiltonian here is built from raw QuTiP operators and SI
        constants using the coupling of docs/theory/species.md,
        H_int = (g_e mu_B / 2) (dB/dz) z sigma_z with z = b x_zpf
        (a + a^dag), so nothing in the reference depends on
        `gradient_lamb_dicke_parameters` except the eta being tested.
        The old (dB/dz)/B_0 convention fails this by a factor 8724.
        """
        modes = normal_modes(1, electron_trap)
        species = electron_trap.species
        eta = float(
            gradient_lamb_dicke_parameters(modes, species, GRADIENT, "axial")[
                0, 0
            ]
        )
        omega = modes.modes["axial"].freqs[0]
        x_zpf = np.sqrt(HBAR / (2 * ELECTRON_MASS * omega))

        n_fock = 12
        a = qutip.destroy(n_fock)
        number = qutip.tensor(qutip.qeye(2), a.dag() * a)
        coupling = ELECTRON_G_FACTOR * BOHR_MAGNETON * GRADIENT / (2 * HBAR)
        H = omega * number + coupling * qutip.tensor(
            qutip.sigmaz(), x_zpf * (a + a.dag())
        )

        psi0 = qutip.tensor(qutip.basis(2, 0), qutip.basis(n_fock, 0))
        result = qutip.sesolve(
            H,
            psi0,
            [0.0, np.pi / omega],
            e_ops=[number],
            options={"atol": 1e-14, "rtol": 1e-12},
        )
        assert result.expect[0][-1] == pytest.approx(eta**2, rel=1e-6)

    def test_eta_is_independent_of_the_bias_field(self):
        """eta must not depend on B_0 at fixed gradient and frequency.

        The spin-motion coupling (g_e mu_B / 2)(dB/dz) z sigma_z
        contains no reference to the bias field, so changing B_0 while
        holding dB/dz and the mode frequency fixed cannot change eta.
        The (dB/dz)/B_0 parametrisation scales as 1/B_0 and fails this
        by the field ratio.
        """
        etas = []
        for magnetic_field in (0.01, 0.1, 1.0):
            trap = PaulTrap(
                v_rf=7.8,
                omega_rf=TWO_PI * 1.6e9,
                r0=300e-6,
                omega_axial=TWO_PI * 30e6,
                species=ElectronSpecies(magnetic_field=magnetic_field),
            )
            modes = normal_modes(1, trap)
            etas.append(
                float(
                    gradient_lamb_dicke_parameters(
                        modes, trap.species, GRADIENT, "axial"
                    )[0, 0]
                )
            )
        assert etas[1] == pytest.approx(etas[0], rel=1e-12)
        assert etas[2] == pytest.approx(etas[0], rel=1e-12)

    def test_weidt_magic_gate_eta(self):
        """Weidt et al., PRL 117, 220501 (2016), p. 3.

        Two Yb-171 ions, dB/dz = 23.6 T/m, stretch mode
        nu_s = sqrt(3) nu_z = 2pi x 459.34 kHz: they quote
        z_0 = sqrt(hbar / 2 m nu_s) = 8.021 nm and
        eta_eff = z_0 mu_B (dB/dz) / (sqrt(2) hbar nu_s) = 0.0041.
        Their differential magnetic moment is one Bohr magneton
        (microwave-dressed hyperfine states, not a free electron), so
        the comparison uses ``g_factor=1``; the 1/sqrt(2) is the
        stretch-mode participation b = 1/sqrt(2), which the helper
        supplies from the eigenvector.
        """
        yb = get_species("Yb171")
        nu_s = TWO_PI * 459.34e3
        trap = PaulTrap(
            v_rf=300.0,
            omega_rf=TWO_PI * 30e6,
            r0=0.5e-3,
            omega_axial=nu_s / np.sqrt(3),
            species=yb,
        )
        modes = normal_modes(2, trap)
        assert modes.modes["axial"].freqs[1] == pytest.approx(nu_s, rel=1e-6)
        eta = gradient_lamb_dicke_parameters(
            modes, yb, 23.6, "axial", g_factor=1.0
        )
        assert abs(eta[0, 1]) == pytest.approx(0.0041, rel=0.01)


class TestElectronGradientGate:
    """Entangling gate on two trapped electrons via gradient coupling.

    The magnetic gradient naturally produces a sigma_z-dependent force,
    so the native gate is a light-shift (ZZ) gate, not an MS (XX) gate.
    """

    def test_zz_gate_entangles(self, electron_trap):
        """Light-shift gate from gradient coupling should entangle
        |+,+> into a maximally entangled state with ZZ correlations.

        The gradient Hamiltonian is sigma_z-dependent, so sigma_z
        eigenstates (|0>, |1>) are displaced in opposite directions
        in phase space. Starting from sigma_x eigenstates (|+>, |->)
        produces entanglement. sigma_x sigma_x commutes with the
        generator, so <sx sx> = 1 must survive the gate exactly - a
        check that the interaction is a pure ZZ force.
        """
        modes = normal_modes(2, electron_trap)
        species = electron_trap.species
        eta_matrix = gradient_lamb_dicke_parameters(
            modes, species, GRADIENT, "axial"
        )
        # COM mode: equal participation, eta ~ 0.0439 at 120 T/m.
        assert eta_matrix[0, 0] == pytest.approx(eta_matrix[1, 0], rel=1e-12)
        assert abs(eta_matrix[0, 0]) == pytest.approx(0.0439, rel=0.01)

        hs = HilbertSpace(n_ions=2, n_modes=1, n_fock=15)
        ops = OperatorFactory(hs)

        eta = [float(eta_matrix[0, 0]), float(eta_matrix[1, 0])]
        delta = TWO_PI * 15e3
        Omega = delta / (4 * abs(eta[0]))
        tau = ms_gate_duration(delta)

        H = light_shift_gate_hamiltonian(
            ops,
            [0, 1],
            0,
            eta,
            Omega,
            delta,
        )

        # Start in |+,+> (sigma_x eigenstates)
        plus = (qutip.basis(2, 0) + qutip.basis(2, 1)).unit()
        psi0 = qutip.tensor(plus, plus, qutip.basis(15, 0))
        tlist = np.linspace(0, tau, 500)
        result = qutip.sesolve(
            H,
            psi0,
            tlist,
            options={"max_step": tau / 100},
        )

        # Maximally entangled: single-qubit state is maximally mixed
        # while the two-qubit block stays pure (motion disentangled).
        rho_spin = result.states[-1].ptrace([0, 1])
        rho_single = rho_spin.ptrace(0)
        assert (rho_single**2).tr().real == pytest.approx(0.5, abs=1e-4)
        assert (rho_spin**2).tr().real == pytest.approx(1.0, abs=1e-4)
        assert qutip.expect(
            ops.sigma_x(0) * ops.sigma_x(1), result.states[-1]
        ) == pytest.approx(1.0, abs=1e-4)

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
            ops,
            [0, 1],
            0,
            [eta, eta],
            Omega,
            delta,
        )

        plus = (qutip.basis(2, 0) + qutip.basis(2, 1)).unit()
        psi0 = qutip.tensor(plus, plus, qutip.basis(20, 0))
        r = qutip.sesolve(
            H,
            psi0,
            np.linspace(0, tau, 500),
            options={"max_step": tau / 100},
        )
        n_final = qutip.expect(ops.number(0), r.states[-1])
        assert n_final == pytest.approx(0.0, abs=1e-4)

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
            ops,
            [0, 1],
            0,
            [eta, eta],
            Omega,
            delta,
        )

        plus = (qutip.basis(2, 0) + qutip.basis(2, 1)).unit()
        psi0 = qutip.tensor(plus, plus, qutip.basis(15, 0))
        tlist = np.linspace(0, tau, 500)

        r_clean = qutip.sesolve(
            H,
            psi0,
            tlist,
            options={"max_step": tau / 100},
        )
        purity_clean = (r_clean.states[-1].ptrace([0, 1]) ** 2).tr().real

        t2 = 100e-6
        c_ops = [
            qubit_dephasing_op(ops, 0, t2=t2),
            qubit_dephasing_op(ops, 1, t2=t2),
        ]
        r_noisy = qutip.mesolve(
            H,
            psi0,
            tlist,
            c_ops=c_ops,
            options={"max_step": tau / 100},
        )
        purity_noisy = (r_noisy.states[-1].ptrace([0, 1]) ** 2).tr().real

        assert purity_noisy < purity_clean
        # The collapse operators commute with the light-shift
        # generator, so dephasing acts on the ideal output exactly.
        # All 16 elements of the maximally entangled spin block have
        # magnitude 1/4; a coherence between computational states
        # differing in one qubit decays at 1/T2 and one differing in
        # both at 2/T2, so
        #   Tr(rho^2) = 1/4 + exp(-4 tau/T2)/4 + exp(-2 tau/T2)/2.
        expected = (
            0.25 + 0.25 * np.exp(-4 * tau / t2) + 0.5 * np.exp(-2 * tau / t2)
        )
        assert purity_noisy == pytest.approx(expected, rel=1e-3)


class TestElectronAnalyticalExactness:
    """Tight numerical checks against known analytical results for
    trapped electrons.

    Every test drives library code: trap properties, equilibrium
    positions, normal modes, Lamb-Dicke parameters or a solver run.
    Pure arithmetic over `tiqs.constants` belongs in
    tests/test_constants.py, and the Penning-trap electron frequencies
    are covered by tests/test_penning.py.

    References: CODATA 2022, Leibfried et al. RMP 75 281 (2003),
    Huang et al. arXiv:2503.12379 (2025), Yu et al. PRA 105 022420
    (2022), Mikhailovskii et al. arXiv:2508.16407 (2025).
    """

    def test_mathieu_q_scaling(self):
        """q = 2 e V_rf / (m Omega_rf^2 r0^2) - check the exponents.

        Restating the formula cannot detect a wrong exponent, so this
        varies one factor at a time: q is linear in V_rf, quadratic in
        1/Omega_rf, quadratic in 1/r0, and linear in 1/m (the last via
        the electron/Ca-40 mass ratio at identical geometry).
        """

        def make(species, v_rf=7.8, f_rf=1.6e9, r0=300e-6):
            return PaulTrap(
                v_rf=v_rf,
                omega_rf=TWO_PI * f_rf,
                r0=r0,
                omega_axial=TWO_PI * 30e6,
                species=species,
            )

        electron = ElectronSpecies(0.1)
        base = make(electron).mathieu_q
        assert make(electron, v_rf=15.6).mathieu_q == pytest.approx(
            2 * base, rel=1e-12
        )
        assert make(electron, f_rf=3.2e9).mathieu_q == pytest.approx(
            base / 4, rel=1e-12
        )
        assert make(electron, r0=600e-6).mathieu_q == pytest.approx(
            base / 4, rel=1e-12
        )

        ca = get_species("Ca40")
        heavy = make(ca, v_rf=7.8, f_rf=1.6e9)
        assert base / heavy.mathieu_q == pytest.approx(
            ca.mass_kg / ELECTRON_MASS, rel=1e-12
        )

    def test_pseudopotential_depth(self, electron_trap):
        """Trap depth = q V_rf / 8, so it grows as V_rf^2.

        The 7.8 V electron trap is 0.294 eV deep, i.e. ~3400 K - deep
        compared with a 4 K cryostat but a factor ~30 shallower than a
        typical ion trap, which is why electron traps run at GHz RF.
        """
        assert electron_trap.pseudopotential_depth_eV == pytest.approx(
            0.294, rel=0.01
        )
        deeper = PaulTrap(
            v_rf=15.6,
            omega_rf=TWO_PI * 1.6e9,
            r0=300e-6,
            omega_axial=TWO_PI * 30e6,
            species=ElectronSpecies(0.1),
        )
        assert deeper.pseudopotential_depth_eV == pytest.approx(
            4 * electron_trap.pseudopotential_depth_eV, rel=1e-12
        )

    def test_two_electron_spacing_analytical(self, electron_trap):
        """Two-particle spacing: d = 2 * (1/2)^(2/3) * l_0 where
        l_0 = (e^2 / (4*pi*eps0 * m_e * omega_z^2))^(1/3)."""
        omega_z = electron_trap.omega_axial
        pos = equilibrium_positions(2, electron_trap)
        l_scale = (COULOMB_CONSTANT / (ELECTRON_MASS * omega_z**2)) ** (1 / 3)
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
            v_rf=300,
            omega_rf=TWO_PI * 30e6,
            r0=0.5e-3,
            omega_axial=omega_z,
            species=ca,
        )
        trap_e = PaulTrap(
            v_rf=7.8,
            omega_rf=TWO_PI * 1.6e9,
            r0=300e-6,
            omega_axial=omega_z,
            species=e,
        )
        d_ca = equilibrium_positions(2, trap_ca)
        d_e = equilibrium_positions(2, trap_e)
        spacing_ratio = (d_e[1] - d_e[0]) / (d_ca[1] - d_ca[0])
        mass_ratio = (ca.mass_kg / ELECTRON_MASS) ** (1 / 3)
        assert spacing_ratio == pytest.approx(mass_ratio, rel=0.001)

    def test_huang_zero_point_spread(self):
        """x_zpf from Huang et al. arXiv:2503.12379 (2025): 554 nm at
        30 MHz and 175 nm at 300 MHz for a single electron.

        `lamb_dicke_parameters` with k_eff = 1 rad/m returns
        b * x_zpf, and b = 1 for a single particle, so this reads the
        library's zero-point spread directly. Two frequencies pin the
        omega^(-1/2) exponent as well as the prefactor.
        """
        cases = [
            (TWO_PI * 30e6, 7.8, TWO_PI * 1.6e9, 300e-6, 554e-9),
            (TWO_PI * 300e6, 7.1, TWO_PI * 10e9, 45.8e-6, 175e-9),
        ]
        for omega_axial, v_rf, omega_rf, r0, x_ref in cases:
            trap = PaulTrap(
                v_rf=v_rf,
                omega_rf=omega_rf,
                r0=r0,
                omega_axial=omega_axial,
                species=ElectronSpecies(0.1),
            )
            modes = normal_modes(1, trap)
            x_zpf = lamb_dicke_parameters(modes, trap.species, 1.0, "axial")
            assert x_zpf[0, 0] == pytest.approx(x_ref, rel=0.005)

    def test_zero_point_spread_vs_ions(self):
        """x_zpf scales as m^(-1/2): electron/Ca-40 = 269.9 at equal
        frequency, read out of `lamb_dicke_parameters` (k_eff = 1)."""
        ca = get_species("Ca40")
        omega = TWO_PI * 1e6
        trap_ca = PaulTrap(
            v_rf=300,
            omega_rf=TWO_PI * 30e6,
            r0=0.5e-3,
            omega_axial=omega,
            species=ca,
        )
        trap_e = PaulTrap(
            v_rf=7.8,
            omega_rf=TWO_PI * 1.6e9,
            r0=300e-6,
            omega_axial=omega,
            species=ElectronSpecies(0.1),
        )
        x_ca = lamb_dicke_parameters(
            normal_modes(1, trap_ca), ca, 1.0, "axial"
        )[0, 0]
        x_e = lamb_dicke_parameters(
            normal_modes(1, trap_e), trap_e.species, 1.0, "axial"
        )[0, 0]
        assert x_e / x_ca == pytest.approx(
            np.sqrt(ca.mass_kg / ELECTRON_MASS), rel=1e-9
        )
        assert x_e / x_ca == pytest.approx(269.9, rel=0.01)

    def test_huang_coulomb_length_scale(self):
        """l_0 from Huang et al. arXiv:2503.12379 (2025): 19.25 um at
        30 MHz and 4.15 um at 300 MHz.

        Recovered from the two-electron equilibrium separation,
        d = 2 (1/2)^(2/3) l_0, so this exercises
        `equilibrium_positions` rather than restating
        l_0 = (e^2 / 4 pi eps0 m omega^2)^(1/3). Two frequencies pin
        the omega^(-2/3) exponent.
        """
        cases = [
            (TWO_PI * 30e6, 7.8, TWO_PI * 1.6e9, 300e-6, 19.25e-6),
            (TWO_PI * 300e6, 7.1, TWO_PI * 10e9, 45.8e-6, 4.15e-6),
        ]
        for omega_axial, v_rf, omega_rf, r0, l0_ref in cases:
            trap = PaulTrap(
                v_rf=v_rf,
                omega_rf=omega_rf,
                r0=r0,
                omega_axial=omega_axial,
                species=ElectronSpecies(0.1),
            )
            pos = equilibrium_positions(2, trap)
            l0 = (pos[1] - pos[0]) / (2 * (1 / 2) ** (2 / 3))
            assert l0 == pytest.approx(l0_ref, rel=0.005)

    def test_zz_gate_fidelity_and_motional_closure(self):
        """ZZ gate from gradient coupling must produce a maximally
        entangled state and return the motion to vacuum.

        Uses the light-shift Hamiltonian (sigma_z force) which is the
        native interaction from magnetic gradient coupling."""
        hs = HilbertSpace(n_ions=2, n_modes=1, n_fock=20)
        ops = OperatorFactory(hs)

        eta = 0.05
        delta = TWO_PI * 15e3
        Omega = delta / (4 * eta)
        tau = ms_gate_duration(delta)
        H = light_shift_gate_hamiltonian(
            ops,
            [0, 1],
            0,
            [eta, eta],
            Omega,
            delta,
        )

        # ZZ gate entangles sigma_x eigenstates, not sigma_z
        plus = (qutip.basis(2, 0) + qutip.basis(2, 1)).unit()
        psi0 = qutip.tensor(plus, plus, qutip.basis(20, 0))
        r = qutip.sesolve(
            H,
            psi0,
            np.linspace(0, tau, 500),
            options={"max_step": tau / 100},
        )

        # Motion returns to vacuum
        n_final = qutip.expect(ops.number(0), r.states[-1])
        assert n_final == pytest.approx(0.0, abs=1e-4)

        # Spin state is maximally entangled: single-qubit purity 0.5
        rho_spin = r.states[-1].ptrace([0, 1])
        rho_single = rho_spin.ptrace(0)
        purity = (rho_single * rho_single).tr().real
        assert purity == pytest.approx(0.5, abs=1e-4)

    def test_carrier_rabi_exact(self):
        """sigma_z = cos(Omega*t) for microwave carrier drive on
        electron spin. Validates the Hamiltonian convention."""
        hs = HilbertSpace(n_ions=1, n_modes=1, n_fock=3)
        ops = OperatorFactory(hs)
        sf = StateFactory(hs)
        Omega = TWO_PI * 500e3
        H = carrier_hamiltonian(ops, 0, Omega)
        tlist = np.linspace(0, 4 * np.pi / Omega, 400)
        result = qutip.sesolve(
            H,
            sf.ground_state(),
            tlist,
            e_ops=[ops.sigma_z(0)],
        )
        expected = np.cos(Omega * tlist)
        np.testing.assert_allclose(result.expect[0], expected, atol=0.01)

    def test_yu_hahn_radial_frequency(self):
        """Yu et al. PRA 105, 022420 (2022): V_rf = 14 V,
        Omega_rf/(2pi) = 10.6 GHz, q = 0.53, omega_r/(2pi) ~ 2 GHz.

        r0 = 45.8 um is reverse-fitted to their quoted q, not taken
        from the paper (their 2 GHz implies q ~ 0.534, which also
        rounds to 0.53), so this is a consistency check against a
        design study, not a measurement.

        The pseudopotential returns 1.9721 GHz, 1.39% below their
        2 GHz. Only 0.57 percentage points of that come from the
        Mathieu a term of the 300 MHz axial confinement (dropping a
        gives 1.9835 GHz); the rest is the fitted r0. Neither is the
        dominant error: beta ~= sqrt(a + q^2/2) drops a +25 q^4/128
        term (Leibfried RMP 75, 281 (2003) Eqs. 11-15), and at
        q = 0.53 the exact Floquet solution is beta = 0.396315, i.e.
        2.1005 GHz. The pseudopotential is therefore 6.1% LOW against
        exact Mathieu, and agrees with the paper only because their
        2 GHz is itself a pseudopotential estimate.
        """
        trap = PaulTrap(
            v_rf=14.0,
            omega_rf=TWO_PI * 10.6e9,
            r0=45.8e-6,
            omega_axial=TWO_PI * 300e6,
            species=ElectronSpecies(magnetic_field=0.0036),
        )
        assert trap.mathieu_q == pytest.approx(0.53, rel=0.01)
        assert trap.omega_radial / TWO_PI == pytest.approx(1.9721e9, rel=1e-3)
        # The a term is worth -0.57%, not the whole -1.39% gap.
        omega_no_a = (trap.omega_rf / 2) * np.sqrt(trap.mathieu_q**2 / 2)
        assert trap.omega_radial / omega_no_a == pytest.approx(
            1 - 0.00574, rel=0.01
        )
        # Exact Floquet beta at this (a, q) is 0.396315.
        omega_exact = 0.396315 * trap.omega_rf / 2
        assert trap.omega_radial / omega_exact == pytest.approx(
            1 - 0.0611, rel=0.01
        )

    def test_mikhailovskii_pseudopotential_vs_measurement(self):
        """Mikhailovskii et al. arXiv:2508.16407 (2025): measured
        electron radial frequency 72 MHz at q = 0.11,
        Omega_rf = 1.6 GHz.

        The pseudopotential predicts 59.4 MHz, so the measurement is
        21% ABOVE the prediction. Their PCB slot geometry is not an
        ideal quadrupole, and higher-order multipoles stiffen the real
        well, so the ideal-quadrupole pseudopotential is a lower bound
        here. The 59.4 MHz figure is a regression pin on TIQS, not a
        published value; only the 72 MHz comparison is from the paper.
        """
        trap = PaulTrap(
            v_rf=6.4,
            omega_rf=TWO_PI * 1.6e9,
            r0=0.45e-3,
            omega_axial=TWO_PI * 26e6,
            species=ElectronSpecies(magnetic_field=0.1),
        )
        assert trap.mathieu_q == pytest.approx(0.11, rel=0.01)
        # Regression pin, not a published value.
        omega_r_pseudo = trap.omega_radial / TWO_PI
        assert omega_r_pseudo == pytest.approx(59.45e6, rel=0.01)
        assert 72e6 / omega_r_pseudo == pytest.approx(1.21, rel=0.02)

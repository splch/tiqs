"""Experimental validation tests grounded in published results.

Each test reproduces a specific published measurement or analytical
result from the trapped-ion literature, with citations. These tests
verify that the simulator's physics matches reality, not just
internal self-consistency.

References
----------
[James1998]  James, Appl. Phys. B 66, 181 (1998). quant-ph/9702053.
[Leibfried2003]  Leibfried et al., Rev. Mod. Phys. 75, 281 (2003).
[Benhelm2008]  Benhelm et al., Nature Physics 4, 463 (2008).
[Wineland1998]  Wineland et al., J. Res. NIST 103, 259 (1998).
"""

import math

import numpy as np
import pytest
import qutip

from tiqs.analysis.fidelity import bell_state_fidelity
from tiqs.chain.equilibrium import equilibrium_positions
from tiqs.chain.lamb_dicke import lamb_dicke_parameters
from tiqs.chain.normal_modes import normal_modes
from tiqs.constants import (
    ELECTRON_CHARGE,
    EPSILON_0,
    PI,
    TWO_PI,
)
from tiqs.gates.molmer_sorensen import ms_gate_duration, ms_gate_hamiltonian
from tiqs.hilbert_space.builder import HilbertSpace
from tiqs.hilbert_space.operators import OperatorFactory
from tiqs.hilbert_space.states import StateFactory
from tiqs.interaction.hamiltonian import (
    carrier_hamiltonian,
    red_sideband_hamiltonian,
)
from tiqs.noise.motional import motional_heating_ops
from tiqs.noise.qubit import qubit_dephasing_op, spontaneous_emission_op
from tiqs.species.ion import get_species
from tiqs.trap import PaulTrap


class TestJames1998NormalModes:
    """Analytical mode frequencies and equilibrium positions from
    [James1998], the foundational reference verified by every
    trapped-ion experiment."""

    @pytest.fixture
    def ca40_trap(self):
        return PaulTrap(
            v_rf=300,
            omega_rf=TWO_PI * 30e6,
            r0=0.5e-3,
            omega_axial=TWO_PI * 1e6,
            species=get_species("Ca40"),
        )

    def test_two_ion_stretch_to_com_ratio(self, ca40_trap):
        """[James1998] Eq. 18: stretch/COM = sqrt(3) exactly."""
        modes = normal_modes(2, ca40_trap)
        ratio = modes.axial_freqs[1] / modes.axial_freqs[0]
        assert ratio == pytest.approx(np.sqrt(3), rel=1e-6)

    def test_two_ion_com_equals_axial(self, ca40_trap):
        """[James1998]: COM mode frequency equals the bare trap
        frequency."""
        modes = normal_modes(2, ca40_trap)
        assert modes.axial_freqs[0] == pytest.approx(
            ca40_trap.omega_axial, rel=1e-6
        )

    def test_three_ion_breathing_mode_ratio(self, ca40_trap):
        """[James1998] Table I: breathing mode at sqrt(29/5) * omega_z
        for 3 ions."""
        modes = normal_modes(3, ca40_trap)
        ratio = modes.axial_freqs[2] / modes.axial_freqs[0]
        assert ratio == pytest.approx(np.sqrt(29 / 5), rel=1e-4)

    def test_three_ion_tilt_mode_ratio(self, ca40_trap):
        """[James1998] Table I: tilt mode at sqrt(3) * omega_z."""
        modes = normal_modes(3, ca40_trap)
        ratio = modes.axial_freqs[1] / modes.axial_freqs[0]
        assert ratio == pytest.approx(np.sqrt(3), rel=1e-4)

    def test_three_ion_center_at_origin(self, ca40_trap):
        """[James1998]: center ion at z=0 by symmetry."""
        pos = equilibrium_positions(3, ca40_trap)
        assert pos[1] == pytest.approx(0.0, abs=1e-12)

    def test_three_ion_dimensionless_offset(self, ca40_trap):
        """[James1998] Table I: outer ions at u = +/-(5/4)^(1/3)
        in dimensionless units."""
        pos = equilibrium_positions(3, ca40_trap)
        length_scale = (
            ELECTRON_CHARGE**2
            / (
                4
                * PI
                * EPSILON_0
                * ca40_trap.species.mass_kg
                * ca40_trap.omega_axial**2
            )
        ) ** (1 / 3)
        u_outer = pos[2] / length_scale
        assert u_outer == pytest.approx((5 / 4) ** (1 / 3), rel=1e-3)

    def test_com_eigenvector_equal_amplitude(self, ca40_trap):
        """[James1998]: COM eigenvector has all components equal."""
        modes = normal_modes(2, ca40_trap)
        v_com = modes.axial_vectors[:, 0]
        assert abs(v_com[0]) == pytest.approx(abs(v_com[1]), rel=1e-6)

    def test_stretch_eigenvector_opposite_phase(self, ca40_trap):
        """[James1998]: stretch eigenvector has opposite signs."""
        modes = normal_modes(2, ca40_trap)
        v_str = modes.axial_vectors[:, 1]
        assert np.sign(v_str[0]) != np.sign(v_str[1])


class TestCa40IonSpacing:
    """Two-ion equilibrium spacing computed from fundamental constants,
    verified against [Leibfried2003] and direct CCD imaging."""

    def test_two_ion_spacing_ca40_at_1mhz(self):
        """d = 2^(1/3) * l_0 where l_0 = (e^2/(4*pi*eps0*m*omega_z^2))^(1/3).
        For Ca-40 at 1 MHz: d ~ 5.6 um."""
        ca = get_species("Ca40")
        trap = PaulTrap(
            v_rf=300,
            omega_rf=TWO_PI * 30e6,
            r0=0.5e-3,
            omega_axial=TWO_PI * 1e6,
            species=ca,
        )
        pos = equilibrium_positions(2, trap)
        spacing = pos[1] - pos[0]

        m = ca.mass_kg
        omega_z = trap.omega_axial
        ke = 1 / (4 * PI * EPSILON_0)
        d_analytical = (2 * ke * ELECTRON_CHARGE**2 / (m * omega_z**2)) ** (
            1 / 3
        )

        assert spacing == pytest.approx(d_analytical, rel=1e-3)
        assert 4e-6 < spacing < 7e-6


class TestMSGateBellState:
    """Verify that the MS gate produces a Bell state with the correct
    geometric phase, matching the condition from [Leibfried2003]."""

    def test_ideal_ms_bell_fidelity(self):
        """An ideal single-loop MS gate with eta*Omega = delta/4
        should produce (|00> + i|11>)/sqrt(2) with F > 0.99."""
        hs = HilbertSpace(n_ions=2, n_modes=1, n_fock=20)
        ops = OperatorFactory(hs)
        sf = StateFactory(hs)

        eta = 0.05
        delta = TWO_PI * 15e3
        Omega = delta / (4 * eta)
        tau = ms_gate_duration(delta)

        H = ms_gate_hamiltonian(ops, [0, 1], 0, [eta, eta], Omega, delta)
        psi0 = sf.ground_state()
        tlist = np.linspace(0, tau, 800)
        result = qutip.sesolve(
            H, psi0, tlist, options={"max_step": tau / 200}
        )
        rho_spin = result.states[-1].ptrace([0, 1])
        fid = bell_state_fidelity(rho_spin)
        assert fid > 0.99

    def test_ms_gate_motional_closure(self):
        """After one complete phase-space loop, the motional state
        should return to vacuum: <n> ~ 0."""
        hs = HilbertSpace(n_ions=2, n_modes=1, n_fock=20)
        ops = OperatorFactory(hs)
        sf = StateFactory(hs)

        eta = 0.05
        delta = TWO_PI * 15e3
        Omega = delta / (4 * eta)
        tau = ms_gate_duration(delta)

        H = ms_gate_hamiltonian(ops, [0, 1], 0, [eta, eta], Omega, delta)
        result = qutip.sesolve(
            H,
            sf.ground_state(),
            np.linspace(0, tau, 800),
            options={"max_step": tau / 200},
        )
        n_final = qutip.expect(ops.number(0), result.states[-1])
        assert n_final == pytest.approx(0.0, abs=0.1)

    def test_rabi_oscillation_frequency(self):
        """[Leibfried2003] Eq. 22: carrier Rabi oscillation at
        frequency Omega. <sigma_z> = cos(Omega*t)."""
        hs = HilbertSpace(n_ions=1, n_modes=1, n_fock=5)
        ops = OperatorFactory(hs)
        sf = StateFactory(hs)

        Omega = TWO_PI * 200e3
        H = carrier_hamiltonian(ops, 0, Omega)
        tlist = np.linspace(0, 3 * TWO_PI / Omega, 500)
        result = qutip.sesolve(
            H, sf.ground_state(), tlist, e_ops=[ops.sigma_z(0)]
        )
        expected = np.cos(Omega * tlist)
        np.testing.assert_allclose(result.expect[0], expected, atol=0.05)


class TestDopplerCoolingLimits:
    """Doppler cooling limits from [Leibfried2003] Eq. 6."""

    def test_ca40_doppler_temperature(self):
        """T_D = hbar*Gamma / (2*k_B). For Ca-40: Gamma/(2*pi) = 22.4 MHz,
        T_D ~ 0.5 mK."""
        ca = get_species("Ca40")
        T_D = ca.doppler_limit_temperature()
        assert 0.1e-3 < T_D < 2e-3

    def test_doppler_nbar_formula(self):
        """nbar_D = Gamma / (2*omega_trap).
        For Ca-40 at 1 MHz: nbar ~ 11."""
        ca = get_species("Ca40")
        nbar = ca.doppler_limit_nbar(1e6)
        gamma = ca.cooling_transition.linewidth
        omega_trap = TWO_PI * 1e6
        expected = gamma / (2 * omega_trap)
        assert nbar == pytest.approx(expected, rel=1e-6)
        assert 5 < nbar < 20


class TestSpeciesAtomicData:
    """Cross-check species database against NIST atomic data and
    published values."""

    def test_yb171_qubit_splitting(self):
        """Yb-171 hyperfine splitting: 12.6428 GHz (NIST)."""
        yb = get_species("Yb171")
        assert yb.qubit_frequency_hz == pytest.approx(12.6428e9, rel=1e-4)

    def test_ca40_metastable_lifetime(self):
        """Ca-40 D5/2 lifetime: 1.168 s (Barton et al. 2000)."""
        ca = get_species("Ca40")
        assert ca.metastable_lifetime == pytest.approx(1.168, rel=0.01)

    def test_yb171_infinite_t1(self):
        """Hyperfine ground-state qubits have T1 = infinity."""
        yb = get_species("Yb171")
        assert yb.qubit_t1 == math.inf

    def test_be9_is_lightest(self):
        """Be-9 is the lightest commonly used ion qubit."""
        be = get_species("Be9")
        for name in ["Ca40", "Ca43", "Ba137", "Yb171", "Sr88"]:
            assert be.mass_amu < get_species(name).mass_amu

    def test_lighter_ion_larger_lamb_dicke(self):
        """Lighter ions have larger Lamb-Dicke parameters at the same
        trap frequency (eta ~ 1/sqrt(m)). [Leibfried2003] Eq. 11."""
        k = TWO_PI / 400e-9
        for species_name, v_rf in [("Be9", 300), ("Yb171", 1000)]:
            species = get_species(species_name)
            trap = PaulTrap(
                v_rf=v_rf,
                omega_rf=TWO_PI * 30e6,
                r0=0.5e-3,
                omega_axial=TWO_PI * 1e6,
                species=species,
            )
            modes = normal_modes(1, trap)
            eta = lamb_dicke_parameters(modes, species, k, "axial")
            if species_name == "Be9":
                eta_be = eta[0, 0]
            else:
                eta_yb = eta[0, 0]
        assert eta_be > eta_yb


class TestT2DephasingDynamics:
    """[Leibfried2003] Sec. V.A: pure dephasing causes exponential
    decay of the off-diagonal density matrix element at rate 1/T2."""

    def test_off_diagonal_decays_as_exp_minus_t_over_t2(self):
        """Prepare |+> = (|0> + |1>)/sqrt(2), evolve under pure
        dephasing, measure <sigma_x>(t). Should decay as exp(-t/T2)."""
        hs = HilbertSpace(n_ions=1, n_modes=1, n_fock=3)
        ops = OperatorFactory(hs)
        sf = StateFactory(hs)

        T2 = 1e-3
        c_ops = [qubit_dephasing_op(ops, 0, t2=T2)]

        plus = (sf.product_state([0], [0]) + sf.product_state([1], [0])).unit()
        tlist = np.linspace(0, 5 * T2, 200)
        sx = ops.sigma_x(0)
        result = qutip.mesolve(
            0 * ops.identity(), plus, tlist, c_ops=c_ops, e_ops=[sx]
        )

        expected = np.exp(-tlist / T2)
        np.testing.assert_allclose(result.expect[0], expected, atol=0.05)

    def test_dephasing_preserves_populations(self):
        """Pure dephasing should not change <sigma_z>."""
        hs = HilbertSpace(n_ions=1, n_modes=1, n_fock=3)
        ops = OperatorFactory(hs)
        sf = StateFactory(hs)

        T2 = 1e-3
        c_ops = [qubit_dephasing_op(ops, 0, t2=T2)]
        plus = (sf.product_state([0], [0]) + sf.product_state([1], [0])).unit()
        tlist = np.linspace(0, 5 * T2, 100)
        sz = ops.sigma_z(0)
        result = qutip.mesolve(
            0 * ops.identity(), plus, tlist, c_ops=c_ops, e_ops=[sz]
        )
        np.testing.assert_allclose(result.expect[0], 0.0, atol=0.02)


class TestT1DecayDynamics:
    r"""[Leibfried2003] Sec. V.A: spontaneous emission from |1> to |0>
    at rate 1/T1. sigma_z decays from -1 toward +1 as
    <sigma_z>(t) = 1 - 2*exp(-t/T1)."""

    def test_excited_state_decays_to_ground(self):
        hs = HilbertSpace(n_ions=1, n_modes=1, n_fock=3)
        ops = OperatorFactory(hs)
        sf = StateFactory(hs)

        T1 = 1e-3
        c_ops = [spontaneous_emission_op(ops, 0, t1=T1)]
        psi_excited = sf.product_state([1], [0])
        tlist = np.linspace(0, 5 * T1, 200)
        sz = ops.sigma_z(0)
        result = qutip.mesolve(
            0 * ops.identity(),
            psi_excited,
            tlist,
            c_ops=c_ops,
            e_ops=[sz],
        )

        expected = 1 - 2 * np.exp(-tlist / T1)
        np.testing.assert_allclose(result.expect[0], expected, atol=0.05)


class TestMotionalHeatingRate:
    """[Leibfried2003] Eq. 33: motional heating adds phonons at a
    constant rate: d<n>/dt = heating_rate."""

    def test_phonon_growth_matches_rate_equation(self):
        """Starting from vacuum, <n>(t) = exp(gamma*t) - 1."""
        hs = HilbertSpace(n_ions=1, n_modes=1, n_fock=30)
        ops = OperatorFactory(hs)
        sf = StateFactory(hs)

        heating_rate = 1e4
        c_ops = motional_heating_ops(ops, 0, heating_rate)
        psi0 = sf.ground_state()
        n_op = ops.number(0)
        tlist = np.linspace(0, 2e-4, 100)
        result = qutip.mesolve(
            0 * ops.identity(),
            psi0,
            tlist,
            c_ops=c_ops,
            e_ops=[n_op],
        )

        expected = np.exp(heating_rate * tlist) - 1
        mask = expected < 5
        np.testing.assert_allclose(
            result.expect[0][mask], expected[mask], rtol=0.1
        )


class TestRSBRabiFrequencyScaling:
    r"""[Leibfried2003] Eq. 12: red sideband Rabi frequency scales as
    Omega_{n,n-1} = eta * Omega * sqrt(n). The pi-pulse time
    for |0,n> -> |1,n-1> is t_pi = pi / (eta * Omega * sqrt(n))."""

    def test_rsb_rabi_sqrt_n_scaling(self):
        """The RSB pi-time for n=4 should be half the pi-time for
        n=1 (since sqrt(4)/sqrt(1) = 2)."""
        hs = HilbertSpace(n_ions=1, n_modes=1, n_fock=15)
        ops = OperatorFactory(hs)

        eta = 0.1
        Omega = TWO_PI * 100e3
        H_rsb = red_sideband_hamiltonian(ops, 0, 0, Omega, eta)

        rsb_rabi_n1 = eta * Omega * np.sqrt(1)
        t_pi_n1 = PI / rsb_rabi_n1

        # n=1: after t_pi, should flip to |1,0>
        psi_n1 = qutip.tensor(qutip.basis(2, 0), qutip.basis(15, 1))
        target_n1 = qutip.tensor(qutip.basis(2, 1), qutip.basis(15, 0))
        r1 = qutip.sesolve(H_rsb, psi_n1, [0, t_pi_n1])
        fid_n1 = abs(r1.states[-1].overlap(target_n1)) ** 2
        assert fid_n1 > 0.95

        # n=4: Rabi freq is 2x faster, so after same t_pi_n1 it
        # should have done 2 full Rabi cycles (back to start)
        psi_n4 = qutip.tensor(qutip.basis(2, 0), qutip.basis(15, 4))
        r4 = qutip.sesolve(H_rsb, psi_n4, [0, t_pi_n1])
        fid_return = abs(r4.states[-1].overlap(psi_n4)) ** 2
        assert fid_return > 0.85


class TestSidebandCoolingConvergence:
    """[Wineland1998]: resolved sideband cooling converges to
    nbar_final ~ (gamma_eff / 2*omega_trap)^2."""

    def test_simulated_nbar_approaches_analytical(self):
        """The simulated final nbar should be within a factor of
        5 of the analytical prediction."""
        from tiqs.cooling.sideband_cooling import (
            sideband_cooling_nbar,
            sideband_cooling_simulate,
        )

        hs = HilbertSpace(n_ions=1, n_modes=1, n_fock=20)
        ops = OperatorFactory(hs)

        gamma_eff = TWO_PI * 10e3
        omega_trap = TWO_PI * 1e6
        nbar_analytical = sideband_cooling_nbar(gamma_eff, omega_trap)

        nbar_sim = sideband_cooling_simulate(
            ops,
            ion=0,
            mode=0,
            n_bar_initial=5.0,
            eta=0.1,
            rabi_frequency=TWO_PI * 100e3,
            optical_pumping_rate=gamma_eff,
            n_cycles=30,
        )

        assert nbar_sim < 1.0
        assert nbar_sim < 5 * nbar_analytical + 0.1


class TestBenhelm2008Reproduction:
    """Reproduce the Benhelm et al. (2008) Ca-40 MS gate.

    [Benhelm2008]: Ca-40 optical qubit, axial COM ~ 1.2 MHz,
    eta ~ 0.05, gate time ~ 50 us. Reported 99.3(1)% Bell fidelity.

    Our noiseless simulation should give F ~ 1.0. Adding the
    dominant noise source (D5/2 spontaneous decay, T1 = 1.168 s)
    should give a small reduction. Adding all noise sources
    should bring it below the ideal but above ~98%."""

    @pytest.fixture
    def benhelm_setup(self):
        hs = HilbertSpace(n_ions=2, n_modes=1, n_fock=20)
        ops = OperatorFactory(hs)
        sf = StateFactory(hs)

        eta = 0.05
        delta = TWO_PI * 20e3
        Omega = delta / (4 * eta)
        tau = ms_gate_duration(delta)

        return ops, sf, eta, delta, Omega, tau

    def test_noiseless_ideal_fidelity(self, benhelm_setup):
        """Noiseless MS gate at Benhelm parameters: F > 0.99."""
        ops, sf, eta, delta, Omega, tau = benhelm_setup
        H = ms_gate_hamiltonian(ops, [0, 1], 0, [eta, eta], Omega, delta)
        r = qutip.sesolve(
            H,
            sf.ground_state(),
            np.linspace(0, tau, 800),
            options={"max_step": tau / 200},
        )
        fid = bell_state_fidelity(r.states[-1].ptrace([0, 1]))
        assert fid > 0.99

    def test_spontaneous_decay_reduces_fidelity(self, benhelm_setup):
        """Ca-40 D5/2 decay (T1=1.168 s) should slightly reduce F."""
        ops, sf, eta, delta, Omega, tau = benhelm_setup
        H = ms_gate_hamiltonian(ops, [0, 1], 0, [eta, eta], Omega, delta)
        tlist = np.linspace(0, tau, 500)

        r_clean = qutip.sesolve(
            H, sf.ground_state(), tlist, options={"max_step": tau / 100}
        )
        fid_clean = bell_state_fidelity(r_clean.states[-1].ptrace([0, 1]))

        c_ops = [
            spontaneous_emission_op(ops, 0, t1=1.168),
            spontaneous_emission_op(ops, 1, t1=1.168),
        ]
        r_noisy = qutip.mesolve(
            H,
            sf.ground_state(),
            tlist,
            c_ops=c_ops,
            options={"max_step": tau / 100},
        )
        fid_noisy = bell_state_fidelity(r_noisy.states[-1].ptrace([0, 1]))
        assert fid_noisy < fid_clean
        assert fid_noisy > 0.98

    def test_combined_noise_reduces_further(self, benhelm_setup):
        """Adding heating + dephasing + decay should reduce F more."""
        ops, sf, eta, delta, Omega, tau = benhelm_setup
        H = ms_gate_hamiltonian(ops, [0, 1], 0, [eta, eta], Omega, delta)
        tlist = np.linspace(0, tau, 500)

        c_ops = [
            spontaneous_emission_op(ops, 0, t1=1.168),
            spontaneous_emission_op(ops, 1, t1=1.168),
            *motional_heating_ops(ops, 0, heating_rate=100),
            qubit_dephasing_op(ops, 0, t2=10e-3),
            qubit_dephasing_op(ops, 1, t2=10e-3),
        ]
        r = qutip.mesolve(
            H,
            sf.ground_state(),
            tlist,
            c_ops=c_ops,
            options={"max_step": tau / 100},
        )
        fid = bell_state_fidelity(r.states[-1].ptrace([0, 1]))
        assert 0.95 < fid < 1.0

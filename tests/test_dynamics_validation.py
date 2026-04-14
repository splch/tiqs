r"""Dynamical validation tests: simulation output vs analytical predictions.

These tests verify that the time-evolution produced by QuTiP with
TIQS-constructed Hamiltonians and Lindblad operators matches known
analytical solutions for decoherence rates, cooling dynamics,
Lamb-Dicke corrections, and gate protocols.

Every assertion is derived from a textbook formula or published
experimental result cited in-line.

References
----------
[Leibfried2003]  Leibfried et al., RMP 75, 281 (2003).
[Benhelm2008]  Benhelm et al., Nature Physics 4, 463 (2008).
[Wineland1998]  Wineland et al., J. Res. NIST 103, 259 (1998).
"""

import numpy as np
import pytest
import qutip

from tiqs.analysis.fidelity import bell_state_fidelity
from tiqs.constants import PI, TWO_PI
from tiqs.cooling.sideband_cooling import (
    sideband_cooling_nbar,
    sideband_cooling_simulate,
)
from tiqs.gates.microwave_ms import (
    microwave_ms_gate_hamiltonian,
    microwave_ms_gate_hamiltonian_full,
)
from tiqs.gates.molmer_sorensen import ms_gate_duration, ms_gate_hamiltonian
from tiqs.hilbert_space.builder import HilbertSpace
from tiqs.hilbert_space.operators import OperatorFactory
from tiqs.hilbert_space.states import StateFactory
from tiqs.interaction.hamiltonian import (
    full_interaction_hamiltonian,
    red_sideband_hamiltonian,
)
from tiqs.noise.motional import motional_heating_ops
from tiqs.noise.qubit import qubit_dephasing_op, spontaneous_emission_op
from tiqs.species.ion import get_species
from tiqs.trap import PaulTrap


class TestT2DephasingDynamics:
    """[Leibfried2003] Sec. V.A: pure dephasing causes exponential
    decay of the off-diagonal density matrix element at rate 1/T2."""

    def test_off_diagonal_decays_as_exp_minus_t_over_t2(self):
        """Prepare |+> = (|0> + |1>)/sqrt(2), evolve under pure
        dephasing, measure <sigma_x>(t).  Should decay as exp(-t/T2)."""
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
        # |+> has <sz> = 0 throughout
        np.testing.assert_allclose(result.expect[0], 0.0, atol=0.02)


class TestT1DecayDynamics:
    r"""[Leibfried2003] Sec. V.A: spontaneous emission from |1> to |0>
    at rate 1/T1.  sigma_z decays from -1 toward +1 as
    $\langle\sigma_z\rangle(t) = 1 - 2\,e^{-t/T_1}$."""

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

        # |1> has <sz> = -1 initially, decays to +1 (ground)
        expected = 1 - 2 * np.exp(-tlist / T1)
        np.testing.assert_allclose(result.expect[0], expected, atol=0.05)


class TestMotionalHeatingRate:
    """[Leibfried2003] Eq. 33: motional heating adds phonons at a
    constant rate: d<n>/dt = heating_rate."""

    def test_phonon_growth_matches_rate_equation(self):
        """For L = sqrt(gamma) * a_dag starting from vacuum,
        <n>(t) = exp(gamma*t) - 1."""
        hs = HilbertSpace(n_ions=1, n_modes=1, n_fock=30)
        ops = OperatorFactory(hs)
        sf = StateFactory(hs)

        heating_rate = 1e4
        c_ops = motional_heating_ops(ops, 0, heating_rate)
        psi0 = sf.ground_state()
        n_op = ops.number(0)
        # Keep t short enough that <n> stays well below Fock cutoff
        tlist = np.linspace(0, 2e-4, 100)
        result = qutip.mesolve(
            0 * ops.identity(),
            psi0,
            tlist,
            c_ops=c_ops,
            e_ops=[n_op],
        )

        expected = np.exp(heating_rate * tlist) - 1
        # Only compare early times where <n> < 5 (Fock truncation)
        mask = expected < 5
        np.testing.assert_allclose(
            result.expect[0][mask], expected[mask], rtol=0.1
        )


class TestRSBRabiFrequencyScaling:
    r"""[Leibfried2003] Eq. 12: red sideband Rabi frequency scales as
    $\Omega_{n,n-1} = \eta\,\Omega\,\sqrt{n}$.  The pi-pulse time
    for |0,n> -> |1,n-1> is $t_\pi = \pi / (\eta\,\Omega\,\sqrt{n})$."""

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


class TestDebyeWallerCorrection:
    r"""[Leibfried2003] Eq. 8: the carrier Rabi frequency is reduced
    by the Debye-Waller factor:
    $\Omega_n \approx \Omega\,[1 - \eta^2(2n+1)/2]$.

    For a thermal state with mean phonon number n_bar, the effective
    carrier frequency is reduced by $\eta^2(2\bar{n}+1)/2$."""

    def test_second_order_carrier_slower_than_first_order(self):
        """With lamb_dicke_order=2 and n_bar > 0, the carrier Rabi
        oscillation should be slower (longer pi-pulse time) than
        with lamb_dicke_order=1."""
        hs = HilbertSpace(n_ions=1, n_modes=1, n_fock=15)
        ops = OperatorFactory(hs)

        eta = 0.15
        Omega = TWO_PI * 200e3
        omega_m = TWO_PI * 1e6
        n_bar = 5

        # First-order Hamiltonian (on resonance, detuning=0)
        H1 = full_interaction_hamiltonian(
            ops, 0, 0, Omega, eta, 0.0, omega_m, lamb_dicke_order=1
        )
        # Second-order Hamiltonian
        H2 = full_interaction_hamiltonian(
            ops, 0, 0, Omega, eta, 0.0, omega_m, lamb_dicke_order=2
        )

        # Thermal initial state
        parts = [
            qutip.ket2dm(qutip.basis(2, 0)),
            qutip.thermal_dm(15, n_bar),
        ]
        rho0 = qutip.tensor(parts)

        t_pi_bare = PI / Omega
        tlist = np.linspace(0, 2 * t_pi_bare, 300)
        sz = ops.sigma_z(0)

        r1 = qutip.mesolve(H1, rho0, tlist, e_ops=[sz])
        r2 = qutip.mesolve(H2, rho0, tlist, e_ops=[sz])

        # With second-order correction, <sigma_z> should reach its
        # minimum later (slower effective Rabi frequency)
        idx_min_1 = np.argmin(r1.expect[0])
        idx_min_2 = np.argmin(r2.expect[0])
        assert idx_min_2 > idx_min_1


class TestSidebandCoolingConvergence:
    """[Wineland1998]: resolved sideband cooling converges to
    nbar_final ~ (gamma_eff / 2*omega_trap)^2."""

    def test_simulated_nbar_approaches_analytical(self):
        """The simulated final nbar should be within a factor of
        5 of the analytical prediction."""
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

        # Simulated should be much less than initial
        assert nbar_sim < 1.0
        # And should be in the same ballpark as analytical
        # (within a factor of 5, since the formula is approximate)
        assert nbar_sim < 5 * nbar_analytical + 0.1


class TestDressedFrameMSConvergence:
    r"""The RWA MS gate (dressed-frame approximation) should produce
    entanglement when the dressing hierarchy is satisfied.

    The full Hamiltonian test is omitted because resolving the
    fast dressing oscillations (period ~ 1/Omega_dress ~ 100 ns)
    over the gate duration (~ 100 us) requires prohibitively fine
    time steps for a unit test."""

    def test_rwa_ms_gate_entangles(self):
        """The dressed-frame MS gate should produce a Bell state."""
        hs = HilbertSpace(n_ions=2, n_modes=1, n_fock=15)
        ops = OperatorFactory(hs)
        sf = StateFactory(hs)

        eta_val = 0.05
        eta = [eta_val, eta_val]
        delta = TWO_PI * 15e3
        Omega_gate = delta / (4 * eta_val)
        tau = ms_gate_duration(delta)

        H_rwa = microwave_ms_gate_hamiltonian(
            ops,
            [0, 1],
            0,
            eta,
            gate_rabi_frequency=Omega_gate,
            detuning=delta,
            dressing_rabi_frequency=TWO_PI * 10e6,
        )
        r = qutip.sesolve(
            H_rwa,
            sf.ground_state(),
            np.linspace(0, tau, 500),
            options={"max_step": tau / 100},
        )
        fid = bell_state_fidelity(r.states[-1].ptrace([0, 1]))
        assert fid > 0.95

    def test_full_hamiltonian_has_more_terms_than_rwa(self):
        """The full Hamiltonian includes static dressing terms not
        present in the RWA version."""
        hs = HilbertSpace(n_ions=2, n_modes=1, n_fock=10)
        ops = OperatorFactory(hs)

        eta = [0.05, 0.05]
        H_rwa = microwave_ms_gate_hamiltonian(
            ops,
            [0, 1],
            0,
            eta,
            gate_rabi_frequency=1e5,
            detuning=TWO_PI * 10e3,
            dressing_rabi_frequency=TWO_PI * 10e6,
        )
        H_full = microwave_ms_gate_hamiltonian_full(
            ops,
            [0, 1],
            0,
            eta,
            gate_rabi_frequency=1e5,
            detuning=TWO_PI * 10e3,
            dressing_rabi_frequency=TWO_PI * 10e6,
        )
        assert len(H_full) > len(H_rwa)


class TestBenhelm2008CaParameters:
    """Reproduce the ideal (noiseless) MS gate at Benhelm et al.
    (2008) Ca-40 parameters, then verify noise degrades fidelity.

    [Benhelm2008]: Ca-40 optical qubit, axial COM ~ 1.2 MHz,
    eta ~ 0.05, reported 99.3% Bell fidelity including all noise.
    Our noiseless simulation should give ~100%."""

    @pytest.fixture
    def benhelm_system(self):
        """Approximate Benhelm 2008 parameters."""
        ca = get_species("Ca40")
        trap = PaulTrap(
            v_rf=300,
            omega_rf=TWO_PI * 30e6,
            r0=0.5e-3,
            omega_axial=TWO_PI * 1.2e6,
            species=ca,
        )
        hs = HilbertSpace(n_ions=2, n_modes=1, n_fock=20)
        ops = OperatorFactory(hs)
        sf = StateFactory(hs)
        return trap, hs, ops, sf

    def test_noiseless_bell_fidelity_near_unity(self, benhelm_system):
        """Ideal MS gate at Ca-40 parameters should give F > 0.99."""
        _trap, _hs, ops, sf = benhelm_system

        eta = 0.05
        delta = TWO_PI * 20e3
        Omega = delta / (4 * eta)
        tau = ms_gate_duration(delta)

        H = ms_gate_hamiltonian(ops, [0, 1], 0, [eta, eta], Omega, delta)
        result = qutip.sesolve(
            H,
            sf.ground_state(),
            np.linspace(0, tau, 800),
            options={"max_step": tau / 200},
        )
        fid = bell_state_fidelity(result.states[-1].ptrace([0, 1]))
        assert fid > 0.99

    def test_heating_degrades_fidelity(self, benhelm_system):
        """Adding motional heating should reduce Bell fidelity
        below the noiseless value."""
        _trap, _hs, ops, sf = benhelm_system

        eta = 0.05
        delta = TWO_PI * 20e3
        Omega = delta / (4 * eta)
        tau = ms_gate_duration(delta)
        H = ms_gate_hamiltonian(ops, [0, 1], 0, [eta, eta], Omega, delta)
        tlist = np.linspace(0, tau, 500)

        # Noiseless
        r_clean = qutip.sesolve(
            H,
            sf.ground_state(),
            tlist,
            options={"max_step": tau / 100},
        )
        fid_clean = bell_state_fidelity(r_clean.states[-1].ptrace([0, 1]))

        # With heating (typical lab: ~100 quanta/s at 50 um height)
        c_ops = motional_heating_ops(ops, 0, heating_rate=1e5)
        r_noisy = qutip.mesolve(
            H,
            sf.ground_state(),
            tlist,
            c_ops=c_ops,
            options={"max_step": tau / 100},
        )
        fid_noisy = bell_state_fidelity(r_noisy.states[-1].ptrace([0, 1]))

        assert fid_noisy < fid_clean

    def test_dephasing_degrades_fidelity(self, benhelm_system):
        """Adding qubit dephasing should reduce fidelity."""
        _trap, _hs, ops, sf = benhelm_system

        eta = 0.05
        delta = TWO_PI * 20e3
        Omega = delta / (4 * eta)
        tau = ms_gate_duration(delta)
        H = ms_gate_hamiltonian(ops, [0, 1], 0, [eta, eta], Omega, delta)
        tlist = np.linspace(0, tau, 500)

        # Noiseless
        r_clean = qutip.sesolve(
            H,
            sf.ground_state(),
            tlist,
            options={"max_step": tau / 100},
        )
        fid_clean = bell_state_fidelity(r_clean.states[-1].ptrace([0, 1]))

        # With T2 = 1 ms (optical qubit, Ca-40 D5/2 ~ 1.2 s,
        # but laser linewidth limits T2 to ~ ms scale)
        c_ops = [
            qubit_dephasing_op(ops, 0, t2=1e-3),
            qubit_dephasing_op(ops, 1, t2=1e-3),
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


class TestPenningMultiIonRadialModes:
    """Verify that Penning trap multi-ion radial mode computation
    produces the correct number of modes in each family."""

    def test_two_ion_has_two_cyclotron_and_two_magnetron(self):
        """2 ions should have 2 cyclotron + 2 magnetron modes."""
        from tiqs.chain.normal_modes import normal_modes
        from tiqs.trap import PenningTrap

        trap = PenningTrap(
            magnetic_field=3.0,
            species=get_species("Be9"),
            d=300e-6,
            omega_axial=TWO_PI * 1.5e6,
        )
        modes = normal_modes(2, trap)
        assert len(modes.cyclotron_freqs) == 2
        assert len(modes.magnetron_freqs) == 2

    def test_cyclotron_above_magnetron_for_multi_ion(self):
        """All cyclotron frequencies > all magnetron frequencies."""
        from tiqs.chain.normal_modes import normal_modes
        from tiqs.trap import PenningTrap

        trap = PenningTrap(
            magnetic_field=3.0,
            species=get_species("Be9"),
            d=300e-6,
            omega_axial=TWO_PI * 1.5e6,
        )
        modes = normal_modes(2, trap)
        assert min(modes.cyclotron_freqs) > max(modes.magnetron_freqs)

    def test_two_ion_axial_com_matches_single_ion(self):
        """The axial COM mode for 2 ions should equal omega_z,
        regardless of the radial mode structure."""
        from tiqs.chain.normal_modes import normal_modes
        from tiqs.trap import PenningTrap

        trap = PenningTrap(
            magnetic_field=3.0,
            species=get_species("Be9"),
            d=300e-6,
            omega_axial=TWO_PI * 1.5e6,
        )
        modes = normal_modes(2, trap)
        assert modes.axial_freqs[0] == pytest.approx(
            trap.omega_axial, rel=1e-4
        )

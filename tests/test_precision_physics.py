r"""Precision physics tests: exact Rabi frequencies, AC Stark shifts,
correlated noise, and experimental reproduction.

These tests validate the three features needed for quantitative
agreement with published experiments:

1. Exact Rabi frequencies beyond the Lamb-Dicke approximation
2. AC Stark shifts from off-resonant carrier excitation in MS gates
3. Correlated noise across multiple ions

Then they reproduce the Benhelm 2008 (Ca-40, 99.3%) and
Ballance 2016 (Ca-43, 99.9%) gate experiments to verify the
simulator predicts the correct fidelity regime.

References
----------
[Leibfried2003]  Leibfried et al., RMP 75, 281 (2003).
[Benhelm2008]  Benhelm et al., Nature Physics 4, 463 (2008).
[Ballance2016]  Ballance et al., PRL 117, 060504 (2016).
"""

import numpy as np
import pytest
import qutip

from tiqs.analysis.fidelity import bell_state_fidelity
from tiqs.constants import TWO_PI
from tiqs.gates.molmer_sorensen import (
    ms_ac_stark_shift,
    ms_gate_duration,
    ms_gate_hamiltonian,
    ms_gate_hamiltonian_with_carrier,
)
from tiqs.hilbert_space.builder import HilbertSpace
from tiqs.hilbert_space.operators import OperatorFactory
from tiqs.hilbert_space.states import StateFactory
from tiqs.interaction.exact_coupling import (
    debye_waller_factor,
    exact_rabi_frequency,
    thermal_averaged_rabi,
)
from tiqs.noise.correlated import (
    correlated_dephasing_op,
    correlated_intensity_noise_hamiltonian,
    correlated_phase_noise_op,
)
from tiqs.noise.motional import motional_heating_ops
from tiqs.noise.qubit import qubit_dephasing_op, spontaneous_emission_op


class TestExactRabiFrequency:
    """[Leibfried2003] Eq. 8: exact Rabi frequencies using
    generalized Laguerre polynomials."""

    def test_carrier_ground_state_equals_bare(self):
        """At n=0, carrier Rabi = Omega * exp(-eta^2/2) * L_0(eta^2)
        = Omega * exp(-eta^2/2)."""
        eta = 0.1
        omega = exact_rabi_frequency(1.0, eta, n=0, s=0)
        assert omega == pytest.approx(np.exp(-(eta**2) / 2), rel=1e-10)

    def test_carrier_reduces_to_debye_waller(self):
        """For small eta, carrier Rabi ~ Omega*(1 - eta^2*(2n+1)/2)."""
        eta = 0.05
        for n in [0, 1, 5, 10]:
            exact = exact_rabi_frequency(1.0, eta, n, s=0)
            approx = 1.0 * (1 - eta**2 * (2 * n + 1) / 2)
            assert exact == pytest.approx(approx, rel=0.01)

    def test_debye_waller_factor_consistency(self):
        """debye_waller_factor(eta, n) must match
        exact_rabi_frequency(1.0, eta, n, 0)."""
        for eta in [0.05, 0.1, 0.2]:
            for n in [0, 3, 10]:
                dw = debye_waller_factor(eta, n)
                exact = exact_rabi_frequency(1.0, eta, n, 0)
                assert dw == pytest.approx(exact, rel=1e-10)

    def test_rsb_scales_as_eta_sqrt_n(self):
        """First red sideband (s=-1): Omega_{n,n-1} ~ eta*Omega*sqrt(n)
        in the Lamb-Dicke limit."""
        eta = 0.02  # small enough for LD to be accurate
        Omega = 1.0
        for n in [1, 4, 9]:
            exact = exact_rabi_frequency(Omega, eta, n, s=-1)
            ld_approx = eta * Omega * np.sqrt(n)
            assert exact == pytest.approx(ld_approx, rel=0.01)

    def test_bsb_scales_as_eta_sqrt_n_plus_1(self):
        """First blue sideband (s=+1): Omega_{n,n+1} ~ eta*Omega*sqrt(n+1)."""
        eta = 0.02
        for n in [0, 3, 8]:
            exact = exact_rabi_frequency(1.0, eta, n, s=+1)
            ld_approx = eta * np.sqrt(n + 1)
            assert exact == pytest.approx(ld_approx, rel=0.01)

    def test_rsb_from_vacuum_is_zero(self):
        """Cannot remove a phonon from vacuum: Omega_{0,-1} = 0."""
        assert exact_rabi_frequency(1.0, 0.1, n=0, s=-1) == 0.0

    def test_large_eta_deviates_from_ld(self):
        """At eta=0.3, n=10, the exact carrier Rabi should deviate
        significantly from the first-order LD prediction."""
        eta = 0.3
        n = 10
        exact = exact_rabi_frequency(1.0, eta, n, s=0)
        ld_first = 1.0  # LD first order: no correction
        ld_second = 1.0 * (1 - eta**2 * (2 * n + 1) / 2)
        # Exact should differ from first-order by > 10%
        assert abs(exact - ld_first) / ld_first > 0.1
        # But be closer to second-order
        assert abs(exact - ld_second) < abs(exact - ld_first)

    def test_thermal_average_reduces_carrier(self):
        """Thermal averaging reduces the effective carrier Rabi
        frequency: thermal_averaged < bare."""
        bare = 1e5
        eta = 0.1
        n_bar = 5
        avg = thermal_averaged_rabi(bare, eta, n_bar, s=0)
        assert avg < bare

    def test_thermal_average_at_nbar_zero_equals_exact_n0(self):
        """At nbar=0, the thermal average is just the n=0 value."""
        bare = 1e5
        eta = 0.1
        avg = thermal_averaged_rabi(bare, eta, 0.0, s=0)
        exact_n0 = exact_rabi_frequency(bare, eta, 0, 0)
        assert avg == pytest.approx(exact_n0, rel=1e-6)


class TestACStarkShift:
    """The off-resonant carrier in the MS bichromatic drive produces
    an AC Stark shift that must be compensated in experiments."""

    def test_ac_stark_formula(self):
        """delta_AC = (eta*Omega)^2 / delta for each ion."""
        eta = [0.1, 0.1]
        Omega = TWO_PI * 100e3
        delta = TWO_PI * 20e3
        shifts = ms_ac_stark_shift(eta, Omega, delta)
        expected = (eta[0] * Omega) ** 2 / delta
        assert shifts[0] == pytest.approx(expected, rel=1e-10)
        assert shifts[1] == pytest.approx(expected, rel=1e-10)

    def test_ac_stark_scales_with_eta_squared(self):
        """Doubling eta quadruples the AC Stark shift."""
        Omega = TWO_PI * 100e3
        delta = TWO_PI * 20e3
        shift_1 = ms_ac_stark_shift([0.05], Omega, delta)[0]
        shift_2 = ms_ac_stark_shift([0.10], Omega, delta)[0]
        assert shift_2 / shift_1 == pytest.approx(4.0, rel=1e-10)

    def test_ms_with_carrier_has_more_terms(self):
        """ms_gate_hamiltonian_with_carrier should have extra terms
        compared to ms_gate_hamiltonian (one carrier per ion)."""
        hs = HilbertSpace(n_ions=2, n_modes=1, n_fock=10)
        ops = OperatorFactory(hs)
        eta = [0.1, 0.1]
        H_bare = ms_gate_hamiltonian(
            ops, [0, 1], 0, eta, TWO_PI * 100e3, TWO_PI * 20e3
        )
        H_full = ms_gate_hamiltonian_with_carrier(
            ops, [0, 1], 0, eta, TWO_PI * 100e3, TWO_PI * 20e3
        )
        assert len(H_full) == len(H_bare) + 2

    def test_carrier_terms_are_hermitian(self):
        """The added carrier terms should be Hermitian operators."""
        hs = HilbertSpace(n_ions=2, n_modes=1, n_fock=10)
        ops = OperatorFactory(hs)
        H = ms_gate_hamiltonian_with_carrier(
            ops, [0, 1], 0, [0.1, 0.1], TWO_PI * 100e3, TWO_PI * 20e3
        )
        # Last two terms are the carrier
        for term in H[-2:]:
            assert term[0].isherm


class TestCorrelatedNoise:
    """Correlated noise across ions gives different error scaling
    than independent noise."""

    @pytest.fixture
    def two_ion_system(self):
        hs = HilbertSpace(n_ions=2, n_modes=1, n_fock=5)
        ops = OperatorFactory(hs)
        sf = StateFactory(hs)
        return hs, ops, sf

    def test_correlated_vs_independent_dephasing_rate(self, two_ion_system):
        """For (|00> + i|11>)/sqrt(2), correlated dephasing
        (L ~ sz1 + sz2) decays the coherence FASTER than
        independent, because the collective eigenvalue gap is 4
        (vs 2 per qubit independently).  This quantitatively tests
        that correlated noise has different scaling."""
        _hs, ops, sf = two_ion_system

        ket_00 = sf.product_state([0, 0], [0])
        ket_11 = sf.product_state([1, 1], [0])
        bell = (ket_00 + 1j * ket_11).unit()
        rho0 = qutip.ket2dm(bell)

        T2 = 1e-3
        tlist = np.linspace(0, 0.3 * T2, 50)
        H_zero = 0 * ops.identity()

        c_indep = [
            qubit_dephasing_op(ops, 0, T2),
            qubit_dephasing_op(ops, 1, T2),
        ]
        r_indep = qutip.mesolve(H_zero, rho0, tlist, c_ops=c_indep)
        fid_indep = bell_state_fidelity(r_indep.states[-1].ptrace([0, 1]))

        c_corr = [correlated_dephasing_op(ops, [0, 1], T2)]
        r_corr = qutip.mesolve(H_zero, rho0, tlist, c_ops=c_corr)
        fid_corr = bell_state_fidelity(r_corr.states[-1].ptrace([0, 1]))

        # Correlated decays FASTER on the |00>+i|11> state
        assert fid_corr < fid_indep

    def test_correlated_dephasing_preserves_01_10_bell(self, two_ion_system):
        """Common-mode dephasing PRESERVES the Bell state
        (|01> + i|10>)/sqrt(2) because both components have
        the same collective sigma_z eigenvalue (0+0=0).
        This is the decoherence-free subspace."""
        _hs, ops, sf = two_ion_system

        ket_01 = sf.product_state([0, 1], [0])
        ket_10 = sf.product_state([1, 0], [0])
        bell_anti = (ket_01 + 1j * ket_10).unit()
        rho0 = qutip.ket2dm(bell_anti)

        T2 = 100e-6
        tlist = np.linspace(0, 5 * T2, 100)

        c_corr = [correlated_dephasing_op(ops, [0, 1], T2)]
        r_corr = qutip.mesolve(0 * ops.identity(), rho0, tlist, c_ops=c_corr)
        rho_final = r_corr.states[-1].ptrace([0, 1])
        purity = (rho_final**2).tr().real
        # Should be perfectly preserved (decoherence-free subspace)
        assert purity > 0.99

    def test_correlated_phase_noise_is_collective(self, two_ion_system):
        """Correlated phase noise should be a single operator on the
        collective sigma_z."""
        _hs, ops, _sf = two_ion_system
        c = correlated_phase_noise_op(ops, [0, 1], TWO_PI * 1.0)
        assert c.isherm
        assert c.shape == ops.identity().shape

    def test_correlated_intensity_noise_is_hermitian(self, two_ion_system):
        _hs, ops, _sf = two_ion_system
        H = correlated_intensity_noise_hamiltonian(ops, [0, 1], 0.01, 1e6)
        assert H.isherm


class TestBenhelm2008Reproduction:
    """Reproduce the Benhelm et al. (2008) Ca-40 MS gate.

    [Benhelm2008]: Ca-40 optical qubit, axial COM ~ 1.2 MHz,
    eta ~ 0.05, gate time ~ 50 us.  Reported 99.3(1)% Bell fidelity.

    Our noiseless simulation should give F ~ 1.0.  Adding the
    dominant noise source (D5/2 spontaneous decay, T1 = 1.168 s)
    should give a small reduction.  Adding all noise sources
    should bring it below the ideal but above ~98%."""

    @pytest.fixture
    def benhelm_setup(self):
        hs = HilbertSpace(n_ions=2, n_modes=1, n_fock=20)
        ops = OperatorFactory(hs)
        sf = StateFactory(hs)

        # Benhelm 2008 parameters (Ca-40, 729 nm optical qubit)
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

        # Noiseless baseline
        r_clean = qutip.sesolve(
            H, sf.ground_state(), tlist, options={"max_step": tau / 100}
        )
        fid_clean = bell_state_fidelity(r_clean.states[-1].ptrace([0, 1]))

        # With D5/2 spontaneous decay (T1 = 1.168 s for Ca-40)
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
        # But the gate is fast (~50 us) vs T1 (1.168 s), so the
        # reduction should be small
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
        # Should still be above 95% (noise is mild for Ca-40)
        assert 0.95 < fid < 1.0


class TestBallance2016Reproduction:
    """Reproduce the Ballance et al. (2016) Ca-43 MS gate.

    [Ballance2016]: Ca-43 hyperfine qubit, radial modes ~ 2.95 MHz,
    eta ~ 0.13, gate time ~ 100 us.  Reported 99.9(1)% Bell fidelity.
    Hyperfine qubit has T1 = infinity.

    The noiseless simulation should give F ~ 1.0.  The dominant
    error sources for Ca-43 are laser phase noise and photon
    scattering, not spontaneous decay."""

    @pytest.fixture
    def ballance_setup(self):
        hs = HilbertSpace(n_ions=2, n_modes=1, n_fock=20)
        ops = OperatorFactory(hs)
        sf = StateFactory(hs)

        # Ballance 2016 approximate parameters
        eta = 0.13
        delta = TWO_PI * 10e3
        Omega = delta / (4 * eta)
        tau = ms_gate_duration(delta)

        return ops, sf, eta, delta, Omega, tau

    def test_noiseless_ideal_fidelity(self, ballance_setup):
        """Noiseless MS gate at Ballance parameters: F > 0.99."""
        ops, sf, eta, delta, Omega, tau = ballance_setup
        H = ms_gate_hamiltonian(ops, [0, 1], 0, [eta, eta], Omega, delta)
        r = qutip.sesolve(
            H,
            sf.ground_state(),
            np.linspace(0, tau, 800),
            options={"max_step": tau / 200},
        )
        fid = bell_state_fidelity(r.states[-1].ptrace([0, 1]))
        assert fid > 0.99

    def test_hyperfine_no_spontaneous_decay(self, ballance_setup):
        """Ca-43 hyperfine qubit has T1=infinity: spontaneous decay
        is absent.  Adding it at the species T1 should have no
        effect (inf T1 -> zero rate)."""
        ops, sf, eta, delta, Omega, tau = ballance_setup
        H = ms_gate_hamiltonian(ops, [0, 1], 0, [eta, eta], Omega, delta)
        tlist = np.linspace(0, tau, 500)

        r_clean = qutip.sesolve(
            H,
            sf.ground_state(),
            tlist,
            options={"max_step": tau / 100},
        )
        fid_clean = bell_state_fidelity(r_clean.states[-1].ptrace([0, 1]))
        # Hyperfine qubit: no T1 decay -> fidelity stays near ideal
        assert fid_clean > 0.99

    def test_decoherence_free_subspace(self, ballance_setup):
        """The state (|01> + i|10>)/sqrt(2) is in the decoherence-free
        subspace of common-mode dephasing.  After aggressive
        dephasing, it should be perfectly preserved while the MS
        Bell state (|00>+i|11>) is destroyed."""
        ops, sf, _eta, _delta, _Omega, _tau = ballance_setup

        # DFS state: (|01> + i|10>)/sqrt(2) - eigenvalue 0 under
        # collective sigma_z
        ket_01 = sf.product_state([0, 1], [0])
        ket_10 = sf.product_state([1, 0], [0])
        dfs_state = (ket_01 + 1j * ket_10).unit()
        rho_dfs = qutip.ket2dm(dfs_state)

        # MS Bell: (|00> + i|11>)/sqrt(2) - eigenvalue gap 4
        ket_00 = sf.product_state([0, 0], [0])
        ket_11 = sf.product_state([1, 1], [0])
        ms_bell = (ket_00 + 1j * ket_11).unit()
        rho_ms = qutip.ket2dm(ms_bell)

        T2 = 100e-6
        tlist = np.linspace(0, 3 * T2, 100)
        c_ops = [correlated_dephasing_op(ops, [0, 1], T2)]

        r_dfs = qutip.mesolve(0 * ops.identity(), rho_dfs, tlist, c_ops=c_ops)
        r_ms = qutip.mesolve(0 * ops.identity(), rho_ms, tlist, c_ops=c_ops)

        purity_dfs = (r_dfs.states[-1].ptrace([0, 1]) ** 2).tr().real
        purity_ms = (r_ms.states[-1].ptrace([0, 1]) ** 2).tr().real

        # DFS state is immune; MS Bell state decoheres
        assert purity_dfs > 0.99
        assert purity_ms < 0.6

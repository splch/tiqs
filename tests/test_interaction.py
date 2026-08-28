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

N_FOCK_DYN = 8


@pytest.fixture
def simple_system():
    """One ion, one motional mode, Fock cutoff 15."""
    hs = HilbertSpace(n_ions=1, n_modes=1, n_fock=15)
    ops = OperatorFactory(hs)
    return hs, ops


@pytest.fixture
def dyn_system():
    """One ion, one mode, Fock cutoff 8 (cheap time-dependent solves)."""
    hs = HilbertSpace(n_ions=1, n_modes=1, n_fock=N_FOCK_DYN)
    ops = OperatorFactory(hs)
    return hs, ops


def _ket(qubit: int, n: int, n_fock: int = N_FOCK_DYN) -> qutip.Qobj:
    """Basis state $|qubit, n\\rangle$."""
    return qutip.tensor(qutip.basis(2, qubit), qutip.basis(n_fock, n))


def _evolve(H, psi0, duration, n_points=150):
    """Solve the Schrodinger equation and return every sampled state."""
    return qutip.sesolve(
        H,
        psi0,
        np.linspace(0, duration, n_points),
        options={"atol": 1e-10, "rtol": 1e-8},
    ).states


def _max_population(states, target):
    """Largest population of ``target`` over a list of states."""
    return max(abs(state.overlap(target)) ** 2 for state in states)


def _exact_interaction_state(
    ops,
    psi0,
    rabi_frequency,
    eta,
    detuning,
    mode_frequency,
    duration,
    phase=0.0,
    n_steps=2000,
):
    r"""Evolve under the *untruncated* interaction-picture Hamiltonian.

    Independent reference for `full_interaction_hamiltonian`: no
    Lamb-Dicke expansion and no shared ODE solver. The generator is
    Leibfried RMP 75, 281 (2003) Eq. (69) verbatim,

    $$
    H(t) = \frac{\Omega}{2}\sigma_- e^{i\phi} e^{-i\delta t}
        D\bigl(i\eta e^{i\omega_m t}\bigr) + \text{h.c.},
    $$

    where the displacement operator is generated from
    $D(i\eta) = \exp[i\eta(a + a^\dagger)]$ by the free-mode rotation
    $D(i\eta e^{i\omega_m t}) = e^{i\omega_m t n} D(i\eta)
    e^{-i\omega_m t n}$. The propagator is the midpoint product
    $\prod_k \exp[-i\,H(t_k + \Delta t/2)\,\Delta t]$, which converges
    as $\Delta t^2$ (checked: 1e-6 state error at ``n_steps=2000``
    for the parameters used here).
    """
    sm = ops.sigma_minus(0)
    a = ops.annihilate(0)
    ad = ops.create(0)
    n_op = ops.number(0)
    occupation = np.real(np.diag(n_op.full()))
    displace = (1j * eta * (a + ad)).expm()
    dt = duration / n_steps
    psi = psi0
    for step in range(n_steps):
        t = (step + 0.5) * dt
        rotation = qutip.Qobj(
            np.diag(np.exp(1j * mode_frequency * t * occupation)),
            dims=n_op.dims,
        )
        term = (
            (rabi_frequency / 2)
            * np.exp(1j * phase)
            * np.exp(-1j * detuning * t)
            * sm
            * (rotation * displace * rotation.dag())
        )
        psi = (-1j * dt * (term + term.dag())).expm() * psi
    return psi


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

    @pytest.mark.parametrize("phase", [0.0, 0.4, np.pi / 2, 2.3])
    def test_phase_is_rotation_axis_azimuth(self, dyn_system, phase):
        """$H = (\\Omega/2)(\\sigma_x\\cos\\phi + \\sigma_y\\sin\\phi)$.

        A drive whose azimuth is $\\phi$ must generate a rotation about
        $\\hat{n} = (\\cos\\phi, \\sin\\phi, 0)$; mirroring the phase
        (attaching $e^{+i\\phi}$ to the de-excitation operator instead)
        flips the axis to $-\\phi$ and is what this pins down.
        """
        hs, ops = dyn_system
        H = carrier_hamiltonian(ops, ion=0, rabi_frequency=2.0, phase=phase)
        expected = ops.sigma_x(0) * np.cos(phase) + ops.sigma_y(0) * np.sin(
            phase
        )
        assert (H - expected).norm() < 1e-12

    def test_pi_over_two_pulse_rotates_about_plus_y(self, dyn_system):
        """Right-hand rule: a pi/2 rotation about +y sends +z to +x."""
        hs, ops = dyn_system
        Omega = TWO_PI * 100e3
        H = carrier_hamiltonian(ops, 0, Omega, phase=np.pi / 2)
        final = _evolve(H, _ket(0, 0), np.pi / (2 * Omega), 2)[-1]
        assert qutip.expect(ops.sigma_x(0), final) == pytest.approx(
            1.0, abs=1e-3
        )
        assert qutip.expect(ops.sigma_y(0), final) == pytest.approx(
            0.0, abs=1e-3
        )


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

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_rsb_rabi_frequency_scales_as_sqrt_n(self, dyn_system, n):
        """Jaynes-Cummings law: the |0,n> -> |1,n-1> Rabi frequency is
        $\\eta\\Omega\\sqrt{n}$ (Leibfried RMP Eq. (75)), so the
        pi-time shrinks as $1/\\sqrt{n}$.
        """
        hs, ops = dyn_system
        eta = 0.05
        Omega = TWO_PI * 100e3
        H = red_sideband_hamiltonian(ops, 0, 0, Omega, eta)
        t_pi = np.pi / (eta * Omega * np.sqrt(n))
        final = _evolve(H, _ket(0, n), t_pi, 2)[-1]
        assert abs(final.overlap(_ket(1, n - 1))) ** 2 == pytest.approx(
            1.0, abs=1e-3
        )


class TestFullInteractionHamiltonian:
    @pytest.mark.parametrize(
        "detuning, order, n_entries",
        [
            (TWO_PI * 1e5, 1, 6),
            (TWO_PI * 1e5, 2, 12),
            # On the carrier resonance the carrier and Debye-Waller
            # pairs are static, so each collapses to one entry.
            (0.0, 1, 5),
            (0.0, 2, 10),
        ],
    )
    def test_term_structure(self, dyn_system, detuning, order, n_entries):
        """One conjugate pair per expansion term: carrier plus two
        sidebands at first order, plus Debye-Waller and both second
        sidebands at second order.
        """
        hs, ops = dyn_system
        H = full_interaction_hamiltonian(
            ops,
            ion=0,
            mode=0,
            rabi_frequency=TWO_PI * 100e3,
            eta=0.1,
            detuning=detuning,
            mode_frequency=TWO_PI * 1e6,
            phase=0.3,
            lamb_dicke_order=order,
        )
        assert len(H) == n_entries

    @pytest.mark.parametrize("t", [0.0, 1.3e-7, 4.7e-7])
    def test_hamiltonian_is_hermitian_at_all_times(self, dyn_system, t):
        """Each term is paired with its h.c. and conjugate coefficient,
        so the summed generator is Hermitian at every instant.
        """
        hs, ops = dyn_system
        H = qutip.QobjEvo(
            full_interaction_hamiltonian(
                ops,
                ion=0,
                mode=0,
                rabi_frequency=TWO_PI * 100e3,
                eta=0.1,
                detuning=TWO_PI * 1e5,
                mode_frequency=TWO_PI * 1e6,
                phase=0.4,
                lamb_dicke_order=2,
            )
        )
        assert (H(t) - H(t).dag()).norm() < 1e-9

    @pytest.mark.parametrize("order", [0, 3, -1])
    def test_invalid_lamb_dicke_order_raises(self, dyn_system, order):
        hs, ops = dyn_system
        with pytest.raises(ValueError, match="lamb_dicke_order"):
            full_interaction_hamiltonian(
                ops,
                ion=0,
                mode=0,
                rabi_frequency=1.0,
                eta=0.1,
                detuning=0.0,
                mode_frequency=TWO_PI * 1e6,
                lamb_dicke_order=order,
            )

    def test_carrier_term_matches_carrier_hamiltonian(self, dyn_system):
        """At delta = 0 and eta = 0 the only term left is the carrier,
        which must be the standalone `carrier_hamiltonian` operator.
        """
        hs, ops = dyn_system
        Omega = TWO_PI * 100e3
        H = full_interaction_hamiltonian(
            ops,
            ion=0,
            mode=0,
            rabi_frequency=Omega,
            eta=0.0,
            detuning=0.0,
            mode_frequency=TWO_PI * 1e6,
            phase=0.9,
        )
        assert (
            H[0] - carrier_hamiltonian(ops, 0, Omega, phase=0.9)
        ).norm() < 1e-12

    def test_first_sideband_carries_the_lamb_dicke_factor_i(self, dyn_system):
        """Wineland et al. (1998) Eq. (23): the coupling phase of an
        order-s sideband is phi + s*pi/2. On the red-sideband
        resonance the static term must therefore equal
        `red_sideband_hamiltonian` evaluated at phase + pi/2.
        """
        hs, ops = dyn_system
        Omega = TWO_PI * 100e3
        eta = 0.1
        omega_m = TWO_PI * 1e6
        phase = 0.9
        H = full_interaction_hamiltonian(
            ops,
            ion=0,
            mode=0,
            rabi_frequency=Omega,
            eta=eta,
            detuning=-omega_m,
            mode_frequency=omega_m,
            phase=phase,
        )
        static = [term for term in H if isinstance(term, qutip.Qobj)]
        assert len(static) == 1
        expected = red_sideband_hamiltonian(
            ops, 0, 0, Omega, eta, phase=phase + np.pi / 2
        )
        assert (static[0] - expected).norm() < 1e-12

    def test_resonant_drive_matches_carrier(self, simple_system):
        """On resonance with detuning=0, the full Hamiltonian should
        behave as a carrier.

        A carrier pi-pulse flips |0> -> |1>. In QuTiP's convention,
        |0> has <sigma_z> = +1 and |1> has <sigma_z> = -1, so after
        the pi-pulse sigma_z goes from +1 to -1. The residual is set
        by the off-resonant sidebands, O((eta*Omega/omega_m)^2) ~ 2e-4.
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
        assert final_sz == pytest.approx(-1.0, abs=2e-3)

    @pytest.mark.parametrize("ratio", [1.0, 2.0, 5.0])
    def test_off_resonant_carrier_follows_rabi_formula(
        self, dyn_system, ratio
    ):
        """Exact two-level law (Rabi 1937): a detuned drive inverts at
        most $\\Omega^2/(\\Omega^2 + \\delta^2)$. Reproducing it
        requires the carrier to be a single complex exponential; a
        $\\cos(\\delta t)$ coefficient commutes with itself at all
        times, generates no AC Stark shift, and gives
        $\\sin^2(\\Omega/2\\delta)$ instead.
        """
        hs, ops = dyn_system
        Omega = TWO_PI * 20e3
        detuning = ratio * Omega
        H = full_interaction_hamiltonian(
            ops,
            ion=0,
            mode=0,
            rabi_frequency=Omega,
            eta=0.0,
            detuning=detuning,
            mode_frequency=TWO_PI * 1e6,
        )
        generalized = np.sqrt(Omega**2 + detuning**2)
        states = _evolve(H, _ket(0, 0), 4 * TWO_PI / generalized, 600)
        assert _max_population(states, _ket(1, 0)) == pytest.approx(
            1.0 / (1.0 + ratio**2), rel=1e-3
        )

    @pytest.mark.parametrize(
        "sign, n_final, n_dark, matrix_element",
        [(-1, 0, 2, np.sqrt(1)), (+1, 2, 0, np.sqrt(2))],
        ids=["red", "blue"],
    )
    def test_first_sideband_resonances(
        self, dyn_system, sign, n_final, n_dark, matrix_element
    ):
        """delta = omega_L - omega_0, so the first red sideband sits at
        delta = -omega_m (a phonon is absorbed to reach the qubit
        excited state) and the blue at delta = +omega_m
        (Leibfried RMP Eqs. (74)-(76)). Starting from |g,1>, the red
        resonance must reach |e,0> and leave |e,2> dark; the blue
        resonance the reverse.
        """
        hs, ops = dyn_system
        eta = 0.1
        Omega = TWO_PI * 20e3
        omega_m = TWO_PI * 1e6
        H = full_interaction_hamiltonian(
            ops,
            ion=0,
            mode=0,
            rabi_frequency=Omega,
            eta=eta,
            detuning=sign * omega_m,
            mode_frequency=omega_m,
        )
        coupling = eta * Omega * matrix_element
        states = _evolve(H, _ket(0, 1), np.pi / coupling)
        assert _max_population(states, _ket(1, n_final)) > 0.98
        assert _max_population(states, _ket(1, n_dark)) < 0.02

    @pytest.mark.parametrize(
        "sign, n_final, matrix_element",
        [(-1, 0, np.sqrt(2 * 1)), (+1, 4, np.sqrt(3 * 4))],
        ids=["second-red", "second-blue"],
    )
    def test_second_sideband_resonances(
        self, dyn_system, sign, n_final, matrix_element
    ):
        """(a + a^dag)^2 = a^2 + (a^dag)^2 + 2n + 1, so second order
        carries BOTH two-phonon sidebands, at delta = -2 omega_m and
        delta = +2 omega_m, with Rabi frequency
        (eta^2/2) Omega <n_final|a^2 or a^dag^2|n_initial>
        (Leibfried RMP Sec. III.B.3, Eq. (78)).
        """
        hs, ops = dyn_system
        eta = 0.25
        Omega = TWO_PI * 20e3
        omega_m = TWO_PI * 2e6
        H = full_interaction_hamiltonian(
            ops,
            ion=0,
            mode=0,
            rabi_frequency=Omega,
            eta=eta,
            detuning=sign * 2 * omega_m,
            mode_frequency=omega_m,
            lamb_dicke_order=2,
        )
        coupling = (eta**2 / 2) * Omega * matrix_element
        states = _evolve(H, _ket(0, 2), np.pi / coupling, 120)
        assert _max_population(states, _ket(1, n_final)) > 0.98

    @pytest.mark.parametrize("order, threshold", [(1, 0.99), (2, 0.999)])
    def test_red_sideband_matches_exact_propagator(
        self, dyn_system, order, threshold
    ):
        """RSB pi-pulse against the untruncated Eq. (69) propagator.

        The off-resonant carrier Stark-shifts the sideband by
        ~Omega^2/(4 omega_m) = 2pi*2.5 kHz against a sideband Rabi of
        eta*Omega = 2pi*10 kHz, so only 77.7% of |g,1> reaches |e,0>
        at the nominal pi-time. Both the missing factor i (which
        rotates the |e,0> amplitude by -pi/2) and a cos(delta t)
        carrier (which cancels the Stark shift) show up here.
        """
        hs, ops = dyn_system
        eta = 0.1
        Omega = TWO_PI * 100e3
        omega_m = TWO_PI * 1e6
        phase = 0.8
        duration = np.pi / (eta * Omega)
        reference = _exact_interaction_state(
            ops,
            _ket(0, 1),
            Omega,
            eta,
            -omega_m,
            omega_m,
            duration,
            phase=phase,
        )
        assert abs(reference.overlap(_ket(1, 0))) ** 2 == pytest.approx(
            0.777, abs=0.005
        )
        H = full_interaction_hamiltonian(
            ops,
            ion=0,
            mode=0,
            rabi_frequency=Omega,
            eta=eta,
            detuning=-omega_m,
            mode_frequency=omega_m,
            phase=phase,
            lamb_dicke_order=order,
        )
        final = _evolve(H, _ket(0, 1), duration)[-1]
        assert abs(final.overlap(reference)) ** 2 > threshold

    def test_off_resonant_carrier_matches_exact_propagator(self, dyn_system):
        """Off-resonant carrier (delta = Omega) against the untruncated
        propagator, over six generalized Rabi periods.
        """
        hs, ops = dyn_system
        eta = 0.1
        Omega = TWO_PI * 100e3
        omega_m = TWO_PI * 1e6
        detuning = Omega
        duration = 6 * TWO_PI / np.sqrt(Omega**2 + detuning**2)
        reference = _exact_interaction_state(
            ops,
            _ket(0, 0),
            Omega,
            eta,
            detuning,
            omega_m,
            duration,
            n_steps=3000,
        )
        H = full_interaction_hamiltonian(
            ops,
            ion=0,
            mode=0,
            rabi_frequency=Omega,
            eta=eta,
            detuning=detuning,
            mode_frequency=omega_m,
            lamb_dicke_order=2,
        )
        final = _evolve(H, _ket(0, 0), duration)[-1]
        assert abs(final.overlap(reference)) ** 2 > 0.999


class TestRamanPair:
    """Anchors: Wineland et al., J. Res. NIST 103, 259 (1998),
    Sec. 2.3.3 Eqs. (39)-(41) (two-photon coupling and level shifts);
    Ozeri et al., PRA 75, 042329 (2007), Eq. (2) (scattering rate in
    the single-intermediate-level limit).
    """

    @staticmethod
    def _pair(**overrides):
        kwargs = dict(
            omega_1=TWO_PI * 1e14,
            omega_2=TWO_PI * 1e14 - TWO_PI * 12.6e9,
            rabi_1=TWO_PI * 1e9,
            rabi_2=TWO_PI * 2e9,
            detuning_from_excited=TWO_PI * 100e9,
        )
        kwargs.update(overrides)
        return RamanPair(**kwargs)

    def test_zero_single_photon_detuning_rejected(self):
        with pytest.raises(ValueError, match="detuning_from_excited"):
            self._pair(detuning_from_excited=0.0)

    def test_effective_rabi_hits_a_known_value(self):
        """1 GHz and 2 GHz beams at 100 GHz detuning give a two-photon
        Rabi frequency of 2pi*10 MHz.
        """
        assert self._pair().effective_rabi_frequency == pytest.approx(
            TWO_PI * 10e6, rel=1e-12
        )

    def test_effective_rabi_is_bilinear_and_inverse_in_detuning(self):
        """Doubling either beam doubles Omega_eff; doubling the
        single-photon detuning halves it.
        """
        base = self._pair().effective_rabi_frequency
        assert self._pair(
            rabi_1=TWO_PI * 2e9
        ).effective_rabi_frequency == pytest.approx(2 * base, rel=1e-12)
        assert self._pair(
            rabi_2=TWO_PI * 4e9
        ).effective_rabi_frequency == pytest.approx(2 * base, rel=1e-12)
        assert self._pair(
            detuning_from_excited=TWO_PI * 200e9
        ).effective_rabi_frequency == pytest.approx(base / 2, rel=1e-12)

    def test_effective_rabi_changes_sign_with_the_detuning(self):
        """Omega_eff = Omega_1 Omega_2/(2 Delta) is signed: driving on
        the other side of the intermediate state flips the sign of the
        two-photon coupling (equivalent to a pi phase shift).
        """
        red = self._pair(detuning_from_excited=-TWO_PI * 100e9)
        assert red.effective_rabi_frequency == pytest.approx(
            -self._pair().effective_rabi_frequency, rel=1e-12
        )

    def test_beat_note_matches_the_qubit_splitting(self):
        assert self._pair().frequency_difference == pytest.approx(
            TWO_PI * 12.6e9, rel=1e-12
        )

    def test_ac_stark_shift_hits_a_known_value(self):
        """(Omega_1, Omega_2) = 2pi*(2, 1) GHz at Delta = 2pi*100 GHz
        leaves a differential shift of 2pi*7.5 MHz.
        """
        pair = self._pair(rabi_1=TWO_PI * 2e9, rabi_2=TWO_PI * 1e9)
        assert pair.ac_stark_shift == pytest.approx(TWO_PI * 7.5e6, rel=1e-12)

    def test_ac_stark_shift_vanishes_for_balanced_beams(self):
        """The differential shift is the imbalance: equal single-photon
        Rabi frequencies cancel it exactly, and swapping the beams
        reverses it.
        """
        balanced = self._pair(rabi_2=TWO_PI * 1e9)
        assert balanced.ac_stark_shift == 0.0
        forward = self._pair(rabi_1=TWO_PI * 2e9, rabi_2=TWO_PI * 1e9)
        swapped = self._pair(rabi_1=TWO_PI * 1e9, rabi_2=TWO_PI * 2e9)
        assert swapped.ac_stark_shift == pytest.approx(
            -forward.ac_stark_shift, rel=1e-12
        )

    def test_scattering_rate_hits_a_known_value(self):
        """Two 1 GHz beams 100 GHz off a Gamma = 2pi*20 MHz transition
        scatter at 2pi*1 kHz.
        """
        pair = self._pair(
            rabi_2=TWO_PI * 1e9,
            excited_state_linewidth=TWO_PI * 20e6,
        )
        assert pair.scattering_rate == pytest.approx(TWO_PI * 1e3, rel=1e-12)

    def test_scattering_rate_falls_as_inverse_detuning_squared(self):
        """Gamma_scatter ~ Delta^-2 while Omega_eff ~ Delta^-1, which is
        why large detunings win (Ozeri 2007): the scattering events per
        pi-pulse fall as 1/Delta.
        """
        near = self._pair(excited_state_linewidth=TWO_PI * 20e6)
        far = self._pair(
            detuning_from_excited=TWO_PI * 1000e9,
            excited_state_linewidth=TWO_PI * 20e6,
        )
        assert far.scattering_rate == pytest.approx(
            near.scattering_rate / 100, rel=1e-12
        )
        events_near = near.scattering_rate / near.effective_rabi_frequency
        events_far = far.scattering_rate / far.effective_rabi_frequency
        assert events_far == pytest.approx(events_near / 10, rel=1e-12)

    def test_scattering_rate_is_sign_independent(self):
        """The rate depends on Delta^2, so it is unchanged when the
        beams are tuned to the other side of the intermediate state.
        """
        blue = self._pair(excited_state_linewidth=TWO_PI * 20e6)
        red = self._pair(
            detuning_from_excited=-TWO_PI * 100e9,
            excited_state_linewidth=TWO_PI * 20e6,
        )
        assert red.scattering_rate == pytest.approx(
            blue.scattering_rate, rel=1e-12
        )

    def test_no_linewidth_means_no_scattering(self):
        assert self._pair().scattering_rate == 0.0

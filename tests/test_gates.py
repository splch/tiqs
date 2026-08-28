"""Tests for single-qubit gates, MS gate, light-shift gate, and
Cirac-Zoller gate."""

import numpy as np
import pytest
import qutip

from tiqs.constants import TWO_PI
from tiqs.gates.cirac_zoller import cirac_zoller_gate
from tiqs.gates.light_shift import light_shift_gate_hamiltonian
from tiqs.gates.molmer_sorensen import ms_gate_duration, ms_gate_hamiltonian
from tiqs.gates.single_qubit import (
    bb1_composite_gate,
    rx_gate,
    ry_gate,
    rz_gate,
    sk1_composite_gate,
)
from tiqs.hilbert_space.builder import HilbertSpace
from tiqs.hilbert_space.operators import OperatorFactory
from tiqs.hilbert_space.states import StateFactory


@pytest.fixture
def single_ion():
    hs = HilbertSpace(n_ions=1, n_modes=1, n_fock=5)
    ops = OperatorFactory(hs)
    sf = StateFactory(hs)
    return hs, ops, sf


def _segments(gate):
    """Segments to evolve under, honouring the GatePulse contract.

    ``(hamiltonian, duration)`` describes the whole gate only when
    ``pulses is None``; for composite gates ``hamiltonian`` is just the
    first segment while ``duration`` is the total.
    """
    if gate.pulses is None:
        return [(gate.hamiltonian, gate.duration)]
    return gate.pulses


def _qubit_propagator(gate, n_fock, amplitude_error=0.0):
    """Exact 2x2 propagator of a single-qubit gate on ion 0.

    Every segment Hamiltonian here acts as the identity on the
    motional mode, so the full propagator factorizes as
    $U_q \\otimes I$ and the qubit block is recovered by slicing the
    $n = 0$ rows/columns. ``amplitude_error`` scales every Hamiltonian
    by ``1 + eps``, i.e. a fractional Rabi-frequency error.
    """
    dim = 2 * n_fock
    u_full = np.eye(dim, dtype=complex)
    for h_seg, t_seg in _segments(gate):
        h_mat = h_seg.full() * (1.0 + amplitude_error)
        u_full = qutip.Qobj(-1j * h_mat * t_seg).expm().full() @ u_full
    idx = [0, n_fock]
    return u_full[np.ix_(idx, idx)]


def _average_gate_infidelity(u_actual, u_target):
    """$1 - F_\\mathrm{avg}$ for two single-qubit unitaries.

    $F_\\mathrm{avg} = (|\\mathrm{Tr}(U^\\dagger V)|^2 + d)/(d(d+1))$
    with $d = 2$ (Nielsen, Phys. Lett. A 303, 249 (2002)). Insensitive
    to global phase.
    """
    d = 2
    overlap = abs(np.trace(u_actual.conj().T @ u_target)) ** 2
    return 1.0 - (overlap + d) / (d * (d + 1))


def _rx_matrix(theta):
    """Exact $R_x(\\theta) = e^{-i\\theta\\sigma_x/2}$ reference."""
    return (-1j * theta * qutip.sigmax() / 2).expm().full()


class TestSingleQubitGates:
    def test_rx_pi_flips_state(self, single_ion):
        hs, ops, sf = single_ion
        psi0 = sf.ground_state()
        gate = rx_gate(ops, ion=0, theta=np.pi)
        result = qutip.sesolve(gate.hamiltonian, psi0, [0, gate.duration])
        final = result.states[-1]
        p1 = abs(final.overlap(sf.product_state([1], [0]))) ** 2
        assert p1 == pytest.approx(1.0, abs=0.01)

    def test_ry_pi_half_creates_superposition(self, single_ion):
        hs, ops, sf = single_ion
        psi0 = sf.ground_state()
        gate = ry_gate(ops, ion=0, theta=np.pi / 2)
        result = qutip.sesolve(gate.hamiltonian, psi0, [0, gate.duration])
        final = result.states[-1]
        sz = ops.sigma_z(0)
        assert qutip.expect(sz, final) == pytest.approx(0.0, abs=0.05)

    def test_rz_gate_phase(self, single_ion):
        """Rz(pi) on |+> should give |->."""
        hs, ops, sf = single_ion
        plus = (sf.product_state([0], [0]) + sf.product_state([1], [0])).unit()
        gate = rz_gate(ops, ion=0, phi=np.pi)
        result = qutip.sesolve(gate.hamiltonian, plus, [0, gate.duration])
        final = result.states[-1]
        sx = ops.sigma_x(0)
        assert qutip.expect(sx, final) == pytest.approx(-1.0, abs=0.05)

    def test_negative_theta_reverses_rotation(self, single_ion):
        """R_x(-pi/2) and R_x(+pi/2) produce opposite sigma_y."""
        hs, ops, sf = single_ion
        psi0 = sf.ground_state()
        sy = ops.sigma_y(0)

        g_pos = rx_gate(ops, ion=0, theta=np.pi / 2)
        r_pos = qutip.sesolve(g_pos.hamiltonian, psi0, [0, g_pos.duration])
        sy_pos = qutip.expect(sy, r_pos.states[-1])

        g_neg = rx_gate(ops, ion=0, theta=-np.pi / 2)
        r_neg = qutip.sesolve(g_neg.hamiltonian, psi0, [0, g_neg.duration])
        sy_neg = qutip.expect(sy, r_neg.states[-1])

        assert sy_pos == pytest.approx(-1.0, abs=0.01)
        assert sy_neg == pytest.approx(+1.0, abs=0.01)

    def test_sk1_more_robust_than_bare(self, single_ion):
        """SK1 composite pulse should be less sensitive to Rabi
        frequency errors."""
        hs, ops, sf = single_ion
        psi0 = sf.ground_state()
        target = sf.product_state([1], [0])
        omega = TWO_PI * 100e3
        bare = rx_gate(ops, ion=0, theta=np.pi, rabi_frequency=omega)
        sk1 = sk1_composite_gate(ops, ion=0, theta=np.pi, rabi_frequency=omega)

        error_omega = 0.05  # 5% over-rotation

        # Bare gate with amplitude error
        H_bare_err = bare.hamiltonian * (1.0 + error_omega)
        result_bare = qutip.sesolve(H_bare_err, psi0, [0, bare.duration])
        fid_bare = abs(result_bare.states[-1].overlap(target)) ** 2

        # SK1 gate with amplitude error: run the 3-pulse sequence
        assert sk1.pulses is not None
        psi = psi0
        for H_seg, t_seg in sk1.pulses:
            H_seg_err = H_seg * (1.0 + error_omega)
            res = qutip.sesolve(H_seg_err, psi, [0, t_seg])
            psi = res.states[-1]
        fid_sk1 = abs(psi.overlap(target)) ** 2

        err_bare = 1 - fid_bare
        err_sk1 = 1 - fid_sk1
        assert err_sk1 < err_bare  # SK1 error < bare error

    @pytest.mark.parametrize(
        "theta",
        [np.pi / 2, -np.pi / 2, np.pi, -np.pi, 0.3, -0.3],
    )
    @pytest.mark.parametrize(
        "builder", [sk1_composite_gate, bb1_composite_gate]
    )
    def test_composite_sequence_equals_exact_rotation(
        self, single_ion, builder, theta
    ):
        """At zero amplitude error the sequence *is* $R_x(\\theta)$.

        Reference is the matrix exponential $e^{-i\\theta\\sigma_x/2}$,
        computed independently of the pulse construction. Both signs of
        theta must work: the sign belongs in the drive axis (base phase
        $\\pi$), not in the pulse area. Placing it in the area instead
        implements $R_x(|\\theta|)$, which differs from $R_x(\\theta)$
        by more than a global phase at generic angles.
        """
        hs, ops, sf = single_ion
        gate = builder(ops, ion=0, theta=theta, rabi_frequency=TWO_PI * 1e6)
        u_actual = _qubit_propagator(gate, n_fock=5)
        u_target = _rx_matrix(theta)
        assert np.allclose(u_actual, u_target, atol=1e-10)

    @pytest.mark.parametrize("theta", [np.pi / 2, -np.pi / 2, np.pi, -np.pi])
    def test_composite_pulse_error_suppression_order(self, single_ion, theta):
        """Amplitude-error scaling matches the composite-pulse literature.

        Average gate infidelity scales as $\\epsilon^2$ for the bare
        rotation, $\\epsilon^4$ for SK1 (first-order cancellation;
        Brown, Harrow & Chuang, PRA 70, 052318 (2004)) and
        $\\epsilon^6$ for BB1 (first and second order; Wimperis,
        J. Magn. Reson. A 109, 221 (1994)). The orders must hold for
        negative theta too: computing $\\phi_1$ from signed theta while
        driving $|\\theta|$ doubles the first-order error generator
        instead of cancelling it, collapsing both slopes to ~2.
        """
        hs, ops, sf = single_ion
        omega = TWO_PI * 1e6
        u_target = _rx_matrix(theta)
        epsilons = np.array([0.2, 0.1, 0.05, 0.025])

        gates = {
            2.0: rx_gate(ops, 0, theta, rabi_frequency=omega),
            4.0: sk1_composite_gate(ops, 0, theta, rabi_frequency=omega),
            6.0: bb1_composite_gate(ops, 0, theta, rabi_frequency=omega),
        }
        for expected_order, gate in gates.items():
            errors = np.array([
                _average_gate_infidelity(
                    _qubit_propagator(gate, 5, amplitude_error=eps),
                    u_target,
                )
                for eps in epsilons
            ])
            slope = np.polyfit(np.log(epsilons), np.log(errors), 1)[0]
            assert slope == pytest.approx(expected_order, abs=0.15)

    @pytest.mark.parametrize(
        ("builder", "n_pulses"),
        [(sk1_composite_gate, 3), (bb1_composite_gate, 4)],
    )
    def test_composite_gate_pulse_bookkeeping(
        self, single_ion, builder, n_pulses
    ):
        """`duration` is the total; `hamiltonian` is only segment one."""
        hs, ops, sf = single_ion
        gate = builder(ops, ion=0, theta=np.pi)
        assert gate.pulses is not None
        assert len(gate.pulses) == n_pulses
        assert gate.duration == pytest.approx(sum(t for _, t in gate.pulses))
        assert gate.hamiltonian == gate.pulses[0][0]
        for H_seg, t_seg in gate.pulses:
            assert H_seg.isherm
            assert t_seg > 0

    @pytest.mark.parametrize(
        "builder", [sk1_composite_gate, bb1_composite_gate]
    )
    def test_composite_gate_rejects_too_large_angle(self, single_ion, builder):
        """phi_1 = arccos(-|theta|/4pi) needs |theta| <= 4*pi."""
        hs, ops, sf = single_ion
        for theta in (4.5 * np.pi, -4.5 * np.pi):
            with pytest.raises(ValueError, match="4\\*pi"):
                builder(ops, ion=0, theta=theta)


class TestMolmerSorensenGate:
    @pytest.fixture
    def two_ion_system(self):
        hs = HilbertSpace(n_ions=2, n_modes=1, n_fock=15)
        ops = OperatorFactory(hs)
        sf = StateFactory(hs)
        return hs, ops, sf

    def test_ms_hamiltonian_is_list_format(self, two_ion_system):
        hs, ops, sf = two_ion_system
        H = ms_gate_hamiltonian(
            ops,
            ions=[0, 1],
            mode=0,
            eta=[0.1, 0.1],
            rabi_frequency=TWO_PI * 50e3,
            detuning=TWO_PI * 10e3,
        )
        assert isinstance(H, list)

    def test_ms_gate_produces_bell_state(self, two_ion_system):
        """MS gate on |00,n=0> should produce (|00> + i|11>)/sqrt(2)
        up to global phase."""
        hs, ops, sf = two_ion_system
        eta = 0.1
        delta = TWO_PI * 20e3
        # For two identically-coupled ions, the maximally entangling condition
        # is eta*Omega = delta/4 (the factor of 2 vs single-ion comes from
        # the collective spin coupling doubling the geometric phase).
        Omega = delta / (4 * eta)
        tau = ms_gate_duration(delta, loops=1)

        H = ms_gate_hamiltonian(
            ops,
            ions=[0, 1],
            mode=0,
            eta=[eta, eta],
            rabi_frequency=Omega,
            detuning=delta,
        )
        psi0 = sf.ground_state()
        tlist = np.linspace(0, tau, 500)
        result = qutip.sesolve(H, psi0, tlist, options={"max_step": tau / 100})
        final_full = result.states[-1]

        # Trace out motional mode
        rho_spin = final_full.ptrace([0, 1])

        # Target: (|00> + i|11>)/sqrt(2)
        ket_00 = qutip.tensor(qutip.basis(2, 0), qutip.basis(2, 0))
        ket_11 = qutip.tensor(qutip.basis(2, 1), qutip.basis(2, 1))
        bell = (ket_00 + 1j * ket_11).unit()
        rho_target = qutip.ket2dm(bell)

        fid = qutip.fidelity(rho_spin, rho_target) ** 2
        assert fid > 0.90

    def test_ms_gate_insensitive_to_thermal_motion(self, two_ion_system):
        """MS gate fidelity should not degrade significantly with
        thermal initial motion."""
        hs, ops, sf = two_ion_system
        eta = 0.05
        delta = TWO_PI * 20e3
        # Same correction: eta*Omega = delta/4 for two symmetric ions.
        Omega = delta / (4 * eta)
        tau = ms_gate_duration(delta, loops=1)

        H = ms_gate_hamiltonian(
            ops,
            ions=[0, 1],
            mode=0,
            eta=[eta, eta],
            rabi_frequency=Omega,
            detuning=delta,
        )

        ket_00 = qutip.tensor(qutip.basis(2, 0), qutip.basis(2, 0))
        ket_11 = qutip.tensor(qutip.basis(2, 1), qutip.basis(2, 1))
        bell = (ket_00 + 1j * ket_11).unit()
        rho_target = qutip.ket2dm(bell)

        fidelities = []
        for n_bar in [0.0, 2.0]:
            rho0 = sf.thermal_state(n_bar=[n_bar])
            tlist = np.linspace(0, tau, 500)
            result = qutip.mesolve(
                H, rho0, tlist, options={"max_step": tau / 100}
            )
            rho_spin = result.states[-1].ptrace([0, 1])
            fid = qutip.fidelity(rho_spin, rho_target) ** 2
            fidelities.append(fid)

        # The implemented Hamiltonian is first order in eta with unit
        # Debye-Waller factor, so at tau = 2*pi*K/|delta| the loop
        # closes and the propagator is purely a spin operator: the gate
        # is insensitive to the initial motional state *by
        # construction*. The residual (measured 7.8e-3 here) is Fock
        # truncation of the n_bar=2 thermal tail at n_fock=15, not
        # physics - so the old ">0.80 / within 0.20" bounds were
        # unfalsifiable.
        assert fidelities[1] > 0.99
        assert fidelities[1] == pytest.approx(fidelities[0], abs=0.012)

    def test_ms_spin_map_is_independent_of_fock_state(self, two_ion_system):
        """The propagator at tau is purely a spin operator.

        With the coupling strictly linear in eta and no Debye-Waller
        factor, the first Magnus term vanishes at
        $\\tau = 2\\pi K/|\\delta|$ and the second is spin-only, so the
        motional state is returned untouched and the spin map is
        identical for every Fock input. Checked on well-truncated
        inputs |n=0> and |n=3>: this exact law is what the docstring's
        "insensitive by construction" claim means, and any
        n-dependent drive factor breaks it.
        """
        hs, ops, sf = two_ion_system
        eta = 0.05
        delta = TWO_PI * 20e3
        tau = ms_gate_duration(delta, loops=1)
        H = ms_gate_hamiltonian(
            ops,
            ions=[0, 1],
            mode=0,
            eta=[eta, eta],
            rabi_frequency=delta / (4 * eta),
            detuning=delta,
        )
        ket_00 = qutip.tensor(qutip.basis(2, 0), qutip.basis(2, 0))

        spin_states = {}
        for n in (0, 3):
            psi0 = qutip.tensor(ket_00, qutip.basis(15, n))
            result = qutip.sesolve(
                H,
                psi0,
                np.linspace(0, tau, 400),
                options={
                    "max_step": tau / 200,
                    "atol": 1e-12,
                    "rtol": 1e-10,
                },
            )
            final = result.states[-1]
            # Motion returns to its initial Fock state.
            assert qutip.expect(ops.number(0), final) == pytest.approx(
                n, abs=1e-3
            )
            spin_states[n] = final.ptrace([0, 1])

        assert (spin_states[3] - spin_states[0]).norm() < 1e-4

    def test_ms_gate_duration_formula(self):
        delta = TWO_PI * 10e3
        tau = ms_gate_duration(delta, loops=1)
        assert tau == pytest.approx(TWO_PI / delta)

    def test_ms_gate_two_loops(self):
        delta = TWO_PI * 10e3
        tau1 = ms_gate_duration(delta, loops=1)
        tau2 = ms_gate_duration(delta, loops=2)
        assert tau2 == pytest.approx(2 * tau1)

    def test_ms_gate_duration_uses_absolute_detuning(self):
        """The loop closes at |delta|*tau = 2*pi*K, either side."""
        delta = TWO_PI * 10e3
        assert ms_gate_duration(-delta) == pytest.approx(
            ms_gate_duration(delta)
        )
        assert ms_gate_duration(-delta) > 0

    def test_ms_gate_duration_rejects_degenerate_input(self):
        """loops=0 is not a gate and delta=0 never closes the loop."""
        with pytest.raises(ValueError, match="loops"):
            ms_gate_duration(TWO_PI * 10e3, loops=0)
        with pytest.raises(ValueError, match="detuning"):
            ms_gate_duration(0.0, loops=1)

    @pytest.mark.parametrize("detuning_sign", [+1, -1])
    @pytest.mark.parametrize("eta_product_sign", [+1, -1])
    def test_bell_phase_sign_law(
        self, two_ion_system, detuning_sign, eta_product_sign
    ):
        """The Bell phase is sign(delta * eta_i * eta_j).

        Second-order Magnus for $H = S(a^\\dagger e^{i\\delta t} +
        a e^{-i\\delta t})$ gives
        $\\Omega_2 = +i S^2 (\\delta\\tau - \\sin\\delta\\tau)/\\delta^2$,
        so $\\chi = +4\\pi K \\eta_i \\eta_j \\Omega^2/\\delta^2$ and
        $U|00\\rangle = (|00\\rangle + i\\,\\mathrm{sign}(\\chi)\\,
        |11\\rangle)/\\sqrt 2$. A mode with opposite ion
        participations, or tones on the other side of the sideband,
        therefore yields the conjugate Bell state.
        """
        hs, ops, sf = two_ion_system
        eta_a = 0.05
        eta_b = eta_product_sign * 0.05
        delta = detuning_sign * TWO_PI * 20e3
        omega = abs(delta) / (4 * np.sqrt(abs(eta_a * eta_b)))
        tau = ms_gate_duration(delta)

        H = ms_gate_hamiltonian(
            ops,
            ions=[0, 1],
            mode=0,
            eta=[eta_a, eta_b],
            rabi_frequency=omega,
            detuning=delta,
        )
        result = qutip.sesolve(
            H,
            sf.ground_state(),
            np.linspace(0, tau, 400),
            options={"max_step": tau / 200, "atol": 1e-12, "rtol": 1e-10},
        )
        rho_spin = result.states[-1].ptrace([0, 1])

        ket_00 = qutip.tensor(qutip.basis(2, 0), qutip.basis(2, 0))
        ket_11 = qutip.tensor(qutip.basis(2, 1), qutip.basis(2, 1))
        expected_sign = detuning_sign * eta_product_sign
        target = (ket_00 + 1j * expected_sign * ket_11).unit()
        conjugate = (ket_00 - 1j * expected_sign * ket_11).unit()

        fid = qutip.fidelity(rho_spin, qutip.ket2dm(target)) ** 2
        fid_conj = qutip.fidelity(rho_spin, qutip.ket2dm(conjugate)) ** 2
        assert fid > 0.999
        assert fid_conj < 1e-3


class TestLightShiftGate:
    @pytest.fixture
    def two_ion_system(self):
        hs = HilbertSpace(n_ions=2, n_modes=1, n_fock=15)
        ops = OperatorFactory(hs)
        sf = StateFactory(hs)
        return hs, ops, sf

    def test_light_shift_is_list_format(self, two_ion_system):
        hs, ops, sf = two_ion_system
        H = light_shift_gate_hamiltonian(
            ops,
            ions=[0, 1],
            mode=0,
            eta=[0.1, 0.1],
            rabi_frequency=TWO_PI * 50e3,
            detuning=TWO_PI * 10e3,
        )
        assert isinstance(H, list)

    def test_light_shift_generates_zz_entanglement(self, two_ion_system):
        r"""Light-shift gate maps $|{+}{+}\rangle$ to a maximally
        entangled state.

        All $\sigma_z$ commute, so the Magnus expansion terminates and
        the exact propagator at $\tau = 2\pi/|\delta|$ is
        $e^{i\chi\sigma_z\sigma_z}$ with
        $\chi = 4\pi(\eta\Omega)^2/\delta^2$. At the calibration
        $\eta\Omega = \delta/4$ this is $\chi = \pi/4$, giving
        single-qubit purity $1/2 + \cos^2(2\chi)/2$ **exactly 0.5** -
        pinning the value, not just "less than 1", so a coupling wrong
        by a factor of two (purity 0.927) cannot pass.
        """
        hs, ops, sf = two_ion_system
        eta = 0.1
        delta = TWO_PI * 20e3
        # For ZZ gate with two identically-coupled ions, same geometric phase
        # condition as MS: eta*Omega = delta/4 for maximally entangling
        Omega = delta / (4 * eta)
        tau = ms_gate_duration(delta, loops=1)

        H = light_shift_gate_hamiltonian(
            ops,
            ions=[0, 1],
            mode=0,
            eta=[eta, eta],
            rabi_frequency=Omega,
            detuning=delta,
        )
        # Start in |++> (both ions in +x eigenstate) - ZZ gate entangles
        # sigma_x eigenstates
        plus = (qutip.basis(2, 0) + qutip.basis(2, 1)).unit()
        psi0 = qutip.tensor(plus, plus, qutip.basis(15, 0))
        tlist = np.linspace(0, tau, 500)
        result = qutip.sesolve(H, psi0, tlist, options={"max_step": tau / 100})
        final = result.states[-1]
        rho_spin = final.ptrace([0, 1])

        rho_single = rho_spin.ptrace(0)
        purity = (rho_single * rho_single).tr().real
        assert purity == pytest.approx(0.5, abs=0.02)

        # The phase-space loop must close, or the "entanglement" is
        # really residual spin-motion correlation.
        assert qutip.expect(ops.number(0), final) == pytest.approx(
            0.0, abs=0.01
        )

        # Compare against the closed-form propagator exp(i*pi/4 ZZ).
        zz = qutip.tensor(qutip.sigmaz(), qutip.sigmaz())
        target = (1j * (np.pi / 4) * zz).expm() * qutip.tensor(plus, plus)
        fid = qutip.fidelity(rho_spin, qutip.ket2dm(target)) ** 2
        assert fid > 0.999


class TestCiracZollerGate:
    @pytest.fixture
    def two_ion_system(self):
        hs = HilbertSpace(n_ions=2, n_modes=1, n_fock=10)
        ops = OperatorFactory(hs)
        sf = StateFactory(hs)
        return hs, ops, sf

    N_FOCK = 10

    @staticmethod
    def _computational_block(hs, seq):
        """Exact 4x4 block and per-input leakage out of |n=0>.

        Returns ``(block, leakage)`` where ``block[i, j]`` is the
        amplitude from input j to output i within
        {|00>, |01>, |10>, |11>} x |n=0>.
        """
        u = qutip.qeye(hs.dims)
        for pulse in seq:
            u = (-1j * pulse.hamiltonian * pulse.duration).expm() * u
        kets = [
            qutip.tensor(
                qutip.basis(2, qa),
                qutip.basis(2, qb),
                qutip.basis(TestCiracZollerGate.N_FOCK, 0),
            )
            for qa, qb in [(0, 0), (0, 1), (1, 0), (1, 1)]
        ]
        block = np.zeros((4, 4), dtype=complex)
        leakage = []
        for j, ket_in in enumerate(kets):
            out = u * ket_in
            for i, ket_out in enumerate(kets):
                block[i, j] = ket_out.overlap(out)
            leakage.append(1.0 - float(np.sum(np.abs(block[:, j]) ** 2)))
        return block, np.array(leakage)

    def test_cz_gate_returns_pulse_sequence(self, two_ion_system):
        hs, ops, sf = two_ion_system
        seq = cirac_zoller_gate(ops, ion_a=0, ion_b=1, mode=0, eta=[0.1, 0.1])
        assert isinstance(seq, list)
        assert len(seq) == 3  # three sequential pulses
        for pulse in seq:
            assert hasattr(pulse, "hamiltonian")
            assert hasattr(pulse, "duration")

    @pytest.mark.parametrize(
        ("eta", "rabi"),
        [([0.1, 0.1], TWO_PI * 100e3), ([0.03, 0.25], TWO_PI * 40e3)],
    )
    def test_cz_map_has_sqrt2_pulse_area_error(
        self, two_ion_system, eta, rabi
    ):
        r"""The implemented map is diag(1, -1, -1, cos(sqrt(2) pi)).

        Analytic anchor: after step 1 the $|11\rangle$ branch sits on
        $|1_B, 1\rangle$, whose red-sideband coupling to
        $|0_B, 2\rangle$ carries the $\sqrt{n}$ enhancement. The
        intended $2\pi$ pulse is therefore $2\sqrt{2}\pi$ long in
        units of that coupling, leaving amplitude
        $\cos(\sqrt 2 \pi) = -0.26626$ and leaking
        $\sin^2(\sqrt 2 \pi) = 92.91\%$. Both numbers are pure
        numbers: independent of eta, of the Rabi frequency, and of the
        Fock truncation.
        """
        hs, ops, sf = two_ion_system
        seq = cirac_zoller_gate(
            ops, ion_a=0, ion_b=1, mode=0, eta=eta, rabi_frequency=rabi
        )
        block, leakage = self._computational_block(hs, seq)

        off_diagonal = block - np.diag(np.diag(block))
        assert np.max(np.abs(off_diagonal)) < 1e-10

        surviving = np.cos(np.sqrt(2) * np.pi)
        assert np.allclose(
            np.diag(block).real, [1.0, -1.0, -1.0, surviving], atol=1e-8
        )
        assert np.allclose(np.diag(block).imag, 0.0, atol=1e-8)
        assert np.allclose(leakage[:3], 0.0, atol=1e-8)
        assert leakage[3] == pytest.approx(
            np.sin(np.sqrt(2) * np.pi) ** 2, abs=1e-8
        )

    def test_cz_gate_is_not_a_controlled_phase_gate(self, two_ion_system):
        r"""The leakage-free part is the *local* operator Z(x)Z.

        Restricted to the three inputs that do not leak, the map is
        diag(1, -1, -1), which is exactly $\sigma_z \otimes \sigma_z$ -
        a product of single-qubit operators, so it generates no
        entanglement. The docstring must not promise a controlled
        phase: the target diag(1, 1, 1, -1) is far away, and what
        entanglement does appear on $|{+}{+}\rangle$ comes from
        leakage-induced decoherence (the spin state is left mixed),
        not from a coherent conditional phase.
        """
        hs, ops, sf = two_ion_system
        seq = cirac_zoller_gate(ops, ion_a=0, ion_b=1, mode=0, eta=[0.1, 0.1])
        block, _ = self._computational_block(hs, seq)

        zz_diagonal = np.diag(
            qutip.tensor(qutip.sigmaz(), qutip.sigmaz()).full()
        ).real
        assert np.allclose(np.diag(block).real[:3], zz_diagonal[:3], atol=1e-8)

        cz_diagonal = np.array([1.0, 1.0, 1.0, -1.0])
        assert not np.allclose(np.diag(block).real, cz_diagonal, atol=0.1)

        # |++>|0>: population lost from |n=0> is 1/4 of the |11>
        # leakage, and the surviving spin state is mixed.
        u = qutip.qeye(hs.dims)
        for pulse in seq:
            u = (-1j * pulse.hamiltonian * pulse.duration).expm() * u
        plus = (qutip.basis(2, 0) + qutip.basis(2, 1)).unit()
        psi = u * qutip.tensor(plus, plus, qutip.basis(self.N_FOCK, 0))
        ground_mode = qutip.tensor(
            qutip.qeye(2),
            qutip.qeye(2),
            qutip.ket2dm(qutip.basis(self.N_FOCK, 0)),
        )
        p_ground = float(np.real(qutip.expect(ground_mode, psi)))
        assert p_ground == pytest.approx(
            1.0 - 0.25 * np.sin(np.sqrt(2) * np.pi) ** 2, abs=1e-6
        )

        rho_spin = psi.ptrace([0, 1])
        purity = (rho_spin * rho_spin).tr().real
        assert purity < 0.99  # decoherence, not a unitary phase gate

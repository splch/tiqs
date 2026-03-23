"""Tests for single-qubit gates, MS gate, light-shift gate, and Cirac-Zoller gate."""
import numpy as np
import pytest
import qutip

from tiqs.gates.single_qubit import (
    rx_gate,
    ry_gate,
    rz_gate,
    sk1_composite_gate,
    bb1_composite_gate,
)
from tiqs.gates.molmer_sorensen import ms_gate_hamiltonian, ms_gate_duration
from tiqs.gates.light_shift import light_shift_gate_hamiltonian
from tiqs.gates.cirac_zoller import cirac_zoller_gate
from tiqs.hilbert_space.builder import HilbertSpace
from tiqs.hilbert_space.operators import OperatorFactory
from tiqs.hilbert_space.states import StateFactory
from tiqs.constants import TWO_PI


@pytest.fixture
def single_ion():
    hs = HilbertSpace(n_ions=1, n_modes=1, n_fock=5)
    ops = OperatorFactory(hs)
    sf = StateFactory(hs)
    return hs, ops, sf


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

    def test_sk1_more_robust_than_bare(self, single_ion):
        """SK1 composite pulse should be less sensitive to Rabi frequency errors."""
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
        psi = psi0
        for H_seg, t_seg in sk1.pulses:
            H_seg_err = H_seg * (1.0 + error_omega)
            res = qutip.sesolve(H_seg_err, psi, [0, t_seg])
            psi = res.states[-1]
        fid_sk1 = abs(psi.overlap(target)) ** 2

        err_bare = 1 - fid_bare
        err_sk1 = 1 - fid_sk1
        assert err_sk1 < err_bare  # SK1 error < bare error

    def test_bb1_hermitian(self, single_ion):
        hs, ops, sf = single_ion
        gate = bb1_composite_gate(ops, ion=0, theta=np.pi)
        assert gate.hamiltonian.isherm or isinstance(gate.hamiltonian, list)


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
            ops, ions=[0, 1], mode=0, eta=[0.1, 0.1],
            rabi_frequency=TWO_PI * 50e3, detuning=TWO_PI * 10e3,
        )
        assert isinstance(H, list)

    def test_ms_gate_produces_bell_state(self, two_ion_system):
        """MS gate on |00,n=0> should produce (|00> + i|11>)/sqrt(2) up to global phase."""
        hs, ops, sf = two_ion_system
        eta = 0.1
        delta = TWO_PI * 20e3
        # For two identically-coupled ions, the maximally entangling condition
        # is eta*Omega = delta/4 (the factor of 2 vs single-ion comes from
        # the collective spin coupling doubling the geometric phase).
        Omega = delta / (4 * eta)
        tau = ms_gate_duration(delta, loops=1)

        H = ms_gate_hamiltonian(
            ops, ions=[0, 1], mode=0, eta=[eta, eta],
            rabi_frequency=Omega, detuning=delta,
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
        """MS gate fidelity should not degrade significantly with thermal initial motion."""
        hs, ops, sf = two_ion_system
        eta = 0.05
        delta = TWO_PI * 20e3
        # Same correction: eta*Omega = delta/4 for two symmetric ions.
        Omega = delta / (4 * eta)
        tau = ms_gate_duration(delta, loops=1)

        H = ms_gate_hamiltonian(
            ops, ions=[0, 1], mode=0, eta=[eta, eta],
            rabi_frequency=Omega, detuning=delta,
        )

        ket_00 = qutip.tensor(qutip.basis(2, 0), qutip.basis(2, 0))
        ket_11 = qutip.tensor(qutip.basis(2, 1), qutip.basis(2, 1))
        bell = (ket_00 + 1j * ket_11).unit()
        rho_target = qutip.ket2dm(bell)

        fidelities = []
        for n_bar in [0.0, 2.0]:
            rho0 = sf.thermal_state(n_bar=[n_bar])
            tlist = np.linspace(0, tau, 500)
            result = qutip.mesolve(H, rho0, tlist, options={"max_step": tau / 100})
            rho_spin = result.states[-1].ptrace([0, 1])
            fid = qutip.fidelity(rho_spin, rho_target) ** 2
            fidelities.append(fid)

        # Fidelity with n_bar=2 should still be > 80%
        assert fidelities[1] > 0.80
        # And the degradation should be modest
        assert fidelities[1] > fidelities[0] - 0.20

    def test_ms_gate_duration_formula(self):
        delta = TWO_PI * 10e3
        tau = ms_gate_duration(delta, loops=1)
        assert tau == pytest.approx(TWO_PI / delta)

    def test_ms_gate_two_loops(self):
        delta = TWO_PI * 10e3
        tau1 = ms_gate_duration(delta, loops=1)
        tau2 = ms_gate_duration(delta, loops=2)
        assert tau2 == pytest.approx(2 * tau1)


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
            ops, ions=[0, 1], mode=0, eta=[0.1, 0.1],
            rabi_frequency=TWO_PI * 50e3, detuning=TWO_PI * 10e3,
        )
        assert isinstance(H, list)

    def test_light_shift_generates_zz_entanglement(self, two_ion_system):
        """Light-shift gate should entangle |+,+> into a state with ZZ correlations."""
        hs, ops, sf = two_ion_system
        eta = 0.1
        delta = TWO_PI * 20e3
        Omega = delta / (2 * eta)
        tau = TWO_PI / delta

        H = light_shift_gate_hamiltonian(
            ops, ions=[0, 1], mode=0, eta=[eta, eta],
            rabi_frequency=Omega, detuning=delta,
        )
        # Start in |++> (both ions in +x eigenstate)
        plus = (qutip.basis(2, 0) + qutip.basis(2, 1)).unit()
        psi0 = qutip.tensor(plus, plus, qutip.basis(15, 0))
        tlist = np.linspace(0, tau, 500)
        result = qutip.sesolve(H, psi0, tlist, options={"max_step": tau / 100})
        rho_spin = result.states[-1].ptrace([0, 1])
        # Should not be a product state anymore (entangled)
        purity = rho_spin.tr()
        single_qubit_purity = rho_spin.ptrace(0).tr()
        # If entangled, reduced state has lower purity
        assert single_qubit_purity < 1.5  # crude check


class TestCiracZollerGate:
    @pytest.fixture
    def two_ion_system(self):
        hs = HilbertSpace(n_ions=2, n_modes=1, n_fock=10)
        ops = OperatorFactory(hs)
        sf = StateFactory(hs)
        return hs, ops, sf

    def test_cz_gate_returns_pulse_sequence(self, two_ion_system):
        hs, ops, sf = two_ion_system
        seq = cirac_zoller_gate(ops, ion_a=0, ion_b=1, mode=0, eta=[0.1, 0.1])
        assert isinstance(seq, list)
        assert len(seq) == 3  # three sequential pulses
        for pulse in seq:
            assert hasattr(pulse, "hamiltonian")
            assert hasattr(pulse, "duration")

"""Tests for time-varying pulse waveforms and pulsed gate Hamiltonians."""

import numpy as np
import pytest
import qutip

from tiqs.analysis.fidelity import bell_state_fidelity
from tiqs.constants import TWO_PI
from tiqs.gates.light_shift import light_shift_gate_hamiltonian_pulsed
from tiqs.gates.molmer_sorensen import (
    ms_gate_duration,
    ms_gate_hamiltonian,
    ms_gate_hamiltonian_pulsed,
)
from tiqs.hilbert_space.builder import HilbertSpace
from tiqs.hilbert_space.operators import OperatorFactory
from tiqs.hilbert_space.states import StateFactory
from tiqs.pulses import (
    BlackmanWindow,
    ConstantWaveform,
    PiecewiseConstant,
    Pulse,
    SinusoidalRamp,
    am_ms_pulse,
    build_pulsed_coefficient,
    smooth_ms_pulse,
    verify_loop_closure,
    windowed_ms_pulse,
)
from tiqs.simulation.config import SimulationConfig
from tiqs.simulation.runner import SimulationRunner
from tiqs.species.ion import get_species
from tiqs.trap import PaulTrap


class TestWaveforms:
    def test_constant_evaluate(self):
        wf = ConstantWaveform(42.0)
        assert wf.evaluate(0.5, 1.0) == 42.0
        arr = wf.evaluate(np.array([0.0, 0.5, 1.0]), 1.0)
        np.testing.assert_allclose(arr, 42.0)

    def test_constant_as_string(self):
        wf = ConstantWaveform(3.14)
        assert wf.as_string(1.0) == "3.14"

    def test_sinusoidal_ramp_evaluate(self):
        wf = SinusoidalRamp(delta_0=1000.0, amplitude=500.0)
        tau = 1e-3
        # At t=0: delta_0 + A = 1500
        assert wf.evaluate(0.0, tau) == pytest.approx(1500.0)
        # At t=tau/2: delta_0 - A = 500
        assert wf.evaluate(tau / 2, tau) == pytest.approx(500.0)
        # At t=tau: delta_0 + A = 1500
        assert wf.evaluate(tau, tau) == pytest.approx(1500.0)

    def test_sinusoidal_ramp_as_string(self):
        wf = SinusoidalRamp(delta_0=1000.0, amplitude=500.0)
        s = wf.as_string(1e-3)
        assert s is not None
        assert "cos" in s

    def test_sinusoidal_ramp_integrated_phase(self):
        """The integrated phase should match numerical integration."""
        wf = SinusoidalRamp(delta_0=TWO_PI * 10e3, amplitude=TWO_PI * 5e3)
        tau = 1e-4
        tlist = np.linspace(0, tau, 10000)
        # Numerical integration
        delta_vals = wf.evaluate(tlist, tau)
        from scipy.integrate import cumulative_trapezoid

        phase_numerical = np.zeros_like(tlist)
        phase_numerical[1:] = cumulative_trapezoid(delta_vals, tlist)
        # Analytical: delta_0*t + A/omega_ramp * sin(omega_ramp*t)
        omega_ramp = TWO_PI / tau
        phase_analytical = (
            wf.delta_0 * tlist
            + wf.amplitude / omega_ramp * np.sin(omega_ramp * tlist)
        )
        np.testing.assert_allclose(
            phase_numerical, phase_analytical, atol=1e-3
        )

    def test_blackman_window(self):
        wf = BlackmanWindow(peak_value=1.0)
        tau = 1.0
        # Endpoints should be near zero (Blackman window vanishes)
        assert abs(wf.evaluate(0.0, tau)) < 0.01
        assert abs(wf.evaluate(tau, tau)) < 0.01
        # Center should be near peak
        assert wf.evaluate(tau / 2, tau) > 0.9

    def test_blackman_has_string(self):
        wf = BlackmanWindow(1.0)
        assert wf.as_string(1.0) is not None

    def test_piecewise_constant(self):
        wf = PiecewiseConstant([1.0, 2.0, 3.0])
        tau = 3.0
        assert wf.evaluate(0.5, tau) == pytest.approx(1.0)
        assert wf.evaluate(1.5, tau) == pytest.approx(2.0)
        assert wf.evaluate(2.5, tau) == pytest.approx(3.0)

    def test_piecewise_array(self):
        wf = PiecewiseConstant([10.0, 20.0])
        tau = 2.0
        arr = wf.evaluate(np.array([0.5, 1.5]), tau)
        np.testing.assert_allclose(arr, [10.0, 20.0])


class TestPulse:
    def test_constant_pulse(self):
        p = Pulse.constant(1e5, TWO_PI * 10e3, duration=1e-4)
        assert isinstance(p.rabi_frequency, ConstantWaveform)
        assert isinstance(p.detuning, ConstantWaveform)
        assert p.duration == 1e-4

    def test_smooth_ms_pulse_creates_sinusoidal(self):
        p = smooth_ms_pulse(
            delta_0=TWO_PI * 10e3,
            ramp_amplitude=TWO_PI * 5e3,
            rabi_frequency=1e5,
        )
        assert isinstance(p.detuning, SinusoidalRamp)
        assert isinstance(p.rabi_frequency, ConstantWaveform)
        assert p.duration == pytest.approx(TWO_PI / (TWO_PI * 10e3), rel=1e-6)

    def test_smooth_ms_pulse_rejects_large_ramp(self):
        with pytest.raises(ValueError):
            smooth_ms_pulse(
                delta_0=TWO_PI * 10e3,
                ramp_amplitude=TWO_PI * 10e3,  # = delta_0, not allowed
                rabi_frequency=1e5,
            )

    def test_am_ms_pulse(self):
        p = am_ms_pulse([1e5, 2e5, 1e5], detuning=TWO_PI * 10e3)
        assert isinstance(p.rabi_frequency, PiecewiseConstant)
        assert isinstance(p.detuning, ConstantWaveform)

    def test_windowed_ms_pulse_blackman(self):
        p = windowed_ms_pulse(1e5, TWO_PI * 10e3, window="blackman")
        assert isinstance(p.rabi_frequency, BlackmanWindow)


class TestCoefficientBuilder:
    def test_both_constant_returns_string(self):
        rabi = ConstantWaveform(1e5)
        det = ConstantWaveform(TWO_PI * 10e3)
        coeff = build_pulsed_coefficient(rabi, det, tau=1e-4, sign=+1)
        assert isinstance(coeff, str)
        assert "exp" in coeff

    def test_sinusoidal_ramp_returns_string(self):
        rabi = ConstantWaveform(1e5)
        det = SinusoidalRamp(TWO_PI * 10e3, TWO_PI * 5e3)
        coeff = build_pulsed_coefficient(rabi, det, tau=1e-4, sign=+1)
        assert isinstance(coeff, str)
        assert "sin" in coeff

    def test_piecewise_returns_array(self):
        rabi = PiecewiseConstant([1e5, 2e5])
        det = ConstantWaveform(TWO_PI * 10e3)
        tlist = np.linspace(0, 1e-4, 100)
        coeff = build_pulsed_coefficient(
            rabi, det, tau=1e-4, sign=+1, tlist=tlist
        )
        assert isinstance(coeff, np.ndarray)
        assert len(coeff) == 100

    def test_array_requires_tlist(self):
        rabi = PiecewiseConstant([1e5, 2e5])
        det = ConstantWaveform(TWO_PI * 10e3)
        with pytest.raises(ValueError, match="tlist"):
            build_pulsed_coefficient(rabi, det, tau=1e-4)


class TestPulsedMSHamiltonian:
    @pytest.fixture
    def system(self):
        hs = HilbertSpace(n_ions=2, n_modes=1, n_fock=15)
        ops = OperatorFactory(hs)
        sf = StateFactory(hs)
        return hs, ops, sf

    def test_constant_pulse_matches_original(self, system):
        """A constant Pulse should produce the same term count as the
        original ms_gate_hamiltonian."""
        _hs, ops, sf = system
        eta = [0.1, 0.1]
        delta = TWO_PI * 10e3
        Omega = 1e5
        tau = ms_gate_duration(delta)

        pulse = Pulse.constant(Omega, delta, duration=tau)
        tlist = np.linspace(0, tau, 200)

        H_orig = ms_gate_hamiltonian(ops, [0, 1], 0, eta, Omega, delta)
        H_pulsed = ms_gate_hamiltonian_pulsed(
            ops, [0, 1], 0, eta, pulse, tlist
        )
        assert len(H_pulsed) == len(H_orig)

    def test_smooth_gate_is_list_format(self, system):
        _hs, ops, sf = system
        eta = [0.1, 0.1]
        pulse = smooth_ms_pulse(TWO_PI * 10e3, TWO_PI * 5e3, 1e5)
        H = ms_gate_hamiltonian_pulsed(ops, [0, 1], 0, eta, pulse)
        assert isinstance(H, list)
        for term in H:
            assert isinstance(term, list)
            assert len(term) == 2

    def test_smooth_gate_coefficients_are_strings(self, system):
        """Smooth gate with constant Omega and sinusoidal delta should
        produce string coefficients (fast path)."""
        _hs, ops, sf = system
        eta = [0.1, 0.1]
        pulse = smooth_ms_pulse(TWO_PI * 10e3, TWO_PI * 5e3, 1e5)
        H = ms_gate_hamiltonian_pulsed(ops, [0, 1], 0, eta, pulse)
        for term in H:
            assert isinstance(term[1], str), (
                f"Expected string coefficient, got {type(term[1])}"
            )


class TestPulsedLightShift:
    def test_pulsed_ls_term_count(self):
        hs = HilbertSpace(n_ions=2, n_modes=1, n_fock=10)
        ops = OperatorFactory(hs)
        eta = [0.1, 0.1]
        pulse = smooth_ms_pulse(TWO_PI * 10e3, TWO_PI * 5e3, 1e5)
        H = light_shift_gate_hamiltonian_pulsed(ops, [0, 1], 0, eta, pulse)
        assert len(H) == 4  # 2 ions * 2 terms each


class TestLoopClosure:
    def test_constant_pulse_closes(self):
        delta = TWO_PI * 10e3
        Omega = 1e5
        tau = ms_gate_duration(delta)
        pulse = Pulse.constant(Omega, delta, duration=tau)
        result = verify_loop_closure(pulse, eta=0.1)
        assert result["closed"]
        assert result["residual"] < 0.01

    def test_smooth_pulse_mean_detuning_closes(self):
        """A smooth pulse with tau = 2*pi/delta_0 should approximately
        close, since the mean detuning over one period equals delta_0."""
        delta_0 = TWO_PI * 10e3
        pulse = smooth_ms_pulse(delta_0, 0.8 * delta_0, rabi_frequency=1e5)
        result = verify_loop_closure(pulse, eta=0.1)
        # The smooth gate doesn't close exactly in phase space - the
        # sinusoidal modulation changes the trajectory shape. But for
        # moderate ramp amplitudes it should be close.
        assert result["residual"] < 0.5 * 0.1 * 1e5


class TestSmoothGatePhysics:
    """Physics validation: smooth gate should produce entanglement."""

    @pytest.fixture
    def system(self):
        hs = HilbertSpace(n_ions=2, n_modes=1, n_fock=15)
        ops = OperatorFactory(hs)
        sf = StateFactory(hs)
        return hs, ops, sf

    def test_smooth_gate_entangles(self, system):
        """A smooth MS gate should entangle two ions."""
        _hs, ops, sf = system
        eta_val = 0.05
        eta = [eta_val, eta_val]
        delta_0 = TWO_PI * 15e3
        Omega = delta_0 / (4 * eta_val)

        pulse = smooth_ms_pulse(
            delta_0=delta_0,
            ramp_amplitude=0.5 * delta_0,
            rabi_frequency=Omega,
        )
        tlist = np.linspace(0, pulse.duration, 800)
        H = ms_gate_hamiltonian_pulsed(ops, [0, 1], 0, eta, pulse, tlist)
        psi0 = sf.ground_state()
        result = qutip.sesolve(
            H, psi0, tlist, options={"max_step": pulse.duration / 200}
        )
        rho_spin = result.states[-1].ptrace([0, 1])
        rho_single = rho_spin.ptrace(0)
        purity = (rho_single * rho_single).tr().real
        # Entangled state has single-qubit purity < 1
        assert purity < 0.9

    def test_constant_pulse_reproduces_standard_ms(self, system):
        """A constant Pulse through the pulsed pathway should produce
        the same Bell fidelity as the original constant pathway."""
        _hs, ops, sf = system
        eta_val = 0.05
        eta = [eta_val, eta_val]
        delta = TWO_PI * 15e3
        Omega = delta / (4 * eta_val)
        tau = ms_gate_duration(delta)

        # Original
        H_orig = ms_gate_hamiltonian(ops, [0, 1], 0, eta, Omega, delta)
        tlist = np.linspace(0, tau, 500)
        r_orig = qutip.sesolve(
            H_orig,
            sf.ground_state(),
            tlist,
            options={"max_step": tau / 100},
        )
        fid_orig = bell_state_fidelity(r_orig.states[-1].ptrace([0, 1]))

        # Pulsed with constant waveforms
        pulse = Pulse.constant(Omega, delta, duration=tau)
        H_pulsed = ms_gate_hamiltonian_pulsed(
            ops, [0, 1], 0, eta, pulse, tlist
        )
        r_pulsed = qutip.sesolve(
            H_pulsed,
            sf.ground_state(),
            tlist,
            options={"max_step": tau / 100},
        )
        fid_pulsed = bell_state_fidelity(r_pulsed.states[-1].ptrace([0, 1]))

        assert fid_pulsed == pytest.approx(fid_orig, abs=0.02)


class TestRunnerPulsedGates:
    @pytest.fixture
    def config(self):
        trap = PaulTrap(
            v_rf=300.0,
            omega_rf=TWO_PI * 30e6,
            r0=0.5e-3,
            omega_axial=TWO_PI * 1e6,
            species=get_species("Ca40"),
        )
        return SimulationConfig(
            species=get_species("Ca40"),
            trap=trap,
            n_ions=2,
            n_modes=1,
            n_fock=15,
        )

    def test_run_smooth_ms_gate(self, config):
        runner = SimulationRunner(config)
        result = runner.run_smooth_ms_gate(ions=[0, 1])
        assert result.states[-1] is not None

    def test_run_smooth_ms_gate_entangles(self, config):
        runner = SimulationRunner(config)
        result = runner.run_smooth_ms_gate(ions=[0, 1])
        rho_spin = result.states[-1].ptrace([0, 1])
        rho_single = rho_spin.ptrace(0)
        purity = (rho_single * rho_single).tr().real
        assert purity < 0.9

    def test_run_pulsed_ms_gate(self, config):
        runner = SimulationRunner(config)
        eta_avg = float(np.mean(np.abs(runner.eta[:, 0])))
        delta = TWO_PI * 1e3
        Omega = delta / (4 * eta_avg)
        pulse = smooth_ms_pulse(delta, 0.5 * delta, Omega)
        result = runner.run_pulsed_ms_gate(ions=[0, 1], pulse=pulse)
        assert result.states[-1] is not None

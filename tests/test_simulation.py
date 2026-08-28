import numpy as np
import pytest
import qutip

from tiqs.analysis.fidelity import bell_state_fidelity
from tiqs.constants import (
    BOHR_MAGNETON,
    ELECTRON_G_FACTOR,
    ELECTRON_MASS,
    HBAR,
    TWO_PI,
)
from tiqs.simulation.config import SimulationConfig
from tiqs.simulation.runner import SimulationRunner
from tiqs.species.electron import ElectronSpecies
from tiqs.species.ion import get_species
from tiqs.trap import PaulTrap


def _ca40_trap(omega_axial=TWO_PI * 1.0e6):
    return PaulTrap(
        v_rf=300.0,
        omega_rf=TWO_PI * 30e6,
        r0=0.5e-3,
        omega_axial=omega_axial,
        species=get_species("Ca40"),
    )


@pytest.fixture
def ca40_config():
    return SimulationConfig(
        species=get_species("Ca40"),
        trap=_ca40_trap(),
        n_ions=2,
        n_modes=1,
        n_fock=15,
        solver="sesolve",
    )


def _decay_rate(times, values):
    """Exponential decay rate from a positive-valued time series."""
    return -float(np.polyfit(times, np.log(values), 1)[0])


class TestSimulationConfig:
    def test_create_config(self, ca40_config):
        assert ca40_config.n_ions == 2
        assert ca40_config.solver == "sesolve"

    def test_config_with_noise(self):
        config = SimulationConfig(
            species=get_species("Ca40"),
            trap=_ca40_trap(),
            n_ions=1,
            n_modes=1,
            n_fock=10,
            solver="mesolve",
            heating_rate=10.0,
            t2_qubit=1.0,
        )
        assert config.solver == "mesolve"
        assert config.heating_rate == 10.0

    def test_unknown_solver_rejected(self):
        """A misspelled solver used to fall through to mesolve."""
        with pytest.raises(ValueError, match="Unknown solver"):
            SimulationConfig(
                species=get_species("Ca40"),
                trap=_ca40_trap(),
                n_ions=1,
                solver="mcsolv",
            )

    def test_n_modes_cannot_exceed_n_ions(self):
        """A chain of N ions has exactly N modes per direction.

        Extra modes get Hilbert-space dimensions, heating operators
        and thermal population but have no frequency and no
        Lamb-Dicke parameter.
        """
        with pytest.raises(ValueError, match="n_modes 3 exceeds n_ions 2"):
            SimulationConfig(
                species=get_species("Ca40"),
                trap=_ca40_trap(),
                n_ions=2,
                n_modes=3,
            )

    def test_fock_cutoff_validated(self):
        for bad in (0, 1, -3):
            with pytest.raises(ValueError, match="n_fock"):
                SimulationConfig(
                    species=get_species("Ca40"),
                    trap=_ca40_trap(),
                    n_ions=1,
                    n_fock=bad,
                )

    def test_coherence_times_validated(self):
        """Non-positive T1/T2 gave NaN operators or ZeroDivisionError."""
        with pytest.raises(ValueError, match="t1_qubit must be > 0"):
            SimulationConfig(
                species=get_species("Ca40"),
                trap=_ca40_trap(),
                n_ions=1,
                t1_qubit=-1e-3,
            )
        with pytest.raises(ValueError, match="t2_qubit must be > 0"):
            SimulationConfig(
                species=get_species("Ca40"),
                trap=_ca40_trap(),
                n_ions=1,
                t2_qubit=0.0,
            )

    def test_t2_above_2t1_rejected(self):
        """T2 <= 2*T1 is required: gamma_phi = 1/T2 - 1/(2*T1) >= 0."""
        with pytest.raises(ValueError, match="exceeds 2 \\* t1_qubit"):
            SimulationConfig(
                species=get_species("Ca40"),
                trap=_ca40_trap(),
                n_ions=1,
                t1_qubit=100e-6,
                t2_qubit=1e-3,
            )

    def test_negative_rates_rejected(self):
        """Negative rates were silently dropped or produced NaN states."""
        for kwargs, match in (
            ({"heating_rate": -100.0}, "heating_rate"),
            ({"heating_rates": [-100.0]}, "heating_rates"),
            ({"n_bar_initial": -0.5}, "n_bar_initial"),
            ({"photon_scattering_rate": -1.0}, "photon_scattering_rate"),
            ({"motional_dephasing_rates": [-1.0]}, "motional_dephasing"),
            ({"laser_phase_linewidth": -1.0}, "laser_phase_linewidth"),
        ):
            with pytest.raises(ValueError, match=match):
                SimulationConfig(
                    species=get_species("Ca40"),
                    trap=_ca40_trap(),
                    n_ions=1,
                    n_modes=1,
                    **kwargs,
                )

    def test_ntraj_validated(self):
        with pytest.raises(ValueError, match="ntraj"):
            SimulationConfig(
                species=get_species("Ca40"),
                trap=_ca40_trap(),
                n_ions=1,
                solver="mcsolve",
                ntraj=0,
            )

    def test_potentials_key_validated(self):
        """A potential on a nonexistent mode was silently accepted."""
        from tiqs.potential import DuffingPotential

        pot = DuffingPotential(omega=TWO_PI * 1e6, anharmonicity=0.0)
        with pytest.raises(ValueError, match="potentials key 1"):
            SimulationConfig(
                species=get_species("Ca40"),
                trap=_ca40_trap(),
                n_ions=2,
                n_modes=1,
                potentials={1: pot},
            )

    def test_lamb_dicke_order_field_removed(self):
        """The inert lamb_dicke_order knob is gone, not silently ignored.

        Second-order Lamb-Dicke physics lives in
        ``full_interaction_hamiltonian``, which the runner never
        calls, so the config field could never take effect.
        """
        with pytest.raises(TypeError):
            SimulationConfig(
                species=get_species("Ca40"),
                trap=_ca40_trap(),
                n_ions=1,
                # ty: the point of the test is that this keyword no
                # longer exists, so the static error is the assertion.
                lamb_dicke_order=2,  # ty: ignore[unknown-argument]
            )


class TestSimulationRunner:
    def test_single_qubit_rabi(self, ca40_config):
        runner = SimulationRunner(ca40_config)
        result = runner.run_carrier_pulse(ion=0, theta=np.pi, duration=None)
        sz = runner.ops.sigma_z(0)
        final_sz = qutip.expect(sz, result.states[-1])
        # Pi-pulse flips |0> (sz=+1) to |1> (sz=-1)
        assert final_sz == pytest.approx(-1.0, abs=0.1)

    def test_ms_gate_entangles(self, ca40_config):
        runner = SimulationRunner(ca40_config)
        result = runner.run_ms_gate(ions=[0, 1], mode=0)
        rho_spin = qutip.ket2dm(result.states[-1]).ptrace([0, 1])
        rho_single = rho_spin.ptrace(0)
        purity = (rho_single * rho_single).tr().real
        assert purity < 0.9  # entangled: reduced state is mixed

    def test_runner_with_noise(self):
        config = SimulationConfig(
            species=get_species("Ca40"),
            trap=_ca40_trap(),
            n_ions=1,
            n_modes=1,
            n_fock=10,
            solver="mesolve",
            heating_rate=1e4,
            t2_qubit=1e-3,
        )
        runner = SimulationRunner(config)
        result = runner.run_carrier_pulse(ion=0, theta=np.pi)
        # With noise, fidelity should be less than perfect
        sz = runner.ops.sigma_z(0)
        final_sz = qutip.expect(sz, result.states[-1])
        assert final_sz < 1.0  # imperfect due to noise

    def test_t2_honored_when_t1_also_set(self):
        r"""Coherence must decay at exactly 1/T2, not 1/T2 + 1/(2 T1).

        A diagonal collapse operator $L$ damps $\rho_{01}$ at
        $|l_0 - l_1|^2/2$, so $L=\sqrt{\gamma_\phi/2}\,\sigma_z$
        contributes $\gamma_\phi$ and $L=\sqrt{1/T_1}\,\sigma_+$
        contributes $1/(2 T_1)$; the two add. Inverting that is why
        $\gamma_\phi = 1/T_2 - 1/(2 T_1)$, so the analytic answer is
        $\langle\sigma_x\rangle(t) = e^{-t/T_2}$. Without the $T_1$
        subtraction the measured rate is 1.83x too large here.
        """
        t1, t2 = 6e-4, 1e-3
        config = SimulationConfig(
            species=get_species("Ca40"),
            trap=_ca40_trap(),
            n_ions=1,
            n_modes=1,
            n_fock=3,
            solver="mesolve",
            t1_qubit=t1,
            t2_qubit=t2,
        )
        runner = SimulationRunner(config)
        assert len(runner._c_ops) == 2
        psi = qutip.tensor(
            (qutip.basis(2, 0) + qutip.basis(2, 1)).unit(),
            qutip.basis(3, 0),
        )
        tlist = np.linspace(0, t2 / 2, 40)
        result = qutip.mesolve(
            0 * runner.ops.identity(),
            psi,
            tlist,
            c_ops=runner._c_ops,
            e_ops=[runner.ops.sigma_x(0)],
        )
        analytic = np.exp(-tlist / t2)
        assert np.abs(result.expect[0] - analytic).max() < 1e-4
        assert _decay_rate(tlist, result.expect[0]) == pytest.approx(
            1.0 / t2, rel=1e-3
        )

    def test_ms_gate_on_odd_parity_mode(self):
        """The two-ion stretch mode must work and flip the Bell phase.

        The Magnus expansion of the code's own Hamiltonian gives
        chi = 4 pi K eta_i eta_j Omega^2 / delta^2, so only
        |eta_i eta_j| sets the required drive strength while its sign
        sets the sign of the geometric phase. Taking sqrt of the
        signed product gave NaN for every pair whose participations
        have opposite signs -- always the case on the stretch mode.
        """
        config = SimulationConfig(
            species=get_species("Ca40"),
            trap=_ca40_trap(),
            n_ions=2,
            n_modes=2,
            n_fock=6,
            solver="sesolve",
        )
        runner = SimulationRunner(config)
        detuning = TWO_PI * 1e3
        eta_i, eta_j = runner.eta[0, 1], runner.eta[1, 1]
        assert eta_i * eta_j < 0  # stretch mode is odd-parity
        expected_sign = int(np.sign(detuning * eta_i * eta_j))

        result = runner.run_ms_gate(
            ions=[0, 1], mode=1, detuning=detuning, n_steps=300
        )
        final = result.states[-1]
        rho_spin = final.ptrace([0, 1])
        assert bell_state_fidelity(rho_spin, expected_sign) > 0.999
        assert bell_state_fidelity(rho_spin, -expected_sign) < 1e-3
        # Maximally entangled: each qubit is exactly half-mixed.
        rho_single = rho_spin.ptrace(0)
        purity = (rho_single * rho_single).tr().real
        assert purity == pytest.approx(0.5, abs=1e-3)
        # The phase-space loop closes, so the motion returns to vacuum.
        assert qutip.expect(runner.ops.number(1), final) < 1e-3

    def test_ms_gate_argument_validation(self, ca40_config):
        """Degenerate gate arguments used to 'succeed' or die opaquely."""
        runner = SimulationRunner(ca40_config)
        with pytest.raises(ValueError, match="exactly 2 ions"):
            runner.run_ms_gate(ions=[0])
        with pytest.raises(ValueError, match="two distinct ions"):
            runner.run_ms_gate(ions=[0, 0])
        with pytest.raises(ValueError, match="ion index 5"):
            runner.run_ms_gate(ions=[0, 5])
        with pytest.raises(ValueError, match="mode 1 out of range"):
            runner.run_ms_gate(ions=[0, 1], mode=1)
        with pytest.raises(ValueError, match="loops must be >= 1"):
            runner.run_ms_gate(ions=[0, 1], loops=0)
        with pytest.raises(ValueError, match="detuning must be nonzero"):
            runner.run_ms_gate(ions=[0, 1], detuning=0.0)

    def test_ms_gate_rejects_uncoupled_ion(self):
        """The middle ion of a 3-ion chain has no stretch-mode motion.

        Its participation is a floating-point zero from ``eigh``, so
        the calibration used to divide by ~1e-18 and the sign of the
        result was decided by numerical noise.
        """
        config = SimulationConfig(
            species=get_species("Ca40"),
            trap=_ca40_trap(),
            n_ions=3,
            n_modes=3,
            n_fock=4,
            solver="sesolve",
        )
        runner = SimulationRunner(config)
        assert abs(runner.eta[1, 1]) < 1e-12
        with pytest.raises(ValueError, match="vanishing"):
            runner.run_ms_gate(ions=[0, 1], mode=1)

    def test_carrier_pulse_argument_validation(self, ca40_config):
        runner = SimulationRunner(ca40_config)
        with pytest.raises(ValueError, match="ion index 7"):
            runner.run_carrier_pulse(ion=7, theta=np.pi)
        with pytest.raises(ValueError, match="rabi_frequency must be > 0"):
            runner.run_carrier_pulse(
                ion=0, theta=np.pi, rabi_frequency=-TWO_PI * 1e5
            )
        with pytest.raises(ValueError, match="rabi_frequency must be > 0"):
            runner.run_carrier_pulse(ion=0, theta=np.pi, rabi_frequency=0.0)

    def test_species_without_laser_transition_requires_k_eff(self):
        """Electrons have no laser transition, so no k_eff to guess.

        The old 2*pi/400nm fallback gave eta = 8.7 for the repo's own
        electron trap, ~140x the magnetic-gradient value and far
        outside the eta*sqrt(2*n+1) << 1 regime every first-order
        Hamiltonian in the package assumes.
        """
        electron = ElectronSpecies(magnetic_field=0.1)
        omega_axial = TWO_PI * 30e6
        trap = PaulTrap(
            v_rf=7.8,
            omega_rf=TWO_PI * 1.6e9,
            r0=300e-6,
            omega_axial=omega_axial,
            species=electron,
        )
        with pytest.raises(ValueError, match="k_eff"):
            SimulationRunner(
                SimulationConfig(
                    species=electron,
                    trap=trap,
                    n_ions=1,
                    n_modes=1,
                    n_fock=3,
                )
            )

        # Mintert and Wunderlich, PRL 87, 257904 (2001), as applied in
        # Weidt et al., PRL 117, 220501 (2016): the gradient plays the
        # role of the laser wavevector.
        gradient = 120.0  # T/m
        k_eff = (
            ELECTRON_G_FACTOR * BOHR_MAGNETON * gradient / (HBAR * omega_axial)
        )
        assert k_eff == pytest.approx(1.121e5, rel=1e-3)
        runner = SimulationRunner(
            SimulationConfig(
                species=electron,
                trap=trap,
                n_ions=1,
                n_modes=1,
                n_fock=3,
                k_eff=k_eff,
            )
        )
        x_zpf = np.sqrt(HBAR / (2 * ELECTRON_MASS * omega_axial))
        assert abs(runner.eta[0, 0]) == pytest.approx(k_eff * x_zpf, rel=1e-12)
        assert abs(runner.eta[0, 0]) == pytest.approx(0.0621, abs=1e-4)

    def test_k_eff_list_length_validated(self):
        with pytest.raises(ValueError, match="k_eff list length"):
            SimulationRunner(
                SimulationConfig(
                    species=get_species("Ca40"),
                    trap=_ca40_trap(),
                    n_ions=2,
                    n_modes=1,
                    n_fock=3,
                    k_eff=[1e7],
                )
            )

    def test_eta_spans_all_axial_modes(self):
        """``runner.eta`` is (n_ions, n_ions), not (n_ions, n_modes)."""
        config = SimulationConfig(
            species=get_species("Ca40"),
            trap=_ca40_trap(),
            n_ions=3,
            n_modes=1,
            n_fock=3,
        )
        runner = SimulationRunner(config)
        assert runner.eta.shape == (3, 3)


class TestNoiseWiring:
    """The runner must actually build the channels it is configured for."""

    def _dephasing_config(self, **kwargs):
        return SimulationConfig(
            species=get_species("Ca40"),
            trap=_ca40_trap(),
            n_ions=1,
            n_modes=1,
            n_fock=4,
            solver="mesolve",
            **kwargs,
        )

    def test_photon_scattering_depolarizes_the_qubit(self):
        r"""Raman scattering at rate $\Gamma$ decays $\sigma_z$ as
        $e^{-\Gamma t}$.

        The bidirectional pair $L_\pm = \sqrt{\Gamma/2}\,\sigma_\pm$
        gives $\dot p_1 = (\Gamma/2)(p_0 - p_1)$, hence
        $\langle\sigma_z\rangle(t) =
        \langle\sigma_z\rangle(0)\,e^{-\Gamma t}$. Setting
        ``photon_scattering_rate`` used to build zero collapse
        operators, leaving the state bit-identical.
        """
        rate = 1e4
        runner = SimulationRunner(
            self._dephasing_config(photon_scattering_rate=rate)
        )
        assert len(runner._c_ops) == 2
        psi = qutip.tensor(qutip.basis(2, 1), qutip.basis(4, 0))
        tlist = np.linspace(0, 2 / rate, 40)
        result = qutip.mesolve(
            0 * runner.ops.identity(),
            psi,
            tlist,
            c_ops=runner._c_ops,
            e_ops=[runner.ops.sigma_z(0)],
        )
        analytic = -np.exp(-rate * tlist)
        assert np.abs(result.expect[0] - analytic).max() < 1e-4

    def test_motional_dephasing_scales_as_fock_gap_squared(self):
        r"""$L = \sqrt{\gamma}\,\hat{n}$ damps $\rho_{ab}$ at
        $\gamma (a-b)^2/2$.

        The quadratic dependence on the Fock gap is the signature of
        a number-operator collapse channel, so the |0>-|2> coherence
        must decay exactly 4x faster than |0>-|1>. Setting a motional
        dephasing rate used to build no collapse operator at all.
        """
        gamma = 5e3
        runner = SimulationRunner(
            self._dephasing_config(motional_dephasing_rates=[gamma])
        )
        assert len(runner._c_ops) == 1
        tlist = np.linspace(0, 4 / gamma, 40)
        rates = {}
        for gap in (1, 2):
            psi = qutip.tensor(
                qutip.basis(2, 0),
                (qutip.basis(4, 0) + qutip.basis(4, gap)).unit(),
            )
            result = qutip.mesolve(
                0 * runner.ops.identity(),
                psi,
                tlist,
                c_ops=runner._c_ops,
            )
            coherence = [
                abs(s.ptrace(1).full()[0, gap]) for s in result.states
            ]
            rates[gap] = _decay_rate(tlist, coherence)
        assert rates[1] == pytest.approx(gamma / 2, rel=1e-3)
        assert rates[2] / rates[1] == pytest.approx(4.0, rel=1e-3)

    def test_laser_phase_noise_rate_scales_with_linewidth(self):
        """Qubit dephasing from laser phase noise is linear in linewidth.

        A collapse operator proportional to sqrt(linewidth) * sigma_z
        damps coherences at a rate proportional to the linewidth, so
        doubling the linewidth doubles the decay rate. Setting
        ``laser_phase_linewidth`` used to build no operator at all.
        """
        base = TWO_PI * 1e3
        rates = {}
        for factor in (1, 2):
            runner = SimulationRunner(
                self._dephasing_config(laser_phase_linewidth=factor * base)
            )
            assert len(runner._c_ops) == 1
            psi = qutip.tensor(
                (qutip.basis(2, 0) + qutip.basis(2, 1)).unit(),
                qutip.basis(4, 0),
            )
            tlist = np.linspace(0, 2 / base, 40)
            result = qutip.mesolve(
                0 * runner.ops.identity(),
                psi,
                tlist,
                c_ops=runner._c_ops,
                e_ops=[runner.ops.sigma_x(0)],
            )
            rates[factor] = _decay_rate(tlist, result.expect[0])
        assert rates[1] > 0
        assert rates[2] / rates[1] == pytest.approx(2.0, rel=1e-3)

    def test_noise_channels_are_per_ion_and_per_mode(self):
        """Counts: one dephasing op per ion, one heating pair per mode."""
        config = SimulationConfig(
            species=get_species("Ca40"),
            trap=_ca40_trap(),
            n_ions=3,
            n_modes=2,
            n_fock=3,
            solver="mesolve",
            heating_rates=[10.0, 20.0],
            motional_dephasing_rates=[1e3, 0.0],
            t2_qubit=1e-3,
            photon_scattering_rate=1.0,
            laser_phase_linewidth=1.0,
        )
        runner = SimulationRunner(config)
        # 2 modes * 2 heating ops + 1 motional dephasing (the 0 rate is
        # skipped) + 3 ions * (dephasing + 2 Raman + laser phase)
        assert len(runner._c_ops) == 4 + 1 + 3 * 4


class TestAnharmonicSimulation:
    def test_config_accepts_potentials(self):
        """SimulationConfig can be created with a potentials dict."""
        from tiqs.potential import DuffingPotential

        pot = DuffingPotential(
            omega=TWO_PI * 1e6,
            anharmonicity=-TWO_PI * 50e3,
        )
        config = SimulationConfig(
            species=get_species("Ca40"),
            trap=_ca40_trap(),
            n_ions=1,
            n_modes=1,
            n_fock=15,
            solver="sesolve",
            potentials={0: pot},
        )
        assert 0 in config.potentials

    def test_runner_with_duffing_potential(self):
        """Carrier pulse with anharmonic mode still produces Rabi
        oscillations (anharmonic correction is on the motion,
        not the spin)."""
        from tiqs.potential import DuffingPotential

        pot = DuffingPotential(
            omega=TWO_PI * 1e6,
            anharmonicity=-TWO_PI * 50e3,
        )
        config = SimulationConfig(
            species=get_species("Ca40"),
            trap=_ca40_trap(),
            n_ions=1,
            n_modes=1,
            n_fock=15,
            solver="sesolve",
            potentials={0: pot},
        )
        runner = SimulationRunner(config)
        result = runner.run_carrier_pulse(ion=0, theta=np.pi)
        sz = runner.ops.sigma_z(0)
        final_sz = qutip.expect(sz, result.states[-1])
        assert final_sz == pytest.approx(-1.0, abs=0.15)

    def test_anharmonic_correction_subtracts_the_mode_frequency(self):
        """H_correction = H_potential - omega_mode * n, always.

        The harmonic part removed is the physical normal-mode
        frequency, so a Duffing potential contributes
        (omega_pot - omega_mode) n + (alpha/2) n(n-1). A branch that
        subtracted ``potential.omega`` instead for out-of-range mode
        indices made the linear term a function of an array length
        rather than of physics; those indices are now rejected
        outright.
        """
        from tiqs.potential import DuffingPotential

        omega_pot = TWO_PI * 7e6
        alpha = -TWO_PI * 50e3
        pot = DuffingPotential(omega=omega_pot, anharmonicity=alpha)
        config = SimulationConfig(
            species=get_species("Ca40"),
            trap=_ca40_trap(),
            n_ions=2,
            n_modes=2,
            n_fock=8,
            solver="sesolve",
            potentials={1: pot},
        )
        runner = SimulationRunner(config)
        omega_mode = runner.modes.modes["axial"].freqs[1]
        H = runner._anharmonic_H
        for n in (0, 1, 2, 3):
            psi = qutip.tensor(
                qutip.basis(2, 0),
                qutip.basis(2, 0),
                qutip.basis(8, 0),
                qutip.basis(8, n),
            )
            expected = (omega_pot - omega_mode) * n + (alpha / 2) * n * (n - 1)
            assert qutip.expect(H, psi) == pytest.approx(
                expected, rel=1e-10, abs=1e-6
            )

    def test_runner_with_t1_qubit(self):
        """t1_qubit adds spontaneous emission collapse operators."""
        config = SimulationConfig(
            species=get_species("Ca40"),
            trap=_ca40_trap(),
            n_ions=1,
            n_modes=1,
            n_fock=5,
            solver="mesolve",
            t1_qubit=100e-6,
        )
        runner = SimulationRunner(config)
        assert len(runner._c_ops) == 1
        result = runner.run_carrier_pulse(ion=0, theta=np.pi, n_steps=50)
        assert len(result.states) > 0

    def test_mcsolve_agrees_with_mesolve(self):
        """Trajectory averaging must reproduce the master equation.

        The two solvers integrate the same Lindblad generator, so
        their expectation values agree up to the Monte-Carlo standard
        error sqrt(Var/ntraj); Var <= 1 - <sz>^2 because
        <sz> is bounded by 1. The previous test only asserted that
        some states came back, and ntraj was hard-coded to 100 with
        no way to change it.
        """
        ntraj = 300
        options = {
            "max_step": 0.0,
            "nsteps": 5000,
            "progress_bar": False,
        }
        kwargs = dict(
            species=get_species("Ca40"),
            trap=_ca40_trap(),
            n_ions=1,
            n_modes=1,
            n_fock=3,
            t2_qubit=2e-5,
        )
        mc_runner = SimulationRunner(
            SimulationConfig(
                solver="mcsolve",
                ntraj=ntraj,
                solver_options=options,
                **kwargs,
            )
        )
        me_runner = SimulationRunner(
            SimulationConfig(solver="mesolve", **kwargs)
        )
        mc = mc_runner.run_carrier_pulse(ion=0, theta=np.pi, n_steps=40)
        me = me_runner.run_carrier_pulse(ion=0, theta=np.pi, n_steps=40)
        assert mc.num_trajectories == ntraj
        sz = mc_runner.ops.sigma_z(0)
        mc_sz = qutip.expect(sz, mc.states[-1])
        me_sz = qutip.expect(sz, me.states[-1])
        std_err = np.sqrt(max(1 - me_sz**2, 0.0) / ntraj)
        assert abs(mc_sz - me_sz) < 5 * std_err

    def test_anharmonic_ms_gate(self):
        """Anharmonic correction applied to list-format (time-dependent)
        Hamiltonian from MS gate."""
        from tiqs.potential import DuffingPotential

        pot = DuffingPotential(
            omega=TWO_PI * 1e6,
            anharmonicity=-TWO_PI * 50e3,
        )
        config = SimulationConfig(
            species=get_species("Ca40"),
            trap=_ca40_trap(),
            n_ions=2,
            n_modes=1,
            n_fock=15,
            solver="sesolve",
            potentials={0: pot},
        )
        runner = SimulationRunner(config)
        result = runner.run_ms_gate(ions=[0, 1])
        rho_spin = result.states[-1].ptrace([0, 1])
        purity = (rho_spin * rho_spin).tr().real
        # Should still entangle (purity < 1)
        assert purity < 0.9

    def test_identity_potential_reproduces_the_harmonic_run(self):
        """A zero-anharmonicity potential at the mode frequency is a no-op.

        ``H_correction = omega_mode * n - omega_mode * n = 0``
        exactly, so the two runs must agree state-by-state. The
        previous version of this test asserted only that
        ``config.potentials == {}``.
        """
        from tiqs.potential import DuffingPotential

        base = dict(
            species=get_species("Ca40"),
            trap=_ca40_trap(),
            n_ions=1,
            n_modes=1,
            n_fock=10,
            solver="sesolve",
        )
        plain = SimulationRunner(SimulationConfig(**base))
        omega_mode = float(plain.modes.modes["axial"].freqs[0])
        pot = DuffingPotential(omega=omega_mode, anharmonicity=0.0)
        with_pot = SimulationRunner(
            SimulationConfig(potentials={0: pot}, **base)
        )
        assert with_pot._anharmonic_H.norm() < 1e-8 * omega_mode

        a = plain.run_carrier_pulse(ion=0, theta=np.pi, n_steps=25)
        b = with_pot.run_carrier_pulse(ion=0, theta=np.pi, n_steps=25)
        for sa, sb in zip(a.states, b.states, strict=True):
            assert (sa - sb).norm() < 1e-8


class TestSympatheticCoolingRunner:
    def _mixed_config(
        self, species, coolant_indices, trap, n_fock=15, **kwargs
    ):
        return SimulationConfig(
            species=species,
            trap=trap,
            n_ions=len(species),
            n_modes=1,
            n_fock=n_fock,
            solver="mesolve",
            coolant_indices=coolant_indices,
            **kwargs,
        )

    def test_coolant_species_must_be_homogeneous(self):
        """One recoil frequency is factored out of the whole coolant set.

        Be-9 and Ca-40 recoil frequencies differ by ~7x, so scoring
        both with one species' constants is simply wrong.
        """
        be9, ca40 = get_species("Be9"), get_species("Ca40")
        trap = _ca40_trap()
        with pytest.raises(ValueError, match="same species"):
            SimulationRunner(
                self._mixed_config([be9, ca40, ca40], [0, 1], trap)
            )
        runner = SimulationRunner(
            self._mixed_config([be9, be9, ca40], [0, 1], trap)
        )
        assert runner._cooling_rates is not None

    def test_default_path_cools_toward_the_doppler_limit(self):
        """The runner's own defaults must cool, not heat.

        With every ion a coolant the participation is exactly 1, so
        the target is the bare Doppler limit
        n_bar = Gamma / (2 omega_m). At omega_axial = 2*pi*5 MHz that
        is ~1.9 quanta for Be-9's 313 nm transition, comfortably
        inside the Fock cutoff; the default path used to aim at 111
        quanta, above the Fock ceiling, and heat instead.
        """
        be9 = get_species("Be9")
        omega_axial = TWO_PI * 5e6
        trap = PaulTrap(
            v_rf=300.0,
            omega_rf=TWO_PI * 30e6,
            r0=0.5e-3,
            omega_axial=omega_axial,
            species=be9,
        )
        runner = SimulationRunner(
            self._mixed_config([be9, be9], [0, 1], trap, n_fock=25)
        )
        gamma = be9.cooling_transition.linewidth
        omega_com = float(runner.modes.modes["axial"].freqs[0])
        assert runner._n_bar_cooled[0] == pytest.approx(
            gamma / (2 * omega_com), rel=1e-9
        )
        assert runner._n_bar_cooled[0] < runner.config.n_fock / 2

        rho0 = runner.sf.thermal_state(n_bar=[3.0])
        n_op = runner.ops.number(0)
        n_before = qutip.expect(n_op, rho0)
        rate = float(runner._cooling_rates[0])
        cooled = runner.run_sympathetic_cooling(rho0, duration=20 / rate)
        n_after = qutip.expect(n_op, cooled)
        assert n_after < n_before
        assert n_after == pytest.approx(runner._n_bar_cooled[0], rel=0.1)

"""Tests for sympathetic cooling: analytical exactness checks and
simulation validation."""

import numpy as np
import pytest
import qutip

from tiqs.chain.normal_modes import normal_modes
from tiqs.constants import TWO_PI
from tiqs.cooling.doppler import doppler_cooled_nbar
from tiqs.cooling.sympathetic import (
    apply_sympathetic_cooling,
    coolant_participation,
    sympathetic_cooling_rate,
    sympathetic_doppler_nbar,
    sympathetic_sideband_nbar,
)
from tiqs.hilbert_space.builder import HilbertSpace
from tiqs.hilbert_space.operators import OperatorFactory
from tiqs.hilbert_space.states import StateFactory
from tiqs.simulation.config import SimulationConfig
from tiqs.simulation.runner import SimulationRunner
from tiqs.species.ion import get_species
from tiqs.trap import PaulTrap


@pytest.fixture
def ca40():
    return get_species("Ca40")


@pytest.fixture
def be9():
    return get_species("Be9")


@pytest.fixture
def ca40_trap(ca40):
    return PaulTrap(
        v_rf=300.0,
        omega_rf=TWO_PI * 30e6,
        r0=0.5e-3,
        omega_axial=TWO_PI * 1.0e6,
        species=ca40,
    )


class TestCoolantParticipation:
    """Analytical exactness tests for coolant participation."""

    @pytest.mark.parametrize("n_ions", [2, 3])
    def test_single_species_all_coolant(self, ca40_trap, n_ions):
        """When all ions are coolants, P_m = 1 for every mode.
        Follows from eigenvector orthonormality: sum_i |b_{i,m}|^2 = 1."""
        modes = normal_modes(n_ions, ca40_trap)
        axial = modes.modes["axial"]
        P = coolant_participation(axial, list(range(n_ions)))
        np.testing.assert_allclose(P, 1.0, atol=1e-10)

    def test_coolant_plus_logic_equals_one(self, be9, ca40, ca40_trap):
        """P_m(coolant) + P_m(logic) = 1 for every mode."""
        masses = np.array([be9.mass_kg, ca40.mass_kg])
        modes = normal_modes(2, ca40_trap, masses=masses)
        axial = modes.modes["axial"]
        P_coolant = coolant_participation(axial, [0])
        P_logic = coolant_participation(axial, [1])
        np.testing.assert_allclose(P_coolant + P_logic, 1.0, atol=1e-10)

    def test_single_ion_as_coolant(self, ca40_trap):
        """One coolant out of two ions: P_m = |b_{0,m}|^2 < 1."""
        modes = normal_modes(2, ca40_trap)
        axial = modes.modes["axial"]
        P = coolant_participation(axial, [0])
        assert all(P < 1.0)
        assert all(P > 0.0)

    def test_com_mode_equal_participation(self, ca40_trap):
        """For single-species COM mode, each ion contributes 1/N."""
        modes = normal_modes(2, ca40_trap)
        axial = modes.modes["axial"]
        P_one = coolant_participation(axial, [0])
        assert P_one[0] == pytest.approx(0.5, rel=1e-6)

    def test_mixed_species_lighter_dominates_high_mode(
        self, be9, ca40, ca40_trap
    ):
        """Be9 (lighter) has larger participation in the out-of-phase
        mode. This is already validated in test_chain.py but we confirm
        the participation function agrees."""
        masses = np.array([be9.mass_kg, ca40.mass_kg])
        modes = normal_modes(2, ca40_trap, masses=masses)
        axial = modes.modes["axial"]
        P_be = coolant_participation(axial, [0])
        P_ca = coolant_participation(axial, [1])
        # Out-of-phase mode (index 1): Be9 dominates
        assert P_be[1] > P_ca[1]


class TestSympatheticDopplerNbar:
    """Analytical exactness tests for the sympathetic Doppler limit."""

    def test_reduces_to_standard_doppler(self, ca40, ca40_trap):
        """When P_m = 1 (all ions are coolants), sympathetic Doppler
        limit equals the standard Doppler limit exactly."""
        modes = normal_modes(1, ca40_trap)
        axial = modes.modes["axial"]
        P = coolant_participation(axial, [0])
        n_bar = sympathetic_doppler_nbar(ca40, axial.freqs, P)
        n_bar_standard = doppler_cooled_nbar(ca40, axial.freqs[0] / TWO_PI)
        assert n_bar[0] == pytest.approx(n_bar_standard, rel=1e-6)

    def test_lower_participation_higher_nbar(self, be9, ca40, ca40_trap):
        """Mode with lower coolant participation has higher n_bar."""
        masses = np.array([be9.mass_kg, ca40.mass_kg])
        modes = normal_modes(2, ca40_trap, masses=masses)
        axial = modes.modes["axial"]
        # Ca40 (heavier coolant at index 1) has higher participation
        # in COM mode (0) than stretch mode (1)
        P = coolant_participation(axial, [1])
        assert P[0] > P[1]
        n_bar = sympathetic_doppler_nbar(ca40, axial.freqs, P)
        assert n_bar[0] < n_bar[1]

    def test_nbar_scales_as_inverse_participation(self, ca40, ca40_trap):
        """n_bar_m is exactly proportional to 1/P_m at fixed omega."""
        modes = normal_modes(2, ca40_trap)
        axial = modes.modes["axial"]
        # Artificial participation values
        P = np.array([0.5, 0.25])
        n_bar = sympathetic_doppler_nbar(ca40, axial.freqs, P)
        # At different freqs the ratio won't be exactly 2, but
        # n_bar * P should be proportional to 1/omega
        gamma = ca40.cooling_transition.linewidth
        expected = gamma / (2 * axial.freqs * P)
        np.testing.assert_allclose(n_bar, expected, rtol=1e-10)


class TestSympatheticSidebandNbar:
    def test_sideband_limit_formula(self):
        """Direct formula check: n_bar = (gamma_eff/(2*omega))^2 / P."""
        gamma_eff = TWO_PI * 1e3
        freqs = np.array([TWO_PI * 1e6, TWO_PI * 2e6])
        P = np.array([0.8, 0.3])
        n_bar = sympathetic_sideband_nbar(gamma_eff, freqs, P)
        expected = (gamma_eff / (2 * freqs)) ** 2 / P
        np.testing.assert_allclose(n_bar, expected, rtol=1e-10)

    def test_full_participation_matches_standard(self):
        """At P=1, reduces to standard sideband formula."""
        gamma_eff = TWO_PI * 1e3
        freq = np.array([TWO_PI * 1e6])
        P = np.array([1.0])
        n_bar = sympathetic_sideband_nbar(gamma_eff, freq, P)
        expected = (gamma_eff / (2 * freq[0])) ** 2
        assert n_bar[0] == pytest.approx(expected, rel=1e-10)


class TestSympatheticCoolingRate:
    """Analytical exactness tests for per-mode cooling rates."""

    def test_rate_proportional_to_participation(self, ca40):
        """rate_m1 / rate_m2 = P_m1 / P_m2 exactly."""
        P = np.array([0.8, 0.2])
        rates = sympathetic_cooling_rate(ca40, P)
        assert rates[0] / rates[1] == pytest.approx(P[0] / P[1], rel=1e-10)

    def test_rate_formula(self, ca40):
        """Direct formula check: rate = (Gamma/2) * P * s/(1+s)."""
        P = np.array([0.6])
        s = 0.5
        rates = sympathetic_cooling_rate(ca40, P, s)
        gamma = ca40.cooling_transition.linewidth
        expected = (gamma / 2) * P[0] * s / (1 + s)
        assert rates[0] == pytest.approx(expected, rel=1e-10)

    def test_zero_participation_zero_rate(self, ca40):
        """Spectator mode (P=0) has zero cooling rate."""
        P = np.array([0.0])
        rates = sympathetic_cooling_rate(ca40, P)
        assert rates[0] == pytest.approx(0.0, abs=1e-30)

    def test_saturation_parameter_effect(self, ca40):
        """Higher saturation gives higher rate up to the limit."""
        P = np.array([1.0])
        rate_low = sympathetic_cooling_rate(ca40, P, 0.1)
        rate_high = sympathetic_cooling_rate(ca40, P, 10.0)
        assert rate_high[0] > rate_low[0]


class TestApplySympatheticCooling:
    """Simulation tests for the density-matrix cooling channel."""

    @pytest.fixture
    def system(self):
        hs = HilbertSpace(n_ions=2, n_modes=1, n_fock=20)
        return OperatorFactory(hs), StateFactory(hs)

    def test_cooling_reduces_phonon_number(self, system):
        """Starting from n_bar=5, cooling should reduce phonon number."""
        ops, sf = system
        rho0 = sf.thermal_state(n_bar=[5.0])
        n_op = ops.number(0)
        n_before = qutip.expect(n_op, rho0)

        cooling_rates = np.array([TWO_PI * 1e6])
        n_bar_target = np.array([0.5])
        rho_cooled = apply_sympathetic_cooling(
            rho0, ops, cooling_rates, n_bar_target, duration=1e-6
        )
        n_after = qutip.expect(n_op, rho_cooled)
        assert n_after < n_before

    def test_qubit_coherence_preserved(self, system):
        """Sympathetic cooling must preserve qubit off-diagonal
        elements (coherence) since only motional operators are used."""
        ops, sf = system
        # Create a superposition state: (|0> + |1>)/sqrt(2) on ion 0
        plus = (
            sf.product_state([0, 0], [0]) + sf.product_state([1, 0], [0])
        ).unit()
        rho0 = qutip.ket2dm(plus)

        cooling_rates = np.array([TWO_PI * 1e6])
        n_bar_target = np.array([0.1])
        rho_cooled = apply_sympathetic_cooling(
            rho0, ops, cooling_rates, n_bar_target, duration=1e-6
        )

        # Qubit coherence: off-diagonal of single-qubit reduced state
        rho_q_before = rho0.ptrace(0)
        rho_q_after = rho_cooled.ptrace(0)
        assert abs(rho_q_after[0, 1]) == pytest.approx(
            abs(rho_q_before[0, 1]), abs=0.01
        )

    def test_long_duration_reaches_steady_state(self):
        """Cooling a qubit+mode system to steady state preserves
        qubit coherence and reaches the target phonon number."""
        hs = HilbertSpace(n_ions=1, n_modes=1, n_fock=10)
        ops = OperatorFactory(hs)
        sf = StateFactory(hs)
        # Qubit in superposition, mode thermally excited
        plus = (sf.product_state([0], [0]) + sf.product_state([1], [0])).unit()
        rho0 = qutip.ket2dm(plus)
        # Heat the motional mode by hand: replace mode part with thermal
        rho_q = rho0.ptrace(0)
        rho_m = qutip.thermal_dm(10, 3.0)
        rho0 = qutip.tensor(rho_q, rho_m)

        n_bar_target = np.array([0.5])
        cooling_rates = np.array([TWO_PI * 500])
        rho_cooled = apply_sympathetic_cooling(
            rho0, ops, cooling_rates, n_bar_target, duration=10e-3
        )

        # Motional mode reaches target
        n_final = qutip.expect(ops.number(0), rho_cooled)
        assert n_final == pytest.approx(n_bar_target[0], rel=0.3)

        # Qubit coherence preserved
        rho_q_after = rho_cooled.ptrace(0)
        assert abs(rho_q_after[0, 1]) == pytest.approx(0.5, abs=0.01)

    def test_zero_duration_returns_unchanged(self, system):
        """Zero duration returns the same state."""
        ops, sf = system
        rho0 = sf.thermal_state(n_bar=[5.0])
        rho_out = apply_sympathetic_cooling(
            rho0, ops, np.array([1e6]), np.array([0.1]), duration=0.0
        )
        assert (rho_out - rho0).norm() == pytest.approx(0.0, abs=1e-12)

    def test_differential_mode_cooling_rates(self):
        """Mode with higher participation cools faster."""
        hs = HilbertSpace(n_ions=1, n_modes=2, n_fock=15)
        ops = OperatorFactory(hs)
        sf = StateFactory(hs)
        rho0 = sf.thermal_state(n_bar=[5.0, 5.0])

        # Mode 0: fast cooling; Mode 1: slow cooling
        cooling_rates = np.array([TWO_PI * 5e6, TWO_PI * 0.5e6])
        n_bar_target = np.array([0.1, 0.1])
        rho_cooled = apply_sympathetic_cooling(
            rho0, ops, cooling_rates, n_bar_target, duration=1e-6
        )

        n0 = qutip.expect(ops.number(0), rho_cooled)
        n1 = qutip.expect(ops.number(1), rho_cooled)
        assert n0 < n1


class TestSimulationRunnerIntegration:
    """Integration tests for sympathetic cooling in SimulationRunner."""

    def test_runner_sympathetic_cooling(self, be9, ca40, ca40_trap):
        """Create a runner with coolant_indices and verify
        run_sympathetic_cooling works."""
        config = SimulationConfig(
            species=[be9, ca40],
            trap=ca40_trap,
            n_ions=2,
            n_modes=1,
            n_fock=15,
            solver="mesolve",
            coolant_indices=[0],
        )
        runner = SimulationRunner(config)

        # Start from a thermal state
        rho0 = runner.sf.thermal_state(n_bar=[5.0])
        n_op = runner.ops.number(0)
        n_before = qutip.expect(n_op, rho0)

        # Use explicit moderate rates to keep the ODE tractable
        rates = np.array([TWO_PI * 1e5])
        targets = np.array([0.5])
        rho_cooled = runner.run_sympathetic_cooling(
            rho0,
            duration=50e-6,
            cooling_rates=rates,
            n_bar_target=targets,
        )
        n_after = qutip.expect(n_op, rho_cooled)
        assert n_after < n_before

    def test_backward_compatibility(self, ca40, ca40_trap):
        """Config without coolant_indices works identically to before."""
        config = SimulationConfig(
            species=ca40,
            trap=ca40_trap,
            n_ions=2,
            n_modes=1,
            n_fock=15,
            solver="sesolve",
        )
        runner = SimulationRunner(config)
        result = runner.run_carrier_pulse(ion=0, theta=np.pi)
        sz = runner.ops.sigma_z(0)
        final_sz = qutip.expect(sz, result.states[-1])
        assert final_sz == pytest.approx(-1.0, abs=0.15)

    def test_per_mode_heating_rates(self, ca40, ca40_trap):
        """Per-mode heating rates produce different heating on
        different modes."""
        config = SimulationConfig(
            species=ca40,
            trap=ca40_trap,
            n_ions=1,
            n_modes=2,
            n_fock=15,
            solver="mesolve",
            heating_rates=[100.0, 5000.0],
        )
        runner = SimulationRunner(config)
        rho0 = runner.sf.thermal_state(n_bar=[0.0, 0.0])
        H = 0 * runner.ops.identity()
        tlist = np.linspace(0, 1e-3, 10)
        result = qutip.mesolve(H, rho0, tlist, c_ops=runner._c_ops)
        n0 = qutip.expect(runner.ops.number(0), result.states[-1])
        n1 = qutip.expect(runner.ops.number(1), result.states[-1])
        assert n1 > n0

    def test_per_mode_initial_nbar(self, ca40, ca40_trap):
        """Per-mode initial n_bar overrides the scalar."""
        config = SimulationConfig(
            species=ca40,
            trap=ca40_trap,
            n_ions=1,
            n_modes=2,
            n_fock=15,
            n_bar_initial_per_mode=[0.5, 3.0],
        )
        runner = SimulationRunner(config)
        state = runner._initial_state()
        n0 = qutip.expect(runner.ops.number(0), state)
        n1 = qutip.expect(runner.ops.number(1), state)
        assert n0 == pytest.approx(0.5, rel=0.1)
        assert n1 == pytest.approx(3.0, rel=0.1)

    def test_no_coolant_raises_on_run(self, ca40, ca40_trap):
        """Running sympathetic cooling without coolant_indices raises."""
        config = SimulationConfig(
            species=ca40,
            trap=ca40_trap,
            n_ions=2,
            n_modes=1,
            n_fock=10,
        )
        runner = SimulationRunner(config)
        rho0 = runner.sf.thermal_state(n_bar=[1.0])
        with pytest.raises(ValueError, match="coolant_indices"):
            runner.run_sympathetic_cooling(rho0, duration=1e-6)

    def test_zero_rate_mode_skipped(self):
        """A mode with rate=0 is skipped; only the other cools."""
        hs = HilbertSpace(n_ions=1, n_modes=2, n_fock=15)
        ops = OperatorFactory(hs)
        sf = StateFactory(hs)
        rho0 = sf.thermal_state(n_bar=[3.0, 3.0])
        n1_before = qutip.expect(ops.number(1), rho0)
        rates = np.array([TWO_PI * 1e5, 0.0])
        targets = np.array([0.1, 0.1])
        rho_cooled = apply_sympathetic_cooling(
            rho0, ops, rates, targets, duration=50e-6
        )
        n0 = qutip.expect(ops.number(0), rho_cooled)
        n1 = qutip.expect(ops.number(1), rho_cooled)
        assert n0 < n1
        assert n1 == pytest.approx(n1_before, rel=0.01)

    def test_all_zero_rates_returns_unchanged(self):
        """All-zero cooling rates return the state unchanged."""
        hs = HilbertSpace(n_ions=1, n_modes=1, n_fock=10)
        ops = OperatorFactory(hs)
        sf = StateFactory(hs)
        rho0 = sf.thermal_state(n_bar=[3.0])
        rates = np.array([0.0])
        targets = np.array([0.1])
        rho_out = apply_sympathetic_cooling(
            rho0, ops, rates, targets, duration=1e-6
        )
        assert (rho_out - rho0).norm() < 1e-12

    def test_wrong_length_heating_rates_raises(self, ca40, ca40_trap):
        """heating_rates with wrong length raises ValueError."""
        with pytest.raises(ValueError, match="heating_rates length"):
            SimulationConfig(
                species=ca40,
                trap=ca40_trap,
                n_ions=1,
                n_modes=2,
                heating_rates=[100.0],
            )

    def test_wrong_length_nbar_per_mode_raises(self, ca40, ca40_trap):
        """n_bar_initial_per_mode with wrong length raises."""
        with pytest.raises(ValueError, match="n_bar_initial_per_mode"):
            SimulationConfig(
                species=ca40,
                trap=ca40_trap,
                n_ions=1,
                n_modes=2,
                n_bar_initial_per_mode=[1.0],
            )

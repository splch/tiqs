"""Tests for sympathetic cooling.

The anchors here are independent of the implementation:

* the per-mode damping rate is compared against a numerical
  derivative of the velocity-dependent radiation-pressure force
  (Wineland & Itano, Phys. Rev. A 20, 1521 (1979)) and against its
  analytic ceiling ``omega_R/2`` at ``s = 2``;
* the steady-state limits are checked to be *independent* of the
  coolant participation (Wübbena et al., Phys. Rev. A 85, 043412
  (2012), text at Eq. 26 and Eq. 27) and to match the Doppler
  temperature through equipartition;
* the external-heating term is checked against the exact Lindblad
  steady state of the same channel.
"""

import numpy as np
import pytest
import qutip

from tiqs.chain.normal_modes import normal_modes
from tiqs.constants import BOLTZMANN, HBAR, TWO_PI
from tiqs.cooling.doppler import doppler_cooled_nbar
from tiqs.cooling.sideband_cooling import sideband_cooling_nbar
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


def _recoil_frequency(species) -> float:
    r"""$\omega_R = \hbar k^2/(2m)$ for the cooling transition."""
    k = TWO_PI / species.cooling_transition.wavelength
    return HBAR * k**2 / (2 * species.mass_kg)


def _radiation_pressure_damping(species, s: float, detuning: float) -> float:
    r"""Energy damping rate $\alpha/m$ from the scattering force.

    Central-difference derivative of
    $F(v) = \frac{\hbar k \Gamma}{2}
      \frac{s}{1 + s + (2(\Delta - kv)/\Gamma)^2}$
    at $v = 0$, divided by the mass. This is the classical
    Doppler-cooling damping rate of Wineland & Itano (1979) and owes
    nothing to the implementation under test.
    """
    gamma = species.cooling_transition.linewidth
    k = TWO_PI / species.cooling_transition.wavelength

    def force(v: float) -> float:
        detune = 2 * (detuning - k * v) / gamma
        return HBAR * k * gamma / 2 * s / (1 + s + detune**2)

    dv = 1e-3  # m/s; capture velocity Gamma/k is several m/s
    alpha = -(force(dv) - force(-dv)) / (2 * dv)
    return alpha / species.mass_kg


@pytest.fixture
def ca40():
    return get_species("Ca40")


@pytest.fixture
def be9():
    return get_species("Be9")


@pytest.fixture
def ca40_trap(ca40):
    """Linear Paul trap holding Ca40 (and Be9 in mixed chains).

    ``v_rf`` and ``omega_rf`` are scaled together (600 V at 60 MHz
    rather than 300 V at 30 MHz) so that the radial frequency is
    unchanged while the Mathieu q of the *lighter* Be9 stays below
    the 0.4 pseudopotential-validity threshold: q scales as
    V/omega_rf^2 while omega_radial scales as V/omega_rf.
    """
    return PaulTrap(
        v_rf=600.0,
        omega_rf=TWO_PI * 60e6,
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


class TestSympatheticCoolingRate:
    """The per-mode phonon damping rate is recoil-limited."""

    @pytest.mark.parametrize("symbol", ["Be9", "Ca40", "Yb171"])
    @pytest.mark.parametrize("s", [0.1, 1.0, 2.0, 10.0])
    def test_matches_radiation_pressure_damping(self, symbol, s):
        """rate = P * alpha/m with alpha = -dF/dv from the Lorentzian
        scattering force at the default detuning -Gamma/2."""
        species = get_species(symbol)
        gamma = species.cooling_transition.linewidth
        P = np.array([1.0, 0.37])
        expected = _radiation_pressure_damping(species, s, -gamma / 2) * P
        rates = sympathetic_cooling_rate(species, P, s)
        np.testing.assert_allclose(rates, expected, rtol=1e-5)

    @pytest.mark.parametrize("detuning_ratio", [-0.25, -0.5, -1.0, -4.0])
    def test_matches_damping_at_explicit_detuning(self, ca40, detuning_ratio):
        """Same anchor away from the default detuning."""
        gamma = ca40.cooling_transition.linewidth
        detuning = detuning_ratio * gamma
        P = np.array([0.6])
        expected = _radiation_pressure_damping(ca40, 1.5, detuning) * P
        rates = sympathetic_cooling_rate(ca40, P, 1.5, detuning)
        np.testing.assert_allclose(rates, expected, rtol=1e-5)

    @pytest.mark.parametrize("symbol", ["Be9", "Ca40", "Sr88", "Yb171"])
    def test_never_exceeds_recoil_ceiling(self, symbol):
        """Maximizing -8 w_R s (D/G)/[1+s+(2D/G)^2]^2 over s and D
        gives w_R/2 at s = 2, 2D/G = -1: no laser parameters can damp
        a mode faster than half the recoil frequency."""
        species = get_species(symbol)
        gamma = species.cooling_transition.linewidth
        ceiling = _recoil_frequency(species) / 2
        P = np.array([1.0])
        for s in (0.01, 0.5, 1.0, 2.0, 10.0, 1e3):
            for ratio in (-0.01, -0.5, -1.0, -3.0, -100.0):
                rate = sympathetic_cooling_rate(species, P, s, ratio * gamma)[
                    0
                ]
                assert rate <= ceiling * (1 + 1e-12)

    def test_peaks_at_saturation_two(self, ca40):
        """The global maximum (w_R/2) P is reached at s = 2."""
        P = np.array([0.8])
        ceiling = _recoil_frequency(ca40) / 2 * P[0]
        best = sympathetic_cooling_rate(ca40, P, 2.0)[0]
        assert best == pytest.approx(ceiling, rel=1e-12)
        for s in (0.1, 1.0, 4.0, 100.0):
            assert sympathetic_cooling_rate(ca40, P, s)[0] < best

    def test_rate_proportional_to_participation(self, ca40):
        """rate_m1 / rate_m2 = P_m1 / P_m2 exactly."""
        P = np.array([0.8, 0.2])
        rates = sympathetic_cooling_rate(ca40, P)
        assert rates[0] / rates[1] == pytest.approx(P[0] / P[1], rel=1e-10)

    def test_zero_participation_zero_rate(self, ca40):
        """Spectator mode (P=0) has zero cooling rate."""
        P = np.array([0.0])
        rates = sympathetic_cooling_rate(ca40, P)
        assert rates[0] == pytest.approx(0.0, abs=1e-30)

    def test_slow_compared_to_mode_frequency(self, be9, ca40, ca40_trap):
        """The secular Doppler treatment needs Gamma_m << omega_m. Even
        at the rate-maximizing saturation, a real Be9/Ca40 chain damps
        each axial mode in many trap periods."""
        masses = np.array([be9.mass_kg, ca40.mass_kg])
        axial = normal_modes(2, ca40_trap, masses=masses).modes["axial"]
        P = coolant_participation(axial, [0])
        rates = sympathetic_cooling_rate(be9, P, 2.0)
        assert np.all(rates / axial.freqs < 0.05)

    def test_non_positive_saturation_raises(self, ca40):
        with pytest.raises(ValueError, match="saturation_parameter"):
            sympathetic_cooling_rate(ca40, np.array([1.0]), 0.0)

    def test_blue_detuning_raises(self, ca40):
        gamma = ca40.cooling_transition.linewidth
        with pytest.raises(ValueError, match="detuning"):
            sympathetic_cooling_rate(ca40, np.array([1.0]), 1.0, gamma / 2)


class TestSympatheticDopplerNbar:
    """The Doppler limit is participation-independent."""

    def test_reduces_to_standard_doppler(self, ca40, ca40_trap):
        """With one ion the sympathetic limit equals the single-ion
        Doppler limit (also pinning the Hz vs rad/s convention of
        doppler_cooled_nbar)."""
        modes = normal_modes(1, ca40_trap)
        axial = modes.modes["axial"]
        P = coolant_participation(axial, [0])
        n_bar = sympathetic_doppler_nbar(ca40, axial.freqs, P)
        n_bar_standard = doppler_cooled_nbar(ca40, axial.freqs[0] / TWO_PI)
        assert n_bar[0] == pytest.approx(n_bar_standard, rel=1e-12)

    def test_limit_independent_of_participation(self, be9, ca40, ca40_trap):
        """Laser damping and photon recoil enter mode m through the
        same factor |b_{c,m}|^2/m_c, so it cancels from the balance:
        the limit is the same for a fully participating mode and for a
        weakly participating one (Wübbena et al., text at Eq. 26)."""
        masses = np.array([be9.mass_kg, ca40.mass_kg])
        axial = normal_modes(2, ca40_trap, masses=masses).modes["axial"]
        P = coolant_participation(axial, [0])
        assert P[0] < 0.1 < 0.9 < P[1]  # strongly asymmetric mode pair
        weak = sympathetic_doppler_nbar(be9, axial.freqs, P)
        full = sympathetic_doppler_nbar(be9, axial.freqs, np.ones_like(P))
        np.testing.assert_allclose(weak, full, rtol=1e-12)

    def test_matches_doppler_temperature(self, be9, ca40_trap):
        """n_bar_m = k_B T_D/(hbar omega_m) by equipartition, with
        T_D = hbar Gamma/(2 k_B) from IonSpecies."""
        freqs = np.array([TWO_PI * 1e6, TWO_PI * 3e6])
        P = np.array([0.4, 0.4])
        n_bar = sympathetic_doppler_nbar(be9, freqs, P)
        t_doppler = be9.doppler_limit_temperature()
        expected = BOLTZMANN * t_doppler / (HBAR * freqs)
        np.testing.assert_allclose(n_bar, expected, rtol=1e-12)
        assert n_bar[0] / n_bar[1] == pytest.approx(3.0, rel=1e-12)

    def test_external_heating_matches_lindblad_steady_state(self):
        """With external heating the limit becomes
        base + ndot_ext/Gamma_m. Check it against the exact steady
        state of the cooling channel plus the library's
        infinite-temperature heating pair (which feeds a constant
        ndot quanta/s, independent of occupation)."""
        ops = OperatorFactory(HilbertSpace(n_ions=1, n_modes=1, n_fock=24))
        a = ops.annihilate(0)
        base = 0.05
        gamma_eff = 2 * TWO_PI * 1e6 * np.sqrt(base)
        freqs = np.array([TWO_PI * 1e6])
        rates = np.array([1e4])
        ndot = np.array([500.0])

        predicted = sympathetic_sideband_nbar(
            gamma_eff,
            freqs,
            np.array([0.3]),
            ndot_ext=ndot,
            cooling_rates=rates,
        )
        c_ops = [
            np.sqrt(rates[0] * (base + 1)) * a,
            np.sqrt(rates[0] * base) * a.dag(),
            np.sqrt(ndot[0]) * a.dag(),
            np.sqrt(ndot[0]) * a,
        ]
        rho_ss = qutip.steadystate(qutip.qzero_like(a), c_ops)
        n_ss = qutip.expect(ops.number(0), rho_ss)
        assert predicted[0] == pytest.approx(n_ss, rel=1e-6)
        assert predicted[0] == pytest.approx(base + ndot[0] / rates[0])

    def test_ndot_ext_requires_cooling_rates(self, be9):
        freqs = np.array([TWO_PI * 1e6])
        P = np.array([0.5])
        with pytest.raises(ValueError, match="cooling_rates"):
            sympathetic_doppler_nbar(be9, freqs, P, ndot_ext=10.0)
        with pytest.raises(ValueError, match="ndot_ext"):
            sympathetic_doppler_nbar(
                be9, freqs, P, cooling_rates=np.array([1e4])
            )

    def test_externally_heated_spectator_mode_raises(self, be9):
        """An uncooled mode under external heating has no steady
        state; the old code silently returned ~1e30 instead."""
        freqs = np.array([TWO_PI * 1e6])
        with (
            pytest.warns(UserWarning, match="zero coolant"),
            pytest.raises(ValueError, match="no steady state"),
        ):
            sympathetic_doppler_nbar(
                be9,
                freqs,
                np.array([0.0]),
                ndot_ext=10.0,
                cooling_rates=np.array([0.0]),
            )

    def test_spectator_mode_warns(self, be9):
        freqs = np.array([TWO_PI * 1e6, TWO_PI * 2e6])
        P = np.array([0.5, 0.0])
        with pytest.warns(UserWarning, match="zero coolant"):
            sympathetic_doppler_nbar(be9, freqs, P)

    def test_shape_mismatch_raises(self, be9):
        with pytest.raises(ValueError, match="participation shape"):
            sympathetic_doppler_nbar(
                be9,
                np.array([TWO_PI * 1e6, TWO_PI * 2e6]),
                np.array([0.5]),
            )


class TestSympatheticSidebandNbar:
    def test_matches_single_mode_formula(self):
        """At any participation it agrees with the single-mode
        sideband_cooling_nbar for the same gamma_eff and omega."""
        gamma_eff = TWO_PI * 1e3
        freq = TWO_PI * 1e6
        n_bar = sympathetic_sideband_nbar(
            gamma_eff, np.array([freq]), np.array([0.3])
        )
        assert n_bar[0] == pytest.approx(
            sideband_cooling_nbar(gamma_eff, freq), rel=1e-12
        )

    def test_limit_independent_of_participation(self):
        """A_+ and A_- both carry eta_{c,m}^2 ~ P_m, so it cancels
        from n = A_+/(A_- - A_+)."""
        gamma_eff = TWO_PI * 1e3
        freqs = np.array([TWO_PI * 1e6, TWO_PI * 2e6])
        weak = sympathetic_sideband_nbar(
            gamma_eff, freqs, np.array([0.05, 0.95])
        )
        full = sympathetic_sideband_nbar(gamma_eff, freqs, np.ones(2))
        np.testing.assert_allclose(weak, full, rtol=1e-12)

    def test_absolute_value_and_quadratic_scaling(self):
        """(2pi*10 kHz / (2 * 2pi*1 MHz))^2 = 2.5e-5, and the limit is
        quadratic in both arguments."""
        freqs = np.array([TWO_PI * 1e6, TWO_PI * 2e6])
        P = np.array([0.7, 0.7])
        n_bar = sympathetic_sideband_nbar(TWO_PI * 10e3, freqs, P)
        assert n_bar[0] == pytest.approx(2.5e-5, rel=1e-12)
        assert n_bar[0] / n_bar[1] == pytest.approx(4.0, rel=1e-12)
        halved = sympathetic_sideband_nbar(TWO_PI * 5e3, freqs, P)
        assert n_bar[0] / halved[0] == pytest.approx(4.0, rel=1e-12)

    def test_external_heating_adds_ndot_over_rate(self):
        freqs = np.array([TWO_PI * 1e6])
        P = np.array([0.25])
        rates = sympathetic_cooling_rate(get_species("Be9"), P, 2.0)
        base = sympathetic_sideband_nbar(TWO_PI * 10e3, freqs, P)
        heated = sympathetic_sideband_nbar(
            TWO_PI * 10e3, freqs, P, ndot_ext=100.0, cooling_rates=rates
        )
        assert heated[0] - base[0] == pytest.approx(100.0 / rates[0])

    def test_non_positive_gamma_eff_raises(self):
        with pytest.raises(ValueError, match="gamma_eff"):
            sympathetic_sideband_nbar(
                0.0, np.array([TWO_PI * 1e6]), np.array([1.0])
            )


class TestApplySympatheticCooling:
    """Simulation tests for the density-matrix cooling channel."""

    @pytest.fixture
    def system(self):
        hs = HilbertSpace(n_ions=2, n_modes=1, n_fock=20)
        return OperatorFactory(hs), StateFactory(hs)

    def test_cooling_reduces_phonon_number(self, system):
        """Starting from n_bar=2, cooling should reduce phonon number."""
        ops, sf = system
        rho0 = sf.thermal_state(n_bar=[2.0])
        n_op = ops.number(0)
        n_before = qutip.expect(n_op, rho0)

        cooling_rates = np.array([TWO_PI * 1e6])
        n_bar_target = np.array([0.5])
        rho_cooled = apply_sympathetic_cooling(
            rho0, ops, cooling_rates, n_bar_target, duration=1e-6
        )
        n_after = qutip.expect(n_op, rho_cooled)
        assert n_after < n_before

    def test_relaxation_is_exponential_at_the_given_rate(self):
        """The channel must relax as n(t) = n_t + (n_0-n_t)exp(-Gamma t):
        this is what makes `cooling_rates` the phonon damping rate
        rather than a photon scattering rate."""
        hs = HilbertSpace(n_ions=1, n_modes=1, n_fock=40)
        ops = OperatorFactory(hs)
        sf = StateFactory(hs)
        rho0 = sf.thermal_state(n_bar=[4.0])
        n0 = qutip.expect(ops.number(0), rho0)
        rate, target = 1e5, 0.5
        for t in (2e-6, 5e-6, 20e-6):
            rho = apply_sympathetic_cooling(
                rho0, ops, np.array([rate]), np.array([target]), duration=t
            )
            n_t = qutip.expect(ops.number(0), rho)
            expected = target + (n0 - target) * np.exp(-rate * t)
            assert n_t == pytest.approx(expected, rel=5e-3)

    def test_target_above_fock_cutoff_raises(self):
        """A Doppler-limit target of 8.2 cannot be represented with
        n_fock=15: the truncated channel would pin population at the
        cutoff and report a fictitious steady state."""
        hs = HilbertSpace(n_ions=1, n_modes=1, n_fock=15)
        ops = OperatorFactory(hs)
        sf = StateFactory(hs)
        rho0 = sf.thermal_state(n_bar=[1.0])
        with pytest.raises(ValueError, match="n_bar_target"):
            apply_sympathetic_cooling(
                rho0, ops, np.array([2.2e6]), np.array([8.18]), duration=1e-6
            )

    def test_length_mismatch_raises(self):
        hs = HilbertSpace(n_ions=1, n_modes=2, n_fock=10)
        ops = OperatorFactory(hs)
        sf = StateFactory(hs)
        rho0 = sf.thermal_state(n_bar=[1.0, 1.0])
        with pytest.raises(ValueError, match="length n_modes"):
            apply_sympathetic_cooling(
                rho0, ops, np.array([1e5]), np.array([0.1]), duration=1e-6
            )

    def test_accepts_ket_input(self, system):
        """Ket input is converted to density matrix internally."""
        ops, sf = system
        psi0 = sf.ground_state()
        cooling_rates = np.array([TWO_PI * 1e6])
        n_bar_target = np.array([0.5])
        rho_cooled = apply_sympathetic_cooling(
            psi0, ops, cooling_rates, n_bar_target, duration=1e-6
        )
        assert rho_cooled.type == "oper"

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
        rho0 = sf.thermal_state(n_bar=[2.0])
        rho_out = apply_sympathetic_cooling(
            rho0, ops, np.array([1e6]), np.array([0.1]), duration=0.0
        )
        assert (rho_out - rho0).norm() == pytest.approx(0.0, abs=1e-12)

    def test_differential_mode_cooling_rates(self):
        """Mode with higher participation cools faster."""
        hs = HilbertSpace(n_ions=1, n_modes=2, n_fock=15)
        ops = OperatorFactory(hs)
        sf = StateFactory(hs)
        rho0 = sf.thermal_state(n_bar=[2.0, 2.0])

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

    @staticmethod
    def _mixed_config(be9, ca40, trap, n_fock):
        return SimulationConfig(
            species=[be9, ca40],
            trap=trap,
            n_ions=2,
            n_modes=1,
            n_fock=n_fock,
            solver="mesolve",
            coolant_indices=[0],
        )

    def test_default_rates_and_targets_are_physical(
        self, be9, ca40, ca40_trap
    ):
        """The rates/targets the runner derives from coolant_indices
        must be the recoil-limited damping rate and the ordinary
        (participation-independent) Doppler limit."""
        runner = SimulationRunner(self._mixed_config(be9, ca40, ca40_trap, 40))
        axial = runner.modes.modes["axial"]
        P = coolant_participation(axial, [0])[:1]
        assert P[0] < 0.1  # Be9 barely participates in the COM mode

        target = runner._n_bar_cooled[0]
        assert target == pytest.approx(
            doppler_cooled_nbar(be9, axial.freqs[0] / TWO_PI), rel=1e-12
        )
        rate = runner._cooling_rates[0]
        assert rate <= _recoil_frequency(be9) / 2 * P[0] * (1 + 1e-12)
        assert rate / axial.freqs[0] < 0.01

    def test_default_path_cools_toward_doppler_limit(
        self, be9, ca40, ca40_trap
    ):
        """The documented default path (no overrides) must run and
        drive the mode to the sympathetic Doppler limit.

        A sympathetic Doppler limit of ~8 quanta needs a large Fock
        space; starting *above* it truncates the requested thermal
        state, which the state factory correctly reports.
        """
        runner = SimulationRunner(self._mixed_config(be9, ca40, ca40_trap, 40))
        target = runner._n_bar_cooled[0]
        with pytest.warns(UserWarning, match="truncates the thermal state"):
            rho0 = runner.sf.thermal_state(n_bar=[1.4 * target])
        n_op = runner.ops.number(0)
        n_before = qutip.expect(n_op, rho0)

        rho_cooled = runner.run_sympathetic_cooling(rho0, duration=50e-6)
        n_after = qutip.expect(n_op, rho_cooled)
        assert n_after < n_before
        assert n_after == pytest.approx(target, rel=0.1)

    def test_default_path_rejects_too_small_fock_cutoff(
        self, be9, ca40, ca40_trap
    ):
        """The library default n_fock=15 cannot represent a Doppler
        limit of ~8 quanta, so the default path must refuse rather
        than pin population at the cutoff."""
        runner = SimulationRunner(self._mixed_config(be9, ca40, ca40_trap, 15))
        rho0 = runner.sf.thermal_state(n_bar=[1.0])
        with pytest.raises(ValueError, match="n_bar_target"):
            runner.run_sympathetic_cooling(rho0, duration=20e-6)

    def test_explicit_overrides_bypass_configured_values(
        self, be9, ca40, ca40_trap
    ):
        """Explicit rates/targets replace the configured ones (used to
        model a resolved-sideband stage, whose limit is far below the
        Doppler value)."""
        runner = SimulationRunner(self._mixed_config(be9, ca40, ca40_trap, 15))
        rho0 = runner.sf.thermal_state(n_bar=[2.0])
        n_op = runner.ops.number(0)
        n_before = qutip.expect(n_op, rho0)

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

    def test_no_coolant_config_still_runs_gates(self, ca40, ca40_trap):
        """Without coolant_indices the runner is unaffected: a pi
        carrier pulse inverts the addressed ion exactly and leaves
        its neighbour alone."""
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
        final = result.states[-1]
        assert qutip.expect(runner.ops.sigma_z(0), final) == pytest.approx(
            -1.0, abs=1e-6
        )
        assert qutip.expect(runner.ops.sigma_z(1), final) == pytest.approx(
            1.0, abs=1e-6
        )

    def test_per_mode_heating_rates(self, ca40, ca40_trap):
        """Per-mode heating rates produce different heating on
        different modes."""
        config = SimulationConfig(
            species=ca40,
            trap=ca40_trap,
            n_ions=2,
            n_modes=2,
            n_fock=8,
            solver="mesolve",
            heating_rates=[100.0, 5000.0],
        )
        runner = SimulationRunner(config)
        rho0 = runner.sf.thermal_state(n_bar=[0.0, 0.0])
        H = 0 * runner.ops.identity()
        tlist = np.linspace(0, 2e-4, 10)
        result = qutip.mesolve(H, rho0, tlist, c_ops=runner._c_ops)
        n0 = qutip.expect(runner.ops.number(0), result.states[-1])
        n1 = qutip.expect(runner.ops.number(1), result.states[-1])
        assert n1 > n0

    def test_per_mode_initial_nbar(self, ca40, ca40_trap):
        """Per-mode initial n_bar overrides the scalar."""
        config = SimulationConfig(
            species=ca40,
            trap=ca40_trap,
            n_ions=2,
            n_modes=2,
            n_fock=15,
            n_bar_initial_per_mode=[0.5, 2.0],
        )
        runner = SimulationRunner(config)
        state = runner._initial_state()
        n0 = qutip.expect(runner.ops.number(0), state)
        n1 = qutip.expect(runner.ops.number(1), state)
        assert n0 == pytest.approx(0.5, rel=0.05)
        assert n1 == pytest.approx(2.0, rel=0.05)

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
        rho0 = sf.thermal_state(n_bar=[2.0, 2.0])
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
        rho0 = sf.thermal_state(n_bar=[1.0])
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
                n_ions=2,
                n_modes=2,
                heating_rates=[100.0],
            )

    def test_wrong_length_nbar_per_mode_raises(self, ca40, ca40_trap):
        """n_bar_initial_per_mode with wrong length raises."""
        with pytest.raises(ValueError, match="n_bar_initial_per_mode"):
            SimulationConfig(
                species=ca40,
                trap=ca40_trap,
                n_ions=2,
                n_modes=2,
                n_bar_initial_per_mode=[1.0],
            )

    def test_empty_coolant_indices_raises(self, ca40, ca40_trap):
        """Empty coolant_indices raises ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            SimulationConfig(
                species=ca40,
                trap=ca40_trap,
                n_ions=2,
                coolant_indices=[],
            )

    def test_out_of_range_coolant_index_raises(self, ca40, ca40_trap):
        """Out-of-range coolant index raises ValueError."""
        with pytest.raises(ValueError, match="coolant index"):
            SimulationConfig(
                species=ca40,
                trap=ca40_trap,
                n_ions=2,
                coolant_indices=[5],
            )

    def test_duplicate_coolant_indices_raises(self, ca40, ca40_trap):
        """Duplicate coolant indices raises ValueError."""
        with pytest.raises(ValueError, match="duplicates"):
            SimulationConfig(
                species=ca40,
                trap=ca40_trap,
                n_ions=2,
                coolant_indices=[0, 0],
            )

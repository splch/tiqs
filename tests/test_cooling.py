"""Tests for the cooling estimators and the sideband-cooling simulation.

The anchors are deliberately independent of the implementations:

* the Doppler limit is checked through the thermodynamic route
  $\\bar{n} = k_B T_D/(\\hbar\\omega)$ using
  ``IonSpecies.doppler_limit_temperature``;
* the EIT floor against the $\\Lambda$-system Fano lineshape and the
  Lamb-Dicke rate equation $\\bar{n} = A_+/(A_- - A_+)$ evaluated in
  the test itself (Leibfried RMP 75, 281 Eqs. 119/121/125/128);
* the pulsed sideband simulation against the exact Jaynes-Cummings
  transfer probability $\\sin^2(\\pi\\sqrt{n}/2)$ of a fixed-duration
  red-sideband pulse.
"""

from itertools import pairwise

import numpy as np
import pytest
import qutip

from tiqs.constants import BOLTZMANN, HBAR, TWO_PI
from tiqs.cooling.doppler import doppler_cooled_nbar
from tiqs.cooling.eit_cooling import eit_cooling_nbar
from tiqs.cooling.sideband_cooling import (
    sideband_cooling_nbar,
    sideband_cooling_simulate,
)
from tiqs.hilbert_space.builder import HilbertSpace
from tiqs.hilbert_space.operators import OperatorFactory
from tiqs.species.ion import get_species


def _thermodynamic_nbar(species, trap_frequency_hz: float) -> float:
    r"""$\bar{n} = k_B T_D/(\hbar\omega)$ from the Doppler temperature.

    Independent route to the Doppler limit: equipartition of a
    classical oscillator at the Doppler temperature
    $T_D = \hbar\Gamma/(2 k_B)$.
    """
    t_doppler = species.doppler_limit_temperature()
    omega = TWO_PI * trap_frequency_hz
    return BOLTZMANN * t_doppler / (HBAR * omega)


def _eit_rate_equation_nbar(
    gamma: float,
    rabi_coupling: float,
    detuning_r: float,
    carrier_suppression: float,
) -> float:
    r"""Exact Lamb-Dicke steady state for an EIT (Fano) lineshape.

    Uses the bright-resonance absorption profile
    $W(\delta) \propto \delta^2 \Gamma /
    [\delta^2\Gamma^2 + 4(\Omega_r^2/4 - \delta\Delta_r)^2]$ - zero at
    $\delta = 0$ (dark state), peaked at the AC Stark shift
    $\nu = \Omega_r^2/(4\Delta_r)$ - and evaluates
    $\bar{n} = A_+/(A_- - A_+)$ with
    $A_\pm = W(\text{carrier}) + W(\mp\nu)$, the trap frequency being
    tuned to $\nu$.
    """

    def absorption(delta: float) -> float:
        return (
            delta**2
            * gamma
            / (
                delta**2 * gamma**2
                + 4 * (rabi_coupling**2 / 4 - delta * detuning_r) ** 2
            )
        )

    nu = rabi_coupling**2 / (4 * detuning_r)
    w_rsb = absorption(nu)
    w_bsb = absorption(-nu)
    w_carrier = carrier_suppression * w_rsb
    return (w_carrier + w_bsb) / (w_rsb - w_bsb)


class TestDopplerCooling:
    def test_ca40_doppler_limit(self):
        """Ca40 at a 1 MHz trap: n_bar = (Gamma/2pi)/(2 f) = 11.2."""
        species = get_species("Ca40")
        n_bar = doppler_cooled_nbar(species, trap_frequency_hz=1e6)
        assert n_bar == pytest.approx(
            _thermodynamic_nbar(species, 1e6), rel=1e-12
        )
        assert n_bar == pytest.approx(11.2, rel=0.1)

    def test_yb171_doppler(self):
        """Yb171 (Gamma/2pi = 19.6 MHz) at 1 MHz: n_bar = 9.8."""
        species = get_species("Yb171")
        n_bar = doppler_cooled_nbar(species, trap_frequency_hz=1e6)
        assert n_bar == pytest.approx(
            _thermodynamic_nbar(species, 1e6), rel=1e-12
        )
        assert n_bar == pytest.approx(9.8, rel=0.1)

    def test_nbar_inversely_proportional_to_trap_frequency(self):
        """n_bar ~ 1/omega exactly: tripling the frequency thirds it."""
        species = get_species("Ca40")
        n1 = doppler_cooled_nbar(species, trap_frequency_hz=1e6)
        n2 = doppler_cooled_nbar(species, trap_frequency_hz=3e6)
        assert n1 / n2 == pytest.approx(3.0, rel=1e-12)

    def test_non_positive_frequency_raises(self):
        species = get_species("Ca40")
        with pytest.raises(ValueError, match="trap_frequency_hz"):
            doppler_cooled_nbar(species, trap_frequency_hz=0.0)


class TestSidebandCoolingLimit:
    def test_sbc_analytical_nbar(self):
        """(Gamma_eff/(2 omega))^2 = (1e3/2e6)^2 = 2.5e-7 exactly."""
        n_bar = sideband_cooling_nbar(
            gamma_eff=TWO_PI * 1e3,
            trap_frequency=TWO_PI * 1e6,
        )
        assert n_bar == pytest.approx(2.5e-7, rel=1e-12)

    def test_quadratic_in_both_arguments(self):
        """Halving gamma_eff or doubling omega divides n_bar by 4."""
        base = sideband_cooling_nbar(TWO_PI * 1e3, TWO_PI * 1e6)
        half_gamma = sideband_cooling_nbar(TWO_PI * 0.5e3, TWO_PI * 1e6)
        double_omega = sideband_cooling_nbar(TWO_PI * 1e3, TWO_PI * 2e6)
        assert base / half_gamma == pytest.approx(4.0, rel=1e-12)
        assert base / double_omega == pytest.approx(4.0, rel=1e-12)

    @pytest.mark.parametrize(
        "gamma_eff,trap_frequency",
        [(0.0, TWO_PI * 1e6), (TWO_PI * 1e3, 0.0), (-1.0, TWO_PI * 1e6)],
    )
    def test_non_positive_arguments_raise(self, gamma_eff, trap_frequency):
        with pytest.raises(ValueError):
            sideband_cooling_nbar(gamma_eff, trap_frequency)


class TestSidebandCoolingSimulation:
    """The pulsed RSB-pulse / optical-pumping protocol."""

    ETA = 0.1
    RABI = TWO_PI * 100e3
    PUMP = TWO_PI * 50e3
    N_FOCK = 30
    N_BAR_0 = 3.0

    @pytest.fixture
    def ops(self):
        return OperatorFactory(
            HilbertSpace(n_ions=1, n_modes=1, n_fock=self.N_FOCK)
        )

    def _simulate(self, ops, n_cycles, eta=None):
        return sideband_cooling_simulate(
            ops,
            ion=0,
            mode=0,
            n_bar_initial=self.N_BAR_0,
            eta=self.ETA if eta is None else eta,
            rabi_frequency=self.RABI,
            optical_pumping_rate=self.PUMP,
            n_cycles=n_cycles,
        )

    def _initial_populations(self):
        return qutip.thermal_dm(self.N_FOCK, self.N_BAR_0).diag().real

    def test_one_cycle_matches_jaynes_cummings_transfer(self, ops):
        """One cycle removes sum_n p_n sin^2(pi sqrt(n)/2) quanta.

        The RSB pulse drives the two-level system
        |0,n> <-> |1,n-1> at Rabi frequency eta*Omega*sqrt(n), so a
        pulse of duration pi/(eta*Omega) transfers
        sin^2(pi*sqrt(n)/2); optical pumping then restores the spin
        without touching the phonon populations (the pumping
        dissipator commutes with n). Continuous cooling with H and
        the collapse operator acting together removes far more than
        this in the same interval.
        """
        p_n = self._initial_populations()
        n = np.arange(self.N_FOCK)
        transferred = np.sin(np.pi * np.sqrt(n) / 2) ** 2
        expected = float((p_n * (n - transferred)).sum())

        assert self._simulate(ops, 1) == pytest.approx(expected, rel=1e-5)

    @pytest.mark.parametrize("n_cycles", [1, 2, 3])
    def test_at_most_one_quantum_per_cycle(self, ops, n_cycles):
        """A coherent RSB pulse conserves n + |1><1| and pumping does
        not change n, so each cycle removes at most one quantum."""
        n_initial = float(
            (self._initial_populations() * np.arange(self.N_FOCK)).sum()
        )
        removed = n_initial - self._simulate(ops, n_cycles)
        assert 0.0 < removed <= n_cycles + 1e-9

    def test_independent_of_eta_sign(self, ops):
        """The sign of eta is a gauge choice: the parity transform
        a -> -a flips H but leaves n, the thermal state and the
        (spin-only) collapse operator invariant."""
        positive = self._simulate(ops, 5, eta=self.ETA)
        negative = self._simulate(ops, 5, eta=-self.ETA)
        assert negative == pytest.approx(positive, rel=1e-9)

    def test_cools_monotonically(self, ops):
        """More cycles never heat, and the sequence cools well below
        its starting point."""
        values = [self._simulate(ops, nc) for nc in (1, 3, 10, 20)]
        assert all(
            later <= earlier + 1e-9 for earlier, later in pairwise(values)
        )
        assert values[-1] < 0.6 * self.N_BAR_0

    def test_never_returns_negative(self, ops):
        """A phonon number is non-negative even when the state is
        driven to the (heating-free) dark state."""
        cold = sideband_cooling_simulate(
            ops,
            ion=0,
            mode=0,
            n_bar_initial=0.01,
            eta=self.ETA,
            rabi_frequency=self.RABI,
            optical_pumping_rate=self.PUMP,
            n_cycles=20,
        )
        assert cold >= 0.0

    def test_zero_coupling_raises(self, ops):
        with pytest.raises(ValueError, match="pi-pulse"):
            self._simulate(ops, 5, eta=0.0)

    def test_non_positive_pumping_rate_raises(self, ops):
        with pytest.raises(ValueError, match="optical_pumping_rate"):
            sideband_cooling_simulate(
                ops,
                ion=0,
                mode=0,
                n_bar_initial=self.N_BAR_0,
                eta=self.ETA,
                rabi_frequency=self.RABI,
                optical_pumping_rate=0.0,
                n_cycles=5,
            )

    def test_zero_cycles_raises(self, ops):
        with pytest.raises(ValueError, match="n_cycles"):
            self._simulate(ops, 0)


class TestEITCooling:
    # Ca40 4P1/2 linewidth with a strong coupling beam far detuned:
    # Gamma/2pi = 22.4 MHz, Omega_r = 2 Gamma, Delta_r = 20 Gamma, so
    # the bright resonance sits at nu = Omega_r^2/(4 Delta_r) = Gamma/20
    # (2pi * 1.12 MHz) with FWHM Gamma Omega_r^2/(4 Delta_r^2) = Gamma/400.
    GAMMA = TWO_PI * 22.4e6
    RABI_R = 2 * GAMMA
    DETUNING_R = 20 * GAMMA
    NU = RABI_R**2 / (4 * DETUNING_R)
    FWHM = GAMMA * RABI_R**2 / (4 * DETUNING_R**2)

    def test_ideal_floor_matches_leibfried_eq128(self):
        """With a perfect dark state, n_bar = (Gamma/(4 Delta_r))^2.

        Leibfried RMP 75, 281 Eq. (128) / Morigi PRA 67, 033402
        Eq. (32). The identity gamma_FWHM/omega = Gamma/Delta_r holds
        when the bright resonance is tuned to the sideband.
        """
        n_bar = eit_cooling_nbar(self.FWHM, self.NU, 0.0)
        assert n_bar == pytest.approx(
            (self.GAMMA / (4 * self.DETUNING_R)) ** 2, rel=1e-12
        )
        assert n_bar == pytest.approx(1.5625e-4, rel=1e-9)

    @pytest.mark.parametrize("eps", [0.0, 1e-3, 1e-2, 1e-1])
    def test_matches_lamb_dicke_rate_equation(self, eps):
        """Agrees with A_+/(A_- - A_+) for the Fano lineshape."""
        exact = _eit_rate_equation_nbar(
            self.GAMMA, self.RABI_R, self.DETUNING_R, eps
        )
        got = eit_cooling_nbar(self.FWHM, self.NU, eps)
        assert got == pytest.approx(exact, rel=0.02)

    def test_carrier_suppression_is_a_floor(self):
        """Residual carrier absorption cancels from the cooling rate
        but survives in the numerator of the steady state, so the
        dark-state quality bounds n_bar from below."""
        eps = 0.01
        n_bar = eit_cooling_nbar(self.FWHM, self.NU, eps)
        assert n_bar >= eps
        # Narrow bright resonance: the ideal term is negligible and
        # the floor is set by eps alone.
        narrow = eit_cooling_nbar(self.FWHM / 100, self.NU, eps)
        assert narrow == pytest.approx(eps, rel=1e-3)

    def test_broadband_cooling_of_two_modes(self):
        """The carrier floor is frequency independent, so two modes
        an octave apart both land near eps - this is the bandwidth
        advantage over resolved sideband cooling. The ideal term,
        which does depend on frequency, falls as 1/omega^2."""
        eps = 0.01
        n_low = eit_cooling_nbar(TWO_PI * 200e3, TWO_PI * 1e6, eps)
        n_high = eit_cooling_nbar(TWO_PI * 200e3, TWO_PI * 2e6, eps)
        assert n_low == pytest.approx(eps, rel=0.3)
        assert n_high == pytest.approx(eps, rel=0.3)
        ideal_ratio = (n_low - eps) / (n_high - eps)
        assert ideal_ratio == pytest.approx(4.0, rel=1e-9)

    @pytest.mark.parametrize(
        "gamma_eit,trap_frequency,eps",
        [
            (0.0, TWO_PI * 1e6, 0.01),
            (TWO_PI * 1e5, 0.0, 0.01),
            (TWO_PI * 1e5, TWO_PI * 1e6, -0.01),
        ],
    )
    def test_invalid_arguments_raise(self, gamma_eit, trap_frequency, eps):
        with pytest.raises(ValueError):
            eit_cooling_nbar(gamma_eit, trap_frequency, eps)

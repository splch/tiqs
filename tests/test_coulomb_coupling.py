"""Tests for Coulomb coupling between separated trapped particles.

Both couplings are anchored two ways that do not restate the
implementation: against Taylor coefficients obtained by numerically
differentiating the exact $1/r$ potential, and against published
measurements - Brown et al. Nature 471, 196 (2011) for the
beam-splitter coupling and Osada et al. Phys. Rev. Research 4,
033245 (2022) Table II for the optomechanical coupling.
"""

import numpy as np
import pytest

from tiqs.constants import (
    COULOMB_CONSTANT,
    ELECTRON_MASS,
    HBAR,
    TWO_PI,
)
from tiqs.interaction.coulomb_coupling import (
    beam_splitter_coupling,
    optomechanical_coupling,
)
from tiqs.species.ion import get_species


def _coulomb_energy(x_1, x_2, separation):
    """Exact attractive Coulomb energy V = -C/(L + x1 - x2) in J."""
    return -COULOMB_CONSTANT / (separation + x_1 - x_2)


def _d2_dx1_dx2(separation, step):
    """Central-difference d^2 V / dx1 dx2 at the equilibrium."""
    return (
        _coulomb_energy(step, step, separation)
        - _coulomb_energy(step, -step, separation)
        - _coulomb_energy(-step, step, separation)
        + _coulomb_energy(-step, -step, separation)
    ) / (4 * step**2)


def _d3_dx1sq_dx2(separation, step):
    """Central-difference d^3 V / dx1^2 dx2 at the equilibrium."""

    def d2_dx1sq(x_2):
        return (
            _coulomb_energy(step, x_2, separation)
            - 2 * _coulomb_energy(0.0, x_2, separation)
            + _coulomb_energy(-step, x_2, separation)
        ) / step**2

    return (d2_dx1sq(step) - d2_dx1sq(-step)) / (2 * step)


def _zero_point_extent(mass, omega):
    """Ground-state position spread x_zpf = sqrt(hbar/(2 m omega))."""
    return np.sqrt(HBAR / (2 * mass * omega))


class TestBeamSplitterCoupling:
    def test_matches_numerically_differentiated_coulomb(self):
        """Rebuild g_bs from the exact 1/r potential, no formula reuse.

        The x1*x2 Taylor coefficient of V is d^2V/dx1dx2, so
        quantizing x_i = x_zpf,i (a_i + a_i^dag) gives
        hbar*g_bs = (d^2V/dx1dx2) * x_zpf1 * x_zpf2. Evaluating the
        derivative by central differences on V(x1, x2) makes no
        reference to the implemented closed form, so a prefactor
        wrong by any constant factor - the factor-of-2 class of
        error these expansions invite - fails here. The residual is
        the O(step^2) difference truncation error.
        """
        mass_1 = ELECTRON_MASS
        mass_2 = get_species("Be9").mass_kg
        omega_1 = TWO_PI * 800e6
        omega_2 = TWO_PI * 2e6
        separation = 10e-6

        coefficient = _d2_dx1_dx2(separation, separation / 500)
        expected = (
            coefficient
            * _zero_point_extent(mass_1, omega_1)
            * _zero_point_extent(mass_2, omega_2)
            / HBAR
        )
        g_bs = beam_splitter_coupling(
            mass_1, mass_2, omega_1, omega_2, separation
        )
        assert g_bs == pytest.approx(expected, rel=1e-4)

    def test_scales_as_L_cubed_inverse(self):
        """Coupling scales as 1/L^3."""
        m1 = m2 = ELECTRON_MASS
        w1 = w2 = TWO_PI * 1e9
        g1 = beam_splitter_coupling(m1, m2, w1, w2, 10e-6)
        g2 = beam_splitter_coupling(m1, m2, w1, w2, 20e-6)
        assert g1 / g2 == pytest.approx(8.0, rel=1e-10)

    def test_symmetric_in_particles(self):
        """Swapping particle labels gives the same coupling."""
        m1, m2 = ELECTRON_MASS, get_species("Ca40").mass_kg
        w1, w2 = TWO_PI * 800e6, TWO_PI * 2e6
        L = 20e-6
        g_12 = beam_splitter_coupling(m1, m2, w1, w2, L)
        g_21 = beam_splitter_coupling(m2, m1, w2, w1, L)
        assert g_12 == pytest.approx(g_21, rel=1e-10)

    def test_lighter_particle_gives_stronger_coupling(self):
        """At equal frequencies, lighter particles couple more
        strongly (larger zero-point fluctuations)."""
        m_e = ELECTRON_MASS
        m_be = get_species("Be9").mass_kg
        w = TWO_PI * 10e6
        L = 50e-6
        g_ee = beam_splitter_coupling(m_e, m_e, w, w, L)
        g_ii = beam_splitter_coupling(m_be, m_be, w, w, L)
        assert g_ee > g_ii

    def test_brown_2011_mode_splitting(self):
        """Brown et al., Nature 471, 196 (2011): 3.0(5) kHz measured.

        Two 9Be+ ions in wells separated by s0 = 40 um with axial
        frequency omega0/2pi = 4.04 MHz. Their Eq. 2 is this
        function; the measured minimum normal-mode splitting is
        delta_f = Omega_ex/pi = 3.0(5) kHz and their own theory value
        is 3.1 kHz, the residual ~2% being an electrode-screening
        correction this function does not model.

        This is an absolute anchor: a factor-of-2 prefactor error
        gives 6.0 kHz and fails both assertions.
        """
        mass = get_species("Be9").mass_kg
        omega = TWO_PI * 4.04e6
        g_bs = beam_splitter_coupling(mass, mass, omega, omega, 40e-6)
        splitting_hz = g_bs / np.pi
        assert splitting_hz == pytest.approx(3.0e3, abs=0.5e3)
        assert splitting_hz == pytest.approx(3.1e3, rel=0.05)

    def test_brown_2011_exchange_time(self):
        """Brown et al. predict tau_ex = 162 us at 4.04 MHz.

        A full energy swap under g_bs(a1^dag a2 + a1 a2^dag) takes
        tau_ex = pi/(2 g_bs) = 1/(2 delta_f). The paper measures
        155(1) us and predicts 162 us.
        """
        mass = get_species("Be9").mass_kg
        omega = TWO_PI * 4.04e6
        g_bs = beam_splitter_coupling(mass, mass, omega, omega, 40e-6)
        assert np.pi / (2 * g_bs) == pytest.approx(162e-6, rel=0.05)

    def test_identical_particles_shift_equals_coupling(self):
        """The omitted x_i^2 frequency shift equals g_bs exactly.

        Pins the module docstring's warning. The x_1^2 term of the
        Coulomb expansion renormalizes each secular frequency by
        |domega| = C/(m omega L^3) to first order, which for two
        identical particles in identical traps is algebraically the
        same expression as g_bs = C/(L^3 sqrt(m^2 omega^2)). So the
        neglected frequency renormalization is never small compared
        with the coupling, and omega must be the shifted value.
        """
        mass = get_species("Be9").mass_kg
        omega = TWO_PI * 4.04e6
        separation = 40e-6
        shift = COULOMB_CONSTANT / (mass * omega * separation**3)
        g_bs = beam_splitter_coupling(mass, mass, omega, omega, separation)
        assert shift == pytest.approx(g_bs, rel=1e-12)

    def test_osada_regime_violates_resonance_condition(self):
        """Osada's electron-ion pair cannot exchange phonons.

        Osada et al. state that for their 800 MHz electron and 2 MHz
        ion "the beam-splitter and two-mode-squeezing interactions
        are not valid here". Resonant exchange needs
        |omega_1 - omega_2| <~ g_bs; here the detuning is ~640x
        g_bs, so transfer is suppressed by (g_bs/Delta)^2 ~ 2e-6
        even though g_bs itself is a large 1.25 MHz. The weaker
        condition |Delta| << omega_1 + omega_2 is also violated
        (ratio 0.995), which is why it must not be quoted alone.
        """
        m_e = ELECTRON_MASS
        m_i = get_species("Be9").mass_kg
        w_e = TWO_PI * 800e6
        w_i = TWO_PI * 2e6
        g_bs = beam_splitter_coupling(m_e, m_i, w_e, w_i, 10e-6)
        detuning = abs(w_e - w_i)
        assert g_bs / TWO_PI == pytest.approx(1.2513e6, rel=1e-3)
        assert detuning / g_bs > 100.0
        assert (g_bs / detuning) ** 2 < 1e-5
        assert detuning / (w_e + w_i) > 0.9


class TestOptomechanicalCoupling:
    def test_matches_numerically_differentiated_coulomb(self):
        """Rebuild g_0 from the exact 1/r potential, no formula reuse.

        The x1^2*x2 Taylor coefficient of V is
        (1/2!) d^3V/dx1^2dx2 = -3C/L^4. Quantizing and splitting
        (a1 + a1^dag)^2 = 2n1 + 1 gives
        hbar*g_0 = |coefficient| * 2 * x_zpf1^2 * x_zpf2, i.e.
        H_int = -hbar g_0 n1 (a2 + a2^dag) for the attractive case.
        Central differences supply the coefficient independently, so
        both the 3 and the 2 are under test.
        """
        mass_1 = ELECTRON_MASS
        mass_2 = get_species("Be9").mass_kg
        omega_1 = TWO_PI * 800e6
        omega_2 = TWO_PI * 2e6
        separation = 10e-6

        third = _d3_dx1sq_dx2(separation, separation / 500)
        coefficient = third / 2.0
        assert coefficient < 0.0  # -3C/L^4, not +3C/L^4
        expected = (
            abs(coefficient)
            * 2.0
            * _zero_point_extent(mass_1, omega_1) ** 2
            * _zero_point_extent(mass_2, omega_2)
            / HBAR
        )
        g_0 = optomechanical_coupling(
            mass_1, mass_2, omega_1, omega_2, separation
        )
        assert g_0 == pytest.approx(expected, rel=1e-4)

    def test_scales_as_L_fourth_inverse(self):
        """Coupling scales as 1/L^4."""
        m1 = m2 = ELECTRON_MASS
        w1 = w2 = TWO_PI * 1e9
        g1 = optomechanical_coupling(m1, m2, w1, w2, 10e-6)
        g2 = optomechanical_coupling(m1, m2, w1, w2, 20e-6)
        assert g1 / g2 == pytest.approx(16.0, rel=1e-10)

    def test_not_symmetric_in_particles(self):
        """Optomechanical coupling is asymmetric: particle 1 is the
        'cavity' (its number operator couples). Swapping gives a
        different value because x_zpf1^2 * x_zpf2 != x_zpf2^2 * x_zpf1
        when masses differ."""
        m1 = ELECTRON_MASS
        m2 = get_species("Ca40").mass_kg
        w1 = TWO_PI * 800e6
        w2 = TWO_PI * 2e6
        L = 20e-6
        g_12 = optomechanical_coupling(m1, m2, w1, w2, L)
        g_21 = optomechanical_coupling(m2, m1, w2, w1, L)
        assert g_12 != pytest.approx(g_21, rel=0.1)

    def test_osada_2022_table_ii_first_row(self):
        """Osada et al., PRR 4, 033245 (2022) Table II: 33 kHz.

        Row 1 is omega_e/2pi = 800 MHz and L = 10 um, with
        omega_i/2pi = 2 MHz "assumed in common" and a beryllium ion
        (Sec. III.1), giving g_0/2pi = 33 kHz.

        That figure is the total of BOTH terms of their Eq. 10. This
        function returns only the first (pure Coulomb) term, so it
        must land ~22% above the total; the deficit is their
        trap-geometry-dependent -2 hbar g_C beta/(omega_e - omega_i)
        term. Be-9 is the only species consistent with 33 kHz -
        Ca-40 gives 19.1 kHz and Yb-171 9.2 kHz, both of which would
        need a wrong-sign correction.
        """
        m_e = ELECTRON_MASS
        m_i = get_species("Be9").mass_kg
        g_0_hz = (
            optomechanical_coupling(
                m_e, m_i, TWO_PI * 800e6, TWO_PI * 2e6, 10e-6
            )
            / TWO_PI
        )
        assert g_0_hz == pytest.approx(40.28e3, rel=1e-3)
        assert g_0_hz == pytest.approx(33e3, rel=0.25)

    def test_relation_to_beam_splitter(self):
        """g_0 = 3 * g_bs * x_zpf1 / L exactly.

        Internal consistency only: both couplings derive from the
        same 1/r expansion, so this ratio is invariant under a
        prefactor error applied to both. The absolute anchors above
        are what constrain the prefactors.
        """
        m1 = ELECTRON_MASS
        m2 = get_species("Be9").mass_kg
        w1 = TWO_PI * 800e6
        w2 = TWO_PI * 2e6
        L = 10e-6
        g_bs = beam_splitter_coupling(m1, m2, w1, w2, L)
        g0 = optomechanical_coupling(m1, m2, w1, w2, L)
        assert g0 == pytest.approx(
            3 * g_bs * _zero_point_extent(m1, w1) / L, rel=1e-10
        )

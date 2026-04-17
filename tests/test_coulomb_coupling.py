"""Tests for Coulomb coupling between separated trapped particles.

Validates beam-splitter and optomechanical coupling strengths
against known analytical results and the parameters from
Osada et al. Phys. Rev. Research 4, 033245 (2022).
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


class TestBeamSplitterCoupling:
    def test_formula_direct(self):
        """Direct check: g_bs = C / (L^3 * sqrt(m1*m2*w1*w2))."""
        m1 = ELECTRON_MASS
        m2 = get_species("Be9").mass_kg
        w1 = TWO_PI * 500e6
        w2 = TWO_PI * 2e6
        L = 30e-6
        g = beam_splitter_coupling(m1, m2, w1, w2, L)
        expected = COULOMB_CONSTANT / (L**3 * np.sqrt(m1 * m2 * w1 * w2))
        assert g == pytest.approx(expected, rel=1e-10)

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

    def test_osada_electron_ion_order_of_magnitude(self):
        """Osada Table II: electron at 800 MHz, Be-9+ at 2 MHz,
        L = 10 um gives g_0/(2pi) ~ 33 kHz. The beam-splitter
        coupling should be comparable in magnitude."""
        m_e = ELECTRON_MASS
        m_i = get_species("Be9").mass_kg
        w_e = TWO_PI * 800e6
        w_i = TWO_PI * 2e6
        L = 10e-6
        g = beam_splitter_coupling(m_e, m_i, w_e, w_i, L)
        assert g / TWO_PI > 1e3
        assert g / TWO_PI < 10e6


class TestOptomechanicalCoupling:
    def test_formula_direct(self):
        """Direct check: g_0 = 3C * x_zpf1^2 * x_zpf2 / (hbar * L^4)."""
        m1 = ELECTRON_MASS
        m2 = get_species("Be9").mass_kg
        w1 = TWO_PI * 800e6
        w2 = TWO_PI * 2e6
        L = 10e-6
        g0 = optomechanical_coupling(m1, m2, w1, w2, L)
        x1 = np.sqrt(HBAR / (2 * m1 * w1))
        x2 = np.sqrt(HBAR / (2 * m2 * w2))
        expected = 3 * COULOMB_CONSTANT * x1**2 * x2 / (HBAR * L**4)
        assert g0 == pytest.approx(expected, rel=1e-10)

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

    def test_osada_table_ii_800mhz_10um(self):
        """Osada Table II: omega_e/(2pi) = 800 MHz, L = 10 um,
        omega_i/(2pi) = 2 MHz gives g_0/(2pi) ~ 33 kHz."""
        m_e = ELECTRON_MASS
        m_i = get_species("Be9").mass_kg
        w_e = TWO_PI * 800e6
        w_i = TWO_PI * 2e6
        L = 10e-6
        g0 = optomechanical_coupling(m_e, m_i, w_e, w_i, L)
        # Paper states g_0/(2pi) = 33 kHz. Our formula gives
        # the pure Coulomb term without the trap-anharmonicity
        # correction (the second term in Eq. 10), so we check
        # order-of-magnitude agreement.
        assert g0 / TWO_PI == pytest.approx(33e3, rel=0.5)

    def test_osada_table_ii_500mhz_10um(self):
        """Osada Table II: omega_e/(2pi) = 500 MHz, L = 10 um
        gives g_0/(2pi) ~ 39 kHz."""
        m_e = ELECTRON_MASS
        m_i = get_species("Be9").mass_kg
        w_e = TWO_PI * 500e6
        w_i = TWO_PI * 2e6
        L = 10e-6
        g0 = optomechanical_coupling(m_e, m_i, w_e, w_i, L)
        assert g0 / TWO_PI == pytest.approx(39e3, rel=0.5)

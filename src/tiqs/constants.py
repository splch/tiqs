"""SI physical constants used throughout the simulator.

Values are the **CODATA 2022** recommended values (NIST reference on
constants, units and uncertainty, https://physics.nist.gov/cuu/Constants).
Constants that the 2019 SI redefinition fixed exactly (``PLANCK``,
``ELECTRON_CHARGE``, ``BOLTZMANN``, ``SPEED_OF_LIGHT``) carry no
uncertainty; the measured ones are annotated with their CODATA 2022
relative standard uncertainty so that test tolerances can be pinned to
the edition rather than to a literal.
"""

import numpy as np

TWO_PI = 2.0 * np.pi
"""2 pi, used for angular frequency conversions."""

PLANCK = 6.62607015e-34
"""Planck constant in J s (exact by the 2019 SI definition)."""

HBAR = PLANCK / TWO_PI
"""Reduced Planck constant $h/2\\pi$ in J s (exact)."""

ELECTRON_CHARGE = 1.602176634e-19
"""Elementary charge in C (exact by the 2019 SI definition)."""

BOLTZMANN = 1.380649e-23
"""Boltzmann constant in J/K (exact by the 2019 SI definition)."""

SPEED_OF_LIGHT = 299792458.0
"""Speed of light in vacuum in m/s (exact by definition)."""

AMU = 1.66053906892e-27
"""Atomic mass constant in kg. CODATA 2022: 1.66053906892(52)e-27,
relative uncertainty 3.1e-10."""

EPSILON_0 = 8.8541878188e-12
"""Vacuum electric permittivity in F/m. CODATA 2022:
8.8541878188(14)e-12, relative uncertainty 1.6e-10."""

ELECTRON_MASS = 9.1093837139e-31
"""Electron mass in kg. CODATA 2022: 9.1093837139(28)e-31, relative
uncertainty 3.1e-10."""

BOHR_MAGNETON = 9.2740100657e-24
"""Bohr magneton in J/T. CODATA 2022: 9.2740100657(29)e-24, relative
uncertainty 3.1e-10."""

ELECTRON_G_FACTOR = 2.00231930436092
"""Electron spin g-factor, stored as the **magnitude**. CODATA 2022
gives $g_e = -2.00231930436092(36)$; the sign is dropped here because
every consumer uses it as a positive gyromagnetic scale factor."""

COULOMB_CONSTANT = ELECTRON_CHARGE**2 / (4.0 * np.pi * EPSILON_0)
"""Coulomb constant $e^2 / (4\\pi\\epsilon_0)$ in J m."""

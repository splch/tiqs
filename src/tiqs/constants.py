"""SI physical constants used throughout the simulator."""

import numpy as np

HBAR = 1.054571817e-34
"""Reduced Planck constant in J s."""

ELECTRON_CHARGE = 1.602176634e-19
"""Elementary charge in C."""

BOLTZMANN = 1.380649e-23
"""Boltzmann constant in J/K."""

SPEED_OF_LIGHT = 299792458.0
"""Speed of light in m/s."""

AMU = 1.66053906660e-27
"""Atomic mass unit in kg."""

EPSILON_0 = 8.8541878128e-12
"""Vacuum permittivity in F/m."""

ELECTRON_MASS = 9.1093837015e-31
"""Electron mass in kg."""

BOHR_MAGNETON = 9.2740100783e-24
"""Bohr magneton in J/T."""

ELECTRON_G_FACTOR = 2.00231930436256
"""Electron spin g-factor."""

PI = np.pi
"""Pi."""

TWO_PI = 2.0 * np.pi
"""2 pi, used for angular frequency conversions."""

COULOMB_CONSTANT = ELECTRON_CHARGE**2 / (4.0 * np.pi * EPSILON_0)
"""Coulomb constant $e^2 / (4\\pi\\epsilon_0)$ in J m."""

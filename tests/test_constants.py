"""Physical-constant tests pinned to the CODATA 2022 edition.

Two layers, so that neither an accidental edit nor a stale edition can
slip through:

* **Edition-independent relations** that must hold in *any* CODATA
  edition - ``HBAR`` derived from the exact Planck constant, the
  Coulomb-constant identity, and the internal consistency of the
  electromagnetic set checked through three combinations that CODATA
  publishes separately (the fine-structure constant, the Bohr magneton
  as $e\\hbar/2m_e$, and the electron gyromagnetic ratio). Mixing
  editions, or truncating one member of the set, breaks these at the
  1e-9 level while every individual value still "looks right".
* **One edition-pinned table** with relative tolerances at the CODATA
  2022 relative standard uncertainty, so a stale value fails loudly.

Reference: NIST, "Fundamental Physical Constants - Complete Listing,
2022 CODATA adjustment", https://physics.nist.gov/cuu/Constants
(``allascii.txt``).
"""

import numpy as np
import pytest

from tiqs.constants import (
    AMU,
    BOHR_MAGNETON,
    BOLTZMANN,
    COULOMB_CONSTANT,
    ELECTRON_CHARGE,
    ELECTRON_G_FACTOR,
    ELECTRON_MASS,
    EPSILON_0,
    HBAR,
    PLANCK,
    SPEED_OF_LIGHT,
    TWO_PI,
)

# CODATA 2022 recommended values with their relative standard
# uncertainties, used as the test tolerances. Every ``pytest.approx``
# below passes ``abs=0``: its 1e-12 absolute default would otherwise
# swallow the whole comparison for constants of order 1e-27.
_CODATA_2022 = [
    ("AMU", AMU, 1.66053906892e-27, 3.1e-10),
    ("EPSILON_0", EPSILON_0, 8.8541878188e-12, 1.6e-10),
    ("ELECTRON_MASS", ELECTRON_MASS, 9.1093837139e-31, 3.1e-10),
    ("BOHR_MAGNETON", BOHR_MAGNETON, 9.2740100657e-24, 3.1e-10),
    ("ELECTRON_G_FACTOR", ELECTRON_G_FACTOR, 2.00231930436092, 1.8e-13),
]


@pytest.mark.parametrize(
    ("name", "value", "codata_2022", "rel_unc"),
    _CODATA_2022,
    ids=[row[0] for row in _CODATA_2022],
)
def test_codata_2022_measured_values(name, value, codata_2022, rel_unc):
    """Measured constants match CODATA 2022 to its own uncertainty.

    The tolerance is the published relative standard uncertainty, which
    is 4x to 9x smaller than the CODATA 2018 -> 2022 revisions, so a
    stale edition fails here rather than being silently accepted.
    """
    assert value == pytest.approx(codata_2022, rel=rel_unc, abs=0)


def test_defining_constants_are_exact():
    """The 2019 SI redefinition fixed these four exactly."""
    assert PLANCK == 6.62607015e-34
    assert ELECTRON_CHARGE == 1.602176634e-19
    assert BOLTZMANN == 1.380649e-23
    assert SPEED_OF_LIGHT == 299792458.0


def test_hbar_is_planck_over_two_pi():
    """``HBAR`` is derived, not a truncated literal.

    $\\hbar \\equiv h/2\\pi$ with *h* exact, so the round trip must
    reproduce *h* to floating-point precision. A hand-typed
    ``1.054571817e-34`` reconstructs *h* only to 6.1e-10.
    """
    assert HBAR == PLANCK / TWO_PI
    h_round_trip = TWO_PI * HBAR
    assert h_round_trip == pytest.approx(PLANCK, rel=1e-15, abs=0)


def test_coulomb_constant_identity():
    """$k_e = e^2/4\\pi\\epsilon_0$, with the textbook eV nm anchor."""
    k_e = ELECTRON_CHARGE**2 / (4.0 * np.pi * EPSILON_0)
    assert k_e == COULOMB_CONSTANT
    # e^2/4 pi eps_0 = 1.439 964 55 eV nm.
    ev_nm = COULOMB_CONSTANT / ELECTRON_CHARGE * 1e9
    assert ev_nm == pytest.approx(1.43996455, rel=1e-8)


def test_fine_structure_constant_consistency():
    r"""$\alpha^{-1}$ rebuilt from *e*, $\epsilon_0$, $\hbar$, *c*.

    CODATA 2022 publishes $\alpha^{-1} = 137.035999177(21)$
    independently of the four constants it is assembled from here, so
    this is a genuine cross-check of the electromagnetic set. A mixed
    CODATA 2018 set with a truncated ``HBAR`` misses it by 1.3e-9.
    """
    alpha = ELECTRON_CHARGE**2 / (
        4.0 * np.pi * EPSILON_0 * HBAR * SPEED_OF_LIGHT
    )
    assert 1.0 / alpha == pytest.approx(137.035999177, rel=1e-11)


def test_bohr_magneton_identity():
    r"""$\mu_B = e\hbar/2m_e$ ties $\mu_B$, $\hbar$ and $m_e$ together.

    A CODATA 2018 ``BOHR_MAGNETON`` and ``ELECTRON_MASS`` with a
    truncated ``HBAR`` violate this by 6.1e-10.
    """
    derived = ELECTRON_CHARGE * HBAR / (2.0 * ELECTRON_MASS)
    assert derived == pytest.approx(BOHR_MAGNETON, rel=1e-10, abs=0)


def test_electron_gyromagnetic_ratio():
    r"""$\gamma_e = |g_e|\mu_B/\hbar$ against the CODATA 2022 value.

    CODATA 2022: electron gyromagnetic ratio
    1.76085962784(55)e11 s^-1 T^-1. This is the combination that sets
    every electron-qubit Zeeman frequency in the package. A mixed
    CODATA 2018 set misses it by 2.0e-9.
    """
    gamma_e = ELECTRON_G_FACTOR * BOHR_MAGNETON / HBAR
    assert gamma_e == pytest.approx(1.76085962784e11, rel=1e-10)


def test_electron_mass_in_u():
    """``ELECTRON_MASS / AMU`` equals the CODATA 2022 value in u.

    CODATA 2022: m_e = 5.485799090441(97)e-4 u. This is the conversion
    that ``IonSpecies.mass_kg`` relies on when it subtracts one
    electron from a neutral atomic mass.
    """
    m_e_in_u = ELECTRON_MASS / AMU
    assert m_e_in_u == pytest.approx(5.485799090441e-4, rel=1e-10, abs=0)

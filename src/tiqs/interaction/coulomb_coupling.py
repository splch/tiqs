r"""Coulomb coupling between two separated trapped particles.

When two charged particles are confined in separate trapping
potentials at equilibrium separation $L$, their mutual Coulomb
interaction $V = -C / (L + x_1 - x_2)$ (with
$C = e^2 / 4\pi\epsilon_0$) generates coupling between the
motional modes of each particle.

Expanding $V$ in Taylor series around the equilibrium positions
and quantizing the displacements
$x_i = x_{\mathrm{zpf},i}(a_i + a_i^\dagger)$:

**Dipole-dipole (beam-splitter) coupling** from the $x_1 x_2$
cross term:

$$
g_\mathrm{bs} = \frac{C}{L^3}
  \frac{1}{\sqrt{m_1 m_2 \omega_1 \omega_2}}
$$

This is the leading-order exchange interaction that enables
sympathetic cooling of one particle via laser cooling of the
other, even when they are in separate traps.

**Optomechanical coupling** from the $x_1^2 x_2$ term:

$$
g_0 = \frac{3C}{\hbar L^4}\,x_{\mathrm{zpf},1}^2\,x_{\mathrm{zpf},2}
$$

This is the radiation-pressure-like interaction where the
phonon number of particle 1 exerts a force on particle 2,
as in Osada et al. Phys. Rev. Research 4, 033245 (2022).

References
----------
Osada, A. et al. "Feasibility study on ground-state cooling
and single-phonon readout of trapped electrons using hybrid
quantum systems." *Phys. Rev. Research* **4**, 033245 (2022).

Kotler, S. et al. "Hybrid quantum systems with trapped
charged particles." *Phys. Rev. A* **95**, 022327 (2017).
"""

import numpy as np

from tiqs.constants import COULOMB_CONSTANT, HBAR


def beam_splitter_coupling(
    mass_1: float,
    mass_2: float,
    omega_1: float,
    omega_2: float,
    separation: float,
) -> float:
    r"""Dipole-dipole Coulomb coupling between two separated
    trapped particles.

    From the $x_1 x_2$ cross term in the Taylor expansion of
    $V = -C/(L + x_1 - x_2)$:

    $$
    g_\mathrm{bs}
      = \frac{e^2}{4\pi\epsilon_0\,L^3}
        \frac{1}{\sqrt{m_1\,m_2\,\omega_1\,\omega_2}}
    $$

    This coupling drives the beam-splitter interaction
    $g_\mathrm{bs}(a_1^\dagger a_2 + a_1 a_2^\dagger)$ when the
    two modes are near-resonant ($|\omega_1 - \omega_2| \ll
    \omega_1 + \omega_2$).

    Parameters
    ----------
    mass_1, mass_2 : float
        Particle masses in kg.
    omega_1, omega_2 : float
        Secular angular frequencies in rad/s.
    separation : float
        Equilibrium inter-particle distance in meters.

    Returns
    -------
    float
        Beam-splitter coupling strength in rad/s.
    """
    return COULOMB_CONSTANT / (
        separation**3 * np.sqrt(mass_1 * mass_2 * omega_1 * omega_2)
    )


def optomechanical_coupling(
    mass_1: float,
    mass_2: float,
    omega_1: float,
    omega_2: float,
    separation: float,
) -> float:
    r"""Optomechanical Coulomb coupling between two separated
    trapped particles.

    From the $x_1^2 x_2$ term in the Taylor expansion:

    $$
    g_0 = \frac{3\,e^2}{4\pi\epsilon_0\,\hbar\,L^4}
      \,x_{\mathrm{zpf},1}^2\,x_{\mathrm{zpf},2}
    $$

    where $x_{\mathrm{zpf},i} = \sqrt{\hbar / 2 m_i \omega_i}$.
    This drives the interaction
    $g_0\,a_1^\dagger a_1 (a_2 + a_2^\dagger)$: the phonon number
    of particle 1 exerts a force on particle 2.

    Parameters
    ----------
    mass_1, mass_2 : float
        Particle masses in kg. Particle 1 is the one whose
        phonon number couples (the "cavity" in optomechanics).
    omega_1, omega_2 : float
        Secular angular frequencies in rad/s.
    separation : float
        Equilibrium inter-particle distance in meters.

    Returns
    -------
    float
        Optomechanical coupling strength in rad/s.
    """
    x_zpf_1 = np.sqrt(HBAR / (2 * mass_1 * omega_1))
    x_zpf_2 = np.sqrt(HBAR / (2 * mass_2 * omega_2))
    return 3 * COULOMB_CONSTANT * x_zpf_1**2 * x_zpf_2 / (HBAR * separation**4)

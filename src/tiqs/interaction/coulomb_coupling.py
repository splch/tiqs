r"""Coulomb coupling between two separated trapped particles.

When two charged particles are confined in separate trapping
potentials at equilibrium separation $L$, their mutual Coulomb
interaction $V = \pm C / (L + x_1 - x_2)$ (with
$C = e^2 / 4\pi\epsilon_0$; sign depends on charge product)
generates coupling between the motional modes of each particle.

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

**Optomechanical coupling** from the $x_1^2 x_2$ term
(including the factor of 2 from
$(a + a^\dagger)^2 \to 2\hat{n}$):

$$
g_0 = \frac{6C}{\hbar L^4}\,x_{\mathrm{zpf},1}^2\,x_{\mathrm{zpf},2}
$$

This is the coupling in
$H_\mathrm{int} = \hbar\,g_0\,\hat{n}_1\,(a_2 + a_2^\dagger)$,
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

    The $x_1^2 x_2$ cross term in the Taylor expansion of
    $V = -C/(L + x_1 - x_2)$ has coefficient $3C/L^4$.
    Quantizing $x_i = x_{\mathrm{zpf},i}(a_i + a_i^\dagger)$
    and keeping the number-operator part
    $(a + a^\dagger)^2 \to 2\hat{n} + 1$ gives:

    $$
    g_0 = \frac{6\,e^2}{4\pi\epsilon_0\,\hbar\,L^4}
      \,x_{\mathrm{zpf},1}^2\,x_{\mathrm{zpf},2}
    $$

    This is the pure Coulomb contribution to the coupling in
    $H_\mathrm{int} = \hbar\,g_0\,\hat{n}_1\,(a_2 + a_2^\dagger)$,
    corresponding to the first term of Osada et al. Eq. 10.
    Real traps have an additional correction from the effective
    potential anharmonicity (the second term in Eq. 10) that
    depends on trap geometry and can be comparable in magnitude.

    Parameters
    ----------
    mass_1, mass_2 : float
        Particle masses in kg. Particle 1 is the one whose
        phonon number couples to particle 2.
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
    return 6 * COULOMB_CONSTANT * x_zpf_1**2 * x_zpf_2 / (HBAR * separation**4)

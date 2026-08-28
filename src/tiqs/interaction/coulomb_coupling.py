r"""Coulomb coupling between two separated trapped particles.

When two charged particles are confined in separate trapping
potentials at equilibrium separation $L$, their mutual Coulomb
interaction $V = s\,C / (L + x_1 - x_2)$ (with
$C = e^2 / 4\pi\epsilon_0$ and $s = \pm 1$ the sign of the charge
product) generates coupling between the motional modes of each
particle.

Writing $u = (x_1 - x_2)/L$ and expanding
$V = (sC/L)(1 - u + u^2 - u^3 + \dots)$ about the equilibrium
positions, then quantizing the displacements
$x_i = x_{\mathrm{zpf},i}(a_i + a_i^\dagger)$, gives the terms
below, carried through in general $s$. Both functions in this
module return **magnitudes**: the overall sign follows the charge
product and is in any case removable by $a_2 \to -a_2$, so it is
stated per term rather than folded into the return value. The two
function docstrings specialize to the attractive case $s = -1$
(e.g. electron-ion), which is the convention of Osada et al.

**Secular frequency renormalization** from the $x_1^2$ and $x_2^2$
parts of the $u^2$ term, coefficient $sC/L^3$ each. This shifts
$\omega_i^2$ by $2sC/(m_i L^3)$, i.e. to first order

$$
|\delta\omega_i| \simeq \frac{C}{m_i\,\omega_i\,L^3}
$$

This is never a small correction relative to the couplings below.
For two identical particles in identical traps it is *exactly*
$g_\mathrm{bs}$, since both reduce to $C/(m\omega L^3)$: 1.5 kHz
against $g_\mathrm{bs}/2\pi = 1.5$ kHz at Brown et al.'s
$^9$Be$^+$ parameters. For the strongly asymmetric electron-ion
pair of Osada et al. Table II ($L = 10$ um) the shifts are 1.0% =
8.0 MHz for the electron and 9.8% = 195 kHz for the ion, against
$g_\mathrm{bs}/2\pi = 1.25$ MHz and $g_0/2\pi = 40$ kHz.

The ``omega_1``/``omega_2`` arguments must therefore be the
**renormalized** (Coulomb-shifted) secular frequencies, not the
bare single-trap values. Osada et al. make the same stipulation
after their Eq. 9 ("$\omega_e$ now stands for the renormalized
secular frequency"), and Brown et al. note these terms "represent
static changes in the trap frequencies that could also be
compensated with potentials applied to nearby electrodes". TIQS
does not compute the shift for you.

**Dipole-dipole (beam-splitter) coupling** from the $x_1 x_2$ part
of the $u^2$ term, coefficient $-2sC/L^3$:

$$
g_\mathrm{bs} = \frac{C}{L^3}
  \frac{1}{\sqrt{m_1 m_2 \omega_1 \omega_2}}
$$

giving $H_\mathrm{int} = -s\,\hbar\,g_\mathrm{bs}
(a_1^\dagger a_2 + a_1 a_2^\dagger)$ after the rotating-wave
approximation. This is the leading-order exchange interaction that
enables sympathetic cooling of one particle via laser cooling of
the other, even when they are in separate traps; it is Eq. 2 of
Brown et al. Nature 471, 196 (2011).

**Optomechanical coupling** from the $x_1^2 x_2$ part of the $u^3$
term, coefficient $3sC/L^4$, with
$(a_1 + a_1^\dagger)^2 = 2\hat{n}_1 + 1$ supplying the factor 2:

$$
g_0 = \frac{6C}{\hbar L^4}\,x_{\mathrm{zpf},1}^2\,x_{\mathrm{zpf},2}
$$

giving $H_\mathrm{int} = s\,\hbar\,g_0\,\hat{n}_1\,
(a_2 + a_2^\dagger)$, i.e. $-\hbar g_0 \hat{n}_1 (a_2 +
a_2^\dagger)$ for $s = -1$ as in Osada et al. Eq. 9. The $+1$ of
$2\hat{n}_1 + 1$ contributes a static force
$s\,(\hbar g_0/2)(a_2 + a_2^\dagger)$ on mode 2 - a constant
displacement, dropped here because it shifts the equilibrium rather
than coupling the modes.

References
----------
Osada, A. et al. "Feasibility study on ground-state cooling
and single-phonon readout of trapped electrons using hybrid
quantum systems." *Phys. Rev. Research* **4**, 033245 (2022).

Brown, K. R. et al. "Coupled quantized mechanical oscillators."
*Nature* **471**, 196 (2011).

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
    $V = -C/(L + x_1 - x_2)$, whose coefficient is $+2C/L^3$:

    $$
    g_\mathrm{bs}
      = \frac{e^2}{4\pi\epsilon_0\,L^3}
        \frac{1}{\sqrt{m_1\,m_2\,\omega_1\,\omega_2}}
    $$

    This is Eq. 2 of Brown et al. Nature 471, 196 (2011), who
    measured a normal-mode splitting $\delta f = g_\mathrm{bs}/\pi$
    of 3.0(5) kHz for two $^9$Be$^+$ ions at $L = 40$ um and
    $\omega/2\pi = 4.04$ MHz.

    **Validity.** Two separate conditions apply, and only the
    second makes $g_\mathrm{bs}$ an exchange *rate*:

    1. Dropping the counter-rotating $a_1 a_2$ and
       $a_1^\dagger a_2^\dagger$ terms, which oscillate at
       $\omega_1 + \omega_2$, needs
       $g_\mathrm{bs} \ll \omega_1 + \omega_2$. Essentially always
       satisfied.
    2. For the surviving
       $g_\mathrm{bs}(a_1^\dagger a_2 + a_1 a_2^\dagger)$ to
       actually exchange population the modes must be resonant:
       $|\omega_1 - \omega_2| \lesssim g_\mathrm{bs}$. This is far
       stricter. Off resonance, transfer is suppressed by
       $(g_\mathrm{bs}/\Delta)^2$ with
       $\Delta = \omega_1 - \omega_2$: two $^9$Be$^+$ ions at
       $\omega/2\pi = 5$ and 4 MHz with $L = 40$ um give
       $g_\mathrm{bs}/2\pi = 1.4$ kHz against
       $\Delta/2\pi = 1$ MHz, so $(g/\Delta)^2 \approx 2 \times
       10^{-6}$ - no exchange, even though
       $|\Delta|/(\omega_1 + \omega_2) = 0.11$.

    Brown et al. use exactly this: they detune "by 100 kHz $\gg$
    $\Omega_\mathrm{ex}/2\pi$, effectively decoupling the ions'
    motions", then tune into resonance to switch the exchange on.
    Osada et al., whose electron (800 MHz) and ion (2 MHz) are
    hopelessly non-degenerate, state that "the beam-splitter and
    two-mode-squeezing interactions are not valid here" and use
    parametric driving instead.

    Parameters
    ----------
    mass_1, mass_2 : float
        Particle masses in kg.
    omega_1, omega_2 : float
        Renormalized (Coulomb-shifted) secular angular frequencies
        in rad/s. See the module docstring: the shift is of order
        $C/(m\omega^2 L^3)$ and is not applied for you.
    separation : float
        Equilibrium inter-particle distance in meters.

    Returns
    -------
    float
        Beam-splitter coupling magnitude in rad/s. The sign of the
        interaction follows the charge product (module docstring).
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
    $V = -C/(L + x_1 - x_2)$ has coefficient $-3C/L^4$: the
    $-u^3$ term of $-(C/L)(1 - u + u^2 - u^3 + \dots)$ contributes
    $+(C/L^4)(x_1 - x_2)^3$, and $(x_1 - x_2)^3$ carries
    $-3x_1^2 x_2$. Quantizing
    $x_i = x_{\mathrm{zpf},i}(a_i + a_i^\dagger)$ and splitting
    $(a_1 + a_1^\dagger)^2 = 2\hat{n}_1 + 1$ gives:

    $$
    g_0 = \frac{6\,e^2}{4\pi\epsilon_0\,\hbar\,L^4}
      \,x_{\mathrm{zpf},1}^2\,x_{\mathrm{zpf},2}
    $$

    This is the pure Coulomb contribution to the coupling in
    $H_\mathrm{int} = -\hbar\,g_0\,\hat{n}_1\,(a_2 + a_2^\dagger)$,
    matching the sign of Osada et al. Eq. 9 and the first term of
    their Eq. 10. The negative sign holds for the attractive case
    ($V = -C/(L + x_1 - x_2)$); it flips with the charge product
    and is in any case removable by $a_2 \to -a_2$, so the return
    value is the magnitude. Keep it consistent with the sign of the
    definite-signed second term of Eq. 10 and of the self-Kerr
    term when assembling a Hamiltonian by hand.

    The discarded $+1$ of $2\hat{n}_1 + 1$ is a static force
    $-(\hbar g_0/2)(a_2 + a_2^\dagger)$ on mode 2: it displaces
    mode 2's equilibrium but does not couple the modes.

    Real traps have an additional correction from the effective
    potential anharmonicity (the second term in Eq. 10) that
    depends on trap geometry and can be comparable in magnitude.
    For Osada's Table II row 1 - electron at 800 MHz, $^9$Be$^+$ at
    2 MHz, $L = 10$ um - this first term alone is 40.3 kHz against
    their quoted total of 33 kHz.

    Parameters
    ----------
    mass_1, mass_2 : float
        Particle masses in kg. Particle 1 is the one whose
        phonon number couples to particle 2.
    omega_1, omega_2 : float
        Renormalized (Coulomb-shifted) secular angular frequencies
        in rad/s. See the module docstring: the shift is of order
        $C/(m\omega^2 L^3)$ and is not applied for you.
    separation : float
        Equilibrium inter-particle distance in meters.

    Returns
    -------
    float
        Optomechanical coupling magnitude in rad/s.
    """
    x_zpf_1 = np.sqrt(HBAR / (2 * mass_1 * omega_1))
    x_zpf_2 = np.sqrt(HBAR / (2 * mass_2 * omega_2))
    return 6 * COULOMB_CONSTANT * x_zpf_1**2 * x_zpf_2 / (HBAR * separation**4)

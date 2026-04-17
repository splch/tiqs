r"""Sympathetic cooling: indirect cooling of qubit ions via a
co-trapped coolant species.

A coolant ion is laser-cooled directly while qubit ions are cooled
indirectly through the Coulomb interaction that couples all ions
via shared normal modes. The cooling laser addresses only the
coolant species (far off-resonance from qubit transitions), so
qubit quantum states are preserved.

The cooling rate of mode $m$ is proportional to the total squared
participation of coolant ions in that mode:

$$
\Gamma_m^\text{cool}
  = \frac{\Gamma}{2}\,\frac{s}{1+s}\,P_m
$$

where $\Gamma$ is the coolant cooling-transition linewidth, $s$
is the saturation parameter, and the **coolant participation** is

$$
P_m = \sum_{k \in \text{coolant}} |b_{k,m}|^2
$$

with $b_{k,m}$ the mass-weighted eigenvector component of ion $k$
in mode $m$. Modes where the coolant has near-zero participation
("spectator modes") are cooled slowly or not at all.

References
----------
Home, J.P. "Quantum science and metrology with mixed-species ion
chains." *Adv. At. Mol. Opt. Phys.* **62**, 231 (2013).

Bruzewicz, C.D. et al. "Trapped-ion quantum computing: Progress
and challenges." *Appl. Phys. Rev.* **6**, 021314 (2019).

Sosnova, K. et al. "Character of motional modes for entanglement
and sympathetic cooling of mixed-species trapped-ion chains."
*Phys. Rev. A* **103**, 012610 (2021).
"""

import numpy as np
import qutip

from tiqs.chain.normal_modes import ModeGroup
from tiqs.hilbert_space.operators import OperatorFactory
from tiqs.species.ion import IonSpecies


def coolant_participation(
    mode_group: ModeGroup,
    coolant_indices: list[int],
) -> np.ndarray:
    r"""Compute coolant participation fraction for each normal mode.

    $$
    P_m = \sum_{k \in \text{coolant}} |b_{k,m}|^2
    $$

    where $b_{k,m}$ is the mass-weighted eigenvector component of
    coolant ion $k$ in mode $m$. Since the eigenvectors are
    orthonormal, $\sum_i |b_{i,m}|^2 = 1$ for every mode, so
    $0 \le P_m \le 1$.

    When all ions are coolants, $P_m = 1$ for every mode.

    Parameters
    ----------
    mode_group : ModeGroup
        Normal mode result for one direction (e.g. axial).
    coolant_indices : list[int]
        Indices of coolant ions in the chain (0-based).

    Returns
    -------
    np.ndarray
        Coolant participation per mode, shape ``(n_modes,)``.
    """
    vectors = mode_group.vectors
    return np.sum(vectors[coolant_indices, :] ** 2, axis=0)


def sympathetic_doppler_nbar(
    coolant_species: IonSpecies,
    mode_freqs: np.ndarray,
    participation: np.ndarray,
) -> np.ndarray:
    r"""Doppler cooling limit per mode via sympathetic cooling.

    $$
    \bar{n}_m = \frac{\Gamma}{2\,\omega_m\,P_m}
    $$

    The standard Doppler limit $\bar{n} = \Gamma / (2\omega)$
    assumes the laser-addressed ion has full participation in the
    mode ($P_m = 1$). For sympathetic cooling, the effective
    cooling rate on mode $m$ is reduced by $P_m$, raising the
    steady-state occupation by $1/P_m$.

    Parameters
    ----------
    coolant_species : IonSpecies
        The coolant species (provides cooling transition linewidth).
    mode_freqs : np.ndarray
        Mode angular frequencies in rad/s, shape ``(n_modes,)``.
    participation : np.ndarray
        Coolant participation per mode from
        ``coolant_participation()``, shape ``(n_modes,)``.

    Returns
    -------
    np.ndarray
        Per-mode mean phonon number at the sympathetic Doppler
        limit, shape ``(n_modes,)``.
    """
    gamma = coolant_species.cooling_transition.linewidth
    safe_p = np.maximum(participation, 1e-30)
    return gamma / (2 * mode_freqs * safe_p)


def sympathetic_sideband_nbar(
    gamma_eff: float,
    mode_freqs: np.ndarray,
    participation: np.ndarray,
) -> np.ndarray:
    r"""Resolved sideband cooling limit per mode via sympathetic cooling.

    $$
    \bar{n}_m = \frac{1}{P_m}
      \left(\frac{\gamma_\text{eff}}{2\,\omega_m}\right)^2
    $$

    Parameters
    ----------
    gamma_eff : float
        Effective linewidth of the cooling transition (rad/s).
    mode_freqs : np.ndarray
        Mode angular frequencies in rad/s, shape ``(n_modes,)``.
    participation : np.ndarray
        Coolant participation per mode, shape ``(n_modes,)``.

    Returns
    -------
    np.ndarray
        Per-mode mean phonon number at the sympathetic sideband
        cooling limit, shape ``(n_modes,)``.
    """
    safe_p = np.maximum(participation, 1e-30)
    return (gamma_eff / (2 * mode_freqs)) ** 2 / safe_p


def sympathetic_cooling_rate(
    coolant_species: IonSpecies,
    participation: np.ndarray,
    saturation_parameter: float = 1.0,
) -> np.ndarray:
    r"""Per-mode Doppler cooling rate from sympathetic cooling.

    At optimum detuning $\Delta = -\Gamma/2$, the cooling rate
    for mode $m$ is:

    $$
    \Gamma_m^\text{cool}
      = \frac{\Gamma}{2}\,\frac{s}{1+s}\,P_m
    $$

    where $s = I/I_\text{sat}$ is the saturation parameter.
    Each mode's phonon number relaxes exponentially:

    $$
    \bar{n}(t) = \bar{n}_\text{ss}
      + (\bar{n}_0 - \bar{n}_\text{ss})\,
        e^{-\Gamma_m^\text{cool}\,t}
    $$

    Parameters
    ----------
    coolant_species : IonSpecies
        The coolant species (provides cooling transition linewidth).
    participation : np.ndarray
        Coolant participation per mode, shape ``(n_modes,)``.
    saturation_parameter : float
        Laser saturation parameter $s = I/I_\text{sat}$
        (default 1.0, optimum for Doppler cooling).

    Returns
    -------
    np.ndarray
        Per-mode cooling rates in 1/s, shape ``(n_modes,)``.
    """
    gamma = coolant_species.cooling_transition.linewidth
    s = saturation_parameter
    return (gamma / 2) * participation * s / (1 + s)


def apply_sympathetic_cooling(
    rho: qutip.Qobj,
    ops: OperatorFactory,
    n_modes: int,
    cooling_rates: np.ndarray,
    n_bar_target: np.ndarray,
    duration: float,
) -> qutip.Qobj:
    r"""Apply sympathetic cooling as a thermal relaxation channel.

    Models sympathetic cooling as Lindblad dissipation on each
    motional mode, driving it toward the steady-state phonon
    number $\bar{n}_\text{target}$. Only motional operators are
    used -- qubit states are preserved exactly, which is the
    defining property of sympathetic cooling.

    The collapse operators for each mode $m$ are:

    $$
    L_{\downarrow,m}
      = \sqrt{\Gamma_m\,(\bar{n}_m + 1)}\;a_m
      \qquad\text{(phonon loss)}
    $$

    $$
    L_{\uparrow,m}
      = \sqrt{\Gamma_m\,\bar{n}_m}\;a_m^\dagger
      \qquad\text{(recoil heating)}
    $$

    where $\Gamma_m$ is the cooling rate and $\bar{n}_m$ is the
    target occupation. These drive $\langle n_m \rangle$ toward
    $\bar{n}_m$ at rate $\Gamma_m$.

    Parameters
    ----------
    rho : qutip.Qobj
        Input density matrix.
    ops : OperatorFactory
        Operator factory for the composite Hilbert space.
    n_modes : int
        Number of motional modes to cool.
    cooling_rates : np.ndarray
        Per-mode cooling rates in 1/s, shape ``(n_modes,)``.
    n_bar_target : np.ndarray
        Per-mode target mean phonon number, shape ``(n_modes,)``.
    duration : float
        Cooling duration in seconds.

    Returns
    -------
    qutip.Qobj
        Density matrix after sympathetic cooling.
    """
    if duration <= 0:
        return rho

    c_ops = []
    for m in range(n_modes):
        rate = cooling_rates[m]
        n_t = n_bar_target[m]
        if rate <= 0:
            continue
        c_ops.append(np.sqrt(rate * (n_t + 1)) * ops.annihilate(m))
        if n_t > 0:
            c_ops.append(np.sqrt(rate * n_t) * ops.create(m))

    if not c_ops:
        return rho

    H = 0 * ops.identity()
    n_steps = max(2, int(duration * cooling_rates.max() * 10) + 2)
    tlist = np.linspace(0, duration, n_steps)
    result = qutip.mesolve(H, rho, tlist, c_ops=c_ops)
    return result.states[-1]

r"""Sympathetic cooling: indirect cooling of qubit ions via a
co-trapped coolant species.

A coolant ion is laser-cooled directly while qubit ions are cooled
indirectly through the Coulomb interaction that couples all ions
via shared normal modes. The cooling laser addresses only the
coolant species (far off-resonance from qubit transitions), so
qubit quantum states are preserved.

Doppler cooling of the coolant damps mode $m$ at a rate set by the
**recoil frequency** of the coolant's cooling transition, scaled by
the coolant participation $P_m$:

$$
\Gamma_m^\text{cool}
  = 4\,\omega_R\,\frac{s}{(2+s)^2}\,P_m ,
\qquad
\omega_R = \frac{\hbar k_c^2}{2\,m_c}
$$

at the detuning $\Delta = -\Gamma/2$, where $s$ is the saturation
parameter and

$$
P_m = \sum_{k \in \text{coolant}} |b_{k,m}|^2
$$

with $b_{k,m}$ the mass-weighted eigenvector component of ion $k$
in mode $m$. Modes where the coolant has near-zero participation
("spectator modes") are cooled slowly or not at all.

The steady-state occupation, by contrast, is **independent of**
$P_m$: laser damping and photon-recoil heating enter mode $m$
through the same factor $|b_{c,m}|^2/m_c$, which cancels in the
balance. Participation dependence returns only through *external*
(electric-field-noise) heating, which is not proportional to
$|b_{c,m}|^2/m_c$; see ``ndot_ext``.

References
----------
Wineland, D.J. & Itano, W.M. "Laser cooling of atoms."
*Phys. Rev. A* **20**, 1521 (1979) - velocity-dependent
radiation-pressure force and the resulting damping coefficient
$\alpha = -\partial F/\partial v$ used for the rate above.

Wübbena, J.B., Amairi, S., Mandel, O. & Schmidt, P.O. "Sympathetic
cooling of mixed-species two-ion crystals for precision
spectroscopy." *Phys. Rev. A* **85**, 043412 (2012)
(arXiv:1202.2730) - Eqs. (21)-(23) for the participation ($b^2$)
scaling of both cooling and heating, Eqs. (26)-(27) for the
participation-independent cooling limit, Eq. (32) for the
externally heated limit.

Sosnova, K. et al. "Character of motional modes for entanglement
and sympathetic cooling of mixed-species trapped-ion chains."
*Phys. Rev. A* **103**, 012610 (2021) - mode character and the
participation definition (their Eqs. 4 and 12); it contains no
cooling-rate formula.

Home, J.P. "Quantum science and metrology with mixed-species ion
chains." *Adv. At. Mol. Opt. Phys.* **62**, 231 (2013) and
Bruzewicz, C.D. et al. "Trapped-ion quantum computing: Progress
and challenges." *Appl. Phys. Rev.* **6**, 021314 (2019) - general
reviews of mixed-species operation.
"""

import warnings

import numpy as np
import qutip

from tiqs.chain.normal_modes import ModeGroup
from tiqs.constants import HBAR
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
    return np.sum(mode_group.vectors[coolant_indices, :] ** 2, axis=0)


def recoil_frequency(species: IonSpecies) -> float:
    r"""Recoil frequency of a species' cooling transition.

    $$
    \omega_R = \frac{\hbar k^2}{2 m}
    $$

    the angular frequency equivalent of the single-photon recoil
    energy $E_R = \hbar^2 k^2 / (2m)$. It sets the scale of every
    Doppler damping rate.

    Parameters
    ----------
    species : IonSpecies
        Species whose ``cooling_transition`` supplies $k$.

    Returns
    -------
    float
        Recoil frequency in rad/s.
    """
    k = species.cooling_transition.wavevector
    return HBAR * k**2 / (2 * species.mass_kg)


def _check_mode_shapes(
    mode_freqs: np.ndarray, participation: np.ndarray
) -> None:
    """Validate per-mode arrays and warn about spectator modes."""
    freqs = np.atleast_1d(mode_freqs)
    part = np.atleast_1d(participation)
    if freqs.shape != part.shape:
        raise ValueError(
            f"mode_freqs shape {freqs.shape} != participation shape"
            f" {part.shape}"
        )
    if np.any(freqs <= 0):
        raise ValueError(f"mode_freqs must be > 0, got {mode_freqs}")
    if np.any(part < 0):
        raise ValueError(f"participation must be >= 0, got {participation}")
    spectators = np.flatnonzero(part == 0)
    if spectators.size:
        warnings.warn(
            f"Modes {spectators.tolist()} have zero coolant"
            " participation: they are not cooled at all and never"
            " reach the returned limit.",
            UserWarning,
            stacklevel=3,
        )


def _add_external_heating(
    base: np.ndarray,
    ndot_ext: np.ndarray | float | None,
    cooling_rates: np.ndarray | None,
) -> np.ndarray:
    r"""Add the externally heated term $\dot{n}_\text{ext}/\Gamma_m$."""
    if ndot_ext is None:
        if cooling_rates is not None:
            raise ValueError(
                "cooling_rates given without ndot_ext: the cooling"
                " limit is participation- and rate-independent"
                " unless external heating is specified."
            )
        return base
    if cooling_rates is None:
        raise ValueError(
            "ndot_ext requires cooling_rates (the per-mode cooling"
            " rate from sympathetic_cooling_rate)."
        )
    ndot = np.broadcast_to(np.asarray(ndot_ext, dtype=float), base.shape)
    rates = np.asarray(cooling_rates, dtype=float)
    if rates.shape != base.shape:
        raise ValueError(
            f"cooling_rates shape {rates.shape} != mode shape {base.shape}"
        )
    if np.any(ndot < 0):
        raise ValueError(f"ndot_ext must be >= 0, got {ndot_ext}")
    stalled = (rates <= 0) & (ndot > 0)
    if np.any(stalled):
        raise ValueError(
            f"Modes {np.flatnonzero(stalled).tolist()} are externally"
            " heated but have zero cooling rate: no steady state"
            " exists."
        )
    return base + np.divide(
        ndot, rates, out=np.zeros_like(base), where=rates > 0
    )


def sympathetic_doppler_nbar(
    coolant_species: IonSpecies,
    mode_freqs: np.ndarray,
    participation: np.ndarray,
    ndot_ext: np.ndarray | float | None = None,
    cooling_rates: np.ndarray | None = None,
) -> np.ndarray:
    r"""Doppler cooling limit per mode via sympathetic cooling.

    $$
    \bar{n}_m = \frac{\Gamma}{2\,\omega_m}
      + \frac{\dot{n}_{\text{ext},m}}{\Gamma_m^\text{cool}}
    $$

    The first term is the ordinary Doppler limit and does **not**
    depend on the coolant participation: a friction force on the
    coolant damps mode $m$ at $\alpha |b_{c,m}|^2/m_c$ while photon
    recoil feeds it at $D_p |b_{c,m}|^2/m_c$, so the geometric
    factor cancels in the balance $E_\text{ss} = D_p/\alpha$
    (Wübbena Eqs. 21-23; the text at Eq. 26 states this explicitly,
    Eq. 27 gives $E_D = \hbar\Gamma/2$). $P_m$ sets how *fast* a
    mode approaches the limit, not where the limit is.

    The second term is the externally heated limit (Wübbena Eq. 32
    structure): electric-field-noise heating does not carry the
    same $|b_{c,m}|^2/m_c$ factor, so it does not cancel, and the
    achievable occupation of a weakly participating mode degrades.

    Like the rest of this module, the classical result
    $\bar{n} = \Gamma/(2\omega_m)$ drops the $O(1)$ geometry factor
    $(1+\xi)\sqrt{1+s}/2$ of Leibfried RMP Eq. (106) and assumes
    $\bar{n} \gg 1$; see ``tiqs.cooling.doppler``.

    Parameters
    ----------
    coolant_species : IonSpecies
        The coolant species (provides cooling transition linewidth).
    mode_freqs : np.ndarray
        Mode angular frequencies in rad/s, shape ``(n_modes,)``.
    participation : np.ndarray
        Coolant participation per mode from
        ``coolant_participation()``, shape ``(n_modes,)``. The limit
        is participation-independent; this argument is validated
        against ``mode_freqs`` and warns for spectator modes
        ($P_m = 0$), which never reach the returned value.
    ndot_ext : np.ndarray or float or None
        Optional per-mode external (electric-field-noise) heating
        rate in quanta/s: a constant flux independent of occupation,
        as produced by ``tiqs.noise.motional_heating_ops``. Requires
        ``cooling_rates``.
    cooling_rates : np.ndarray or None
        Per-mode cooling rates in 1/s from
        ``sympathetic_cooling_rate()``. Only used with ``ndot_ext``.

    Returns
    -------
    np.ndarray
        Per-mode mean phonon number at the sympathetic Doppler
        limit, shape ``(n_modes,)``.
    """
    _check_mode_shapes(mode_freqs, participation)
    gamma = coolant_species.cooling_transition.linewidth
    base = gamma / (2 * np.atleast_1d(mode_freqs).astype(float))
    return _add_external_heating(base, ndot_ext, cooling_rates)


def sympathetic_sideband_nbar(
    gamma_eff: float,
    mode_freqs: np.ndarray,
    participation: np.ndarray,
    ndot_ext: np.ndarray | float | None = None,
    cooling_rates: np.ndarray | None = None,
) -> np.ndarray:
    r"""Resolved sideband cooling limit per mode via sympathetic cooling.

    $$
    \bar{n}_m
      = \left(\frac{\gamma_\text{eff}}{2\,\omega_m}\right)^2
      + \frac{\dot{n}_{\text{ext},m}}{\Gamma_m^\text{cool}}
    $$

    As for Doppler cooling, the coupling strength cancels from the
    limit: the sideband rates $A_\pm$ both carry
    $\eta_{c,m}^2 \propto P_m$, so $\bar{n} = A_+/(A_- - A_+)$ is
    participation-independent. Only external heating (second term)
    reintroduces a $P_m$ dependence, through
    $\Gamma_m^\text{cool} \propto P_m$.

    The first term drops the $O(1)$ bracket of Leibfried RMP
    Eq. (112); see ``sideband_cooling_nbar``.

    Parameters
    ----------
    gamma_eff : float
        Effective linewidth of the cooling transition (rad/s).
    mode_freqs : np.ndarray
        Mode angular frequencies in rad/s, shape ``(n_modes,)``.
    participation : np.ndarray
        Coolant participation per mode, shape ``(n_modes,)``.
        Validated only; the limit does not depend on it.
    ndot_ext : np.ndarray or float or None
        Optional per-mode external heating rate in quanta/s (a
        constant flux, as in ``motional_heating_ops``). Requires
        ``cooling_rates``.
    cooling_rates : np.ndarray or None
        Per-mode cooling rates in 1/s. Only used with ``ndot_ext``.

    Returns
    -------
    np.ndarray
        Per-mode mean phonon number at the sympathetic sideband
        cooling limit, shape ``(n_modes,)``.
    """
    if gamma_eff <= 0:
        raise ValueError(f"gamma_eff must be > 0, got {gamma_eff}")
    _check_mode_shapes(mode_freqs, participation)
    freqs = np.atleast_1d(mode_freqs).astype(float)
    base = (gamma_eff / (2 * freqs)) ** 2
    return _add_external_heating(base, ndot_ext, cooling_rates)


def sympathetic_cooling_rate(
    coolant_species: IonSpecies,
    participation: np.ndarray,
    saturation_parameter: float = 1.0,
    detuning: float | None = None,
) -> np.ndarray:
    r"""Per-mode Doppler cooling (phonon damping) rate.

    The velocity-dependent radiation-pressure force on the coolant,

    $$
    F(v) = \frac{\hbar k \Gamma}{2}\,
      \frac{s}{1 + s + \left(2(\Delta - k v)/\Gamma\right)^2},
    $$

    has damping coefficient $\alpha = -\partial F/\partial v|_{v=0}$;
    the mode energy (hence $\langle n\rangle$) relaxes at
    $\Gamma_m^\text{cool} = P_m\,\alpha/m_c$:

    $$
    \Gamma_m^\text{cool}
      = -8\,\omega_R\,
        \frac{s\,(\Delta/\Gamma)}
             {\left[1 + s + (2\Delta/\Gamma)^2\right]^2}\,P_m ,
    \qquad
    \omega_R = \frac{\hbar k_c^2}{2\,m_c}
    $$

    At the default $\Delta = -\Gamma/2$ this reduces to
    $4\,\omega_R\,s/(2+s)^2\,P_m$, and the global maximum over both
    $s$ and $\Delta$ is $(\omega_R/2)\,P_m$, attained at $s = 2$.
    The scale is therefore the recoil frequency $\omega_R$, not
    $\Gamma$: the linewidth enters only through the dimensionless
    ratio $2\Delta/\Gamma$. Note that $\Delta = -\Gamma/2$ is the
    *temperature* optimum; the rate alone peaks at
    $2\Delta/\Gamma = -\sqrt{(1+s)/3}$.

    Each mode's phonon number relaxes exponentially:

    $$
    \bar{n}(t) = \bar{n}_\text{ss}
      + (\bar{n}_0 - \bar{n}_\text{ss})\,
        e^{-\Gamma_m^\text{cool}\,t}
    $$

    Model scope: the laser-geometry projection
    $l_x^2 = |\hat{k}\cdot\hat{e}_m|^2$ is taken as 1 (best case),
    and the secular treatment requires
    $\Gamma_m^\text{cool} \ll \omega_m \ll \Gamma$.

    Parameters
    ----------
    coolant_species : IonSpecies
        The coolant species; its ``cooling_transition`` wavevector
        and ``mass_kg`` set $\omega_R$, and its linewidth sets the
        default detuning.
    participation : np.ndarray
        Coolant participation per mode, shape ``(n_modes,)``.
    saturation_parameter : float
        Laser saturation parameter $s = I/I_\text{sat}$
        (default 1.0; the rate peaks at $s = 2$).
    detuning : float or None
        Laser detuning $\Delta = \omega_L - \omega_0$ in rad/s, must
        be negative (red). ``None`` (default) uses $-\Gamma/2$.

    Returns
    -------
    np.ndarray
        Per-mode cooling rates in 1/s, shape ``(n_modes,)``.
    """
    if saturation_parameter <= 0:
        raise ValueError(
            f"saturation_parameter must be > 0, got {saturation_parameter}"
        )
    part = np.atleast_1d(participation).astype(float)
    if np.any(part < 0):
        raise ValueError(f"participation must be >= 0, got {participation}")

    gamma = coolant_species.cooling_transition.linewidth
    if detuning is None:
        detuning = -gamma / 2
    if detuning >= 0:
        raise ValueError(f"detuning must be < 0 (red) to cool, got {detuning}")

    omega_r = recoil_frequency(coolant_species)
    s = saturation_parameter
    x = 2 * detuning / gamma
    return -8 * omega_r * s * (detuning / gamma) / (1 + s + x**2) ** 2 * part


def apply_sympathetic_cooling(
    rho: qutip.Qobj,
    ops: OperatorFactory,
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

    A target occupation comparable to the Fock cutoff cannot be
    represented: the truncated channel then pins population at the
    top of the ladder and produces a fictitious steady state, so
    ``n_bar_target >= fock_dim/2`` raises.

    Parameters
    ----------
    rho : qutip.Qobj
        Input density matrix.
    ops : OperatorFactory
        Operator factory for the composite Hilbert space.
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

    Raises
    ------
    ValueError
        If the array lengths do not match ``ops.hs.n_modes``, a rate
        or target is negative, or a target is too large for the Fock
        truncation.
    """
    rates = np.atleast_1d(np.asarray(cooling_rates, dtype=float))
    targets = np.atleast_1d(np.asarray(n_bar_target, dtype=float))
    n_modes = ops.hs.n_modes
    if rates.size != n_modes or targets.size != n_modes:
        raise ValueError(
            f"cooling_rates ({rates.size}) and n_bar_target"
            f" ({targets.size}) must both have length n_modes"
            f" ({n_modes})"
        )
    if np.any(rates < 0):
        raise ValueError(f"cooling_rates must be >= 0, got {cooling_rates}")
    if np.any(targets < 0):
        raise ValueError(f"n_bar_target must be >= 0, got {n_bar_target}")
    for m, n_t in enumerate(targets):
        fock_dim = ops.hs.fock_dim(m)
        if n_t >= fock_dim / 2:
            raise ValueError(
                f"n_bar_target[{m}] = {n_t:.4g} is not representable"
                f" with n_fock = {fock_dim} for mode {m}: the"
                " truncated thermal state would pin population at"
                f" the cutoff. Use n_fock >> {n_t:.4g} (a few times"
                " the target) or lower the target."
            )

    if duration <= 0:
        return rho
    if rho.isket:
        rho = qutip.ket2dm(rho)

    c_ops = []
    fastest = 0.0
    for m, (rate, n_t) in enumerate(zip(rates, targets, strict=True)):
        if rate <= 0:
            continue
        c_ops.append(np.sqrt(rate * (n_t + 1)) * ops.annihilate(m))
        if n_t > 0:
            c_ops.append(np.sqrt(rate * n_t) * ops.create(m))
        # Largest Lindblad matrix element on the truncated ladder.
        fastest = max(fastest, rate * (n_t + 1) * ops.hs.fock_dim(m))

    if not c_ops:
        return rho

    H = qutip.qzero_like(rho)
    # Resolve the fastest dissipative time scale, not just the
    # cooling rate: the Liouvillian's largest eigenvalue scales as
    # rate * (n_bar + 1) * n_fock, and an under-resolved tlist
    # exhausts the integrator's step budget between output times.
    n_steps = int(np.clip(duration * fastest / 10, 20, 2000))
    tlist = np.linspace(0, duration, n_steps)
    result = qutip.mesolve(H, rho, tlist, c_ops=c_ops)
    return result.states[-1]

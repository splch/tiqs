"""Electromagnetically induced transparency (EIT) cooling."""


def eit_cooling_nbar(
    gamma_eit: float,
    trap_frequency: float,
    carrier_suppression: float,
) -> float:
    r"""Estimate final phonon number from EIT cooling.

    EIT cooling uses a narrow absorption resonance (Fano profile) tuned to
    the red sideband while suppressing carrier absorption via the dark
    state. The Lamb-Dicke rate equation
    $\dot{\bar{n}} = -(A_- - A_+)\bar{n} + A_+$ with
    $A_\pm = \eta^2[W(\Delta) + W(\Delta \mp \omega)]$ has the steady
    state $\bar{n} = A_+/(A_- - A_+)$, i.e.

    $$
    \bar{n} \approx \epsilon
      + \left(\frac{\gamma_\text{EIT}}{4\,\omega_\text{trap}}\right)^2
    $$

    The first term is the carrier-limited floor: residual carrier
    absorption enters $A_+$ and $A_-$ equally, so it cancels from the
    cooling rate but survives in the numerator, giving
    $\bar{n} \to \epsilon$ (up to an $O(1)$ recoil/geometry
    projection). The second term is the ideal (perfect dark state)
    floor $(\Gamma/4\Delta_r)^2$ of Leibfried RMP Eq. (128) and
    Morigi PRA 67, 033402 Eq. (32), rewritten with the bright
    resonance tuned to the sideband ($\omega_\text{trap}$ equals the
    AC Stark shift $\Omega_r^2/4\Delta_r$), where
    $\gamma_\text{EIT}/\omega_\text{trap} = \Gamma/\Delta_r$ exactly.

    Both terms matter: the carrier term dominates for a poor dark
    state and the ideal term for a wide bright resonance.

    The advantage over resolved sideband cooling is broader bandwidth:
    the carrier-limited floor is frequency independent, so all modes
    within the EIT linewidth cool to $\sim\epsilon$ simultaneously.

    Model scope: Lamb-Dicke, weak-probe rate-equation limit with a
    single spectator-free mode; the $O(1)$ prefactor of the carrier
    term (spontaneous-recoil projection $\alpha \approx 1/3$-$2/5$
    and laser-mode projection $\cos^2\theta$, Schmidt-Kaler et al.,
    *Appl. Phys. B* **73**, 807 (2001), arXiv:quant-ph/0107087,
    Eqs. (1)-(3)) is dropped.

    Parameters
    ----------
    gamma_eit : float
        Width (FWHM) of the bright/Fano absorption resonance in
        rad/s. For a $\Lambda$ system with coupling Rabi frequency
        $\Omega_r$ at detuning $\Delta_r \gg \Omega_r, \Gamma$ this
        is $\Gamma\,\Omega_r^2/(4\Delta_r^2)$.
    trap_frequency : float
        Motional mode frequency (rad/s).
    carrier_suppression : float
        Ratio $\epsilon$ of carrier to sideband absorption
        (ideally $\ll 1$). Set by the EIT dark-state quality.

    Returns
    -------
    float
        Estimated final mean phonon number.

    Raises
    ------
    ValueError
        If ``gamma_eit`` or ``trap_frequency`` is not positive, or
        ``carrier_suppression`` is negative.
    """
    if gamma_eit <= 0:
        raise ValueError(f"gamma_eit must be > 0, got {gamma_eit}")
    if trap_frequency <= 0:
        raise ValueError(f"trap_frequency must be > 0, got {trap_frequency}")
    if carrier_suppression < 0:
        raise ValueError(
            f"carrier_suppression must be >= 0, got {carrier_suppression}"
        )
    return carrier_suppression + (gamma_eit / (4 * trap_frequency)) ** 2

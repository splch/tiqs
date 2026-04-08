"""Electromagnetically induced transparency (EIT) cooling."""


def eit_cooling_nbar(
    gamma_eit: float,
    trap_frequency: float,
    carrier_suppression: float,
) -> float:
    r"""Estimate final phonon number from EIT cooling.

    EIT cooling uses a narrow absorption resonance (Fano profile) tuned to
    the red sideband while suppressing carrier absorption via the dark
    state. The cooling limit:

    $$
    \bar{n} \sim \epsilon \cdot \frac{\gamma_\text{EIT}}{2\omega_\text{trap}}
    $$

    where $\epsilon$ is the carrier suppression factor.

    The advantage over resolved sideband cooling is broader bandwidth:
    EIT cooling can simultaneously cool multiple modes within the EIT
    linewidth.

    Parameters
    ----------
    gamma_eit : float
        Effective EIT linewidth (rad/s) - width of the Fano absorption
        feature.
    trap_frequency : float
        Motional mode frequency (rad/s).
    carrier_suppression : float
        Ratio of carrier absorption to sideband absorption (ideally $\ll 1$).
        Set by the EIT dark-state quality.

    Returns
    -------
    float
        Estimated final mean phonon number.
    """
    return carrier_suppression * gamma_eit / (2 * trap_frequency)

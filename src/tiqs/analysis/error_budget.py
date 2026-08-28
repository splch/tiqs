"""Error budget decomposition for trapped-ion operations."""


def compute_error_budget(
    ideal_fidelity: float = 1.0,
    heating_error: float = 0.0,
    dephasing_error: float = 0.0,
    scattering_error: float = 0.0,
    spam_error: float = 0.0,
    crosstalk_error: float = 0.0,
    laser_noise_error: float = 0.0,
    motional_dephasing_error: float = 0.0,
) -> dict[str, float]:
    r"""Aggregate error contributions into a total error budget.

    For small errors, the total gate infidelity is approximately additive:
    $\epsilon_\text{total} \approx \sum_i \epsilon_i$
    (Ballance et al., *Phys. Rev. Lett.* **117**, 060504 (2016),
    Table 1). The approximation degrades once the total approaches
    unity, where the returned ``total_error`` can exceed 1 even
    though every input is a valid infidelity.

    Model scope: the seven enumerated channels are the ones TIQS can
    estimate. There is no micromotion channel (micromotion is a
    standalone diagnostic in `tiqs.trap` and feeds into no Rabi
    frequency or Lamb-Dicke parameter), no leakage/shelving channel
    (ions are strictly two-level), and no term for the omitted MS-gate
    physics (off-resonant carrier, spectator modes, Debye-Waller, AC
    Stark). Those must be supplied by the caller if they matter.

    Parameters
    ----------
    ideal_fidelity : float
        Fidelity in the absence of all noise (should be ~1 for a correct
        gate). Must lie in [0, 1]; it enters the budget as
        ``ideal_infidelity = 1 - ideal_fidelity``.
    heating_error, dephasing_error, ... : float
        Individual error contributions (infidelities from each source).
        Each must be non-negative.

    Returns
    -------
    dict[str, float]
        Dictionary with each error source plus a ``"total_error"``
        entry holding their sum. ``"total_error"`` shares the
        dictionary with the components it sums, so iterating the
        result (e.g. to plot a pie chart) double-counts unless that
        key is removed first.

    Raises
    ------
    ValueError
        If *ideal_fidelity* is outside [0, 1] or any error
        contribution is negative.
    """
    if not 0.0 <= ideal_fidelity <= 1.0:
        raise ValueError(
            f"ideal_fidelity must be in [0, 1], got {ideal_fidelity}"
        )
    contributions = (
        ("heating_error", "heating", heating_error),
        ("dephasing_error", "dephasing", dephasing_error),
        ("scattering_error", "photon_scattering", scattering_error),
        ("spam_error", "spam", spam_error),
        ("crosstalk_error", "crosstalk", crosstalk_error),
        ("laser_noise_error", "laser_noise", laser_noise_error),
        (
            "motional_dephasing_error",
            "motional_dephasing",
            motional_dephasing_error,
        ),
    )
    for name, _key, value in contributions:
        if value < 0:
            raise ValueError(f"{name} must be >= 0, got {value}")

    sources = {"ideal_infidelity": 1 - ideal_fidelity}
    for _name, key, value in contributions:
        sources[key] = value
    sources["total_error"] = sum(sources.values())
    return sources

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
    """Aggregate error contributions into a total error budget.

    For small errors, the total gate infidelity is approximately additive:
    epsilon_total ~ sum of individual error contributions.

    Parameters
    ----------
    ideal_fidelity : float
        Fidelity in the absence of all noise (should be ~1 for a correct gate).
    heating_error, dephasing_error, ... : float
        Individual error contributions (infidelities from each source).

    Returns
    -------
    dict[str, float]
        Dictionary with each error source and the total.
    """
    sources = {
        "ideal_infidelity": 1 - ideal_fidelity,
        "heating": heating_error,
        "dephasing": dephasing_error,
        "photon_scattering": scattering_error,
        "spam": spam_error,
        "crosstalk": crosstalk_error,
        "laser_noise": laser_noise_error,
        "motional_dephasing": motional_dephasing_error,
    }
    sources["total_error"] = sum(sources.values())
    return sources

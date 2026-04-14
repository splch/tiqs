"""Error budget decomposition for trapped-ion operations."""

import numpy as np


def residual_displacement_infidelity(
    alpha_residual: complex | float,
    n_bar: float = 0.0,
) -> float:
    r"""Infidelity from residual spin-motion entanglement.

    When a phase-space loop does not close, the residual
    displacement $\alpha_\text{res}$ leaves the spin and motion
    entangled.  The resulting infidelity of the spin-only state
    is:

    $$
    1 - F \approx 1 - \exp\bigl(
      -2\,|\alpha_\text{res}|^2\,(2\bar{n}+1)\bigr)
    $$

    This formula is exact for a single mode in a coherent state
    and approximate for thermal states.  The $(2\bar{n}+1)$
    factor is why detuning errors compound with temperature.

    Parameters
    ----------
    alpha_residual : complex or float
        Residual phase-space displacement.
    n_bar : float
        Mean phonon number.

    Returns
    -------
    float
        Estimated infidelity (1 - F).

    References
    ----------
    Kirchmair et al., NJP 11, 023002 (2009), Eq. 5.
    """
    alpha_sq = abs(alpha_residual) ** 2
    return 1 - np.exp(-2 * alpha_sq * (2 * n_bar + 1))


def ms_residual_alpha(
    eta: float,
    rabi_frequency: float,
    detuning: float,
    gate_time: float,
) -> complex:
    r"""Residual displacement after an MS gate with imperfect closure.

    $$
    \alpha(\tau) = \frac{\eta\,\Omega}{\delta}
      \bigl(e^{i\delta\tau} - 1\bigr)
    $$

    For perfect closure ($\delta\tau = 2\pi K$), this is zero.

    Parameters
    ----------
    eta : float
    rabi_frequency : float
    detuning : float
    gate_time : float

    Returns
    -------
    complex
        Residual displacement in phase space.
    """
    if detuning == 0:
        return 0.0
    coupling = eta * rabi_frequency
    return (coupling / detuning) * (np.exp(1j * detuning * gate_time) - 1)


def off_resonant_mode_error(
    eta: np.ndarray,
    rabi_frequency: float,
    detunings: list[float],
    gate_time: float,
    n_bar: list[float] | None = None,
) -> dict[str, float]:
    r"""Infidelity from off-resonant modes via residual displacement.

    Computes the residual phase-space displacement for each mode
    using the exact formula
    $\alpha_p = (\bar\eta_p\,\Omega/\delta_p)(e^{i\delta_p\tau}-1)$,
    then converts to infidelity via
    $1 - \exp(-2|\alpha_p|^2(2\bar{n}_p+1))$.

    Parameters
    ----------
    eta : np.ndarray
        Lamb-Dicke matrix, shape ``(N_ions, N_modes)``.
    rabi_frequency : float
        Drive Rabi frequency (rad/s).
    detunings : list[float]
        Per-mode detuning ``delta_p`` (rad/s).
    gate_time : float
        Gate duration (s).
    n_bar : list[float] or None
        Mean phonon number per mode.  Defaults to 0 for all modes.

    Returns
    -------
    dict[str, float]
        Per-mode errors keyed by ``"mode_0"``, ``"mode_1"``, etc.,
        plus ``"total"``.
    """
    _n_ions, n_modes = eta.shape
    if n_bar is None:
        n_bar = [0.0] * n_modes

    result: dict[str, float] = {}
    total = 0.0
    for p in range(n_modes):
        delta_p = detunings[p]
        if delta_p == 0:
            result[f"mode_{p}"] = 0.0
            continue
        eta_avg = float(np.mean(np.abs(eta[:, p])))
        alpha = ms_residual_alpha(eta_avg, rabi_frequency, delta_p, gate_time)
        eps = residual_displacement_infidelity(alpha, n_bar[p])
        result[f"mode_{p}"] = eps
        total += eps

    result["total"] = total
    return result


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
    $\epsilon_\text{total} \approx \sum_i \epsilon_i$.

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

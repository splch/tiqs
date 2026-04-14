"""Calibration routines for multi-mode entangling gates."""

import numpy as np

from tiqs.constants import PI, TWO_PI
from tiqs.gates.molmer_sorensen import (
    ms_gate_duration,
    ms_geometric_phase,
    ms_residual_displacement,
)


def calibrate_single_tone_ms(
    eta: np.ndarray,
    mode_frequencies: np.ndarray,
    target_mode: int,
    ion_pair: tuple[int, int],
    loops: int = 1,
    target_chi: float = PI / 4,
    detuning: float | None = None,
) -> dict:
    r"""Calibrate a single-tone multi-mode MS gate.

    One bichromatic drive tone is placed near ``target_mode``.  The
    target mode's phase-space loop closes exactly; other modes
    accumulate residual spin-motion entanglement.

    The Rabi frequency is chosen so the total geometric phase
    (including off-resonant mode contributions) equals
    ``target_chi``.

    Parameters
    ----------
    eta : np.ndarray
        Lamb-Dicke matrix, shape ``(N_ions, N_modes)``.
    mode_frequencies : np.ndarray
        Mode angular frequencies, shape ``(N_modes,)``.
    target_mode : int
        Index into ``mode_frequencies`` of the mode to close
        exactly.
    ion_pair : tuple[int, int]
        ``(j, k)`` indices into the eta matrix rows.
    loops : int
        Number of phase-space loops for the target mode.
    target_chi : float
        Desired geometric phase (default ``pi/4``).
    detuning : float or None
        Detuning from the target mode in rad/s.  If ``None``, a
        default is chosen midway between the target mode and its
        nearest neighbour (or ``2*pi * 10 kHz`` for a single mode).

    Returns
    -------
    dict
        ``mu``, ``detunings``, ``rabi_frequency``, ``gate_time``,
        ``chi_matrix``, ``residuals``.
    """
    n_modes = len(mode_frequencies)
    j, k = ion_pair

    if detuning is None:
        if n_modes == 1:
            detuning = TWO_PI * 10e3
        else:
            diffs = np.abs(mode_frequencies - mode_frequencies[target_mode])
            diffs[target_mode] = np.inf
            nearest_gap = np.min(diffs)
            detuning = nearest_gap / 2

    delta_target = detuning
    mu = mode_frequencies[target_mode] + delta_target
    detunings = [float(mu - omega_p) for omega_p in mode_frequencies]
    tau = ms_gate_duration(delta_target, loops)

    # Solve for Omega from: sum_p 4*pi*K_eff_p*eta_j*eta_k*Omega^2/delta_p^2
    #                      = target_chi
    phase_sum = 0.0
    for p in range(n_modes):
        delta_p = detunings[p]
        if delta_p == 0:
            continue
        k_eff = delta_p * tau / TWO_PI
        phase_sum += 4 * PI * k_eff * eta[j, p] * eta[k, p] / delta_p**2

    if phase_sum <= 0:
        raise ValueError(
            "Cannot calibrate: total phase contribution is "
            "non-positive.  Check eta matrix signs."
        )
    omega = np.sqrt(target_chi / phase_sum)

    chi = ms_geometric_phase(eta, omega, detunings, tau)
    residuals = ms_residual_displacement(detunings, tau)

    return {
        "mu": mu,
        "detunings": detunings,
        "rabi_frequency": omega,
        "gate_time": tau,
        "chi_matrix": chi,
        "residuals": residuals,
    }


def calibrate_multi_tone_ms(
    eta: np.ndarray,
    mode_frequencies: np.ndarray,
    ion_pair: tuple[int, int],
    loops: int | list[int] = 1,
    target_chi: float = PI / 4,
    base_detuning: float | None = None,
) -> dict:
    r"""Calibrate a multi-tone MS gate closing all modes.

    Each mode gets its own drive tone with detuning chosen so
    every phase-space loop closes in the same gate time.  A
    uniform Rabi frequency is used across all tones.

    Parameters
    ----------
    eta : np.ndarray
        Lamb-Dicke matrix, shape ``(N_ions, N_modes)``.
    mode_frequencies : np.ndarray
        Mode angular frequencies, shape ``(N_modes,)``.
    ion_pair : tuple[int, int]
        ``(j, k)`` indices into the eta matrix rows.
    loops : int or list[int]
        Per-mode loop counts.  A scalar applies to all modes.
    target_chi : float
        Desired geometric phase (default ``pi/4``).
    base_detuning : float or None
        Detuning for mode 0.  If ``None``, defaults to
        ``2*pi * 10 kHz``.

    Returns
    -------
    dict
        ``detunings``, ``rabi_frequency`` (list), ``gate_time``,
        ``chi_matrix``, ``residuals`` (all zeros by construction).
    """
    n_modes = len(mode_frequencies)
    j, k = ion_pair

    if isinstance(loops, int):
        loops_list = [loops] * n_modes
    else:
        loops_list = list(loops)

    if base_detuning is None:
        base_detuning = TWO_PI * 10e3

    tau = ms_gate_duration(base_detuning, loops_list[0])
    detunings = [TWO_PI * lp / tau for lp in loops_list]

    # Uniform Omega: sum_p 4*pi*K_p*eta_j*eta_k*Omega^2/delta_p^2 = target_chi
    phase_sum = 0.0
    for p in range(n_modes):
        delta_p = detunings[p]
        if delta_p == 0:
            continue
        k_p = loops_list[p]
        phase_sum += 4 * PI * k_p * eta[j, p] * eta[k, p] / delta_p**2

    if phase_sum <= 0:
        raise ValueError(
            "Cannot calibrate: total phase contribution is non-positive."
        )
    omega = np.sqrt(target_chi / phase_sum)
    rabi_frequencies = [float(omega)] * n_modes

    chi = ms_geometric_phase(eta, rabi_frequencies, detunings, tau)
    residuals = ms_residual_displacement(detunings, tau)

    return {
        "detunings": detunings,
        "rabi_frequency": rabi_frequencies,
        "gate_time": tau,
        "chi_matrix": chi,
        "residuals": residuals,
    }

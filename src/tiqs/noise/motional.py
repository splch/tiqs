"""Motional decoherence: anomalous heating and motional dephasing."""

import numpy as np
import qutip

from tiqs.constants import ELECTRON_CHARGE, HBAR
from tiqs.hilbert_space.operators import OperatorFactory


def motional_heating_ops(
    ops: OperatorFactory,
    mode: int,
    heating_rate: float,
    n_bar_env: float = 0.0,
) -> list[qutip.Qobj]:
    """Lindblad collapse operators for motional heating of a single mode.

    Models a thermal bath that adds phonons at rate n_dot (quanta/s).
    For a zero-temperature bath (n_bar_env=0): only phonon creation.
    For finite temperature: both creation (rate ~ n_bar+1) and destruction (rate ~ n_bar).

    The collapse operators are:
        L_up   = sqrt(heating_rate * (n_bar_env + 1)) * a^dag  (phonon gain)
        L_down = sqrt(heating_rate * n_bar_env) * a             (phonon loss)

    Parameters
    ----------
    ops : OperatorFactory
    mode : int
        Motional mode index.
    heating_rate : float
        Heating rate in quanta/second (n_dot).
    n_bar_env : float
        Mean phonon number of the thermal environment.

    Returns
    -------
    list[qutip.Qobj]
        Collapse operators.
    """
    c_ops = []
    rate_up = heating_rate * (n_bar_env + 1)
    if rate_up > 0:
        c_ops.append(np.sqrt(rate_up) * ops.create(mode))
    rate_down = heating_rate * n_bar_env
    if rate_down > 0:
        c_ops.append(np.sqrt(rate_down) * ops.annihilate(mode))
    return c_ops


def motional_dephasing_op(
    ops: OperatorFactory,
    mode: int,
    rate: float,
) -> qutip.Qobj:
    """Collapse operator for motional dephasing: L = sqrt(rate) * n_mode.

    Models fluctuations in the trap frequency that cause dephasing of
    motional superposition states without changing the phonon number.
    """
    return np.sqrt(rate) * ops.number(mode)


def heating_rate_from_noise(
    spectral_density: float,
    distance: float,
    frequency: float,
    mass_kg: float = 40 * 1.66e-27,
    alpha: float = 1.0,
    reference_distance: float = 100e-6,
    reference_frequency: float = 1e6,
) -> float:
    """Estimate heating rate from electric field noise spectral density.

    n_dot = e^2 * S_E(omega) / (4 * m * hbar * omega)

    where S_E scales as d^{-4} with distance and f^{-alpha} with frequency.

    Parameters
    ----------
    spectral_density : float
        Electric field noise spectral density S_E at reference distance and frequency
        in V^2 m^-2 Hz^-1.
    distance : float
        Ion-electrode distance in meters.
    frequency : float
        Motional mode frequency in Hz.
    mass_kg : float
        Ion mass in kg.
    alpha : float
        Frequency scaling exponent (typically 1-2 for 1/f noise).
    reference_distance : float
        Reference distance for the spectral density.
    reference_frequency : float
        Reference frequency for the spectral density.

    Returns
    -------
    float
        Heating rate in quanta/second.
    """
    d_scaling = (reference_distance / distance) ** 4
    f_scaling = (reference_frequency / frequency) ** alpha
    S_E = spectral_density * d_scaling * f_scaling
    omega = 2 * np.pi * frequency
    return ELECTRON_CHARGE**2 * S_E / (4 * mass_kg * HBAR * omega)

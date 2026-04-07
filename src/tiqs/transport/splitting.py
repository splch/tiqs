"""Crystal splitting and merging operations."""

import numpy as np


def split_crystal_excitation(
    trap_frequency: float,
    split_duration: float,
) -> float:
    """Estimate motional excitation from splitting a two-ion crystal.

    Crystal splitting is more heating-prone than linear shuttling because
    the axial potential must be reshaped from a single well to a double well.
    The excitation depends on how adiabatically the potential is transformed.

    Parameters
    ----------
    trap_frequency : float
        Axial secular angular frequency (rad/s).
    split_duration : float
        Time for the splitting operation (s).

    Returns
    -------
    float
        Estimated added motional quanta.
    """
    adiabaticity = trap_frequency * split_duration
    if adiabaticity > 50:
        return 0.05
    return max(0.05, 2.0 * np.exp(-adiabaticity / 5))

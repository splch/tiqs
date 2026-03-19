"""Lamb-Dicke parameter calculation for ion-mode-laser combinations."""
import numpy as np

from tiqs.chain.normal_modes import NormalModeResult
from tiqs.constants import HBAR
from tiqs.species.data import IonSpecies


def lamb_dicke_parameters(
    modes: NormalModeResult,
    species: IonSpecies,
    k_eff: float,
    direction: str = "axial",
) -> np.ndarray:
    """Compute Lamb-Dicke parameters eta_{i,m} for each ion i and mode m.

    eta_{i,m} = k_eff * b_{i,m} * sqrt(hbar / (2 * M * omega_m))

    where b_{i,m} is the participation of ion i in mode m, M is the ion mass,
    omega_m is the mode frequency, and k_eff is the effective laser wavevector
    component along the mode direction.

    Parameters
    ----------
    modes : NormalModeResult
        Result from normal_modes().
    species : IonSpecies
        Ion species (for mass).
    k_eff : float
        Effective laser wavevector magnitude along the mode direction (rad/m).
        For counter-propagating Raman beams: k_eff = 2 * k_laser.
        For co-propagating: k_eff ~ 0 (no motional coupling).
        For single beam on optical qubit: k_eff = k_laser * cos(theta).
    direction : str
        Which modes to compute for: "axial", "radial_x", or "radial_y".

    Returns
    -------
    np.ndarray
        Matrix of Lamb-Dicke parameters, shape (N_ions, N_modes).
        eta[i, m] is the Lamb-Dicke parameter for ion i and mode m.
    """
    if direction == "axial":
        freqs = modes.axial_freqs
        vectors = modes.axial_vectors
    elif direction == "radial_x":
        freqs = modes.radial_x_freqs
        vectors = modes.radial_x_vectors
    elif direction == "radial_y":
        freqs = modes.radial_y_freqs
        vectors = modes.radial_y_vectors
    else:
        raise ValueError(f"Unknown direction: {direction}")

    m = species.mass_kg
    n_ions = len(freqs)
    eta = np.zeros((n_ions, n_ions))

    for mode_idx in range(n_ions):
        omega_m = freqs[mode_idx]
        if omega_m <= 0:
            continue
        x_zpf = np.sqrt(HBAR / (2 * m * omega_m))
        for ion_idx in range(n_ions):
            b_im = vectors[ion_idx, mode_idx]
            eta[ion_idx, mode_idx] = abs(k_eff) * abs(b_im) * x_zpf

    return eta

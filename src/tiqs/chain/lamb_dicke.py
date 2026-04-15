"""Lamb-Dicke parameter calculation for ion-mode-laser combinations."""

import numpy as np

from tiqs.chain.normal_modes import NormalModeResult
from tiqs.constants import HBAR
from tiqs.species.protocol import Species


def lamb_dicke_parameters(
    modes: NormalModeResult,
    species: Species,
    k_eff: float,
    direction: str = "axial",
) -> np.ndarray:
    r"""Compute Lamb-Dicke parameters $\eta_{i,m}$
    for each ion $i$ and mode $m$.

    $$
    \eta_{i,m} = k_\mathrm{eff} \, b_{i,m} \sqrt{\frac{\hbar}{2 M \omega_m}}
    $$

    where $b_{i,m}$ is the participation of ion $i$
    in mode $m$, $M$ is the particle mass,
    $\omega_m$ is the mode frequency, and
    $k_\mathrm{eff}$ is the effective wavevector
    component along the mode direction.

    Parameters
    ----------
    modes : NormalModeResult
        Result from normal_modes().
    species : Species
        Particle species (for mass).
    k_eff : float
        Effective wavevector magnitude along the mode direction (rad/m).
        For counter-propagating Raman beams:
        $k_\mathrm{eff} = 2 k_\mathrm{laser}$.
        For co-propagating:
        $k_\mathrm{eff} \approx 0$ (no motional
        coupling). For single beam on optical qubit:
        $k_\mathrm{eff} = k_\mathrm{laser} \cos\theta$.
    direction : str
        Key into ``modes.modes``: e.g. ``"axial"``, ``"radial_x"``,
        ``"modified_cyclotron"``.

    Returns
    -------
    np.ndarray
        Matrix of Lamb-Dicke parameters, shape
        $(N_\mathrm{ions}, N_\mathrm{modes})$.
        $\eta[i, m]$ is the Lamb-Dicke parameter for ion $i$ and mode $m$.
    """
    if direction not in modes.modes:
        raise ValueError(
            f"Unknown direction: {direction!r}. "
            f"Available: {list(modes.modes.keys())}"
        )
    group = modes.modes[direction]
    freqs = group.freqs
    vectors = group.vectors

    m = species.mass_kg
    x_zpf = np.zeros_like(freqs)
    mask = freqs > 0
    x_zpf[mask] = np.sqrt(HBAR / (2 * m * freqs[mask]))
    return k_eff * vectors * x_zpf

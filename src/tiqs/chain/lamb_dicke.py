"""Lamb-Dicke parameter calculation for ion-mode-laser combinations."""

import numpy as np

from tiqs.chain.normal_modes import NormalModeResult
from tiqs.constants import HBAR
from tiqs.species.protocol import Species


def lamb_dicke_parameters(
    modes: NormalModeResult,
    species: Species | list[Species],
    k_eff: float | list[float],
    direction: str = "axial",
) -> np.ndarray:
    r"""Compute Lamb-Dicke parameters $\eta_{i,m}$
    for each ion $i$ and mode $m$.

    $$
    \eta_{i,m} = k_{\mathrm{eff},i} \, b_{i,m}
    \sqrt{\frac{\hbar}{2\, m_i\, \omega_m}}
    $$

    where $b_{i,m}$ is the mass-weighted participation of ion $i$
    in mode $m$, $m_i$ is the mass of ion $i$,
    $\omega_m$ is the mode frequency, and
    $k_{\mathrm{eff},i}$ is the effective wavevector for ion $i$.

    For single-species chains (scalar ``species`` and ``k_eff``),
    all ions share the same mass and wavevector.

    Parameters
    ----------
    modes : NormalModeResult
        Result from normal_modes().
    species : Species or list[Species]
        Particle species for mass. A single ``Species`` applies
        to all ions. A list provides per-ion species for
        mixed-species chains.
    k_eff : float or list[float]
        Effective wavevector magnitude along the mode direction
        (rad/m). A single ``float`` applies to all ions. A list
        provides per-ion values for mixed-species chains where
        different ions use different laser wavelengths.
        For counter-propagating Raman beams:
        $k_\mathrm{eff} = 2 k_\mathrm{laser}$.
        For co-propagating:
        $k_\mathrm{eff} \approx 0$ (no motional
        coupling). For single beam on optical qubit:
        $k_\mathrm{eff} = k_\mathrm{laser} \cos\theta$.
        For magnetic-gradient coupling
        (Mintert and Wunderlich, PRL 87, 257904):
        $k_\mathrm{eff} = g\,\mu_B\,(\partial B/\partial z)
        / (\hbar\,\omega_m)$.
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
    n_ions = vectors.shape[0]

    if isinstance(species, list):
        if len(species) != n_ions:
            raise ValueError(
                f"species list length {len(species)} != n_ions {n_ions}"
            )
        masses = np.array([s.mass_kg for s in species])
    else:
        masses = np.full(n_ions, species.mass_kg)

    if isinstance(k_eff, list):
        if len(k_eff) != n_ions:
            raise ValueError(
                f"k_eff list length {len(k_eff)} != n_ions {n_ions}"
            )
        k_arr = np.array(k_eff)
    else:
        k_arr = np.full(n_ions, k_eff)

    # x_zpf[i, m] = sqrt(hbar / (2 * m_i * omega_m))
    mask = freqs > 0
    x_zpf = np.zeros((n_ions, len(freqs)))
    x_zpf[:, mask] = np.sqrt(
        HBAR / (2 * masses[:, np.newaxis] * freqs[np.newaxis, mask])
    )
    return k_arr[:, np.newaxis] * vectors * x_zpf

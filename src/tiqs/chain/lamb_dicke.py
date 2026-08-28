"""Lamb-Dicke parameter calculation for ion-mode-laser combinations."""

from collections.abc import Sequence

import numpy as np

from tiqs.chain.normal_modes import NormalModeResult
from tiqs.constants import BOHR_MAGNETON, ELECTRON_G_FACTOR, HBAR
from tiqs.species.protocol import Species


def _mode_geometry(
    modes: NormalModeResult,
    species: Species | Sequence[Species],
    direction: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Validated mode frequencies, eigenvectors and zero-point spreads.

    Returns ``(freqs, vectors, x_zpf)`` where
    ``x_zpf[i, m]`` $= \sqrt{\hbar/(2 m_i \omega_m)}$.
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

    # eta diverges as omega -> 0, so a non-positive frequency has no
    # Lamb-Dicke parameter at all. Returning 0 would be the opposite of
    # the physical limit and would hide an unstable mode.
    bad = np.where(freqs <= 0)[0]
    if len(bad) > 0:
        raise ValueError(
            f"Modes {bad.tolist()} of group {direction!r} have "
            f"non-positive frequencies "
            f"({freqs[bad].tolist()!r} rad/s): the zero-point spread "
            f"sqrt(hbar / (2 m omega)) diverges, so the Lamb-Dicke "
            f"parameter is undefined."
        )

    if hasattr(species, "mass_kg"):
        masses = np.full(n_ions, species.mass_kg)
    else:
        per_ion = list(species)
        if len(per_ion) != n_ions:
            raise ValueError(
                f"species sequence length {len(per_ion)} != n_ions {n_ions}"
            )
        masses = np.array([s.mass_kg for s in per_ion])

    x_zpf = np.sqrt(HBAR / (2 * masses[:, np.newaxis] * freqs[np.newaxis, :]))
    return freqs, vectors, x_zpf


def lamb_dicke_parameters(
    modes: NormalModeResult,
    species: Species | Sequence[Species],
    k_eff: float | Sequence[float] | np.ndarray,
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
    all ions share the same mass and wavevector, reducing to
    the standard formula with a single $M$ and $k_\mathrm{eff}$.

    Parameters
    ----------
    modes : NormalModeResult
        Result from normal_modes().
    species : Species or sequence of Species
        Particle species for mass. A single ``Species`` applies
        to all ions. A sequence (list, tuple or array) provides
        per-ion species for mixed-species chains.
    k_eff : float or sequence of float
        Effective wavevector magnitude along the mode direction
        (rad/m). A scalar applies to all ions. A length-``n_ions``
        sequence provides per-ion values for mixed-species chains
        where different ions use different laser wavelengths.
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

    Raises
    ------
    ValueError
        If ``direction`` is not a key of ``modes.modes``, if any mode in
        that group has a non-positive frequency, or if a per-ion
        ``species``/``k_eff`` sequence has the wrong length.

    Notes
    -----
    ``k_eff`` is per ion, not per mode, so it cannot express a coupling
    whose strength depends on the mode frequency. Magnetic-gradient
    (MAGIC) coupling is such a case - use
    :func:`gradient_lamb_dicke_parameters` for it.

    Applying this to the Penning ``"magnetron"`` group gives the
    zero-point spread of a NEGATIVE-energy mode; see
    :class:`tiqs.chain.normal_modes.NormalModeResult`.
    """
    _, vectors, x_zpf = _mode_geometry(modes, species, direction)
    n_ions = vectors.shape[0]

    k_arr = np.atleast_1d(np.asarray(k_eff, dtype=float))
    if k_arr.ndim != 1 or k_arr.size not in (1, n_ions):
        raise ValueError(
            f"k_eff must be a scalar or a sequence of length "
            f"n_ions {n_ions}, got shape {np.shape(k_eff)}"
        )
    if k_arr.size == 1:
        k_arr = np.full(n_ions, k_arr[0])

    return k_arr[:, np.newaxis] * vectors * x_zpf


def gradient_lamb_dicke_parameters(
    modes: NormalModeResult,
    species: Species | Sequence[Species],
    gradient: float,
    direction: str = "axial",
    g_factor: float = ELECTRON_G_FACTOR,
) -> np.ndarray:
    r"""Lamb-Dicke parameters for magnetic-gradient (MAGIC) coupling.

    A static magnetic-field gradient makes the qubit splitting depend on
    position, which couples spin to motion with no laser at all
    (Mintert and Wunderlich, PRL 87, 257904 (2001)). The effective
    wavevector is set by the gradient divided by the mode energy,

    $$
    k_{\mathrm{eff},m} = \frac{g\,\mu_B\,(\partial B/\partial z)}
    {\hbar\,\omega_m}
    $$

    so it is a property of the MODE, not of the ion:

    $$
    \eta_{i,m} = \frac{g\,\mu_B\,(\partial B/\partial z)\, b_{i,m}}
    {\hbar\,\omega_m}
    \sqrt{\frac{\hbar}{2\, m_i\, \omega_m}}
    \;\propto\; \omega_m^{-3/2}
    $$

    That $\omega_m^{-3/2}$ scaling is why this cannot be obtained from
    :func:`lamb_dicke_parameters`, whose per-ion ``k_eff`` is applied
    across the whole frequency axis and therefore yields
    $\omega_m^{-1/2}$. A scalar ``k_eff`` there is only correct for one
    named mode at a time.

    Parameters
    ----------
    modes : NormalModeResult
        Result from normal_modes().
    species : Species or sequence of Species
        Particle species for mass, as in
        :func:`lamb_dicke_parameters`.
    gradient : float
        Magnetic field gradient $\partial B/\partial z$ along the mode
        direction, in T/m. Must be non-negative.
    direction : str
        Key into ``modes.modes``.
    g_factor : float, optional
        Lande $g$ factor of the qubit transition. Defaults to the free
        electron value, appropriate for the $\Delta m = 1$ Zeeman
        qubits used in MAGIC experiments.

    Returns
    -------
    np.ndarray
        Matrix of Lamb-Dicke parameters, shape
        $(N_\mathrm{ions}, N_\mathrm{modes})$.

    Raises
    ------
    ValueError
        If ``gradient`` is negative, or for the same reasons as
        :func:`lamb_dicke_parameters`.
    """
    if gradient < 0:
        raise ValueError(f"gradient must be non-negative, got {gradient}")

    freqs, vectors, x_zpf = _mode_geometry(modes, species, direction)
    k_mode = g_factor * BOHR_MAGNETON * gradient / (HBAR * freqs)
    return k_mode[np.newaxis, :] * vectors * x_zpf

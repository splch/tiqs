"""Normal mode analysis of a Coulomb crystal."""

import warnings
from dataclasses import dataclass

import numpy as np

from tiqs.chain.equilibrium import equilibrium_positions
from tiqs.constants import ELECTRON_CHARGE, EPSILON_0, PI
from tiqs.trap import PaulTrap, PenningTrap

_COULOMB_PREFACTOR = ELECTRON_CHARGE**2 / (4 * PI * EPSILON_0)


@dataclass
class ModeGroup:
    """A set of normal modes along one degree of freedom.

    Attributes
    ----------
    freqs : np.ndarray
        Mode angular frequencies in rad/s, shape (N,), sorted
        ascending.
    vectors : np.ndarray
        Mode eigenvectors, shape (N, N). Column m is the
        participation vector for mode m:
        vectors[i, m] = b_{i,m}.
    """

    freqs: np.ndarray
    vectors: np.ndarray


@dataclass
class NormalModeResult:
    """Results of normal mode analysis.

    Attributes
    ----------
    positions : np.ndarray
        Equilibrium positions in meters, shape (N,).
    modes : dict[str, ModeGroup]
        Mode groups keyed by physical name. For a Paul trap:
        ``"axial"``, ``"radial_x"``, ``"radial_y"``. For a
        Penning trap: ``"axial"``, ``"modified_cyclotron"``,
        ``"magnetron"``.
    """

    positions: np.ndarray
    modes: dict[str, ModeGroup]


def _dynamical_matrix(
    n_ions: int,
    pos: np.ndarray,
    omega_diag: np.ndarray,
    masses: np.ndarray,
    axial: bool,
) -> np.ndarray:
    r"""Build the mass-weighted dynamical matrix $D = M^{-1/2} V M^{-1/2}$.

    For axial modes: $D_{ii} = \omega_{z,i}^2 + \sum 2C/(m_i\,d^3)$,
    $D_{ij} = -2C/(\sqrt{m_i m_j}\,d^3)$.
    For radial: signs flip and factor changes from 2 to 1.
    $C = e^2/(4\pi\epsilon_0)$ is mass-independent.
    Reduces to $H = V/m$ for single-species chains.
    """
    sign = -1 if axial else +1
    factor = 2 if axial else 1

    D = np.zeros((n_ions, n_ions))
    for i in range(n_ions):
        coulomb_sum = 0.0
        for k in range(n_ions):
            if k != i:
                d3 = abs(pos[i] - pos[k]) ** 3
                coulomb_sum += factor * _COULOMB_PREFACTOR / (masses[i] * d3)
                D[i, k] = (
                    sign
                    * factor
                    * _COULOMB_PREFACTOR
                    / (np.sqrt(masses[i] * masses[k]) * d3)
                )
        if axial:
            D[i, i] = omega_diag[i] ** 2 + coulomb_sum
        else:
            D[i, i] = omega_diag[i] ** 2 - coulomb_sum

    return D


def _diagonalize_to_modes(
    n_ions: int, omega_diag: np.ndarray, D: np.ndarray
) -> ModeGroup:
    """Diagonalize a dynamical matrix into a ModeGroup."""
    if n_ions == 1:
        return ModeGroup(
            freqs=np.array([omega_diag[0]]),
            vectors=np.array([[1.0]]),
        )
    eigenvalues, eigenvectors = np.linalg.eigh(D)
    freqs = np.sqrt(np.maximum(eigenvalues, 0.0))
    return ModeGroup(freqs=freqs, vectors=eigenvectors)


def _penning_transverse_modes(
    n_ions: int, omega_transverse: np.ndarray
) -> ModeGroup:
    """Per-ion transverse modes (single-particle approximation).

    Full N-particle mode structure with rotating-frame Coulomb
    coupling is a future extension.
    """
    if n_ions > 1:
        warnings.warn(
            "Penning transverse modes use a single-particle approximation. "
            "Inter-particle coupling is not included.",
            stacklevel=3,
        )
    order = np.argsort(omega_transverse)
    freqs = omega_transverse[order]
    vectors = np.eye(n_ions)[:, order]
    return ModeGroup(freqs=freqs, vectors=vectors)


def normal_modes(
    n_ions: int,
    trap: PaulTrap | PenningTrap,
    masses: np.ndarray | None = None,
) -> NormalModeResult:
    r"""Compute all normal modes of an N-ion crystal.

    Constructs the mass-weighted dynamical matrix
    $D = M^{-1/2}\,V\,M^{-1/2}$ of the total potential (harmonic
    trap + Coulomb) evaluated at the equilibrium positions, then
    diagonalizes it to find mode frequencies and participation
    vectors. For single-species chains this is equivalent to
    the standard Hessian $H = V/m$. Axial modes are computed
    identically for all trap types; transverse modes use
    trap-specific physics (radial pseudopotential for Paul traps,
    per-ion cyclotron/magnetron frequencies for Penning traps).

    Parameters
    ----------
    n_ions : int
        Number of ions.
    trap : PaulTrap or PenningTrap
        Trap configuration. ``trap.species`` serves as the
        reference species for electrode-derived quantities
        (axial spring constant, Mathieu parameters).
    masses : np.ndarray or None, optional
        Per-ion masses in kg, shape ``(n_ions,)``. When ``None``
        (default), all ions use ``trap.species.mass_kg``. For
        mixed-species chains, pass an array with different masses
        (e.g. ``np.array([m_Be, m_Ca])``). Ordering matches
        the sorted equilibrium positions: ``masses[0]`` is the
        leftmost ion, ``masses[-1]`` the rightmost.

    Returns
    -------
    NormalModeResult
        Equilibrium positions and mode groups keyed by physical name.
    """
    if masses is None:
        masses = np.full(n_ions, trap.species.mass_kg)
    else:
        masses = np.asarray(masses, dtype=float)
        if masses.shape != (n_ions,):
            raise ValueError(
                f"masses must have shape ({n_ions},), got {masses.shape}"
            )

    pos = equilibrium_positions(n_ions, trap)

    # Axial spring constant K = m_ref * omega_z_ref^2 is mass-independent
    # (K = kappa * e * U_dc / z_0^2 for DC confinement).
    spring_constant = trap.species.mass_kg * trap.omega_axial**2
    omega_z = np.sqrt(spring_constant / masses)

    D_axial = _dynamical_matrix(n_ions, pos, omega_z, masses, axial=True)
    axial = _diagonalize_to_modes(n_ions, omega_z, D_axial)

    if isinstance(trap, PenningTrap):
        # Per-ion cyclotron frequency: omega_c_i = eB / m_i.
        omega_c = ELECTRON_CHARGE * trap.magnetic_field / masses
        omega_c_half = omega_c / 2
        disc = omega_c_half**2 - omega_z**2 / 2
        unstable = np.where(disc < 0)[0]
        if len(unstable) > 0:
            raise ValueError(
                f"Penning-unstable ions at indices {unstable.tolist()}: "
                f"omega_c < sqrt(2)*omega_z. Heavier species may "
                f"require a stronger magnetic field."
            )
        omega_plus = omega_c_half + np.sqrt(disc)
        omega_minus = omega_c_half - np.sqrt(disc)
        modes = {
            "axial": axial,
            "modified_cyclotron": _penning_transverse_modes(
                n_ions, omega_plus
            ),
            "magnetron": _penning_transverse_modes(n_ions, omega_minus),
        }
    elif isinstance(trap, PaulTrap):
        # Per-ion radial frequency from mass-dependent Mathieu parameters.
        q = (
            2
            * ELECTRON_CHARGE
            * trap.v_rf
            / (masses * trap.omega_rf**2 * trap.r0**2)
        )
        a = -2 * omega_z**2 / trap.omega_rf**2
        beta_sq = a + q**2 / 2
        large_q = np.where(q > 0.4)[0]
        if len(large_q) > 0:
            warnings.warn(
                f"Mathieu q > 0.4 for ions at indices "
                f"{large_q.tolist()} (q = {q[large_q].tolist()!r}). "
                f"The pseudopotential approximation loses accuracy "
                f"above q ~ 0.4.",
                stacklevel=2,
            )
        unstable = np.where(beta_sq <= 0)[0]
        if len(unstable) > 0:
            raise ValueError(
                f"Radially unstable ions at indices {unstable.tolist()}: "
                f"beta^2 <= 0. Lighter species may require different "
                f"RF parameters."
            )
        omega_r = (trap.omega_rf / 2) * np.sqrt(beta_sq)

        D_radial = _dynamical_matrix(n_ions, pos, omega_r, masses, axial=False)
        radial = _diagonalize_to_modes(n_ions, omega_r, D_radial)
        modes = {
            "axial": axial,
            "radial_x": radial,
            "radial_y": ModeGroup(
                freqs=radial.freqs.copy(), vectors=radial.vectors.copy()
            ),
        }
    else:
        raise TypeError(f"Unknown trap type: {type(trap)}")

    return NormalModeResult(positions=pos, modes=modes)

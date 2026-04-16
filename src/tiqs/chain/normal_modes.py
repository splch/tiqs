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


def _coulomb_hessian(
    n_ions: int,
    pos: np.ndarray,
    omega_diag: float,
    mass_kg: float,
    axial: bool,
) -> np.ndarray:
    """Build the Coulomb-coupled Hessian matrix for one direction.

    H_ij = d^2 V / (m * dx_i dx_j)  where  C = e^2 / (4*pi*eps0*m).

    For axial modes (focusing Coulomb coupling):
        Diagonal:     omega_z^2 + sum_{k!=i} 2C / |z_i - z_k|^3
        Off-diagonal: -2C / |z_i - z_j|^3

    For radial modes (defocusing Coulomb coupling):
        Diagonal:     omega_r^2 - sum_{k!=i} C / |z_i - z_k|^3
        Off-diagonal: +C / |z_i - z_j|^3

    Note the sign difference: radial Coulomb coupling is
    repulsive/defocusing.
    """
    C = _COULOMB_PREFACTOR / mass_kg
    sign = -1 if axial else +1
    factor = 2 if axial else 1

    H = np.zeros((n_ions, n_ions))
    for i in range(n_ions):
        coulomb_sum = 0.0
        for k in range(n_ions):
            if k != i:
                d3 = abs(pos[i] - pos[k]) ** 3
                coulomb_sum += factor * C / d3
                # axial: -2C/d^3, radial: +C/d^3
                H[i, k] = sign * factor * C / d3
        if axial:
            # omega_z^2 + sum 2C/|z_i - z_k|^3
            H[i, i] = omega_diag**2 + coulomb_sum
        else:
            # omega_r^2 - sum C/|z_i - z_k|^3
            H[i, i] = omega_diag**2 - coulomb_sum

    return H


def _diagonalize_to_modes(
    n_ions: int, omega_single: float, H: np.ndarray | None
) -> ModeGroup:
    """Diagonalize a Hessian into a ModeGroup, or return single-ion result."""
    if n_ions == 1:
        return ModeGroup(
            freqs=np.array([omega_single]),
            vectors=np.array([[1.0]]),
        )
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    freqs = np.sqrt(np.maximum(eigenvalues, 0.0))
    return ModeGroup(freqs=freqs, vectors=eigenvectors)


def _penning_transverse_modes(
    n_ions: int, omega_transverse: float
) -> ModeGroup:
    """Compute transverse modes for a Penning trap (single-particle
    approximation).

    For a single-particle or weakly-coupled chain, the transverse modes
    are approximately at the single-particle frequency. Full N-particle
    Penning mode structure with rotation-frame Coulomb coupling is a
    future extension.
    """
    if n_ions > 1:
        warnings.warn(
            "Penning transverse modes use a single-particle approximation. "
            "Inter-particle coupling is not included.",
            stacklevel=3,
        )
    freqs = np.full(n_ions, omega_transverse)
    vectors = np.eye(n_ions)
    return ModeGroup(freqs=freqs, vectors=vectors)


def normal_modes(
    n_ions: int, trap: PaulTrap | PenningTrap
) -> NormalModeResult:
    """Compute all normal modes of an N-ion crystal.

    Constructs the Hessian matrix of the total potential (harmonic trap +
    Coulomb) evaluated at the equilibrium positions, then diagonalizes it
    to find mode frequencies and participation vectors. Axial modes are
    computed identically for all trap types; transverse modes use
    trap-specific physics (radial pseudopotential for Paul traps,
    cyclotron/magnetron frequencies for Penning traps).

    Parameters
    ----------
    n_ions : int
        Number of ions.
    trap : PaulTrap or PenningTrap
        Trap configuration.

    Returns
    -------
    NormalModeResult
        Equilibrium positions and mode groups keyed by physical name.
    """
    pos = equilibrium_positions(n_ions, trap)
    m = trap.species.mass_kg
    omega_z = trap.omega_axial

    H_axial = _coulomb_hessian(n_ions, pos, omega_z, m, axial=True)
    axial = _diagonalize_to_modes(n_ions, omega_z, H_axial)

    if isinstance(trap, PenningTrap):
        modes = {
            "axial": axial,
            "modified_cyclotron": _penning_transverse_modes(
                n_ions, trap.omega_modified_cyclotron
            ),
            "magnetron": _penning_transverse_modes(
                n_ions, trap.omega_magnetron
            ),
        }
    elif isinstance(trap, PaulTrap):
        H_radial = _coulomb_hessian(
            n_ions, pos, trap.omega_radial, m, axial=False
        )
        radial = _diagonalize_to_modes(n_ions, trap.omega_radial, H_radial)
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

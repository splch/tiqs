"""Normal mode analysis of a Coulomb crystal."""

from dataclasses import dataclass

import numpy as np

from tiqs.chain.equilibrium import equilibrium_positions
from tiqs.constants import ELECTRON_CHARGE, EPSILON_0, PI
from tiqs.trap import PaulTrap, PenningTrap, Trap


@dataclass
class ModeGroup:
    r"""A set of normal modes along one degree of freedom.

    Attributes
    ----------
    freqs : np.ndarray
        Mode angular frequencies in rad/s, shape $(N,)$, sorted ascending.
    vectors : np.ndarray
        Mode eigenvectors, shape $(N, N)$. Column $m$ is the participation
        vector for mode $m$: ``vectors[i, m]`` $= b_{i,m}$.
    """

    freqs: np.ndarray
    vectors: np.ndarray


@dataclass
class NormalModeResult:
    r"""Results of normal mode analysis.

    Attributes
    ----------
    positions : np.ndarray
        Equilibrium positions in meters, shape $(N,)$.
    modes : dict[str, ModeGroup]
        Mode groups keyed by physical name. For a Paul trap:
        ``"axial"``, ``"radial_x"``, ``"radial_y"``. For a Penning
        trap: ``"axial"``, ``"modified_cyclotron"``, ``"magnetron"``.
    """

    positions: np.ndarray
    modes: dict[str, ModeGroup]


def _axial_modes(
    n_ions: int, pos: np.ndarray, omega_z: float, mass_kg: float
) -> ModeGroup:
    """Compute axial normal modes from equilibrium positions.

    Shared by all trap types since the axial Hessian depends only on
    the axial frequency and Coulomb repulsion.
    """
    if n_ions == 1:
        return ModeGroup(
            freqs=np.array([omega_z]),
            vectors=np.array([[1.0]]),
        )

    coulomb_prefactor = ELECTRON_CHARGE**2 / (4 * PI * EPSILON_0)
    C = coulomb_prefactor / mass_kg

    H_axial = np.zeros((n_ions, n_ions))
    for i in range(n_ions):
        for j in range(n_ions):
            if i == j:
                coulomb_sum = 0.0
                for k in range(n_ions):
                    if k != i:
                        coulomb_sum += 2 * C / abs(pos[i] - pos[k]) ** 3
                H_axial[i, i] = omega_z**2 + coulomb_sum
            else:
                H_axial[i, j] = -2 * C / abs(pos[i] - pos[j]) ** 3

    eigenvalues, eigenvectors = np.linalg.eigh(H_axial)
    freqs = np.sqrt(np.maximum(eigenvalues, 0.0))
    return ModeGroup(freqs=freqs, vectors=eigenvectors)


def _paul_radial_modes(
    n_ions: int, pos: np.ndarray, omega_r: float, mass_kg: float
) -> ModeGroup:
    """Compute radial normal modes for a Paul trap."""
    if n_ions == 1:
        return ModeGroup(
            freqs=np.array([omega_r]),
            vectors=np.array([[1.0]]),
        )

    coulomb_prefactor = ELECTRON_CHARGE**2 / (4 * PI * EPSILON_0)
    C = coulomb_prefactor / mass_kg

    H_radial = np.zeros((n_ions, n_ions))
    for i in range(n_ions):
        for j in range(n_ions):
            if i == j:
                coulomb_sum = 0.0
                for k in range(n_ions):
                    if k != i:
                        coulomb_sum += C / abs(pos[i] - pos[k]) ** 3
                H_radial[i, i] = omega_r**2 - coulomb_sum
            else:
                H_radial[i, j] = C / abs(pos[i] - pos[j]) ** 3

    eigenvalues, eigenvectors = np.linalg.eigh(H_radial)
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
    freqs = np.full(n_ions, omega_transverse)
    vectors = np.eye(n_ions)
    return ModeGroup(freqs=freqs, vectors=vectors)


def normal_modes(n_ions: int, trap: Trap) -> NormalModeResult:
    """Compute all normal modes of an N-particle crystal.

    Parameters
    ----------
    n_ions : int
        Number of particles.
    trap : Trap
        Trap configuration (PaulTrap or PenningTrap).

    Returns
    -------
    NormalModeResult
        Equilibrium positions and mode groups keyed by physical name.
    """
    pos = equilibrium_positions(n_ions, trap)
    m = trap.species.mass_kg
    omega_z = trap.omega_axial

    axial = _axial_modes(n_ions, pos, omega_z, m)

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
        radial = _paul_radial_modes(n_ions, pos, trap.omega_radial, m)
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

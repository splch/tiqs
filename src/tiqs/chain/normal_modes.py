"""Normal mode analysis of an ion Coulomb crystal."""

from dataclasses import dataclass

import numpy as np

from tiqs.chain.equilibrium import equilibrium_positions
from tiqs.constants import ELECTRON_CHARGE, EPSILON_0, PI
from tiqs.trap import PaulTrap


@dataclass
class NormalModeResult:
    """Results of normal mode analysis.

    Attributes
    ----------
    positions : np.ndarray
        Equilibrium positions in meters, shape (N,).
    axial_freqs : np.ndarray
        Axial mode angular frequencies in rad/s, shape (N,), sorted ascending.
    axial_vectors : np.ndarray
        Axial mode eigenvectors, shape (N, N). Column m is the participation
        vector for mode m: axial_vectors[i, m] = b_{i,m}.
    radial_x_freqs : np.ndarray
        Radial-x mode angular frequencies, shape (N,), sorted ascending.
    radial_x_vectors : np.ndarray
        Radial-x mode eigenvectors, shape (N, N).
    radial_y_freqs : np.ndarray
        Radial-y mode angular frequencies, shape (N,), sorted ascending.
    radial_y_vectors : np.ndarray
        Radial-y mode eigenvectors, shape (N, N).
    """

    positions: np.ndarray
    axial_freqs: np.ndarray
    axial_vectors: np.ndarray
    radial_x_freqs: np.ndarray
    radial_x_vectors: np.ndarray
    radial_y_freqs: np.ndarray
    radial_y_vectors: np.ndarray


def normal_modes(n_ions: int, trap: PaulTrap) -> NormalModeResult:
    """Compute all normal modes of an N-ion crystal in a linear Paul trap.

    Constructs the Hessian matrix of the total potential (harmonic trap +
    Coulomb) evaluated at the equilibrium positions, then diagonalizes it
    to find mode frequencies and participation vectors.

    Parameters
    ----------
    n_ions : int
        Number of ions.
    trap : PaulTrap
        Trap configuration.

    Returns
    -------
    NormalModeResult
        Mode frequencies and eigenvectors for axial and both radial directions.
    """
    pos = equilibrium_positions(n_ions, trap)

    if n_ions == 1:
        return NormalModeResult(
            positions=pos,
            axial_freqs=np.array([trap.omega_axial]),
            axial_vectors=np.array([[1.0]]),
            radial_x_freqs=np.array([trap.omega_radial]),
            radial_x_vectors=np.array([[1.0]]),
            radial_y_freqs=np.array([trap.omega_radial]),
            radial_y_vectors=np.array([[1.0]]),
        )

    m = trap.species.mass_kg
    omega_z = trap.omega_axial
    omega_r = trap.omega_radial
    coulomb_prefactor = ELECTRON_CHARGE**2 / (4 * PI * EPSILON_0)

    # Axial Hessian: H_ij = d^2 V / (m * dx_i dx_j)
    # Diagonal: omega_z^2 + sum_{k!=i} 2*C / |z_i - z_k|^3
    # Off-diagonal: -2*C / |z_i - z_j|^3
    # where C = e^2 / (4*pi*eps0 * m)
    C = coulomb_prefactor / m
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

    eigenvalues_ax, eigenvectors_ax = np.linalg.eigh(H_axial)
    axial_freqs = np.sqrt(np.maximum(eigenvalues_ax, 0.0))

    # Radial Hessian: transverse direction (x or y perpendicular to
    # chain axis)
    # Diagonal: omega_r^2 - sum_{k!=i} C / |z_i - z_k|^3
    # Off-diagonal: +C / |z_i - z_j|^3
    # (Note the SIGN DIFFERENCE from axial: radial Coulomb coupling
    # is repulsive/defocusing)
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

    eigenvalues_rad, eigenvectors_rad = np.linalg.eigh(H_radial)
    radial_freqs = np.sqrt(np.maximum(eigenvalues_rad, 0.0))

    return NormalModeResult(
        positions=pos,
        axial_freqs=axial_freqs,
        axial_vectors=eigenvectors_ax,
        radial_x_freqs=radial_freqs,
        radial_x_vectors=eigenvectors_rad,
        radial_y_freqs=radial_freqs.copy(),
        radial_y_vectors=eigenvectors_rad.copy(),
    )

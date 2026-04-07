"""Compute equilibrium positions of N ions in a linear Paul trap."""

import numpy as np
from scipy.optimize import root

from tiqs.constants import ELECTRON_CHARGE, EPSILON_0, PI
from tiqs.trap.paul_trap import PaulTrap


def equilibrium_positions(n_ions: int, trap: PaulTrap) -> np.ndarray:
    """Find the axial equilibrium positions of N ions in a harmonic trap with Coulomb repulsion.

    Solves the dimensionless equilibrium equation for each ion i:
        u_i - sum_{j != i} sign(u_i - u_j) / (u_i - u_j)^2 = 0

    Then rescales to physical units using the length scale:
        l = (e^2 / (4*pi*eps0 * m * omega_z^2))^(1/3)

    Parameters
    ----------
    n_ions : int
        Number of ions.
    trap : PaulTrap
        Trap configuration providing mass and axial frequency.

    Returns
    -------
    np.ndarray
        Sorted array of equilibrium positions in meters, shape (n_ions,).
    """
    if n_ions == 0:
        return np.array([])
    if n_ions == 1:
        return np.array([0.0])

    length_scale = (
        ELECTRON_CHARGE**2
        / (4 * PI * EPSILON_0 * trap.species.mass_kg * trap.omega_axial**2)
    ) ** (1 / 3)

    def equations(u):
        """Dimensionless force balance: d/du_i [ sum_i u_i^2/2 + sum_{i<j} 1/|u_i-u_j| ] = 0."""
        f = np.zeros(n_ions)
        for i in range(n_ions):
            f[i] = u[i]
            for j in range(n_ions):
                if i != j:
                    diff = u[i] - u[j]
                    f[i] -= np.sign(diff) / diff**2
        return f

    u0 = np.linspace(-(n_ions - 1) / 2, (n_ions - 1) / 2, n_ions) * 1.5
    sol = root(equations, u0, method="hybr", tol=1e-12)
    residual = np.max(np.abs(equations(sol.x)))
    if not sol.success and residual > 1e-10:
        raise RuntimeError(f"Failed to find equilibrium positions: {sol.message}")

    u_sorted = np.sort(sol.x)
    return u_sorted * length_scale

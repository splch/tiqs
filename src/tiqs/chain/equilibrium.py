"""Compute equilibrium positions of N ions in a linear trap."""

import numpy as np
from scipy.optimize import root

from tiqs.constants import COULOMB_CONSTANT
from tiqs.trap import Trap


def equilibrium_positions(n_ions: int, trap: Trap) -> np.ndarray:
    r"""Find the axial equilibrium positions of N ions in a linear trap.

    Solves for a harmonic trap with Coulomb repulsion.

    Solves the dimensionless equilibrium equation for each ion $i$:

    $$
    u_i - \sum_{j \neq i} \frac{\mathrm{sign}(u_i - u_j)}{(u_i - u_j)^2} = 0
    $$

    Then rescales to physical units using the length scale:

    $$
    \ell = \left( \frac{e^2}{4\pi\epsilon_0\, m\, \omega_z^2} \right)^{1/3}
    $$

    Parameters
    ----------
    n_ions : int
        Number of ions.
    trap : Trap
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
        COULOMB_CONSTANT / (trap.species.mass_kg * trap.omega_axial**2)
    ) ** (1 / 3)

    def equations(u):
        r"""Dimensionless force balance.

        $$
        \frac{\partial}{\partial u_i} \left[
        \sum_i \frac{u_i^2}{2}
        + \sum_{i<j} \frac{1}{|u_i - u_j|}
        \right] = 0
        $$
        """
        diff = u[:, np.newaxis] - u[np.newaxis, :]
        np.fill_diagonal(diff, np.inf)
        return u - np.sum(np.sign(diff) / diff**2, axis=1)

    u0 = np.linspace(-(n_ions - 1) / 2, (n_ions - 1) / 2, n_ions) * 1.5
    sol = root(equations, u0, method="hybr", tol=1e-12)
    residual = np.max(np.abs(equations(sol.x)))
    if not sol.success and residual > 1e-10:
        raise RuntimeError(
            f"Failed to find equilibrium positions: {sol.message}"
        )

    u_sorted = np.sort(sol.x)
    return u_sorted * length_scale

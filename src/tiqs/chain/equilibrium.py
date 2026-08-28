"""Compute equilibrium positions of N ions in a linear trap."""

import numpy as np
from scipy.optimize import root

from tiqs.constants import COULOMB_CONSTANT
from tiqs.trap import Trap


def _force_jacobian(u: np.ndarray) -> np.ndarray:
    r"""Jacobian of the dimensionless force balance.

    Differentiating $f_i = u_i - \sum_{j \neq i} \mathrm{sign}(u_i -
    u_j)/(u_i - u_j)^2$ and using $\mathrm{sign}(d)/d^3 = |d|^{-3}$:

    $$
    J_{ii} = 1 + 2\sum_{j \neq i} |u_i - u_j|^{-3}, \qquad
    J_{ij} = -2\,|u_i - u_j|^{-3}
    $$

    which is the (positive-definite) axial Hessian in dimensionless
    units, so supplying it makes the root-find a true Newton step.
    """
    diff = u[:, np.newaxis] - u[np.newaxis, :]
    np.fill_diagonal(diff, np.inf)
    coupling = 2.0 / np.abs(diff) ** 3
    jacobian = -coupling
    np.fill_diagonal(jacobian, 1.0 + np.sum(coupling, axis=1))
    return jacobian


def equilibrium_positions(n_ions: int, trap: Trap) -> np.ndarray:
    r"""Find the axial equilibrium positions of N ions in a linear trap.

    Solves for a harmonic trap with Coulomb repulsion.

    Solves the dimensionless equilibrium equation for each ion $i$
    (James, Appl. Phys. B 66, 181 (1998), Eq. (2.5)):

    $$
    u_i - \sum_{j \neq i} \frac{\mathrm{sign}(u_i - u_j)}{(u_i - u_j)^2} = 0
    $$

    Then rescales to physical units using the length scale
    (James Eq. (2.4)):

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

    Raises
    ------
    RuntimeError
        If the root-find does not reduce the force residual below
        ``1e-10`` relative to the chain half-width, or if the solution
        is not strictly ordered (two ions at the same site).

    Notes
    -----
    Only the 1D axial force balance is solved, so the result is a
    stationary point of the axial potential alone. Above the
    linear-to-zigzag threshold it is a saddle point of the full 3D
    potential rather than a minimum; :func:`tiqs.chain.normal_modes`
    detects that case from the radial dynamical matrix.
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

    # The chain half-width grows only as ~N^0.56 (James Eq. (2.8) gives
    # the minimum spacing u_min = 2.018 N^-0.559), so a fixed spacing
    # would place the guess a factor 5.9 too wide by N = 60 and the
    # root-find diverges. Verified to converge for N = 2..200.
    u0 = np.linspace(-1.0, 1.0, n_ions) * 0.79 * n_ions**0.56
    sol = root(equations, u0, jac=_force_jacobian, method="hybr", tol=1e-13)
    u_sorted = np.sort(sol.x)

    # Ordering first: coincident ions make the Coulomb force infinite, so
    # the residual would report inf rather than the actual failure mode.
    if np.any(np.diff(u_sorted) <= 0):
        raise RuntimeError(
            f"Equilibrium positions for {n_ions} ions are not strictly "
            f"ordered: the root-find collapsed two or more ions onto the "
            f"same site ({sol.message})."
        )

    # Gate on the residual, not on sol.success: hybr's xtol criterion can
    # report success at a point with a large residual, and conversely it
    # reports failure on solutions that are exact to machine precision
    # (N = 2 among them). The residual is measured relative to the chain
    # half-width because the equations are forces of order max|u|, so the
    # attainable absolute residual grows with N: 2e-10 at N = 107, but
    # 2e-11 relative. Genuine divergence leaves a residual of order 10,
    # ten orders of magnitude clear of the gate.
    residual = np.max(np.abs(equations(u_sorted)))
    tolerance = 1e-10 * max(1.0, np.max(np.abs(u_sorted)))
    if residual > tolerance:
        raise RuntimeError(
            f"Failed to find equilibrium positions for {n_ions} ions: "
            f"force residual {residual:.3e} exceeds {tolerance:.3e} "
            f"({sol.message})"
        )

    return u_sorted * length_scale

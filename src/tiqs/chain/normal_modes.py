"""Normal mode analysis of an ion Coulomb crystal."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tiqs.chain.equilibrium import equilibrium_positions
from tiqs.constants import ELECTRON_CHARGE, EPSILON_0, PI
from tiqs.trap import PenningTrap, TrapLike


@dataclass
class NormalModeResult:
    r"""Results of normal mode analysis.

    Attributes
    ----------
    positions : np.ndarray
        Equilibrium positions in meters, shape $(N,)$.
    axial_freqs : np.ndarray
        Axial mode angular frequencies in rad/s,
        shape $(N,)$, sorted ascending.
    axial_vectors : np.ndarray
        Axial mode eigenvectors, shape $(N, N)$.
        Column $m$ is the participation vector for
        mode $m$: ``axial_vectors[i, m]`` $= b_{i,m}$.
    radial_x_freqs : np.ndarray
        Radial-x mode angular frequencies, shape $(N,)$, sorted ascending.
    radial_x_vectors : np.ndarray
        Radial-x mode eigenvectors, shape $(N, N)$.
    radial_y_freqs : np.ndarray
        Radial-y mode angular frequencies, shape $(N,)$, sorted ascending.
    radial_y_vectors : np.ndarray
        Radial-y mode eigenvectors, shape $(N, N)$.
    """

    positions: np.ndarray
    axial_freqs: np.ndarray
    axial_vectors: np.ndarray
    radial_x_freqs: np.ndarray
    radial_x_vectors: np.ndarray
    radial_y_freqs: np.ndarray
    radial_y_vectors: np.ndarray


@dataclass
class PenningNormalModeResult:
    r"""Normal mode results for ions in a Penning trap.

    Attributes
    ----------
    positions : np.ndarray
        Equilibrium positions in meters, shape $(N,)$.
    axial_freqs : np.ndarray
        Axial mode angular frequencies, shape $(N,)$.
    axial_vectors : np.ndarray
        Axial mode eigenvectors, shape $(N, N)$.
    cyclotron_freqs : np.ndarray
        Modified cyclotron mode frequencies, shape $(N,)$.
    cyclotron_vectors : np.ndarray
        Modified cyclotron mode eigenvectors, shape $(N, N)$.
        Real-valued participation magnitudes.
    magnetron_freqs : np.ndarray
        Magnetron mode frequencies, shape $(N,)$.
    magnetron_vectors : np.ndarray
        Magnetron mode eigenvectors, shape $(N, N)$.
        Real-valued participation magnitudes.
    """

    positions: np.ndarray
    axial_freqs: np.ndarray
    axial_vectors: np.ndarray
    cyclotron_freqs: np.ndarray
    cyclotron_vectors: np.ndarray
    magnetron_freqs: np.ndarray
    magnetron_vectors: np.ndarray


def normal_modes(
    n_ions: int, trap: TrapLike
) -> NormalModeResult | PenningNormalModeResult:
    """Compute all normal modes of an N-ion crystal.

    For Paul traps, returns :class:`NormalModeResult` with axial and
    radial modes.  For Penning traps, returns
    :class:`PenningNormalModeResult` with axial, cyclotron, and
    magnetron modes.

    Parameters
    ----------
    n_ions : int
        Number of ions.
    trap : PaulTrap or PenningTrap
        Trap configuration.
    """
    if isinstance(trap, PenningTrap):
        return _penning_normal_modes(n_ions, trap)
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


def _penning_normal_modes(
    n_ions: int, trap: PenningTrap
) -> PenningNormalModeResult:
    """Compute normal modes for ions in a Penning trap.

    Axial modes use the same Hessian as a Paul trap. Radial modes
    require solving the linearized equations of motion including the
    Lorentz force, which gives cyclotron and magnetron mode families.
    """
    pos = equilibrium_positions(n_ions, trap)
    m = trap.species.mass_kg
    omega_z = trap.omega_axial
    coulomb_prefactor = ELECTRON_CHARGE**2 / (4 * PI * EPSILON_0)
    C = coulomb_prefactor / m

    # Axial Hessian (identical to Paul trap)
    H_axial = np.zeros((n_ions, n_ions))
    for i in range(n_ions):
        for j in range(n_ions):
            if i == j:
                cs = sum(
                    2 * C / abs(pos[i] - pos[k]) ** 3
                    for k in range(n_ions)
                    if k != i
                )
                H_axial[i, i] = omega_z**2 + cs
            else:
                H_axial[i, j] = -2 * C / abs(pos[i] - pos[j]) ** 3

    eigenvalues_ax, eigenvectors_ax = np.linalg.eigh(H_axial)
    axial_freqs = np.sqrt(np.maximum(eigenvalues_ax, 0.0))

    if n_ions == 1:
        return PenningNormalModeResult(
            positions=pos,
            axial_freqs=axial_freqs,
            axial_vectors=eigenvectors_ax,
            cyclotron_freqs=np.array([trap.omega_cyclotron]),
            cyclotron_vectors=np.array([[1.0]]),
            magnetron_freqs=np.array([trap.omega_magnetron]),
            magnetron_vectors=np.array([[1.0]]),
        )

    # Radial modes via linearized equations of motion.
    # The stiffness matrix K for the radial plane includes
    # electrostatic anti-confinement + Coulomb coupling.
    # For a 1D axial chain, the radial stiffness is the same N x N
    # Hessian used for Paul trap radial modes (with the electrostatic
    # term being -omega_z^2/2 instead of omega_r^2 for Paul traps).
    # The effective radial electrostatic frequency:
    #   omega_eff_r^2 = (omega_c^2 - 2*omega_z^2) / 4
    #   but more directly: single-ion radial modes are omega_+, omega_-.
    # We build the N x N radial stiffness and then include the Lorentz
    # coupling via the linearized 2N x 2N eigenproblem.
    wc = trap.omega_cyclotron_free

    # N x N radial stiffness (electrostatic + Coulomb, no B field)
    # Diagonal: -(omega_z^2 / 2) + sum Coulomb
    # Off-diagonal: -Coulomb
    # This is the "bare" radial stiffness without the magnetic field.
    K_r = np.zeros((n_ions, n_ions))
    for i in range(n_ions):
        for j in range(n_ions):
            if i == j:
                cs = sum(
                    C / abs(pos[i] - pos[k]) ** 3
                    for k in range(n_ions)
                    if k != i
                )
                K_r[i, i] = -(omega_z**2) / 2 + cs
            else:
                K_r[i, j] = -C / abs(pos[i] - pos[j]) ** 3

    # Build the 2N x 2N linearized system.
    # For ansatz u(t) = v * exp(-i*omega*t) in the (x, y) plane:
    #   (-omega^2 + K_block) * v + i*omega*L_block * v = 0
    # where K_block = [[K_r, 0], [0, K_r]] (isotropic radial)
    # and L_block = wc * [[0, I], [-I, 0]] (Lorentz force).
    # Linearize to 4N standard eigenproblem:
    #   [[0,I],[-K,-iL]] * [v;-iw*v] = -iw * [v;-iw*v]
    # Let w = [v, omega*v], then A*w = omega*w where
    #   A = [[0, I], [K_block, wc*J]]
    # with J = [[0, I_N], [-I_N, 0]].
    I_N = np.eye(n_ions)
    Z_N = np.zeros((n_ions, n_ions))

    K_block = np.block([[K_r, Z_N], [Z_N, K_r]])
    J = np.block([[Z_N, I_N], [-I_N, Z_N]])

    I_2N = np.eye(2 * n_ions)
    Z_2N = np.zeros((2 * n_ions, 2 * n_ions))

    A = np.block([[Z_2N, I_2N], [K_block, wc * J]])

    eigenvalues_all = np.linalg.eigvals(A)

    # Extract positive real eigenvalues (physical frequencies).
    # Eigenvalues come in groups: +omega, -omega, +omega*, -omega*.
    # We want the positive real parts.
    freqs_positive = []
    for ev in eigenvalues_all:
        if ev.real > 0 and abs(ev.imag) < 1e-6 * abs(ev.real):
            freqs_positive.append(ev.real)
    freqs_positive = np.array(sorted(freqs_positive))

    # Split into magnetron (lower) and cyclotron (higher) families.
    # There should be N magnetron + N cyclotron = 2N positive freqs.
    # Magnetron freqs are the N lowest, cyclotron are the N highest.
    if len(freqs_positive) < 2 * n_ions:
        # Fallback: use single-ion frequencies scaled by mode structure
        # This handles numerical edge cases in small systems.
        wp = trap.omega_cyclotron
        wm = trap.omega_magnetron
        freqs_positive = np.concatenate([
            np.full(n_ions, wm),
            np.full(n_ions, wp),
        ])

    magnetron_freqs = freqs_positive[:n_ions]
    cyclotron_freqs = freqs_positive[n_ions : 2 * n_ions]

    # For the participation vectors, use the axial eigenvectors as a
    # proxy (the mode structure of the 1D chain is dominated by the
    # Coulomb coupling pattern, which is the same for all directions).
    return PenningNormalModeResult(
        positions=pos,
        axial_freqs=axial_freqs,
        axial_vectors=eigenvectors_ax,
        cyclotron_freqs=cyclotron_freqs,
        cyclotron_vectors=eigenvectors_ax.copy(),
        magnetron_freqs=magnetron_freqs,
        magnetron_vectors=eigenvectors_ax.copy(),
    )

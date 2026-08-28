"""Normal mode analysis of a Coulomb crystal."""

import warnings
from dataclasses import dataclass

import numpy as np

from tiqs.chain.equilibrium import equilibrium_positions
from tiqs.constants import COULOMB_CONSTANT, ELECTRON_CHARGE
from tiqs.trap import PaulTrap, PenningTrap


@dataclass
class ModeGroup:
    r"""A set of normal modes along one degree of freedom.

    Attributes
    ----------
    freqs : np.ndarray
        Mode angular frequencies in rad/s, shape (N,), sorted
        ascending.
    vectors : np.ndarray
        Mass-weighted mode eigenvectors of
        $D = M^{-1/2} V M^{-1/2}$, shape (N, N). Column m is the
        participation vector for mode m: ``vectors[i, m]`` =
        $b_{i,m}$, orthonormal over $i$. These are NOT physical
        displacements: ion $i$ moves by
        ``vectors[i, m] / sqrt(m_i)`` times a common amplitude
        (Home, Adv. At. Mol. Opt. Phys. 62, 231 (2013), Eqs. (8),
        (14); Sosnova, PRA 103, 012610 (2021), Eqs. (4)-(5)). For a
        single-species chain the two differ only by a global
        constant, so ``vectors`` is then proportional to the
        displacement pattern.
    """

    freqs: np.ndarray
    vectors: np.ndarray


@dataclass
class NormalModeResult:
    r"""Results of normal mode analysis.

    Attributes
    ----------
    positions : np.ndarray
        Equilibrium positions in meters, shape (N,).
    modes : dict[str, ModeGroup]
        Mode groups keyed by physical name. For a Paul trap:
        ``"axial"``, ``"radial_x"``, ``"radial_y"``. For a
        Penning trap: ``"axial"``, ``"modified_cyclotron"``,
        ``"magnetron"``.

    Notes
    -----
    All groups except ``"magnetron"`` are ordinary positive-energy
    oscillators, $H = \sum_p \hbar\omega_p (n_p + 1/2)$. The Penning
    magnetron mode is an $\mathbf{E}\times\mathbf{B}$ drift whose
    total energy DECREASES with radius, so it enters as
    $-\hbar\omega_-(n_- + 1/2)$: the motion is only metastable, the
    red/blue sideband roles are exchanged for it, and cooling it means
    raising $n_-$ (Dehmelt, Nobel lecture 1989, Fig. 4 caption;
    Brown and Gabrielse, Rev. Mod. Phys. 58, 233 (1986), Sec. II;
    Jain et al., Nature 627, 510 (2024)). ``freqs`` stores the
    unsigned $\omega_-$; nothing downstream applies the sign for you.
    """

    positions: np.ndarray
    modes: dict[str, ModeGroup]


def _dynamical_matrix(
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

    diff = np.abs(pos[:, np.newaxis] - pos[np.newaxis, :])
    np.fill_diagonal(diff, np.inf)
    d3 = diff**3

    mass_geom = np.sqrt(masses[:, np.newaxis] * masses[np.newaxis, :])
    D = sign * factor * COULOMB_CONSTANT / (mass_geom * d3)
    coulomb_diag = np.sum(
        factor * COULOMB_CONSTANT / (masses[:, np.newaxis] * d3), axis=1
    )
    if axial:
        np.fill_diagonal(D, omega_diag**2 + coulomb_diag)
    else:
        np.fill_diagonal(D, omega_diag**2 - coulomb_diag)

    return D


def _diagonalize_to_modes(
    D: np.ndarray, label: str, hint: str = ""
) -> ModeGroup:
    """Diagonalize a dynamical matrix into a ModeGroup.

    A negative eigenvalue is an imaginary frequency: the equilibrium is
    a saddle point of the potential along ``label``, not a minimum, so
    no normal modes exist there. Raise instead of clamping - a clamped
    zero propagates as a legitimate-looking mode with an undefined
    (formally divergent) zero-point amplitude.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(D)
    tol = 1e-9 * max(1.0, np.max(np.abs(eigenvalues)))
    negative = np.where(eigenvalues < -tol)[0]
    if len(negative) > 0:
        raise ValueError(
            f"{label} dynamical matrix has {len(negative)} negative "
            f"eigenvalue(s) (most negative {eigenvalues[0]:.4e} "
            f"rad^2/s^2): these are imaginary frequencies, so the "
            f"equilibrium is a saddle point along {label} and no normal "
            f"modes exist.{hint}"
        )
    freqs = np.sqrt(np.maximum(eigenvalues, 0.0))
    return ModeGroup(freqs=freqs, vectors=eigenvectors)


def _zigzag_hint(
    axial: ModeGroup, omega_z: np.ndarray, omega_r: np.ndarray
) -> str:
    r"""Actionable suffix for the linear-to-zigzag buckling error.

    In units of $\omega_z^2$ the axial and radial dynamical matrices are
    $I + 2A$ and $(\omega_r/\omega_z)^2 I - A$ with the same Coulomb
    matrix $A$, so the linear chain is a radial minimum iff

    $$
    \frac{\omega_r}{\omega_z} > c_N
    = \sqrt{\frac{\mu_{N,\max} - 1}{2}}
    $$

    with $\mu_{N,\max}$ the largest axial eigenvalue in units of
    $\omega_z^2$ (James, Appl. Phys. B 66, 181 (1998), Table 2). Exact
    values: $c_2 = 1$, $c_3 = 1.5492$, $c_4 = 2.0382$, $c_5 = 2.4975$.
    The single ratio only exists for a single species, so the
    mixed-species case gets the general statement instead.
    """
    if not (
        np.allclose(omega_z, omega_z[0]) and np.allclose(omega_r, omega_r[0])
    ):
        return (
            " The linear chain has buckled into a zigzag. Mixed-species "
            "chains have no single anisotropy ratio; stiffen the radial "
            "confinement (larger v_rf or smaller omega_axial) or shorten "
            "the chain until the radial dynamical matrix is positive "
            "definite."
        )
    mu_max = (axial.freqs[-1] / omega_z[0]) ** 2
    c_n = np.sqrt((mu_max - 1) / 2)
    return (
        f" The linear chain has buckled into a zigzag: "
        f"omega_radial/omega_axial = {omega_r[0] / omega_z[0]:.4f} is "
        f"below the threshold c_N = sqrt((mu_max - 1)/2) = {c_n:.4f}. "
        f"Stiffen the radial confinement (larger v_rf or smaller "
        f"omega_axial) or shorten the chain."
    )


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
        For a single-species chain the lowest axial mode is the
        center-of-mass mode at exactly ``trap.omega_axial`` with
        $b_{i} = 1/\sqrt{N}$; neither holds for mixed species, where
        the eigenvectors are mass-weighted. ``"radial_y"`` is exactly
        degenerate with ``"radial_x"`` by construction: the model
        hard-codes the radially symmetric DC split
        $a = -2\omega_z^2/\Omega_\mathrm{rf}^2$ (Wuebbena, PRA 85,
        043412 (2012), Eq. (6) with $\alpha = 1/2$) and ``PaulTrap``
        exposes no asymmetry parameter, so the key exists only for API
        symmetry.

    Raises
    ------
    ValueError
        If any ion falls outside the first Mathieu stability region
        ($q \geq 0.908$) or loses radial confinement
        ($\beta^2 \leq 0$); if a Penning ion violates
        $\omega_c > \sqrt{2}\,\omega_z$; or if the radial dynamical
        matrix is not positive definite, which means the linear chain
        has buckled into a zigzag and no linear normal modes exist.
        In the buckled regime the axial modes are unphysical too - the
        1D equilibrium is a saddle point of the 3D potential - so the
        whole result is rejected rather than the radial part alone.
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
    # (K = 2 * kappa * e * U_dc / z_0^2 for DC confinement; Wineland,
    # J. Res. NIST 103, 259 (1998), Eq. (2)).
    spring_constant = trap.species.mass_kg * trap.omega_axial**2
    omega_z = np.sqrt(spring_constant / masses)

    D_axial = _dynamical_matrix(pos, omega_z, masses, axial=True)
    axial = _diagonalize_to_modes(D_axial, "axial")

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
        # First Mathieu stability region: q < 0.908 at a ~ 0, the same
        # bound PaulTrap.is_stable() enforces. beta^2 > 0 alone is the
        # lowest-order pseudopotential condition and stays satisfied well
        # outside the region, so a light ion could otherwise be handed
        # back a secular frequency while not being trapped at all.
        outside_region = np.where(q >= 0.908)[0]
        if len(outside_region) > 0:
            raise ValueError(
                f"Ions at indices {outside_region.tolist()} are outside "
                f"the first Mathieu stability region "
                f"(q = {q[outside_region].tolist()!r}, must be < 0.908): "
                f"their radial motion is unbounded. Heavier species, "
                f"lower v_rf or higher omega_rf are required."
            )
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

        D_radial = _dynamical_matrix(pos, omega_r, masses, axial=False)
        radial = _diagonalize_to_modes(
            D_radial, "radial", _zigzag_hint(axial, omega_z, omega_r)
        )
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

r"""Frequency-shifts matrix for the elliptical Penning trap.

Implements the Verdu (2011) Appendix B formulas for energy-dependent
frequency shifts from electrostatic anharmonicities in a Penning
trap with broken radial symmetry.

The total frequency-shifts matrix relates energy changes to
frequency changes:

$$
\begin{pmatrix} \Delta\nu_+ \\ \Delta\nu_z \\ \Delta\nu_- \end{pmatrix}
= M \cdot
\begin{pmatrix} \Delta E_+ \\ \Delta E_z \\ \Delta E_- \end{pmatrix}
$$

where $M$ is the sum of nine sub-matrices, one per anharmonic
coefficient $C_{ijk}$ up to fourth order.

References
----------
Verdu, J. New J. Phys. 13, 113029 (2011), Appendix B.
Kretzschmar, M. Int. J. Mass Spectrom. 275, 21 (2008).
"""

from dataclasses import dataclass

import numpy as np

from tiqs.constants import ELECTRON_CHARGE


@dataclass(frozen=True)
class OrbitParams:
    r"""Orbit shape parameters for the elliptical Penning trap.

    Attributes
    ----------
    xi_p, xi_m : float
        x-amplitude coefficients for the modified cyclotron and
        magnetron orbits.
    eta_p, eta_m : float
        y-amplitude coefficients for the modified cyclotron and
        magnetron orbits.
    gamma_p : float
        $1 - \omega_z^2 / (2\,\omega_+^2)$, approximately 1.
    """

    xi_p: float
    xi_m: float
    eta_p: float
    eta_m: float
    gamma_p: float


def orbit_params(
    omega_c: float,
    omega_z: float,
    omega_p: float,
    epsilon: float,
) -> OrbitParams:
    r"""Compute orbit shape parameters from Kretzschmar's theory.

    Parameters
    ----------
    omega_c : float
        Free cyclotron angular frequency.
    omega_z : float
        Axial angular frequency.
    omega_p : float
        Modified cyclotron angular frequency.
    epsilon : float
        Ellipticity parameter.

    Returns
    -------
    OrbitParams
    """
    omega_1_sq = omega_c**2 - 2 * omega_z**2
    omega_1 = np.sqrt(omega_1_sq)
    D = np.sqrt(omega_c**2 * omega_1_sq + epsilon**2 * omega_z**4)
    norm = 2 * (omega_p / omega_1) * D

    xi_p = np.sqrt((omega_c**2 + epsilon * omega_z**2 + D) / norm)
    xi_m = np.sqrt(abs(omega_c**2 + epsilon * omega_z**2 - D) / norm)
    eta_p = np.sqrt((omega_c**2 - epsilon * omega_z**2 + D) / norm)
    eta_m = np.sqrt(abs(omega_c**2 - epsilon * omega_z**2 - D) / norm)
    gamma_p = 1 - omega_z**2 / (2 * omega_p**2)

    return OrbitParams(
        xi_p=xi_p,
        xi_m=xi_m,
        eta_p=eta_p,
        eta_m=eta_m,
        gamma_p=gamma_p,
    )


@dataclass(frozen=True)
class AnharmonicCoeffs:
    r"""Electrostatic anharmonic coefficients $C_{ijk}$.

    All coefficients have units consistent with the potential
    expansion $\phi = \sum C_{ijk}\,x^i\,y^j\,z^k$ where
    $C_{ijk}$ has units of $\mathrm{V/m}^{i+j+k}$.

    Not all coefficients are independent; the Laplace equation
    constrains them (Verdu Eq. 4-5).

    Attributes
    ----------
    c002 : float
        Axial quadrupole (defines omega_z).
    c004 : float
        Axial octupole.
    c220, c202, c022, c400, c040 : float
        Even anharmonicities (first-order perturbation theory).
    c012, c210, c030 : float
        Odd anharmonicities (second-order perturbation theory).
    """

    c002: float
    c004: float = 0.0
    c220: float = 0.0
    c202: float = 0.0
    c022: float = 0.0
    c400: float = 0.0
    c040: float = 0.0
    c012: float = 0.0
    c210: float = 0.0
    c030: float = 0.0


def frequency_shifts_matrix(
    nu_p: float,
    nu_z: float,
    nu_m: float,
    orb: OrbitParams,
    coeffs: AnharmonicCoeffs,
    mass: float,
) -> np.ndarray:
    r"""Compute the total 3x3 frequency-shifts matrix.

    Returns M such that
    $(\Delta\nu_+, \Delta\nu_z, \Delta\nu_-)^T
    = M \cdot (\Delta E_+, \Delta E_z, \Delta E_-)^T$.

    All frequencies are in Hz (not angular). Energies are in
    joules. The matrix elements have units of Hz/J.

    Parameters
    ----------
    nu_p, nu_z, nu_m : float
        Modified cyclotron, axial, magnetron frequencies in Hz.
    orb : OrbitParams
        Orbit shape parameters.
    coeffs : AnharmonicCoeffs
        Electrostatic anharmonic coefficients.
    mass : float
        Particle mass in kg.

    Returns
    -------
    np.ndarray
        3x3 frequency-shifts matrix in Hz/J.

    References
    ----------
    Verdu, J. New J. Phys. 13, 113029 (2011), Eqs. B.1-B.15.
    """
    q = ELECTRON_CHARGE
    xp = orb.xi_p
    xm = orb.xi_m
    ep = orb.eta_p
    em = orb.eta_m
    gp = orb.gamma_p

    # Shorthand for repeated denominators
    vm2 = nu_m**2
    vp2 = nu_p**2
    vz2 = nu_z**2
    vm2_half = vm2 - vz2 / 2  # nu_m^2 - nu_z^2/2

    pi4 = np.pi**4
    pi6 = np.pi**6
    m = mass

    M = np.zeros((3, 3))

    # B.1: M^004
    if coeffs.c004 != 0:
        pf = q * coeffs.c004 / (16 * pi4 * m**2 * nu_z**3)
        M[1, 1] += pf * 3

    # B.2: M^220
    if coeffs.c220 != 0:
        pf = q * coeffs.c220 / (16 * pi4 * m**2 * nu_p)
        M[0, 0] += pf * ep**2 * xp**2 / (gp**2 * vp2)
        cross = xm**2 * ep**2 + em**2 * xp**2
        M[0, 2] += pf * cross / (gp * vm2_half)
        M[2, 0] += pf * nu_m * cross / (gp * nu_p * vm2_half)
        M[2, 2] += pf * nu_m * nu_p * em**2 * xm**2 / vm2_half**2

    # B.3: M^202
    if coeffs.c202 != 0:
        pf = q * coeffs.c202 / (16 * pi4 * m**2 * nu_z)
        M[0, 1] += pf * xp**2 / (gp * nu_p * nu_z)
        M[1, 0] += pf * xp**2 / (gp * vp2)
        M[1, 2] += pf * xm**2 / vm2_half
        M[2, 1] += pf * nu_m * xm**2 / (nu_z * vm2_half)

    # B.4: M^022
    if coeffs.c022 != 0:
        pf = q * coeffs.c022 / (16 * pi4 * m**2 * nu_z)
        M[0, 1] += pf * ep**2 / (gp * nu_p * nu_z)
        M[1, 0] += pf * ep**2 / (gp * vp2)
        M[1, 2] += pf * em**2 / vm2_half
        M[2, 1] += pf * nu_m * em**2 / (nu_z * vm2_half)

    # B.5: M^400
    if coeffs.c400 != 0:
        pf = q * coeffs.c400 / (16 * pi4 * m**2 * nu_p)
        M[0, 0] += pf * 3 * xp**4 / (gp**2 * vp2)
        M[0, 2] += pf * 6 * xm**2 * xp**2 / (gp * vm2_half)
        M[2, 0] += pf * 6 * xm**2 * xp**2 * nu_m / (gp * nu_p * vm2_half)
        M[2, 2] += pf * 3 * xm**4 * nu_m * nu_p / vm2_half**2

    # B.6: M^040
    if coeffs.c040 != 0:
        pf = q * coeffs.c040 / (16 * pi4 * m**2 * nu_p)
        M[0, 0] += pf * 3 * ep**4 / (gp**2 * vp2)
        M[0, 2] += pf * 6 * em**2 * ep**2 / (gp * vm2_half)
        M[2, 0] += pf * 6 * em**2 * ep**2 * nu_m / (gp * nu_p * vm2_half)
        M[2, 2] += pf * 3 * em**4 * nu_m * nu_p / vm2_half**2

    # B.7: M^012 (second-order, depends on C_012^2)
    if coeffs.c012 != 0:
        pf = q**2 * coeffs.c012**2 / (32 * pi6 * m**3 * nu_z**5)
        M[0, 1] += pf * ep**2 / (gp * nu_p * (vp2 - 4 * vz2))
        M[1, 0] += pf * ep**2 * nu_z / (gp * vp2 * (vp2 - 4 * vz2))
        M[1, 1] += pf * (
            -(em**2 * (3 * vm2 - 8 * vz2))
            / (4 * nu_z * (vm2 - 4 * vz2) * vm2_half)
            - (ep**2 * (3 * vp2 - 8 * vz2))
            / (4 * gp * vp2 * nu_z * (vp2 - 4 * vz2))
        )
        M[1, 2] += pf * em**2 * nu_z / ((vm2 - 4 * vz2) * vm2_half)
        M[2, 1] += pf * em**2 * nu_m / ((vm2 - 4 * vz2) * vm2_half)

    # B.8-B.12: M^210 (second-order, depends on C_210^2)
    if coeffs.c210 != 0:
        pf = q**2 * coeffs.c210**2 / (32 * pi6 * m**3 * gp**2 * nu_p**3)
        # a_{1,1} (B.9)
        a11_numer = xp**2 * (
            3 * ep**2 * xp**2 * (vm2 - 4 * vp2) * (2 * vm2 - vz2)
            + 2
            * gp
            * vp2
            * (
                4 * vm2 * xm**2 * ep**2
                - 8 * em * nu_m * xm * ep * nu_p * xp
                + em**2 * xp**2 * (3 * vm2 - 8 * vp2)
            )
        )
        a11_denom = 4 * gp * vp2 * (vm2 - 4 * vp2) * (2 * vm2 - vz2)
        a11 = -a11_numer / a11_denom

        # a_{3,1} (B.10)
        term1 = (
            xp**2
            * (
                -(xm**2) * ep**2 * (vm2 - 12 * vp2)
                - 4 * em * ep * xm * xp * nu_m * nu_p
                + 2 * em**2 * xp**2 * vp2
            )
            / (nu_p * (vm2 - 4 * vp2))
        )
        term2 = (
            xm**2
            * gp
            * nu_p
            * (
                4 * em * xm * ep * xp * nu_m * nu_p
                + em**2 * xp**2 * (vp2 - 12 * vm2)
                - 2 * vm2 * xm**2 * ep**2
            )
            / ((4 * vm2 - vp2) * vm2_half)
        )
        a31 = nu_m / (2 * vm2_half) * (term1 + term2)

        # a_{1,3} (B.11)
        term1_13 = (
            xp**2
            * (
                xm**2 * (-(ep**2)) * (vm2 - 12 * vp2)
                - 4 * em * xm * ep * xp * nu_m * nu_p
                + 2 * em**2 * vp2 * xp**2
            )
            / (nu_p * (vm2 - 4 * vp2))
        )
        term2_13 = (
            2
            * xm**2
            * gp
            * nu_p
            * (
                4 * em * xm * ep * xp * nu_m * nu_p
                + em**2 * xp**2 * (vp2 - 12 * vm2)
                - 2 * vm2 * xm**2 * ep**2
            )
            / ((4 * vm2 - vp2) * (2 * vm2 - vz2))
        )
        a13 = 1 / (2 * vm2_half) * (term1_13 + term2_13)

        # a_{3,3} (B.12)
        a33_inner = nu_m * xm**2 * gp * nu_p * (2 * vm2 - vz2)
        a33_numer = a33_inner * (
            2
            * em**2
            * vp2
            * (2 * xp**2 * (vz2 - 2 * vm2) + 3 * xm**2 * gp * (4 * vm2 - vp2))
            + 8 * em * nu_m * xm * ep * nu_p * xp * (2 * vm2 - vz2)
            + xm**2 * ep**2 * (8 * vm2 - 3 * vp2) * (2 * vm2 - vz2)
        )
        a33 = a33_numer / (vp2 - 4 * vm2)

        M[0, 0] += pf * a11
        M[0, 2] += pf * a13
        M[2, 0] += pf * a31
        M[2, 2] += pf * a33

    # B.13-B.15: M^030 (second-order, depends on C_030^2)
    if coeffs.c030 != 0:
        pf = q**2 * coeffs.c030**2 / (32 * pi6 * m**3 * nu_p**3)
        # b_{1,1} (B.14)
        b11 = (
            -(3 * ep**4)
            / (4 * gp**3 * vp2)
            * (
                6
                * em**2
                * gp
                * vp2
                * (3 * vm2 - 8 * vp2)
                / ((vm2 - 4 * vp2) * (2 * vm2 - vz2))
                + 5 * ep**2
            )
        )

        # b_{1,3} (B.14)
        poly_p = -25 * vm2 * vp2 + 4 * vm2**2 + 6 * vp2**2
        poly_m = -25 * vm2 * vp2 + 6 * vm2**2 + 4 * vp2**2
        denom_b = (
            gp**2
            * (-17 * vm2 * vp2 + 4 * vm2**2 + 4 * vp2**2)
            * (vz2 - 2 * vm2) ** 2
        )
        b13_numer = (
            9
            * em**2
            * ep**2
            * (
                ep**2 * poly_p * (2 * vm2 - vz2)
                + 2 * em**2 * gp * vp2 * poly_m
            )
        )
        b13 = -b13_numer / denom_b

        # b_{3,1} (B.15)
        b31 = (
            -9
            * em**2
            * nu_m
            * ep**2
            * (
                ep**2 * poly_p * (2 * vm2 - vz2)
                + 2 * em**2 * gp * vp2 * poly_m
            )
            / (
                gp**2
                * nu_p
                * (-17 * vm2 * vp2 + 4 * vm2**2 + 4 * vp2**2)
                * (vz2 - 2 * vm2) ** 2
            )
        )

        # b_{3,3} (B.15)
        b33 = (3 * em**4 * nu_m * nu_p**3 / (vz2 - 2 * vm2) ** 3) * (
            3
            * ep**2
            * (3 * vp2 - 8 * vm2)
            * (2 * vm2 - vz2)
            / (gp * vp2 * (vp2 - 4 * vm2))
            + 10 * em**2
        )

        M[0, 0] += pf * b11
        M[0, 2] += pf * b13
        M[2, 0] += pf * b31
        M[2, 2] += pf * b33

    return M

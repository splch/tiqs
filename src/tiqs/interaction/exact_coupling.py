r"""Exact laser-ion coupling beyond the Lamb-Dicke approximation.

The exact Rabi frequency for the $s$-th order sideband transition
$|g,n\rangle \leftrightarrow |e,n+s\rangle$ is [Leibfried2003]_
Eq. 8:

$$
\Omega_{n,n+s} = \Omega\,e^{-\eta^2/2}\,\eta^{|s|}
  \sqrt{\frac{n_< !}{n_> !}}
  \,\bigl|L_{n_<}^{|s|}(\eta^2)\bigr|
$$

where $n_< = \min(n, n+s)$, $n_> = \max(n, n+s)$, and
$L_n^\alpha(x)$ is the generalized Laguerre polynomial.

.. [Leibfried2003] Leibfried et al., RMP 75, 281 (2003).
"""

import numpy as np
from scipy.special import assoc_laguerre, factorial


def exact_rabi_frequency(
    bare_rabi: float,
    eta: float,
    n: int,
    s: int = 0,
) -> float:
    r"""Exact Rabi frequency for the transition $|n\rangle \to |n+s\rangle$.

    Parameters
    ----------
    bare_rabi : float
        Bare (free-space) Rabi frequency $\Omega$ in rad/s.
    eta : float
        Lamb-Dicke parameter.
    n : int
        Initial motional Fock state.
    s : int
        Sideband order: 0 = carrier, +1 = blue (adds phonon),
        -1 = red (removes phonon), +2 = second blue, etc.

    Returns
    -------
    float
        Effective Rabi frequency $\Omega_{n,n+s}$ in rad/s.
    """
    n_final = n + s
    if n_final < 0:
        return 0.0
    n_lo = min(n, n_final)
    n_hi = max(n, n_final)
    abs_s = abs(s)

    laguerre = float(assoc_laguerre(eta**2, n_lo, abs_s))
    prefactor = np.exp(-(eta**2) / 2) * eta**abs_s
    fock_ratio = np.sqrt(
        factorial(n_lo, exact=True) / factorial(n_hi, exact=True)
    )

    return bare_rabi * prefactor * fock_ratio * abs(laguerre)


def debye_waller_factor(eta: float, n: int) -> float:
    r"""Debye-Waller factor: ratio of exact carrier Rabi frequency
    to bare Rabi frequency for Fock state $|n\rangle$.

    $$
    \frac{\Omega_n}{\Omega} = e^{-\eta^2/2}\,L_n^0(\eta^2)
    $$

    For small $\eta$: $\approx 1 - \eta^2(2n+1)/2$.

    Parameters
    ----------
    eta : float
    n : int

    Returns
    -------
    float
        Ratio $\Omega_n / \Omega$.
    """
    laguerre = float(assoc_laguerre(eta**2, n, 0))
    return np.exp(-(eta**2) / 2) * laguerre


def _signed_matrix_element(eta: float, n: int, s: int) -> float:
    """Signed coupling matrix element (preserves Laguerre sign)."""
    n_final = n + s
    if n_final < 0:
        return 0.0
    n_lo = min(n, n_final)
    n_hi = max(n, n_final)
    abs_s = abs(s)

    laguerre = float(assoc_laguerre(eta**2, n_lo, abs_s))
    prefactor = np.exp(-(eta**2) / 2) * eta**abs_s
    fock_ratio = np.sqrt(
        factorial(n_lo, exact=True) / factorial(n_hi, exact=True)
    )
    return prefactor * fock_ratio * laguerre  # signed, no abs()


def exact_sideband_operator(
    eta: float,
    n_fock: int,
    s: int = 1,
) -> np.ndarray:
    r"""Exact sideband coupling matrix in the Fock basis.

    Replaces the first-order Lamb-Dicke operator $\eta\,a^\dagger$
    (for blue sideband, s=+1) or $\eta\,a$ (for red, s=-1) with the
    exact n-dependent matrix elements.  The Laguerre polynomial
    sign is **preserved** so that the Hamiltonian operator has
    the correct phase evolution across Fock states.

    Parameters
    ----------
    eta : float
        Lamb-Dicke parameter.
    n_fock : int
        Fock space dimension.
    s : int
        Sideband order (+1 = blue, -1 = red).

    Returns
    -------
    np.ndarray
        Matrix of shape ``(n_fock, n_fock)``.
    """
    M = np.zeros((n_fock, n_fock), dtype=complex)
    for n in range(n_fock):
        n_final = n + s
        if 0 <= n_final < n_fock:
            M[n_final, n] = _signed_matrix_element(eta, n, s)
    return M


def exact_carrier_operator(
    eta: float,
    n_fock: int,
) -> np.ndarray:
    r"""Exact carrier coupling: diagonal Debye-Waller factors.

    $$
    M_{n,n} = e^{-\eta^2/2}\,L_n^0(\eta^2)
    $$

    Parameters
    ----------
    eta : float
    n_fock : int

    Returns
    -------
    np.ndarray
        Diagonal matrix of shape ``(n_fock, n_fock)``.
    """
    M = np.zeros((n_fock, n_fock), dtype=complex)
    for n in range(n_fock):
        M[n, n] = debye_waller_factor(eta, n)
    return M


def thermal_averaged_rabi(
    bare_rabi: float,
    eta: float,
    n_bar: float,
    s: int = 0,
    n_max: int = 50,
) -> float:
    r"""Thermal-averaged Rabi frequency.

    $$
    \bar\Omega_s = \sum_{n=0}^{n_\max} P(n)\,\Omega_{n,n+s}
    $$

    where $P(n)$ is the thermal (Bose-Einstein) distribution.

    Parameters
    ----------
    bare_rabi : float
    eta : float
    n_bar : float
        Mean phonon number of the thermal state.
    s : int
        Sideband order.
    n_max : int
        Truncation of the sum.

    Returns
    -------
    float
        Thermally averaged Rabi frequency.
    """
    if n_bar <= 0:
        return exact_rabi_frequency(bare_rabi, eta, 0, s)
    total = 0.0
    for n in range(n_max):
        p_n = (n_bar / (n_bar + 1)) ** n / (n_bar + 1)
        omega_n = exact_rabi_frequency(bare_rabi, eta, n, s)
        total += p_n * omega_n
    return total

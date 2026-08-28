r"""Motional potentials: harmonic, Duffing (Kerr), and arbitrary.

.. include:: ../../docs/theory/potentials.md
"""

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import qutip

from tiqs.hilbert_space.operators import OperatorFactory

# Relative tolerances for structural tests on the truncated
# Hamiltonian. Both are scaled by max|H_ij| because entries carry
# angular-frequency units (~1e7 rad/s), so an absolute tolerance
# (QuTiP's ``isherm`` uses atol = 1e-12) would flag pure round-off.
_HERM_RTOL = 1e-10
_DIAG_RTOL = 1e-12


class Potential(Protocol):
    """Structural interface for any motional potential.

    Any class exposing ``omega`` and ``single_mode_hamiltonian()``
    satisfies this protocol. ``HarmonicPotential``,
    ``DuffingPotential``, and ``ArbitraryPotential`` conform
    without modification.
    """

    @property
    def omega(self) -> float: ...

    def single_mode_hamiltonian(self, n_fock: int) -> qutip.Qobj: ...


@dataclass(frozen=True)
class HarmonicPotential:
    r"""Harmonic oscillator potential with equally-spaced energy levels.

    $$
    H = \omega\,a^\dagger a
    $$

    This is what TIQS implicitly assumes for all modes when no
    potential is explicitly specified. The energy eigenvalues are
    $E_n = n\,\omega$ (zero-point energy excluded by convention).

    *omega* must be a **positive-energy** mode frequency. The ladder
    built here ascends, so feeding it a Penning magnetron frequency
    (`tiqs.trap.PenningTrap.omega_magnetron`, or the ``"magnetron"``
    `tiqs.chain.normal_modes.ModeGroup`) inverts the physics: the
    magnetron mode carries negative total energy,
    $H_\text{radial} = \omega_+(n_+ + \tfrac12)
    - \omega_-(n_- + \tfrac12)$, so its true ladder descends
    (Brown & Gabrielse, *Rev. Mod. Phys.* **58**, 233 (1986), Sec. II).

    Attributes
    ----------
    omega : float
        Oscillation angular frequency in rad/s, positive-energy mode
        only.
    """

    omega: float

    def single_mode_hamiltonian(self, n_fock: int) -> qutip.Qobj:
        r"""Return $H = \omega\,a^\dagger a$ truncated to ``n_fock`` levels."""
        return self.omega * qutip.num(n_fock)


@dataclass(frozen=True)
class DuffingPotential:
    r"""Duffing (Kerr) oscillator: harmonic with a quartic nonlinearity.

    $$
    H = \omega\,\hat{n}
      + \frac{\alpha}{2}\,\hat{n}\,(\hat{n} - 1)
    $$

    The transition frequency from $|n\rangle$ to $|n+1\rangle$ is
    $\omega + \alpha\,n$, so $\alpha$ is the frequency shift per
    excitation quantum.

    For softening ($\alpha < 0$) the ladder turns over at
    $n = \omega/|\alpha|$ and the spectrum is non-monotonic above it,
    so energy-ascending eigenvalue order stops matching Fock order
    once ``n_fock`` exceeds $\omega/|\alpha| + 2$. Use
    ``transition_frequencies`` (which reads the Fock-ordered
    diagonal) rather than differencing ``energy_levels`` there.

    Attributes
    ----------
    omega : float
        Fundamental oscillation angular frequency in rad/s.
    anharmonicity : float
        Anharmonicity $\alpha$ in rad/s. Negative for softening
        (transmon-like), positive for stiffening. A phenomenological
        parameter: TIQS does not derive it from trap geometry.
    """

    omega: float
    anharmonicity: float

    def single_mode_hamiltonian(self, n_fock: int) -> qutip.Qobj:
        r"""Return the Duffing Hamiltonian truncated to ``n_fock`` levels."""
        n = qutip.num(n_fock)
        return self.omega * n + (self.anharmonicity / 2) * n * (n - 1)


@dataclass(frozen=True)
class ArbitraryPotential:
    r"""Arbitrary potential defined in dimensionless coordinates.

    The user supplies $V(q)$ as a callable of the dimensionless
    position operator $q = a + a^\dagger$, returning the **full**
    potential in angular frequency units (rad/s). The Hamiltonian is:

    $$
    H = \omega\,(\hat{n} + \tfrac{1}{2})
      - \frac{\omega}{4}\,q^2 + V(q)
    $$

    For a quartic oscillator, for example,
    $V(q) = \omega/4\,q^2 + \lambda\,q^4$. The harmonic term
    must be included because $V$ is the full potential.

    Choose ``omega`` to match the curvature of $V$ near its
    minimum for best Fock-basis convergence, and verify with
    ``check_convergence()``.

    .. note::
       The coordinate here is $q = a + a^\dagger$, **not** the
       unit-commutator quadrature
       $x = (a + a^\dagger)/\sqrt{2}$ returned by
       ``OperatorFactory.position``. The two differ by
       $q = \sqrt{2}\,x$, i.e. a factor 2 in any quadratic term, so
       expectation values taken with ``ops.position`` must be
       rescaled before being fed to a $V(q)$ written for this class.

    Attributes
    ----------
    v_func : callable
        ``V(q_op) -> qutip.Qobj`` where ``q_op`` is the
        dimensionless position operator $q = a + a^\dagger$.
        Must return the full potential in rad/s.
    omega : float
        Reference harmonic frequency in rad/s. Defines the Fock
        basis and sets the kinetic energy scale.
    """

    v_func: Callable[[qutip.Qobj], qutip.Qobj]
    omega: float

    def single_mode_hamiltonian(self, n_fock: int) -> qutip.Qobj:
        r"""Return $T + V(q)$ truncated to ``n_fock`` levels.

        Raises
        ------
        ValueError
            If $T + V(q)$ is not Hermitian to within
            ``1e-10`` of $\max_{ij}|H_{ij}|$. A non-Hermitian
            ``v_func`` would otherwise yield complex eigenvalues
            whose imaginary parts ``energy_levels`` discards.
        """
        a = qutip.destroy(n_fock)
        n = qutip.num(n_fock)
        q_op = a + a.dag()
        # T = H_ref - V_ref = omega*(n + 1/2) - omega/4 * q^2
        T = self.omega * (n + 0.5) - self.omega / 4 * q_op * q_op
        H = T + self.v_func(q_op)
        dense = H.full()
        scale = float(np.max(np.abs(dense)))
        asymmetry = float(np.max(np.abs(dense - dense.conj().T)))
        if asymmetry > _HERM_RTOL * scale:
            raise ValueError(
                f"v_func gives a non-Hermitian Hamiltonian at "
                f"n_fock={n_fock}: max|H - H^dag| = {asymmetry:.3e} "
                f"exceeds {_HERM_RTOL:.0e} * max|H_ij| = "
                f"{_HERM_RTOL * scale:.3e}. V(q) must be a real "
                f"function of the Hermitian operator q."
            )
        return H


def energy_levels(potential: Potential, n_fock: int) -> np.ndarray:
    r"""Compute energy eigenvalues of a potential.

    Diagonalizes the single-mode Hamiltonian and returns the
    eigenvalues in ascending energy order, in rad/s (units of
    $\hbar = 1$).

    Ascending energy order is **not** Fock order for a spectrum that
    is not monotonic in $n$ (a softening ``DuffingPotential``, for
    instance). Differencing this array therefore does not in general
    give the $|n\rangle \to |n+1\rangle$ ladder; use
    ``transition_frequencies`` for that.

    Parameters
    ----------
    potential : Potential
        The motional potential.
    n_fock : int
        Fock space truncation dimension.

    Returns
    -------
    np.ndarray
        Energy eigenvalues in ascending order, shape
        $(n_\mathrm{fock},)$.
    """
    H = potential.single_mode_hamiltonian(n_fock)
    return np.sort(H.eigenenergies().real)


def transition_frequencies(potential: Potential, n_fock: int) -> np.ndarray:
    r"""Compute transition frequencies $\omega_{n \to n+1}$.

    Returns an array of length ``n_fock - 1``.

    When the truncated Hamiltonian is diagonal in the Fock basis
    (``HarmonicPotential``, ``DuffingPotential``, and any
    ``ArbitraryPotential`` whose $V$ cancels the reference
    curvature exactly) the Fock states *are* the eigenstates, and
    element $n$ is read off the Fock-ordered diagonal so it is
    exactly the $|n\rangle \to |n+1\rangle$ frequency - including
    where that frequency is negative, as above the turnover of a
    softening Duffing ladder.

    Otherwise the Fock states are not eigenstates, no
    $|n\rangle \to |n+1\rangle$ ladder exists, and element $n$ is the
    gap between the $n$-th and $(n+1)$-th eigenvalues in ascending
    energy order. A ``UserWarning`` is issued in that case.

    Parameters
    ----------
    potential : Potential
        The motional potential.
    n_fock : int
        Fock space truncation dimension.

    Returns
    -------
    np.ndarray
        Transition frequencies, shape $(n_\mathrm{fock} - 1,)$.
    """
    H = potential.single_mode_hamiltonian(n_fock)
    dense = H.full()
    diagonal = np.diag(dense)
    scale = float(np.max(np.abs(dense)))
    off_diagonal = float(np.max(np.abs(dense - np.diag(diagonal))))
    if off_diagonal <= _DIAG_RTOL * scale:
        return np.diff(diagonal.real)
    warnings.warn(
        f"{type(potential).__name__} is not diagonal in the Fock "
        f"basis at n_fock={n_fock} (max off-diagonal / max|H_ij| = "
        f"{off_diagonal / scale:.2e}), so Fock states are not "
        f"eigenstates. Returning gaps between adjacent eigenvalues "
        f"in ascending energy order, not |n> -> |n+1> transitions.",
        stacklevel=2,
    )
    return np.diff(np.sort(H.eigenenergies().real))


def check_convergence(
    potential: Potential,
    n_fock: int,
    n_levels: int = 5,
) -> bool:
    r"""Check that the lowest energy levels are converged.

    Compares the lowest ``n_levels`` eigenvalues at ``n_fock`` and at
    ``2 * n_fock``. Returns ``True`` when the largest level shift is
    at most $10^{-6}$ of the largest checked level magnitude,
    $\max_i |E_i|$. Warns with that same ratio if not.

    The doubling step is deliberate: a fixed additive step cannot
    resolve tail contributions that grow slowly with the truncation.

    .. warning::
       A potential that is unbounded below (say $V$ with a negative
       quartic coefficient) has no true ground state, so **no**
       truncation is converged and this check cannot certify it.
       Such a potential can report ``True`` over a wide band of
       ``n_fock`` while the basis is simply too small to reach the
       runaway region. Convergence here is necessary, not sufficient;
       confirm that $V(q) \to +\infty$ as $|q| \to \infty$.

    Parameters
    ----------
    potential : Potential
        The motional potential.
    n_fock : int
        Fock space truncation dimension to test.
    n_levels : int
        Number of lowest levels to check (default 5).

    Returns
    -------
    bool
        Whether the levels are converged.
    """
    if n_levels > n_fock:
        raise ValueError(f"n_levels ({n_levels}) must be <= n_fock ({n_fock})")
    E1 = energy_levels(potential, n_fock)[:n_levels]
    E2 = energy_levels(potential, 2 * n_fock)[:n_levels]
    scale = float(np.max(np.abs(E2)))
    shift = float(np.max(np.abs(E1 - E2)))
    max_diff = shift / scale if scale > 0.0 else 0.0
    converged = max_diff <= 1e-6
    if not converged:
        warnings.warn(
            f"Lowest {n_levels} levels not converged at "
            f"n_fock={n_fock}. Max relative difference: "
            f"{max_diff:.2e}. Increase n_fock.",
            stacklevel=2,
        )
    return converged


def mode_hamiltonian(
    potential: Potential,
    ops: OperatorFactory,
    mode: int,
) -> qutip.Qobj:
    """Lift a single-mode Hamiltonian to the full tensor-product space.

    Constructs the single-mode Hamiltonian from the potential, then
    embeds it in the composite Hilbert space at the given mode index
    using the operator factory.

    Any frequency the potential carries must belong to a
    positive-energy mode -- see `HarmonicPotential` for the Penning
    magnetron caveat.

    Parameters
    ----------
    potential : Potential
        The motional potential for this mode.
    ops : OperatorFactory
        Operator factory for the composite Hilbert space.
    mode : int
        Index of the target motional mode.

    Returns
    -------
    qutip.Qobj
        Hamiltonian acting on the full composite Hilbert space.
    """
    n_fock = ops.hs.fock_dim(mode)
    H_single = potential.single_mode_hamiltonian(n_fock)
    return ops.embed_mode_operator(H_single, mode)

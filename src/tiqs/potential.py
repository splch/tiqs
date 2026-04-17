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

    Attributes
    ----------
    omega : float
        Oscillation angular frequency in rad/s.
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

    Attributes
    ----------
    omega : float
        Fundamental oscillation angular frequency in rad/s.
    anharmonicity : float
        Anharmonicity $\alpha$ in rad/s. Negative for softening
        (transmon-like), positive for stiffening.
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
        r"""Return $T + V(q)$ truncated to ``n_fock`` levels."""
        a = qutip.destroy(n_fock)
        n = qutip.num(n_fock)
        q_op = a + a.dag()
        # T = H_ref - V_ref = omega*(n + 1/2) - omega/4 * q^2
        T = self.omega * (n + 0.5) - self.omega / 4 * q_op * q_op
        return T + self.v_func(q_op)


def energy_levels(potential: Potential, n_fock: int) -> np.ndarray:
    r"""Compute energy eigenvalues of a potential.

    Diagonalizes the single-mode Hamiltonian and returns sorted
    eigenvalues in rad/s (units of $\hbar = 1$).

    Parameters
    ----------
    potential : Potential
        The motional potential.
    n_fock : int
        Fock space truncation dimension.

    Returns
    -------
    np.ndarray
        Sorted energy eigenvalues, shape $(n_\mathrm{fock},)$.
    """
    H = potential.single_mode_hamiltonian(n_fock)
    return np.sort(H.eigenenergies().real)


def transition_frequencies(potential: Potential, n_fock: int) -> np.ndarray:
    r"""Compute transition frequencies $\omega_{n \to n+1}$.

    Returns an array of length ``n_fock - 1`` where element $n$ is
    the frequency of the $|n\rangle \to |n+1\rangle$ transition.

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
    E = energy_levels(potential, n_fock)
    return np.diff(E)


def check_convergence(
    potential: Potential,
    n_fock: int,
    n_levels: int = 5,
) -> bool:
    r"""Check that the lowest energy levels are converged.

    Compares eigenvalues at ``n_fock`` vs ``n_fock + 5``. Returns
    ``True`` if all ``n_levels`` eigenvalues agree to within
    $10^{-6}$ relative tolerance. Warns if not converged.

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
    E2 = energy_levels(potential, n_fock + 5)[:n_levels]
    converged = np.allclose(E1, E2, rtol=1e-6)
    if not converged:
        denom = np.maximum(np.abs(E2), 1e-30)
        max_diff = float(np.max(np.abs(E1 - E2) / denom))
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

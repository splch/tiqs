r"""Motional potentials: harmonic, Duffing (Kerr), and arbitrary.

The ``Potential`` protocol defines the shared interface for any
motional potential. Concrete implementations provide the
single-mode Hamiltonian; utility functions compute energy levels
and transition frequencies by diagonalizing it.

.. include:: ../../docs/theory/potentials.md
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np
import qutip

if TYPE_CHECKING:
    from tiqs.hilbert_space.operators import OperatorFactory

from tiqs.constants import HBAR


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
        return self.omega * qutip.num(n_fock)


@dataclass(frozen=True)
class DuffingPotential:
    r"""Duffing (Kerr) oscillator: harmonic with a quartic nonlinearity.

    $$
    H = \omega\,\hat{n}
      + \frac{\alpha}{2}\,\hat{n}\,(\hat{n} - 1)
    $$

    where $\alpha$ is the anharmonicity. For transmon-like systems,
    $\alpha < 0$ (subharmonic: higher transitions have lower
    frequencies). For stiffening nonlinearities, $\alpha > 0$.

    The transition frequency from $|n\rangle$ to $|n+1\rangle$ is:

    $$
    \omega_{n \to n+1} = \omega + \alpha\,n
    $$

    so $\alpha$ equals the frequency shift per excitation quantum.
    The $|0\rangle \to |1\rangle$ transition is at $\omega$, the
    $|1\rangle \to |2\rangle$ transition is at $\omega + \alpha$,
    and so on.

    Attributes
    ----------
    omega : float
        Fundamental oscillation angular frequency in rad/s.
    anharmonicity : float
        Anharmonicity $\alpha$ in rad/s. Negative for transmon-like
        (subharmonic) systems, positive for stiffening.
    """

    omega: float
    anharmonicity: float

    def single_mode_hamiltonian(self, n_fock: int) -> qutip.Qobj:
        n = qutip.num(n_fock)
        eye = qutip.qeye(n_fock)
        return self.omega * n + (self.anharmonicity / 2) * n * (n - eye)


@dataclass(frozen=True)
class ArbitraryPotential:
    r"""Arbitrary potential energy function $V(x)$.

    Constructs the Hamiltonian in the Fock basis by expressing
    the potential in terms of the position operator
    $\hat{x} = x_\mathrm{zpf}\,(a + a^\dagger)$
    and adding the kinetic energy:

    $$
    H = \frac{\hat{p}^2}{2m} + V(\hat{x})
    $$

    The kinetic energy is computed as $T = H_\mathrm{ref} - V_\mathrm{ref}$
    where $H_\mathrm{ref} = \omega\,(\hat{n} + \tfrac{1}{2})$ is the
    reference harmonic oscillator and
    $V_\mathrm{ref} = \tfrac{1}{2} m \omega^2 \hat{x}^2$ is its potential.

    The user must provide the **full** potential $V(x)$ including any
    harmonic part. For example, a quartic anharmonic oscillator:
    $V(x) = \tfrac{1}{2} m \omega^2 x^2 + \lambda x^4$.

    Convergence of the Fock-basis representation depends on
    ``n_fock``. Use ``check_convergence()`` to verify the
    truncation. Choose ``omega`` to match the curvature of $V(x)$
    near its minimum for best convergence.

    Simulations with ``ArbitraryPotential`` should use the
    Schrodinger picture rather than the interaction picture,
    because the anharmonic correction generally does not commute
    with the free harmonic Hamiltonian.

    Attributes
    ----------
    v_func : callable
        ``V(x_op) -> qutip.Qobj`` where ``x_op`` is the position
        operator as a QuTiP Qobj. Returns the **full** potential
        energy operator in the Fock basis.
    omega : float
        Reference harmonic frequency in rad/s. Defines the Fock
        basis length scale.
    mass_kg : float
        Particle mass in kg.
    """

    v_func: callable
    omega: float
    mass_kg: float

    def single_mode_hamiltonian(self, n_fock: int) -> qutip.Qobj:
        a = qutip.destroy(n_fock)
        n = qutip.num(n_fock)
        eye = qutip.qeye(n_fock)
        x_zpf = np.sqrt(HBAR / (2 * self.mass_kg * self.omega))
        x_op = x_zpf * (a + a.dag())
        H_ref = self.omega * (n + 0.5 * eye)
        V_ref = 0.5 * self.mass_kg * self.omega**2 * x_op * x_op
        return H_ref - V_ref + self.v_func(x_op)


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
        Sorted energy eigenvalues, shape ``(n_fock,)``.
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
        Transition frequencies, shape ``(n_fock - 1,)``.
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
    E1 = energy_levels(potential, n_fock)[:n_levels]
    E2 = energy_levels(potential, n_fock + 5)[:n_levels]
    converged = np.allclose(E1, E2, rtol=1e-6)
    if not converged:
        max_diff = float(np.max(np.abs(E1 - E2) / np.abs(E2)))
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
    r"""Lift a single-mode Hamiltonian to the full tensor-product space.

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
    return ops._full_operator(H_single, ops._mode_index(mode))

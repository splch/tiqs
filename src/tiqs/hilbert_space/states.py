"""Initial state construction for trapped-ion simulations."""

import math
import warnings

import qutip

from tiqs.hilbert_space.builder import HilbertSpace

THERMAL_MEAN_TOL = 0.02
r"""Relative accuracy demanded of a truncated thermal state's mean.

`StateFactory.thermal_state` warns when Fock truncation pulls the
realized $\langle n \rangle$ further than this below the requested
``n_bar``.
"""


def _suggested_fock_dim(n_bar: float) -> int:
    r"""Fock cutoff that keeps a thermal state's mean within tolerance.

    The truncated, renormalized Bose-Einstein distribution has
    $\langle n \rangle$ within `THERMAL_MEAN_TOL` of $\bar{n}$ once the
    cutoff reaches $6\bar{n} + 10$ (verified against the exact
    geometric sum for $0 \le \bar{n} \le 5000$, where the residual
    error saturates at 1.5%).

    Parameters
    ----------
    n_bar : float
        Requested mean phonon number.

    Returns
    -------
    int
        Recommended ``n_fock`` for this mode.
    """
    return math.ceil(6.0 * n_bar) + 10


class StateFactory:
    """Constructs initial quantum states in the composite Hilbert space.

    Parameters
    ----------
    hilbert_space : HilbertSpace
        The composite Hilbert space specification.
    """

    def __init__(self, hilbert_space: HilbertSpace):
        """Store the Hilbert space specification used to build states."""
        self.hs = hilbert_space

    def ground_state(self) -> qutip.Qobj:
        """All qubits in |0> (down), all modes in vacuum |n=0>.

        Returns
        -------
        qutip.Qobj
            Tensor-product ket in the composite Hilbert space.
        """
        parts = [qutip.basis(2, 0) for _ in range(self.hs.n_ions)]
        parts += [
            qutip.basis(self.hs.fock_dim(m), 0) for m in range(self.hs.n_modes)
        ]
        return qutip.tensor(parts)

    def product_state(
        self,
        qubit_states: list[int],
        fock_states: list[int],
    ) -> qutip.Qobj:
        """Construct an arbitrary product state as a ket.

        Parameters
        ----------
        qubit_states : list[int]
            Qubit state for each ion, 0 (down) or 1 (up).
        fock_states : list[int]
            Phonon occupation number for each motional mode.

        Returns
        -------
        qutip.Qobj
            Tensor-product ket in the composite Hilbert space.
        """
        if len(qubit_states) != self.hs.n_ions:
            raise ValueError(
                f"Expected {self.hs.n_ions} qubit states,"
                f" got {len(qubit_states)}"
            )
        if len(fock_states) != self.hs.n_modes:
            raise ValueError(
                f"Expected {self.hs.n_modes} fock states,"
                f" got {len(fock_states)}"
            )
        parts = [qutip.basis(2, q) for q in qubit_states]
        parts += [
            qutip.basis(self.hs.fock_dim(m), n)
            for m, n in enumerate(fock_states)
        ]
        return qutip.tensor(parts)

    def thermal_state(
        self,
        n_bar: list[float],
        qubit_states: list[int] | None = None,
    ) -> qutip.Qobj:
        r"""Qubits in |0>, motional modes in thermal states.

        Returns a density matrix. Each mode carries the Bose-Einstein
        distribution $p_n = \bar{n}^n / (1 + \bar{n})^{n+1}$ truncated
        to ``fock_dim(m)`` levels and renormalized, so the realized
        mean occupation always falls *below* the requested ``n_bar``
        (e.g. ``n_bar = 5`` at ``n_fock = 10`` realizes
        $\langle n \rangle = 3.07$). A ``UserWarning`` names the mode,
        the request, the realized value and a sufficient cutoff
        whenever that shortfall exceeds `THERMAL_MEAN_TOL`.

        Parameters
        ----------
        n_bar : list[float]
            Mean phonon number for each mode; must be >= 0.
        qubit_states : list[int] or None
            Qubit state for each ion (default all 0).

        Returns
        -------
        qutip.Qobj
            Density matrix in the composite Hilbert space.

        Raises
        ------
        ValueError
            If the list lengths do not match the Hilbert space, or if
            any ``n_bar`` is negative (``qutip.thermal_dm`` returns a
            NaN density matrix for negative occupations).

        Warns
        -----
        UserWarning
            If Fock truncation pulls a realized mean occupation more
            than `THERMAL_MEAN_TOL` below the request.
        """
        if qubit_states is None:
            qubit_states = [0] * self.hs.n_ions
        if len(n_bar) != self.hs.n_modes:
            raise ValueError(
                f"Expected {self.hs.n_modes} n_bar values, got {len(n_bar)}"
            )
        if len(qubit_states) != self.hs.n_ions:
            raise ValueError(
                f"Expected {self.hs.n_ions} qubit states,"
                f" got {len(qubit_states)}"
            )

        parts = [qutip.ket2dm(qutip.basis(2, q)) for q in qubit_states]
        parts += [self._thermal_mode(m, nb) for m, nb in enumerate(n_bar)]
        return qutip.tensor(parts)

    def _thermal_mode(self, mode: int, n_bar: float) -> qutip.Qobj:
        """Single-mode thermal density matrix, checked for truncation.

        Parameters
        ----------
        mode : int
            Index of the motional mode.
        n_bar : float
            Requested mean phonon number for this mode.

        Returns
        -------
        qutip.Qobj
            Thermal density matrix of dimension ``fock_dim(mode)``.
        """
        if n_bar < 0:
            raise ValueError(
                f"n_bar[{mode}] must be >= 0, got {n_bar}"
                " (a negative occupation gives a NaN density matrix)"
            )
        dim = self.hs.fock_dim(mode)
        rho = qutip.thermal_dm(dim, n_bar)
        realized = qutip.expect(qutip.num(dim), rho)
        if n_bar > 0 and abs(realized - n_bar) > THERMAL_MEAN_TOL * n_bar:
            warnings.warn(
                f"Fock cutoff {dim} truncates the thermal state of mode"
                f" {mode}: requested n_bar={n_bar:g}, realized"
                f" <n>={realized:.4g}. Use n_fock >="
                f" {_suggested_fock_dim(n_bar)} for this mode.",
                UserWarning,
                stacklevel=3,
            )
        return rho

"""Composite Hilbert space construction for ion qubits + motional modes."""

import math
from dataclasses import dataclass

MIN_FOCK_DIM = 2
"""Smallest usable Fock cutoff.

A one-level mode has no ladder: ``qutip.destroy(1)`` is not
constructible, while ``qutip.num(1)`` and ``qutip.thermal_dm(1, n)``
succeed and silently freeze the motion. Zero or negative cutoffs
propagate into a zero or negative ``total_dim``.
"""


def _checked_fock_dim(value: int, label: str) -> int:
    """Return ``value`` if it is a usable Fock cutoff, else raise.

    Parameters
    ----------
    value : int
        Candidate Fock-space truncation.
    label : str
        Name of the offending input, used in the error message.

    Returns
    -------
    int
        The validated cutoff.

    Raises
    ------
    TypeError
        If ``value`` is not an ``int``. ``bool`` is rejected even
        though it subclasses ``int``, since ``True`` would silently
        mean a one-level mode.
    ValueError
        If ``value`` is below ``MIN_FOCK_DIM``.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            f"{label} must be an int, got {type(value).__name__} {value!r}"
        )
    if value < MIN_FOCK_DIM:
        raise ValueError(
            f"{label} must be >= {MIN_FOCK_DIM}, got {value}"
            " (a mode needs at least |0> and |1> to have a ladder)"
        )
    return value


@dataclass
class HilbertSpace:
    """Defines the tensor-product structure of the composite Hilbert space.

    Convention: ``[qubit_0, qubit_1, ..., mode_0, mode_1, ...]``.
    Qubit subspaces are dimension 2 (see the module docstring for the
    two-level restriction). Motional mode subspaces are dimension
    ``n_fock``.

    Attributes
    ----------
    n_ions : int
        Number of ion qubits.
    n_modes : int
        Number of motional modes included in the simulation.
    n_fock : int or list[int]
        Fock space truncation, at least ``MIN_FOCK_DIM``. If ``int``,
        all modes share the same cutoff. If ``list``, specifies the
        per-mode cutoff.
    """

    n_ions: int
    n_modes: int
    n_fock: int | list[int] = 10

    def __post_init__(self):
        """Validate inputs and expand ``n_fock`` to per-mode dimensions."""
        if self.n_ions < 0:
            raise ValueError(f"n_ions must be >= 0, got {self.n_ions}")
        if self.n_modes < 1:
            raise ValueError(f"n_modes must be >= 1, got {self.n_modes}")
        if isinstance(self.n_fock, int):
            # bool reaches this branch too; _checked_fock_dim rejects it.
            dim = _checked_fock_dim(self.n_fock, "n_fock")
            self._fock_dims = [dim] * self.n_modes
        elif hasattr(self.n_fock, "__len__"):
            if len(self.n_fock) != self.n_modes:
                raise ValueError(
                    f"n_fock list length {len(self.n_fock)}"
                    f" != n_modes {self.n_modes}"
                )
            self._fock_dims = [
                _checked_fock_dim(d, f"n_fock[{m}]")
                for m, d in enumerate(self.n_fock)
            ]
        else:
            raise TypeError(
                "n_fock must be an int or a list of ints, got"
                f" {type(self.n_fock).__name__} {self.n_fock!r}"
            )

    @property
    def dims(self) -> list[int]:
        """Dimension list for each subsystem.

        Format: [2, 2, ..., n_fock_0, n_fock_1, ...]
        """
        return [2] * self.n_ions + self._fock_dims

    @property
    def total_dim(self) -> int:
        """Total Hilbert space dimension (product of all subsystem dims)."""
        return math.prod(self.dims)

    def fock_dim(self, mode_index: int) -> int:
        """Fock space dimension for a given mode.

        Parameters
        ----------
        mode_index : int
            Index of the motional mode.

        Returns
        -------
        int
            Truncated Fock space dimension for the requested mode.
        """
        if mode_index < 0 or mode_index >= self.n_modes:
            raise IndexError(
                f"Mode index {mode_index} out of range [0, {self.n_modes})"
            )
        return self._fock_dims[mode_index]

"""Composite Hilbert space construction for ion qubits + motional modes."""

import math
from dataclasses import dataclass


@dataclass
class HilbertSpace:
    """Defines the tensor-product structure of the composite Hilbert space.

    Convention: ``[qubit_0, qubit_1, ..., mode_0, mode_1, ...]``.
    Qubit subspaces are dimension 2. Motional mode subspaces are
    dimension ``n_fock``.

    Attributes
    ----------
    n_ions : int
        Number of ion qubits.
    n_modes : int
        Number of motional modes included in the simulation.
    n_fock : int or list[int]
        Fock space truncation. If ``int``, all modes share the same cutoff.
        If ``list``, specifies per-mode cutoff.
    """

    n_ions: int
    n_modes: int
    n_fock: int | list[int] = 10

    def __post_init__(self):
        """Validate inputs and expand ``n_fock`` to per-mode dimensions."""
        if self.n_ions < 1:
            raise ValueError(f"n_ions must be >= 1, got {self.n_ions}")
        if self.n_modes < 1:
            raise ValueError(f"n_modes must be >= 1, got {self.n_modes}")
        if isinstance(self.n_fock, int):
            self._fock_dims = [self.n_fock] * self.n_modes
        else:
            if len(self.n_fock) != self.n_modes:
                raise ValueError(
                    f"n_fock list length {len(self.n_fock)}"
                    f" != n_modes {self.n_modes}"
                )
            self._fock_dims = list(self.n_fock)

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
        return self._fock_dims[mode_index]

"""Composite Hilbert space construction for ion qubits + motional modes."""

import math
from dataclasses import dataclass


@dataclass
class HilbertSpace:
    """Defines the tensor-product structure of the composite Hilbert space.

    Convention: ``[ion_0, ion_1, ..., mode_0, mode_1, ...]``.
    Ion subspaces default to dimension 2 (qubits) but can be set
    to arbitrary *d* for qudit simulations via *ion_dims*.  Motional
    mode subspaces have dimension *n_fock*.

    Attributes
    ----------
    n_ions : int
        Number of ions (internal subsystems).
    n_modes : int
        Number of motional modes included in the simulation.
    n_fock : int or list[int]
        Fock space truncation.
    ion_dims : int, list[int], or None
        Dimension of each ion subsystem.  ``None`` (default) means
        all ions are qubits (d=2).  An ``int`` applies uniformly.
        A ``list`` sets per-ion dimensions.
    """

    n_ions: int
    n_modes: int
    n_fock: int | list[int] = 10
    ion_dims: int | list[int] | None = None

    def __post_init__(self):
        """Validate inputs and expand per-subsystem dimensions."""
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
        if self.ion_dims is None:
            self._ion_dims = [2] * self.n_ions
        elif isinstance(self.ion_dims, int):
            self._ion_dims = [self.ion_dims] * self.n_ions
        else:
            if len(self.ion_dims) != self.n_ions:
                raise ValueError(
                    f"ion_dims length {len(self.ion_dims)}"
                    f" != n_ions {self.n_ions}"
                )
            self._ion_dims = list(self.ion_dims)
        for d in self._ion_dims:
            if d < 2:
                raise ValueError(f"Ion dimension must be >= 2, got {d}")

    @property
    def dims(self) -> list[int]:
        """Dimension list for each subsystem.

        Format: ``[d_ion0, d_ion1, ..., n_fock_0, n_fock_1, ...]``
        """
        return list(self._ion_dims) + self._fock_dims

    @property
    def total_dim(self) -> int:
        """Total Hilbert space dimension (product of all subsystem dims)."""
        return math.prod(self.dims)

    def ion_dim(self, ion_index: int) -> int:
        """Internal-state dimension for a given ion.

        Returns 2 for a qubit, *d* for a qudit.

        Parameters
        ----------
        ion_index : int
            Index of the ion.
        """
        return self._ion_dims[ion_index]

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

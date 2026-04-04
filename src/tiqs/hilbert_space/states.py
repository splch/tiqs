"""Initial state construction for trapped-ion simulations."""
from typing import Optional

import numpy as np
import qutip

from tiqs.hilbert_space.builder import HilbertSpace


class StateFactory:
    """Constructs initial quantum states in the composite Hilbert space.

    Parameters
    ----------
    hilbert_space : HilbertSpace
        The composite Hilbert space specification.
    """

    def __init__(self, hilbert_space: HilbertSpace):
        self.hs = hilbert_space

    def ground_state(self) -> qutip.Qobj:
        """All qubits in |0> (down), all modes in vacuum |n=0>. Returns a ket."""
        parts = []
        for _ in range(self.hs.n_ions):
            parts.append(qutip.basis(2, 0))
        for m in range(self.hs.n_modes):
            parts.append(qutip.basis(self.hs.fock_dim(m), 0))
        return qutip.tensor(parts)

    def product_state(
        self,
        qubit_states: list[int],
        fock_states: list[int],
    ) -> qutip.Qobj:
        """Arbitrary product state. qubit_states[i] in {0, 1}, fock_states[m] is phonon number."""
        if len(qubit_states) != self.hs.n_ions:
            raise ValueError(f"Expected {self.hs.n_ions} qubit states, got {len(qubit_states)}")
        if len(fock_states) != self.hs.n_modes:
            raise ValueError(f"Expected {self.hs.n_modes} fock states, got {len(fock_states)}")
        parts = []
        for i, q in enumerate(qubit_states):
            parts.append(qutip.basis(2, q))
        for m, n in enumerate(fock_states):
            parts.append(qutip.basis(self.hs.fock_dim(m), n))
        return qutip.tensor(parts)

    def thermal_state(
        self,
        n_bar: list[float],
        qubit_states: Optional[list[int]] = None,
    ) -> qutip.Qobj:
        """Qubits in |0>, motional modes in thermal states. Returns a density matrix.

        Parameters
        ----------
        n_bar : list[float]
            Mean phonon number for each mode.
        qubit_states : list[int] or None
            Qubit state for each ion (default all 0).
        """
        if qubit_states is None:
            qubit_states = [0] * self.hs.n_ions
        if len(n_bar) != self.hs.n_modes:
            raise ValueError(f"Expected {self.hs.n_modes} n_bar values, got {len(n_bar)}")
        if len(qubit_states) != self.hs.n_ions:
            raise ValueError(f"Expected {self.hs.n_ions} qubit states, got {len(qubit_states)}")

        parts = []
        for q in qubit_states:
            parts.append(qutip.ket2dm(qutip.basis(2, q)))
        for m, nb in enumerate(n_bar):
            parts.append(qutip.thermal_dm(self.hs.fock_dim(m), nb))
        return qutip.tensor(parts)

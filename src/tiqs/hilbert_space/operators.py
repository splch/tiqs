"""Factory for constructing operators in the composite Hilbert space."""

import qutip

from tiqs.hilbert_space.builder import HilbertSpace


class OperatorFactory:
    r"""Lifts single-subsystem operators to the full tensor-product space.

    Each method returns a `qutip.Qobj` acting on the full composite
    space $\mathcal{H} = \mathcal{H}_\text{qubit}^{\otimes N}
    \otimes \mathcal{H}_\text{motion}^{\otimes M}$, with the
    requested operator on one subsystem and identities elsewhere.

    Parameters
    ----------
    hilbert_space : HilbertSpace
        The composite Hilbert space specification.
    """

    def __init__(self, hilbert_space: HilbertSpace):
        self.hs = hilbert_space

    def _full_operator(
        self, op: qutip.Qobj, subsystem_index: int
    ) -> qutip.Qobj:
        """Tensor an operator on one subsystem with identities on others.

        Parameters
        ----------
        op : qutip.Qobj
            Operator acting on a single subsystem.
        subsystem_index : int
            Index into the composite dimension list identifying the target
            subsystem.

        Returns
        -------
        qutip.Qobj
            Operator acting on the full composite Hilbert space.
        """
        dims = self.hs.dims
        n_subsystems = len(dims)
        if subsystem_index < 0 or subsystem_index >= n_subsystems:
            raise IndexError(
                f"Subsystem index {subsystem_index} out of range"
                f" [0, {n_subsystems})"
            )
        op_list = [qutip.qeye(d) for d in dims]
        op_list[subsystem_index] = op
        return qutip.tensor(op_list)

    def _ion_index(self, ion: int) -> int:
        """Validate ion index and return subsystem index."""
        if ion < 0 or ion >= self.hs.n_ions:
            raise IndexError(
                f"Ion index {ion} out of range [0, {self.hs.n_ions})"
            )
        return ion

    def _validate_qubit_ion(self, ion: int) -> None:
        """Raise if the ion is not a 2-level qubit."""
        d = self.hs.ion_dim(ion)
        if d != 2:
            raise ValueError(
                f"sigma_* operators require a 2-level ion, but "
                f"ion {ion} has dimension {d}. "
                f"Use transition() or spin_j() instead."
            )

    def _mode_index(self, mode: int) -> int:
        """Validate mode index and return subsystem index."""
        if mode < 0 or mode >= self.hs.n_modes:
            raise IndexError(
                f"Mode index {mode} out of range [0, {self.hs.n_modes})"
            )
        return self.hs.n_ions + mode

    def sigma_x(self, ion: int) -> qutip.Qobj:
        r"""Pauli $\sigma_x$ on a qubit ion.

        Raises ``ValueError`` for qudit ions (d > 2).
        """
        self._validate_qubit_ion(ion)
        return self._full_operator(qutip.sigmax(), self._ion_index(ion))

    def sigma_y(self, ion: int) -> qutip.Qobj:
        r"""Pauli $\sigma_y$ on a qubit ion.

        Raises ``ValueError`` for qudit ions (d > 2).
        """
        self._validate_qubit_ion(ion)
        return self._full_operator(qutip.sigmay(), self._ion_index(ion))

    def sigma_z(self, ion: int) -> qutip.Qobj:
        r"""Pauli $\sigma_z$ on a qubit ion.

        Raises ``ValueError`` for qudit ions (d > 2).
        """
        self._validate_qubit_ion(ion)
        return self._full_operator(qutip.sigmaz(), self._ion_index(ion))

    def sigma_plus(self, ion: int) -> qutip.Qobj:
        r"""Raising operator $\sigma_+ = |0\rangle\langle 1|$ on a qubit ion.

        Raises ``ValueError`` for qudit ions (d > 2).
        """
        self._validate_qubit_ion(ion)
        return self._full_operator(qutip.sigmap(), self._ion_index(ion))

    def sigma_minus(self, ion: int) -> qutip.Qobj:
        r"""Lowering operator $\sigma_- = |1\rangle\langle 0|$ on a qubit ion.

        Raises ``ValueError`` for qudit ions (d > 2).
        """
        self._validate_qubit_ion(ion)
        return self._full_operator(qutip.sigmam(), self._ion_index(ion))

    def annihilate(self, mode: int) -> qutip.Qobj:
        r"""Bosonic annihilation operator $a$ for the given motional mode.

        Parameters
        ----------
        mode : int
            Index of the target motional mode.

        Returns
        -------
        qutip.Qobj
            Annihilation operator on mode ``mode``, tensored with identities
            on all other subsystems.
        """
        idx = self._mode_index(mode)
        dim = self.hs.fock_dim(mode)
        return self._full_operator(qutip.destroy(dim), idx)

    def create(self, mode: int) -> qutip.Qobj:
        r"""Bosonic creation operator $a^\dagger$ for the given motional mode.

        Parameters
        ----------
        mode : int
            Index of the target motional mode.

        Returns
        -------
        qutip.Qobj
            Creation operator on mode ``mode``, tensored with identities on
            all other subsystems.
        """
        idx = self._mode_index(mode)
        dim = self.hs.fock_dim(mode)
        return self._full_operator(qutip.create(dim), idx)

    def number(self, mode: int) -> qutip.Qobj:
        r"""Number operator $\hat{n} = a^\dagger a$
        for the given motional mode.

        Parameters
        ----------
        mode : int
            Index of the target motional mode.

        Returns
        -------
        qutip.Qobj
            Number operator on mode ``mode``, tensored with identities on all
            other subsystems.
        """
        idx = self._mode_index(mode)
        dim = self.hs.fock_dim(mode)
        return self._full_operator(qutip.num(dim), idx)

    def position(self, mode: int) -> qutip.Qobj:
        r"""Dimensionless position quadrature
        $(a + a^\dagger)/\sqrt{2}$ for the given mode.

        Parameters
        ----------
        mode : int
            Index of the target motional mode.

        Returns
        -------
        qutip.Qobj
            Position quadrature operator on mode ``mode``, tensored with
            identities on all other subsystems.
        """
        a = self.annihilate(mode)
        return (a + a.dag()) / 2**0.5

    def momentum(self, mode: int) -> qutip.Qobj:
        r"""Dimensionless momentum quadrature
        $i(a^\dagger - a)/\sqrt{2}$ for the given mode.

        Parameters
        ----------
        mode : int
            Index of the target motional mode.

        Returns
        -------
        qutip.Qobj
            Momentum quadrature operator on mode ``mode``, tensored with
            identities on all other subsystems.
        """
        a = self.annihilate(mode)
        return 1j * (a.dag() - a) / 2**0.5

    def identity(self) -> qutip.Qobj:
        """Identity operator on the full Hilbert space.

        Returns
        -------
        qutip.Qobj
            Identity operator spanning every subsystem.
        """
        return qutip.tensor([qutip.qeye(d) for d in self.hs.dims])

    def transition(self, ion: int, level_i: int, level_j: int) -> qutip.Qobj:
        r"""Transition operator $|i\rangle\langle j|$ on the given ion.

        The fundamental building block for qudit operations.

        Parameters
        ----------
        ion : int
        level_i, level_j : int
            Level indices (0 to d-1).
        """
        idx = self._ion_index(ion)
        d = self.hs.ion_dim(ion)
        if not (0 <= level_i < d and 0 <= level_j < d):
            raise IndexError(
                f"Level indices ({level_i}, {level_j}) out of "
                f"range for ion {ion} with dimension {d}"
            )
        op = qutip.basis(d, level_i) * qutip.basis(d, level_j).dag()
        return self._full_operator(op, idx)

    def transition_x(self, ion: int, level_i: int, level_j: int) -> qutip.Qobj:
        r"""Hermitian transition operator
        $|i\rangle\langle j| + |j\rangle\langle i|$.

        Analogous to $\sigma_x$ restricted to the
        $\{|i\rangle, |j\rangle\}$ subspace.
        """
        return self.transition(ion, level_i, level_j) + self.transition(
            ion, level_j, level_i
        )

    def transition_y(self, ion: int, level_i: int, level_j: int) -> qutip.Qobj:
        r"""Hermitian transition operator
        $-i|i\rangle\langle j| + i|j\rangle\langle i|$.

        Analogous to $\sigma_y$ restricted to the
        $\{|i\rangle, |j\rangle\}$ subspace.
        """
        return -1j * self.transition(
            ion, level_i, level_j
        ) + 1j * self.transition(ion, level_j, level_i)

    def transition_z(self, ion: int, level_i: int, level_j: int) -> qutip.Qobj:
        r"""Diagonal operator
        $|i\rangle\langle i| - |j\rangle\langle j|$.

        Analogous to $\sigma_z$ restricted to the
        $\{|i\rangle, |j\rangle\}$ subspace.
        """
        return self.projector(ion, level_i) - self.projector(ion, level_j)

    def projector(self, ion: int, level: int) -> qutip.Qobj:
        r"""Projector $|k\rangle\langle k|$ on the given ion."""
        idx = self._ion_index(ion)
        d = self.hs.ion_dim(ion)
        if not (0 <= level < d):
            raise IndexError(
                f"Level {level} out of range for ion {ion} with dimension {d}"
            )
        op = qutip.ket2dm(qutip.basis(d, level))
        return self._full_operator(op, idx)

    def spin_j(self, ion: int, axis: str) -> qutip.Qobj:
        r"""Generalized spin-$j$ operator on the given ion.

        Parameters
        ----------
        ion : int
        axis : str
            One of ``'x'``, ``'y'``, ``'z'``, ``'+'``, ``'-'``.

        Returns
        -------
        qutip.Qobj
            Spin-$j$ operator where $j = (d-1)/2$.
        """
        idx = self._ion_index(ion)
        d = self.hs.ion_dim(ion)
        j = (d - 1) / 2
        op = qutip.jmat(j, axis)
        return self._full_operator(op, idx)

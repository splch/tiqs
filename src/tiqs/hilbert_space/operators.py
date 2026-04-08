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

    def _mode_index(self, mode: int) -> int:
        """Validate mode index and return subsystem index."""
        if mode < 0 or mode >= self.hs.n_modes:
            raise IndexError(
                f"Mode index {mode} out of range [0, {self.hs.n_modes})"
            )
        return self.hs.n_ions + mode

    def sigma_x(self, ion: int) -> qutip.Qobj:
        r"""Pauli $\sigma_x$ on the given ion.

        Parameters
        ----------
        ion : int
            Index of the target ion qubit.

        Returns
        -------
        qutip.Qobj
            Pauli X operator on ion ``ion``, tensored with identities on all
            other subsystems.
        """
        return self._full_operator(qutip.sigmax(), self._ion_index(ion))

    def sigma_y(self, ion: int) -> qutip.Qobj:
        r"""Pauli $\sigma_y$ on the given ion.

        Parameters
        ----------
        ion : int
            Index of the target ion qubit.

        Returns
        -------
        qutip.Qobj
            Pauli Y operator on ion ``ion``, tensored with identities on all
            other subsystems.
        """
        return self._full_operator(qutip.sigmay(), self._ion_index(ion))

    def sigma_z(self, ion: int) -> qutip.Qobj:
        r"""Pauli $\sigma_z$ on the given ion.

        Parameters
        ----------
        ion : int
            Index of the target ion qubit.

        Returns
        -------
        qutip.Qobj
            Pauli Z operator on ion ``ion``, tensored with identities on all
            other subsystems.
        """
        return self._full_operator(qutip.sigmaz(), self._ion_index(ion))

    def sigma_plus(self, ion: int) -> qutip.Qobj:
        r"""Raising operator $\sigma_+ = |1\rangle\langle 0|$ on the given ion.

        Parameters
        ----------
        ion : int
            Index of the target ion qubit.

        Returns
        -------
        qutip.Qobj
            Raising operator on ion ``ion``, tensored with identities on all
            other subsystems.
        """
        return self._full_operator(qutip.sigmap(), self._ion_index(ion))

    def sigma_minus(self, ion: int) -> qutip.Qobj:
        r"""Lowering operator
        $\sigma_- = |0\rangle\langle 1|$ on the given ion.

        Parameters
        ----------
        ion : int
            Index of the target ion qubit.

        Returns
        -------
        qutip.Qobj
            Lowering operator on ion ``ion``, tensored with identities on all
            other subsystems.
        """
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

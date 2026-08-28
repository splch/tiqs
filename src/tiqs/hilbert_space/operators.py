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
        return qutip.expand_operator(op, dims, subsystem_index)

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
        r"""De-excitation operator $\sigma_+ = |0\rangle\langle 1|$.

        This is `qutip.sigmap`, the spin-*raising* operator in the
        matrix convention. TIQS takes ``basis(2, 0)`` $= |0\rangle$ as
        the qubit ground state, so $|0\rangle\langle 1|$ lowers the
        energy: it de-excites the ion and is the jump operator for
        spontaneous decay. Use `sigma_minus` to excite.

        Parameters
        ----------
        ion : int
            Index of the target ion qubit.

        Returns
        -------
        qutip.Qobj
            De-excitation operator on ion ``ion``, tensored with
            identities on all other subsystems.
        """
        return self._full_operator(qutip.sigmap(), self._ion_index(ion))

    def sigma_minus(self, ion: int) -> qutip.Qobj:
        r"""Excitation operator $\sigma_- = |1\rangle\langle 0|$.

        This is `qutip.sigmam`, the spin-*lowering* operator in the
        matrix convention. With ``basis(2, 0)`` $= |0\rangle$ the
        ground state, $|1\rangle\langle 0|$ raises the energy, so this
        is the operator that carries the drive phase $e^{+i\phi}$ in
        the interaction Hamiltonians. Use `sigma_plus` to de-excite.

        Parameters
        ----------
        ion : int
            Index of the target ion qubit.

        Returns
        -------
        qutip.Qobj
            Excitation operator on ion ``ion``, tensored with
            identities on all other subsystems.
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
        r"""Unit-commutator position quadrature
        $x = (a + a^\dagger)/\sqrt{2}$ for the given mode.

        Normalized so that $[x, p] = i$ together with `momentum`, i.e.
        $\langle x^2\rangle = 1/2$ in the vacuum;
        `tiqs.analysis.phase_space_trajectory` uses the same scaling.
        Beware the other common dimensionless coordinate,
        $q = a + a^\dagger = \sqrt{2}\,x$, which is what
        `tiqs.potential.ArbitraryPotential` calls the dimensionless
        position operator. A potential written in $q$ is therefore
        *not* the same function of this ``x``: a quadratic term picks
        up a factor 2.

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
        r"""Unit-commutator momentum quadrature
        $p = i(a^\dagger - a)/\sqrt{2}$ for the given mode.

        Conjugate to `position` with $[x, p] = i$ ($\hbar = 1$). In the
        $q = a + a^\dagger = \sqrt{2}\,x$ convention of
        `tiqs.potential.ArbitraryPotential` the companion momentum is
        $p_q = i(a^\dagger - a) = \sqrt{2}\,p$, for which
        $[q, p_q] = 2i$.

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

    def embed_mode_operator(self, op: qutip.Qobj, mode: int) -> qutip.Qobj:
        """Embed a single-mode operator into the full Hilbert space.

        Parameters
        ----------
        op : qutip.Qobj
            Operator acting on one motional mode.
        mode : int
            Index of the target motional mode.

        Returns
        -------
        qutip.Qobj
            Operator acting on the full composite Hilbert space.
        """
        return self._full_operator(op, self._mode_index(mode))

    def identity(self) -> qutip.Qobj:
        """Identity operator on the full Hilbert space.

        Returns
        -------
        qutip.Qobj
            Identity operator spanning every subsystem.
        """
        return qutip.tensor([qutip.qeye(d) for d in self.hs.dims])

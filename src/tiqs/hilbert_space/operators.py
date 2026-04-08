"""Factory for constructing operators in the composite Hilbert space."""

import qutip

from tiqs.hilbert_space.builder import HilbertSpace


class OperatorFactory:
    """Lifts single-subsystem operators to the full tensor-product space.

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
        """Tensor an operator on one subsystem with identities on others."""
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
        if ion < 0 or ion >= self.hs.n_ions:
            raise IndexError(
                f"Ion index {ion} out of range [0, {self.hs.n_ions})"
            )
        return ion

    def _mode_index(self, mode: int) -> int:
        if mode < 0 or mode >= self.hs.n_modes:
            raise IndexError(
                f"Mode index {mode} out of range [0, {self.hs.n_modes})"
            )
        return self.hs.n_ions + mode

    def sigma_x(self, ion: int) -> qutip.Qobj:
        return self._full_operator(qutip.sigmax(), self._ion_index(ion))

    def sigma_y(self, ion: int) -> qutip.Qobj:
        return self._full_operator(qutip.sigmay(), self._ion_index(ion))

    def sigma_z(self, ion: int) -> qutip.Qobj:
        return self._full_operator(qutip.sigmaz(), self._ion_index(ion))

    def sigma_plus(self, ion: int) -> qutip.Qobj:
        return self._full_operator(qutip.sigmap(), self._ion_index(ion))

    def sigma_minus(self, ion: int) -> qutip.Qobj:
        return self._full_operator(qutip.sigmam(), self._ion_index(ion))

    def annihilate(self, mode: int) -> qutip.Qobj:
        idx = self._mode_index(mode)
        dim = self.hs.fock_dim(mode)
        return self._full_operator(qutip.destroy(dim), idx)

    def create(self, mode: int) -> qutip.Qobj:
        idx = self._mode_index(mode)
        dim = self.hs.fock_dim(mode)
        return self._full_operator(qutip.create(dim), idx)

    def number(self, mode: int) -> qutip.Qobj:
        idx = self._mode_index(mode)
        dim = self.hs.fock_dim(mode)
        return self._full_operator(qutip.num(dim), idx)

    def position(self, mode: int) -> qutip.Qobj:
        a = self.annihilate(mode)
        return (a + a.dag()) / 2**0.5

    def momentum(self, mode: int) -> qutip.Qobj:
        a = self.annihilate(mode)
        return 1j * (a.dag() - a) / 2**0.5

    def identity(self) -> qutip.Qobj:
        return qutip.tensor([qutip.qeye(d) for d in self.hs.dims])

"""Off-resonant photon scattering during laser-driven gates."""

import numpy as np
import qutip

from tiqs.hilbert_space.operators import OperatorFactory


def rayleigh_scattering_op(
    ops: OperatorFactory,
    ion: int,
    rate: float,
) -> qutip.Qobj:
    """Rayleigh (elastic) photon scattering collapse operator: L = sqrt(rate/4) * sigma_z.

    Rayleigh scattering preserves the qubit state but causes dephasing
    through the random phase kick of the scattered photon.
    """
    return np.sqrt(rate / 4) * ops.sigma_z(ion)


def raman_scattering_op(
    ops: OperatorFactory,
    ion: int,
    rate: float,
) -> qutip.Qobj:
    """Raman (inelastic) photon scattering collapse operator: L = sqrt(rate) * sigma_plus.

    In our convention sigma_plus = |0><1| maps |1> to |0>, implementing
    decay from the excited qubit state. Raman scattering changes the qubit
    state (spin flip), producing a bit-flip error. This is the dominant
    fundamental error source for laser-driven gates.
    """
    return np.sqrt(rate) * ops.sigma_plus(ion)

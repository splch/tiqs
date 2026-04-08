r"""Off-resonant photon scattering during laser-driven gates."""

import numpy as np
import qutip

from tiqs.hilbert_space.operators import OperatorFactory


def rayleigh_scattering_op(
    ops: OperatorFactory,
    ion: int,
    rate: float,
) -> qutip.Qobj:
    r"""Rayleigh (elastic) photon scattering collapse operator.

    $$
    L = \sqrt{\gamma_\text{Ray} / 4}\;\sigma_z
    $$

    Rayleigh scattering preserves the qubit state but causes
    dephasing through the random phase kick of the scattered
    photon.

    Parameters
    ----------
    ops : OperatorFactory
        Factory for constructing multi-body operators.
    ion : int
        Index of the target ion.
    rate : float
        Rayleigh scattering rate in events/second.

    Returns
    -------
    qutip.Qobj
        Collapse operator for Rayleigh scattering dephasing.
    """
    return np.sqrt(rate / 4) * ops.sigma_z(ion)


def raman_scattering_op(
    ops: OperatorFactory,
    ion: int,
    rate: float,
) -> qutip.Qobj:
    r"""Raman (inelastic) photon scattering collapse operator.

    $$
    L = \sqrt{\gamma_\text{Ram}}\;\sigma_+
    $$

    In our convention $\sigma_+ = |0\rangle\langle 1|$ maps
    $|1\rangle$ to $|0\rangle$, implementing decay from the
    excited qubit state. Raman scattering changes the qubit state
    (spin flip), producing a bit-flip error. This is the dominant
    fundamental error source for laser-driven gates.

    Parameters
    ----------
    ops : OperatorFactory
        Factory for constructing multi-body operators.
    ion : int
        Index of the target ion.
    rate : float
        Raman scattering rate in events/second.

    Returns
    -------
    qutip.Qobj
        Collapse operator for Raman scattering bit-flip.
    """
    return np.sqrt(rate) * ops.sigma_plus(ion)

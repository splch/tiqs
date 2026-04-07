"""State preparation via optical pumping."""

import numpy as np
import qutip

from tiqs.hilbert_space.operators import OperatorFactory


def optical_pumping_ops(
    ops: OperatorFactory,
    ion: int,
    pumping_rate: float,
) -> list[qutip.Qobj]:
    """Collapse operators modeling optical pumping to |0> (ground/bright state).

    Optical pumping drives |1> -> |0> dissipatively via a cycling transition.
    Modeled as spontaneous decay from |1> to |0> at the pumping rate.

    QuTiP convention: sigmap() = |0><1| drives |1> -> |0> (spontaneous emission).

    Parameters
    ----------
    ops : OperatorFactory
    ion : int
    pumping_rate : float
        Effective optical pumping rate (rad/s).

    Returns
    -------
    list[qutip.Qobj]
        Collapse operators for optical pumping.
    """
    # sigma_plus maps to sigmap() = |0><1|, which takes |1> -> |0>
    return [np.sqrt(pumping_rate) * ops.sigma_plus(ion)]


def prepare_qubit(
    ops: OperatorFactory,
    ion: int,
    initial_state: qutip.Qobj,
    pumping_rate: float,
    duration: float,
) -> qutip.Qobj:
    """Simulate optical pumping to prepare a qubit in |0>.

    Parameters
    ----------
    ops : OperatorFactory
    ion : int
    initial_state : qutip.Qobj
        Initial density matrix.
    pumping_rate : float
        Optical pumping rate.
    duration : float
        Pumping duration in seconds.

    Returns
    -------
    qutip.Qobj
        Final density matrix after pumping.
    """
    c_ops = optical_pumping_ops(ops, ion, pumping_rate)
    H = 0 * ops.identity()
    tlist = np.linspace(0, duration, 20)
    result = qutip.mesolve(H, initial_state, tlist, c_ops=c_ops)
    return result.states[-1]

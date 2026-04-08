r"""Qubit decoherence: dephasing and spontaneous emission."""

import math

import numpy as np
import qutip

from tiqs.hilbert_space.operators import OperatorFactory


def qubit_dephasing_op(
    ops: OperatorFactory,
    ion: int,
    t2: float,
    t1: float = math.inf,
) -> qutip.Qobj:
    r"""Collapse operator for qubit pure dephasing.

    $$
    L = \sqrt{\gamma_\phi / 2}\;\sigma_z
    $$

    $\gamma_\phi = 1/T_2 - 1/(2 T_1)$. For hyperfine qubits with
    $T_1 \to \infty$: $\gamma_\phi = 1/T_2$.
    Causes exponential decay of off-diagonal density matrix elements.

    Parameters
    ----------
    ops : OperatorFactory
        Factory for constructing multi-body operators.
    ion : int
        Index of the target ion.
    t2 : float
        Qubit $T_2$ dephasing time in seconds.
    t1 : float, optional
        Qubit $T_1$ relaxation time in seconds. Defaults to
        infinity (no relaxation, pure dephasing only).

    Returns
    -------
    qutip.Qobj
        Collapse operator for qubit dephasing.
    """
    gamma_phi = 1.0 / t2 - 1.0 / (2 * t1)
    if gamma_phi < 0:
        raise ValueError(
            f"Pure dephasing rate is negative "
            f"(T2={t2} > 2*T1={2 * t1}). "
            f"T2 cannot exceed 2*T1."
        )
    return np.sqrt(gamma_phi / 2) * ops.sigma_z(ion)


def spontaneous_emission_op(
    ops: OperatorFactory,
    ion: int,
    t1: float,
) -> qutip.Qobj:
    r"""Collapse operator for spontaneous emission.

    $L = \sqrt{1/T_1}\;\sigma_+$.

    In our convention $|0\rangle$ = ground, $|1\rangle$ = excited,
    and $\sigma_+ = |0\rangle\langle 1|$ maps $|1\rangle$ to
    $|0\rangle$, implementing decay from the excited state to the
    ground state.

    Relevant for optical qubits (Ca-40 $D_{5/2}$ lifetime ~1.17 s)
    and off-resonant decay during Raman gates.

    Parameters
    ----------
    ops : OperatorFactory
        Factory for constructing multi-body operators.
    ion : int
        Index of the target ion.
    t1 : float
        Energy relaxation time $T_1$ in seconds.

    Returns
    -------
    qutip.Qobj
        Collapse operator for spontaneous emission.
    """
    return np.sqrt(1.0 / t1) * ops.sigma_plus(ion)

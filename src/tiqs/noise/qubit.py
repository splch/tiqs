"""Qubit decoherence: dephasing and spontaneous emission."""

import numpy as np
import qutip

from tiqs.hilbert_space.operators import OperatorFactory


def qubit_dephasing_op(
    ops: OperatorFactory,
    ion: int,
    t2: float,
    t1: float = float("inf"),
) -> qutip.Qobj:
    """Collapse operator for qubit pure dephasing: L = sqrt(gamma_phi/2) * sigma_z.

    gamma_phi = 1/T2 - 1/(2*T1). For hyperfine qubits with T1 ~ inf: gamma_phi = 1/T2.
    Causes exponential decay of off-diagonal density matrix elements.
    """
    gamma_phi = 1.0 / t2 - 1.0 / (2 * t1)
    if gamma_phi < 0:
        raise ValueError(
            f"Pure dephasing rate is negative (T2={t2} > 2*T1={2 * t1}). "
            f"T2 cannot exceed 2*T1."
        )
    return np.sqrt(gamma_phi / 2) * ops.sigma_z(ion)


def spontaneous_emission_op(
    ops: OperatorFactory,
    ion: int,
    t1: float,
) -> qutip.Qobj:
    """Collapse operator for spontaneous emission: L = sqrt(1/T1) * sigma_plus.

    In our convention |0>=ground, |1>=excited, and sigma_plus = |0><1|
    maps |1> to |0>, implementing decay from the excited state to the
    ground state.

    Relevant for optical qubits (Ca40 D5/2 lifetime ~1.17 s) and
    off-resonant decay during Raman gates.
    """
    return np.sqrt(1.0 / t1) * ops.sigma_plus(ion)

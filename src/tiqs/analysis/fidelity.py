"""Fidelity metrics for trapped-ion quantum operations."""

import qutip


def state_fidelity(state1: qutip.Qobj, state2: qutip.Qobj) -> float:
    """Compute state fidelity between two quantum states (kets or density matrices).

    F = |<psi1|psi2>|^2 for pure states, or F^2 from qutip.fidelity for mixed states.
    """
    if state1.type == "ket" and state2.type == "ket":
        return abs(state1.overlap(state2)) ** 2

    rho1 = qutip.ket2dm(state1) if state1.type == "ket" else state1
    rho2 = qutip.ket2dm(state2) if state2.type == "ket" else state2
    return qutip.fidelity(rho1, rho2) ** 2


def gate_fidelity(
    rho_actual: qutip.Qobj,
    rho_target_spin: qutip.Qobj,
    qubit_indices: list[int],
) -> float:
    """Compute gate fidelity by tracing out motional modes and comparing spin states.

    Parameters
    ----------
    rho_actual : qutip.Qobj
        Full density matrix (qubits + motional modes).
    rho_target_spin : qutip.Qobj
        Target spin-only density matrix.
    qubit_indices : list[int]
        Indices of the qubit subsystems.

    Returns
    -------
    float
        Gate fidelity (squared fidelity).
    """
    if rho_actual.type == "ket":
        rho_actual = qutip.ket2dm(rho_actual)
    rho_spin = rho_actual.ptrace(qubit_indices)
    return qutip.fidelity(rho_spin, rho_target_spin) ** 2


def bell_state_fidelity(rho_spin: qutip.Qobj) -> float:
    """Compute fidelity with the (|00> + i|11>)/sqrt(2) Bell state.

    This is the standard target state for an MS gate.

    Parameters
    ----------
    rho_spin : qutip.Qobj
        Two-qubit density matrix.

    Returns
    -------
    float
        Fidelity with the Bell state.
    """
    ket_00 = qutip.tensor(qutip.basis(2, 0), qutip.basis(2, 0))
    ket_11 = qutip.tensor(qutip.basis(2, 1), qutip.basis(2, 1))
    bell = (ket_00 + 1j * ket_11).unit()
    rho_bell = qutip.ket2dm(bell)
    return qutip.fidelity(rho_spin, rho_bell) ** 2

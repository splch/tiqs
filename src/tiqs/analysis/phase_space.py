"""Phase-space analysis: Wigner functions and trajectories."""

import numpy as np
import qutip


def motional_wigner(
    state: qutip.Qobj,
    mode_index: int,
    qubit_indices: list[int],
    xvec: np.ndarray | None = None,
) -> np.ndarray:
    """Compute the Wigner function of a motional mode by tracing out qubits.

    Parameters
    ----------
    state : qutip.Qobj
        Full system state (ket or density matrix).
    mode_index : int
        Index of the motional mode (0-based among modes, not subsystem index).
    qubit_indices : list[int]
        Indices of all qubit subsystems (will be traced out).
    xvec : np.ndarray or None
        Grid points for the Wigner function. Default: linspace(-5, 5, 100).

    Returns
    -------
    np.ndarray
        2D Wigner function W(x, p), shape (len(xvec), len(xvec)).
    """
    if xvec is None:
        xvec = np.linspace(-5, 5, 100)

    # The subsystem index for mode_index is offset by the number of qubits.
    # Convention: [qubit_0, ..., qubit_{n-1}, mode_0, mode_1, ...]
    n_qubits = len(qubit_indices)
    subsystem_index = n_qubits + mode_index
    rho_mode = state.ptrace(subsystem_index)
    return qutip.wigner(rho_mode, xvec, xvec)


def phase_space_trajectory(
    states: list[qutip.Qobj],
    mode_index: int,
    qubit_indices: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Extract the mean position and momentum of a motional mode over time.

    Useful for visualizing phase-space loops during MS gates.

    Parameters
    ----------
    states : list[qutip.Qobj]
        Time series of full system states.
    mode_index : int
        Index of the motional mode (0-based among modes).
    qubit_indices : list[int]
        Indices of all qubit subsystems.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (x_mean, p_mean) arrays of mean position and momentum.
    """
    n_qubits = len(qubit_indices)
    subsystem_index = n_qubits + mode_index

    # Hoist operator construction out of loop (dimension is constant)
    dim = states[0].ptrace(subsystem_index).shape[0]
    a = qutip.destroy(dim)
    x_op = (a + a.dag()) / np.sqrt(2)
    p_op = 1j * (a.dag() - a) / np.sqrt(2)

    x_vals = []
    p_vals = []
    for state in states:
        rho_mode = state.ptrace(subsystem_index)
        x_vals.append(qutip.expect(x_op, rho_mode))
        p_vals.append(qutip.expect(p_op, rho_mode))

    return np.array(x_vals), np.array(p_vals)

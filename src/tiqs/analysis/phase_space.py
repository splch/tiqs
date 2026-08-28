"""Phase-space analysis: Wigner functions and trajectories."""

import numpy as np
import qutip


def _subsystem_index(
    state: qutip.Qobj,
    mode_index: int,
    n_qubits: int,
) -> int:
    """Resolve a mode index to a subsystem index.

    The composite Hilbert space is ordered
    ``[qubit_0, ..., qubit_{n-1}, mode_0, mode_1, ...]``, so the
    subsystem holding mode *mode_index* is ``n_qubits + mode_index``.

    Parameters
    ----------
    state : qutip.Qobj
        Full system state, used to bound-check the result.
    mode_index : int
        Index of the motional mode (0-based among modes).
    n_qubits : int
        Number of qubit subsystems preceding the modes.

    Returns
    -------
    int
        Index of the subsystem holding the requested mode.

    Raises
    ------
    TypeError
        If *n_qubits* is not an integer. A list of qubit indices
        cannot be accepted: a short list (e.g. ``[0]`` for a
        two-qubit state) is indistinguishable from a correct call on
        a smaller register, and would silently select a qubit
        subsystem instead of the requested mode.
    ValueError
        If *n_qubits* is negative or *mode_index* does not name a
        motional subsystem of *state*.
    """
    if isinstance(n_qubits, bool) or not isinstance(n_qubits, int):
        raise TypeError(
            f"n_qubits must be the integer number of qubit subsystems, "
            f"got {n_qubits!r}. Pass n_qubits=<count>, not a list of "
            f"qubit indices."
        )
    if n_qubits < 0:
        raise ValueError(f"n_qubits must be >= 0, got {n_qubits}")
    n_subsystems = len(state.dims[0])
    n_modes = n_subsystems - n_qubits
    if mode_index < 0 or mode_index >= n_modes:
        raise ValueError(
            f"mode_index {mode_index} out of range [0, {n_modes}) for a "
            f"state with {n_subsystems} subsystems and "
            f"{n_qubits} qubits"
        )
    return n_qubits + mode_index


def motional_wigner(
    state: qutip.Qobj,
    mode_index: int,
    n_qubits: int,
    xvec: np.ndarray | None = None,
) -> np.ndarray:
    """Compute the Wigner function of a motional mode by tracing out qubits.

    Parameters
    ----------
    state : qutip.Qobj
        Full system state (ket or density matrix).
    mode_index : int
        Index of the motional mode (0-based among modes, not subsystem index).
    n_qubits : int
        Number of qubit subsystems preceding the modes (they are all
        traced out). An integer count, not a list of indices.
    xvec : np.ndarray or None
        Grid points for the Wigner function. Default: linspace(-5, 5, 100).

    Returns
    -------
    np.ndarray
        2D Wigner function W(x, p), shape (len(xvec), len(xvec)).
    """
    if xvec is None:
        xvec = np.linspace(-5, 5, 100)

    subsystem_index = _subsystem_index(state, mode_index, n_qubits)
    rho_mode = state.ptrace(subsystem_index)
    return qutip.wigner(rho_mode, xvec, xvec)


def phase_space_trajectory(
    states: list[qutip.Qobj],
    mode_index: int,
    n_qubits: int,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Extract the mean position and momentum of a motional mode over time.

    Quadratures are $x = (a + a^\dagger)/\sqrt{2}$ and
    $p = i(a^\dagger - a)/\sqrt{2}$, so a coherent state $|\alpha\rangle$
    sits at $(\sqrt{2}\,\mathrm{Re}\,\alpha, \sqrt{2}\,\mathrm{Im}\,\alpha)$.
    Useful for visualizing phase-space loops during MS gates.

    Parameters
    ----------
    states : list[qutip.Qobj]
        Time series of full system states.
    mode_index : int
        Index of the motional mode (0-based among modes).
    n_qubits : int
        Number of qubit subsystems preceding the modes. An integer
        count, not a list of indices.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (x_mean, p_mean) arrays of mean position and momentum.
    """
    subsystem_index = _subsystem_index(states[0], mode_index, n_qubits)

    # Hoist operator construction out of loop (dimension is constant)
    dim = states[0].dims[0][subsystem_index]
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

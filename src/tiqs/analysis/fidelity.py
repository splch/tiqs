r"""Fidelity metrics for trapped-ion quantum operations."""

import qutip


def _clamp(value: float) -> float:
    """Clamp a fidelity to the physical range [0, 1].

    ``qutip.fidelity`` goes through ``sqrtm`` of a rank-deficient
    density matrix for pure states, which leaves an absolute error
    floor of order 1e-8 and can return values slightly above 1.

    Parameters
    ----------
    value : float
        Raw fidelity estimate.

    Returns
    -------
    float
        *value* restricted to [0, 1].
    """
    return min(1.0, max(0.0, value))


def state_fidelity(state1: qutip.Qobj, state2: qutip.Qobj) -> float:
    r"""Compute state fidelity between two quantum states.

    Accepts kets or density matrices.

    $F = |\langle\psi_1|\psi_2\rangle|^2$ for pure states, or $F^2$ from
    ``qutip.fidelity`` for mixed states.

    Whenever one argument is a ket the exact expressions
    $|\langle\psi_1|\psi_2\rangle|^2$ or
    $\langle\psi|\rho|\psi\rangle$ are used, avoiding the
    ``sqrtm`` of a singular density matrix.

    Parameters
    ----------
    state1 : qutip.Qobj
        First quantum state (ket or density matrix).
    state2 : qutip.Qobj
        Second quantum state (ket or density matrix).

    Returns
    -------
    float
        State fidelity in the range [0, 1].
    """
    if state1.isket and state2.isket:
        return _clamp(abs(state1.overlap(state2)) ** 2)
    if state1.isket:
        return _clamp(qutip.expect(state2, state1))
    if state2.isket:
        return _clamp(qutip.expect(state1, state2))
    return _clamp(qutip.fidelity(state1, state2) ** 2)


def gate_fidelity(
    rho_actual: qutip.Qobj,
    rho_target_spin: qutip.Qobj,
    qubit_indices: list[int],
) -> float:
    r"""Compute gate fidelity by tracing out motional modes.

    Traces out motional modes and compares the resulting spin
    state to the target.

    Parameters
    ----------
    rho_actual : qutip.Qobj
        Full density matrix (qubits + motional modes).
    rho_target_spin : qutip.Qobj
        Target spin state, ket or density matrix. A ket target takes
        the exact $\langle\psi|\rho|\psi\rangle$ path.
    qubit_indices : list[int]
        Indices of the qubit subsystems to keep. Subsets are
        allowed: the values are used, not just their count.

    Returns
    -------
    float
        Gate fidelity (squared fidelity) in the range [0, 1].
    """
    if rho_actual.isket:
        rho_actual = qutip.ket2dm(rho_actual)
    rho_spin = rho_actual.ptrace(qubit_indices)
    if rho_target_spin.isket:
        return _clamp(qutip.expect(rho_spin, rho_target_spin))
    return _clamp(qutip.fidelity(rho_spin, rho_target_spin) ** 2)


def bell_state_fidelity(rho_spin: qutip.Qobj, sign: int = 1) -> float:
    r"""Compute fidelity with the Bell state
    $(|00\rangle + s\,i\,|11\rangle)/\sqrt{2}$, $s = \pm 1$.

    TIQS's Molmer-Sorensen Hamiltonian places the bichromatic tones
    inside the sidebands, as Sorensen and Molmer do
    (*Phys. Rev. A* **62**, 022311 (2000), Eqs. 4 and 9), so the
    geometric phase carries
    $s = \operatorname{sign}(\delta\,\eta_i\,\eta_j)$: the default
    ``sign=+1`` is the output of ``SimulationRunner.run_ms_gate``
    for a positive detuning on an even-parity mode, and a negative
    detuning or an odd-parity mode (participations of opposite sign)
    produces the conjugate state ``sign=-1``. The two targets are
    orthogonal, so the wrong sign scores 0 for a perfect gate.

    Parameters
    ----------
    rho_spin : qutip.Qobj
        Two-qubit state (ket or density matrix).
    sign : int, optional
        Sign of the relative phase $i$ on $|11\rangle$
        (default ``+1``).

    Returns
    -------
    float
        Fidelity with the Bell state, in the range [0, 1].

    Raises
    ------
    ValueError
        If *sign* is not ``+1`` or ``-1``.
    """
    if sign not in (1, -1):
        raise ValueError(f"sign must be +1 or -1, got {sign}")
    ket_00 = qutip.tensor(qutip.basis(2, 0), qutip.basis(2, 0))
    ket_11 = qutip.tensor(qutip.basis(2, 1), qutip.basis(2, 1))
    bell = (ket_00 + sign * 1j * ket_11).unit()
    return state_fidelity(rho_spin, bell)

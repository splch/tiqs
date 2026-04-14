r"""Qudit gate implementations using transition operators.

For a d-level system, gates are constructed from transition
operators $|i\rangle\langle j|$ that drive individual pairs of
levels, replacing the Pauli operators used for qubit gates.
"""

import numpy as np

from tiqs.constants import TWO_PI
from tiqs.gates.single_qubit import GatePulse
from tiqs.hilbert_space.operators import OperatorFactory


def r_transition(
    ops: OperatorFactory,
    ion: int,
    level_i: int,
    level_j: int,
    theta: float,
    phi: float = 0.0,
    rabi_frequency: float = TWO_PI * 1e6,
) -> GatePulse:
    r"""Rotation by angle *theta* in the |i>-|j> subspace of a qudit.

    $$
    H = \frac{\Omega}{2}\bigl(
      |i\rangle\langle j|\,e^{i\phi}
      + |j\rangle\langle i|\,e^{-i\phi}\bigr)
    $$

    Parameters
    ----------
    ops : OperatorFactory
    ion : int
    level_i, level_j : int
        The two levels to couple.
    theta : float
        Rotation angle (rad).
    phi : float
        Drive phase (rad).
    rabi_frequency : float
        Rabi frequency (rad/s).
    """
    H = (rabi_frequency / 2) * (
        ops.transition(ion, level_i, level_j) * np.exp(1j * phi)
        + ops.transition(ion, level_j, level_i) * np.exp(-1j * phi)
    )
    duration = abs(theta) / rabi_frequency
    return GatePulse(hamiltonian=H, duration=duration)


def ms_qudit_gate_hamiltonian(
    ops: OperatorFactory,
    ions: list[int],
    mode: int,
    eta: list[float],
    rabi_frequency: float,
    detuning: float,
    transitions: list[tuple[int, int]],
) -> list:
    r"""MS gate Hamiltonian for qudit ions.

    Replaces $\sigma_x$ with the Hermitian transition operator
    $|i\rangle\langle j| + |j\rangle\langle i|$ on the driven
    transition for each ion.

    Parameters
    ----------
    ops : OperatorFactory
    ions : list[int]
    mode : int
    eta : list[float]
        Lamb-Dicke parameters per ion.
    rabi_frequency : float
    detuning : float
    transitions : list[tuple[int, int]]
        ``(level_i, level_j)`` for each ion defining the driven
        transition.

    Returns
    -------
    list
        QuTiP list-format Hamiltonian.
    """
    a = ops.annihilate(mode)
    ad = ops.create(mode)

    H_terms = []
    for k, ion_idx in enumerate(ions):
        i, j = transitions[k]
        sx_ij = ops.transition_x(ion_idx, i, j)
        coupling = eta[k] * rabi_frequency

        H_terms.append([coupling * ad * sx_ij, f"exp(1j*{detuning}*t)"])
        H_terms.append([coupling * a * sx_ij, f"exp(-1j*{detuning}*t)"])

    return H_terms


def light_shift_qudit_gate_hamiltonian(
    ops: OperatorFactory,
    ions: list[int],
    mode: int,
    eta: list[float],
    rabi_frequency: float,
    detuning: float,
    transitions: list[tuple[int, int]],
) -> list:
    r"""Light-shift gate Hamiltonian for qudit ions.

    Replaces $\sigma_z$ with the diagonal operator
    $|i\rangle\langle i| - |j\rangle\langle j|$ on the driven
    transition.

    Parameters
    ----------
    ops : OperatorFactory
    ions : list[int]
    mode : int
    eta : list[float]
    rabi_frequency : float
    detuning : float
    transitions : list[tuple[int, int]]
        ``(level_i, level_j)`` for each ion.

    Returns
    -------
    list
        QuTiP list-format Hamiltonian.
    """
    a = ops.annihilate(mode)
    ad = ops.create(mode)

    H_terms = []
    for k, ion_idx in enumerate(ions):
        i, j = transitions[k]
        sz_ij = ops.transition_z(ion_idx, i, j)
        coupling = eta[k] * rabi_frequency

        H_terms.append([coupling * ad * sz_ij, f"exp(1j*{detuning}*t)"])
        H_terms.append([coupling * a * sz_ij, f"exp(-1j*{detuning}*t)"])

    return H_terms

r"""Adiabatic smooth entangling gate (AESE) from arXiv:2510.17286.

The gate Hamiltonian in the displaced frame is:

$$
H_g = \delta(t)\,a^\dagger a
  + \frac{\Omega_g(t)}{2}\,S_x\,(a^\dagger + a)
$$

where $S_x = \sum_j \sigma_{x,j}$, $\delta(t)$ is the
time-varying detuning from the motional mode, and $\Omega_g(t)$
is the (slowly varying) Rabi frequency.

This is NOT the interaction-picture MS Hamiltonian.  The detuning
appears as a time-varying mode frequency, not as an oscillating
phase.
"""

import numpy as np

from tiqs.hilbert_space.operators import OperatorFactory


def adiabatic_gate_hamiltonian(
    ops: OperatorFactory,
    ions: list[int],
    mode: int,
    eta: list[float],
    omega_arr: np.ndarray,
    delta_arr: np.ndarray,
    tlist: np.ndarray,
) -> list:
    r"""Construct the adiabatic smooth gate Hamiltonian.

    $$
    H(t) = \delta(t)\,a^\dagger a
      + \sum_j \frac{\eta_j\,\Omega(t)}{2}\,\sigma_{x,j}
        \,(a^\dagger + a)
    $$

    **Coupling convention:** ``omega_arr`` is the sideband Rabi
    frequency $\Omega_g = \eta\,\Omega_\text{carrier}$.  When
    using a pulse from ``calibrate_adiabatic_smooth_gate``, pass
    ``eta=[1.0, 1.0]`` since the Lamb-Dicke factor is already
    absorbed into $\Omega_g$.

    Both $\delta(t)$ and $\Omega(t)$ are provided as arrays
    sampled at ``tlist``.

    Parameters
    ----------
    ops : OperatorFactory
    ions : list[int]
    mode : int
    eta : list[float]
        Lamb-Dicke parameters per ion.
    omega_arr : np.ndarray
        Rabi frequency values at each time point (rad/s).
    delta_arr : np.ndarray
        Detuning values at each time point (rad/s).
    tlist : np.ndarray
        Time points (s).

    Returns
    -------
    list
        QuTiP list-format Hamiltonian.
    """
    a = ops.annihilate(mode)
    ad = ops.create(mode)
    n_op = ad * a
    x_op = a + ad

    H_terms = []

    # Mode frequency term: delta(t) * a_dag * a
    H_terms.append([n_op, delta_arr])

    # Spin-motion coupling: eta_j * Omega(t)/2 * sx_j * (a + a_dag)
    for j, ion_idx in enumerate(ions):
        sx_j = ops.sigma_x(ion_idx)
        H_coupling = (eta[j] / 2) * sx_j * x_op
        H_terms.append([H_coupling, omega_arr])

    return H_terms


def adiabatic_gate_multimode_hamiltonian(
    ops: OperatorFactory,
    ions: list[int],
    modes: list[int],
    eta: np.ndarray,
    omega_arr: np.ndarray,
    delta_arrs: list[np.ndarray],
    tlist: np.ndarray,
) -> list:
    r"""Multi-mode adiabatic gate Hamiltonian.

    $$
    H(t) = \sum_p \delta_p(t)\,a_p^\dagger a_p
      + \sum_p \sum_j \frac{\eta_{j,p}\,\Omega(t)}{2}
        \,\sigma_{x,j}\,(a_p^\dagger + a_p)
    $$

    Each mode has its own detuning profile $\delta_p(t)$.  The
    target mode sweeps from $\delta_\max$ to $\delta_\min$; the
    off-resonant modes remain at their (larger) detunings
    throughout.

    Parameters
    ----------
    ops : OperatorFactory
    ions : list[int]
    modes : list[int]
        Motional mode indices.
    eta : np.ndarray
        Lamb-Dicke matrix, shape ``(len(ions), len(modes))``.
    omega_arr : np.ndarray
        Shared Rabi frequency vs time (rad/s).
    delta_arrs : list[np.ndarray]
        Per-mode detuning arrays, one per mode.
    tlist : np.ndarray
        Time points (s).
    """
    H_terms: list = []

    for p_idx, mode in enumerate(modes):
        a = ops.annihilate(mode)
        ad = ops.create(mode)
        n_op = ad * a
        x_op = a + ad

        H_terms.append([n_op, delta_arrs[p_idx]])

        for j_idx, ion_idx in enumerate(ions):
            sx_j = ops.sigma_x(ion_idx)
            coupling = eta[j_idx, p_idx] / 2
            H_terms.append([coupling * sx_j * x_op, omega_arr])

    return H_terms

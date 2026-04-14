r"""Microwave MS gate via gradient coupling with continuous dressing.

A strong microwave field dresses the qubit, rotating the spin basis
so that the gradient's native $\sigma_z$ force acts as a
$\sigma_x$ force in the dressed frame.  In the regime
$\Omega_\text{dress} \gg \eta\,\Omega_\text{gate}$, the
dressed-frame Hamiltonian is the standard MS Hamiltonian.
"""

from __future__ import annotations

import warnings

import numpy as np

from tiqs.gates.molmer_sorensen import ms_gate_hamiltonian
from tiqs.hilbert_space.operators import OperatorFactory


def microwave_ms_gate_hamiltonian(
    ops: OperatorFactory,
    ions: list[int],
    mode: int,
    eta: list[float],
    gate_rabi_frequency: float,
    detuning: float,
    dressing_rabi_frequency: float,
) -> list:
    r"""Microwave MS gate via gradient coupling with continuous dressing.

    Validates the dressing hierarchy and delegates to
    :func:`~tiqs.gates.molmer_sorensen.ms_gate_hamiltonian`.

    The dressing drive must satisfy
    $\Omega_\text{dress} \gg \eta\,\Omega_\text{gate}$ for the
    rotating-wave approximation to hold.

    Parameters
    ----------
    ops : OperatorFactory
    ions : list[int]
    mode : int
    eta : list[float]
        Gradient-mediated Lamb-Dicke parameters.
    gate_rabi_frequency : float
        Effective gate drive Rabi frequency (rad/s).
    detuning : float
        Detuning from dressed-state sideband (rad/s).
    dressing_rabi_frequency : float
        Continuous dressing drive Rabi frequency (rad/s).

    Returns
    -------
    list
        QuTiP list-format Hamiltonian (MS gate in dressed frame).
    """
    eta_max = max(abs(e) for e in eta)
    gate_coupling = eta_max * gate_rabi_frequency

    if dressing_rabi_frequency < 10 * gate_coupling:
        warnings.warn(
            f"Dressing hierarchy marginal: "
            f"Omega_dress={dressing_rabi_frequency:.0f} < "
            f"10*eta*Omega_gate={10 * gate_coupling:.0f}. "
            f"Dressed-frame approximation may be inaccurate.",
            stacklevel=2,
        )

    return ms_gate_hamiltonian(
        ops,
        ions=ions,
        mode=mode,
        eta=eta,
        rabi_frequency=gate_rabi_frequency,
        detuning=detuning,
    )


def microwave_ms_gate_hamiltonian_full(
    ops: OperatorFactory,
    ions: list[int],
    mode: int,
    eta: list[float],
    gate_rabi_frequency: float,
    detuning: float,
    dressing_rabi_frequency: float,
    dressing_phase: float = 0.0,
) -> list:
    r"""Full Hamiltonian including the dressing drive (no RWA).

    Includes both the continuous dressing tone and the
    gradient-mediated spin-motion coupling:

    $$
    H(t) = \sum_j \frac{\Omega_\text{dress}}{2}
      (\sigma_+^{(j)} e^{i\varphi} + \sigma_-^{(j)} e^{-i\varphi})
    + \sum_j \eta_j\,\Omega_\text{gate}\,\sigma_{z,j}
      (a^\dagger e^{i\delta t} + a\,e^{-i\delta t})
    $$

    Use this when the dressing hierarchy is not well-satisfied and
    the RWA approximation breaks down.

    Parameters
    ----------
    ops : OperatorFactory
    ions : list[int]
    mode : int
    eta : list[float]
    gate_rabi_frequency : float
    detuning : float
    dressing_rabi_frequency : float
    dressing_phase : float
        Phase of the dressing drive (rad).

    Returns
    -------
    list
        QuTiP list-format Hamiltonian.
    """
    a = ops.annihilate(mode)
    ad = ops.create(mode)

    H_terms: list = []

    # Dressing drive (static in the qubit rotating frame)
    for ion_idx in ions:
        sp = ops.sigma_plus(ion_idx)
        sm = ops.sigma_minus(ion_idx)
        H_dress = (dressing_rabi_frequency / 2) * (
            sp * np.exp(1j * dressing_phase)
            + sm * np.exp(-1j * dressing_phase)
        )
        H_terms.append(H_dress)

    # Gradient spin-motion coupling (sigma_z force)
    for j, ion_idx in enumerate(ions):
        sz_j = ops.sigma_z(ion_idx)
        coupling = eta[j] * gate_rabi_frequency

        H_terms.append([coupling * ad * sz_j, f"exp(1j*{detuning}*t)"])
        H_terms.append([coupling * a * sz_j, f"exp(-1j*{detuning}*t)"])

    return H_terms

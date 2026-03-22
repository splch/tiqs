"""Molmer-Sorensen entangling gate Hamiltonian construction."""
import numpy as np
import qutip

from tiqs.constants import TWO_PI
from tiqs.hilbert_space.operators import OperatorFactory


def ms_gate_duration(detuning: float, loops: int = 1) -> float:
    """Gate time for the MS gate: tau = 2*pi*K / delta where K is the number of loops.

    Parameters
    ----------
    detuning : float
        Sideband detuning delta (rad/s).
    loops : int
        Number of phase-space loops (K). More loops = slower but more robust.

    Returns
    -------
    float
        Gate duration in seconds.
    """
    return TWO_PI * loops / abs(detuning)


def ms_gate_hamiltonian(
    ops: OperatorFactory,
    ions: list[int],
    mode: int,
    eta: list[float],
    rabi_frequency: float,
    detuning: float,
    phases: list[float] | None = None,
) -> list:
    """Construct the Molmer-Sorensen gate Hamiltonian in the interaction picture.

    The MS Hamiltonian for two ions coupled to one motional mode:

    H_MS(t) = sum_j eta_j * Omega * sigma_x_j * (a^dag * e^{i*delta*t} + a * e^{-i*delta*t})

    This is a spin-dependent force that displaces the motional state conditioned on the
    collective spin. After time tau = 2*pi*K/delta, the motion returns to its initial state
    and the spins acquire a geometric phase proportional to the enclosed phase-space area.

    The gate is maximally entangling when eta*Omega = delta / 2 (for symmetric coupling
    and a single loop).

    Parameters
    ----------
    ops : OperatorFactory
        Operator factory for the composite Hilbert space.
    ions : list[int]
        Indices of the two ions to entangle.
    mode : int
        Index of the motional mode used as the bus.
    eta : list[float]
        Lamb-Dicke parameters [eta_0, eta_1] for each ion on this mode.
    rabi_frequency : float
        Rabi frequency Omega (rad/s) of the bichromatic drive on each ion.
    detuning : float
        Detuning delta (rad/s) from the motional sideband.
    phases : list[float] or None
        Optical phases for each ion [phi_0, phi_1]. Default [0, 0].

    Returns
    -------
    list
        QuTiP list-format Hamiltonian: [[H_op, coeff_string], ...].
    """
    if phases is None:
        phases = [0.0] * len(ions)

    a = ops.annihilate(mode)
    ad = ops.create(mode)

    H_terms = []

    for j, ion_idx in enumerate(ions):
        sx_j = ops.sigma_x(ion_idx)
        coupling = eta[j] * rabi_frequency

        H_plus = coupling * ad * sx_j
        H_minus = coupling * a * sx_j

        H_terms.append([H_plus, f"exp(1j*{detuning}*t)"])
        H_terms.append([H_minus, f"exp(-1j*{detuning}*t)"])

    return H_terms

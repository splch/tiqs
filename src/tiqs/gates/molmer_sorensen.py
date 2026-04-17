"""Molmer-Sorensen entangling gate Hamiltonian construction."""

from collections.abc import Callable

from tiqs.constants import TWO_PI
from tiqs.hilbert_space.operators import OperatorFactory


def _geometric_phase_hamiltonian(
    ops: OperatorFactory,
    ions: list[int],
    mode: int,
    eta: list[float],
    rabi_frequency: float,
    detuning: float,
    spin_op: Callable[[int], object],
) -> list:
    """Build a spin-dependent force Hamiltonian in the interaction picture.

    Shared implementation for MS ($\\sigma_x$) and light-shift
    ($\\sigma_z$) gates, which differ only in the spin operator.

    Parameters
    ----------
    ops : OperatorFactory
    ions : list[int]
    mode : int
    eta : list[float]
    rabi_frequency : float
    detuning : float
    spin_op : callable
        Function mapping ion index to the spin operator
        (e.g., ``ops.sigma_x`` or ``ops.sigma_z``).

    Returns
    -------
    list
        QuTiP list-format Hamiltonian.
    """
    a = ops.annihilate(mode)
    ad = ops.create(mode)
    H_terms = []
    for j, ion_idx in enumerate(ions):
        s_j = spin_op(ion_idx)
        coupling = eta[j] * rabi_frequency
        H_terms.append([coupling * ad * s_j, f"exp(1j*{detuning}*t)"])
        H_terms.append([coupling * a * s_j, f"exp(-1j*{detuning}*t)"])
    return H_terms


def ms_gate_duration(detuning: float, loops: int = 1) -> float:
    r"""Gate time for the MS gate: $\tau = 2\pi K / \delta$ where $K$ is the
    number of loops.

    Parameters
    ----------
    detuning : float
        Sideband detuning $\delta$ (rad/s).
    loops : int
        Number of phase-space loops ($K$). More loops = slower but more
        robust.

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
) -> list:
    r"""Construct the Molmer-Sorensen gate Hamiltonian in the interaction
    picture.

    The MS Hamiltonian for $N$ ions coupled to one motional mode:

    $$
    H_\mathrm{MS}(t) = \sum_j \eta_j \, \Omega \, \sigma_{x,j}
    \left( a^\dagger e^{i \delta t} + a \, e^{-i \delta t} \right)
    $$

    Note: this implementation absorbs the factor of $1/2$ from the
    theory-doc convention
    ($\frac{\hbar\eta\Omega}{2}\,\sigma_\phi\,[\ldots]$) into the
    definition of $\Omega$, so the ``rabi_frequency`` parameter
    here equals half the bare single-photon Rabi frequency.
    ``SimulationRunner.run_ms_gate`` calibrates $\Omega$ automatically.

    This is a spin-dependent force that displaces the motional state
    conditioned on the collective spin. After time
    $\tau = 2\pi K / \delta$, the motion returns to its initial state and
    the spins acquire a geometric phase proportional to the enclosed
    phase-space area.

    For two identically-coupled ions, the maximally entangling condition is
    $\eta \Omega = \delta / 4$ (single loop).

    Parameters
    ----------
    ops : OperatorFactory
        Operator factory for the composite Hilbert space.
    ions : list[int]
        Indices of the ions to entangle.
    mode : int
        Index of the motional mode used as the bus.
    eta : list[float]
        Lamb-Dicke parameters for each ion on this mode.
    rabi_frequency : float
        Rabi frequency $\Omega$ (rad/s) of the bichromatic drive on each ion.
    detuning : float
        Detuning $\delta$ (rad/s) from the motional sideband.

    Returns
    -------
    list
        QuTiP list-format Hamiltonian: ``[[H_op, coeff_string], ...]``.
    """
    return _geometric_phase_hamiltonian(
        ops, ions, mode, eta, rabi_frequency, detuning, ops.sigma_x
    )

r"""Addressing crosstalk between neighboring ions."""

import numpy as np
import qutip

from tiqs.hilbert_space.operators import OperatorFactory


def crosstalk_hamiltonian(
    ops: OperatorFactory,
    target_ion: int,
    neighbor_ion: int,
    crosstalk_fraction: float,
    rabi_frequency: float,
    phase: float = 0.0,
) -> qutip.Qobj:
    r"""Spurious Hamiltonian on neighbor ion from imperfect beam focusing.

    When addressing target ion with Rabi frequency $\Omega$, the neighbor ion
    sees a fraction $\epsilon$ of the light:
    $\Omega_\text{neighbor} = \epsilon\,\Omega$.

    $$
    H_\text{crosstalk} = \frac{\epsilon\,\Omega}{2}\;\sigma_x^\text{neighbor}
    $$

    Typical crosstalk: $\epsilon \sim 10^{-3}$ to $10^{-2}$ for ~2 um beam
    waist and ~5 um ion spacing.

    Parameters
    ----------
    ops : OperatorFactory
    target_ion : int
        Ion being addressed (not used in the operator, just for clarity).
    neighbor_ion : int
        Ion experiencing crosstalk.
    crosstalk_fraction : float
        Fraction of target Rabi frequency seen by neighbor (0 to 1).
    rabi_frequency : float
        Target Rabi frequency (rad/s).
    phase : float
        Drive phase (same as the target beam).

    Returns
    -------
    qutip.Qobj
        Crosstalk Hamiltonian on the neighbor ion.
    """
    sp = ops.sigma_plus(neighbor_ion)
    sm = ops.sigma_minus(neighbor_ion)
    omega_xt = crosstalk_fraction * rabi_frequency
    return (omega_xt / 2) * (
        sp * np.exp(1j * phase) + sm * np.exp(-1j * phase)
    )

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

    When addressing the target ion with Rabi frequency $\Omega$, the
    neighbor ion sees a fraction $\epsilon$ of the drive amplitude:
    $\Omega_\text{neighbor} = \epsilon\,\Omega$. It is driven on the
    same axis as the target,

    $$
    H_\text{crosstalk}
        = \frac{\epsilon\,\Omega}{2}
          \bigl(\sigma_- e^{i\varphi} + \sigma_+ e^{-i\varphi}\bigr)
        = \frac{\epsilon\,\Omega}{2}
          \bigl(\sigma_x \cos\varphi
                + \sigma_y \sin\varphi\bigr)
    $$

    acting on the neighbor. Convention (repo-wide): the excitation
    operator `OperatorFactory.sigma_minus` $= |1\rangle\langle 0|$
    carries $e^{+i\varphi}$, so at equal $\varphi$ this drives the
    neighbor about the same axis as the single-qubit gates in
    `tiqs.gates.single_qubit` and the carrier drive in
    `tiqs.interaction.hamiltonian`.

    Magnitude: for a Gaussian beam of waist $w_0$ at ion spacing $s$
    the intensity ratio is $e^{-2 s^2 / w_0^2}$, so the
    Rabi-frequency fraction is $\epsilon = e^{-s^2/w_0^2}$ --
    $1.9 \times 10^{-3}$ for $w_0 = 2\;\mu$m and $s = 5\;\mu$m
    (and utterly negligible, $10^{-11}$, for $w_0 = 1\;\mu$m).
    Reported nearest-neighbor crosstalk in addressed-beam systems is
    typically $10^{-3}$--$10^{-2}$, i.e. above the Gaussian estimate,
    because it is limited by aberrations and scattered light rather
    than by the ideal beam profile.

    Parameters
    ----------
    ops : OperatorFactory
        Factory for constructing multi-body operators.
    target_ion : int
        Ion being addressed. Does not enter the operator (only the
        neighbor is driven); validated for consistency.
    neighbor_ion : int
        Ion experiencing crosstalk.
    crosstalk_fraction : float
        Fraction $\epsilon$ of the target Rabi frequency seen by the
        neighbor (0 to 1).
    rabi_frequency : float
        Target Rabi frequency $\Omega$ (rad/s).
    phase : float
        Drive phase $\varphi$ (the same as the target beam).

    Returns
    -------
    qutip.Qobj
        Crosstalk Hamiltonian on the neighbor ion.

    Raises
    ------
    IndexError
        If ``target_ion`` or ``neighbor_ion`` is out of range.
    """
    ops._ion_index(target_ion)
    sp = ops.sigma_plus(neighbor_ion)
    sm = ops.sigma_minus(neighbor_ion)
    omega_xt = crosstalk_fraction * rabi_frequency
    return (omega_xt / 2) * (
        sm * np.exp(1j * phase) + sp * np.exp(-1j * phase)
    )

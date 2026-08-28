r"""Off-resonant photon scattering during laser-driven gates."""

import numpy as np
import qutip

from tiqs.hilbert_space.operators import OperatorFactory


def rayleigh_scattering_op(
    ops: OperatorFactory,
    ion: int,
    rate: float,
) -> qutip.Qobj:
    r"""Rayleigh (elastic) photon scattering dephasing operator.

    $$
    L = \sqrt{\Gamma_\text{el} / 4}\;\sigma_z
    $$

    which decays qubit coherences at $\Gamma_\text{el}/2$ (Uys et
    al., PRL 105, 200401 (2010) Eqs. 6 and 8).

    ``rate`` is the elastic-scattering **decoherence** rate
    $\Gamma_\text{el}$, *not* the Rayleigh scattering rate. Elastic
    scattering exchanges no energy or angular momentum with the
    internal state, so it only dephases through the *difference*
    between the two qubit states' scattering amplitudes: Uys Eq. (7)
    makes $\Gamma_\text{el}$ the square of that difference, hence
    $\Gamma_\text{el} \le \Gamma_\text{Rayleigh}$ always. For
    clock-state qubits the two amplitudes are nearly equal and the
    suppression is severe -- Ozeri et al., PRA 75, 042329 (2007)
    Eq. (66) gives
    $\Gamma_\text{el} / \Gamma_\text{Rayleigh}
    \approx (\omega_0/\Delta)^2$ for Raman detuning
    $\Delta \gg \omega_0$, e.g. $4\times 10^{-5}$ for a $^9$Be$^+$
    clock qubit ($\omega_0/2\pi = 1.25$ GHz) at
    $\Delta = 2\pi \times 197$ GHz. Passing a raw scattering rate
    here therefore overestimates dephasing by orders of magnitude.

    Model scope: the photon-recoil kick that accompanies each elastic
    event (Ozeri's $\epsilon_R$, a motional error) is not modeled.

    Parameters
    ----------
    ops : OperatorFactory
        Factory for constructing multi-body operators.
    ion : int
        Index of the target ion.
    rate : float
        Elastic-scattering decoherence rate $\Gamma_\text{el}$ in
        $\text{s}^{-1}$ (Uys Eq. 7), not the scattering rate.

    Returns
    -------
    qutip.Qobj
        Collapse operator for Rayleigh dephasing.

    Raises
    ------
    ValueError
        If ``rate`` is negative.
    """
    if rate < 0:
        raise ValueError(f"rate must be non-negative, got {rate}")
    return np.sqrt(rate / 4) * ops.sigma_z(ion)


def raman_scattering_ops(
    ops: OperatorFactory,
    ion: int,
    rate: float,
) -> list[qutip.Qobj]:
    r"""Raman (inelastic) photon scattering collapse operators.

    Spontaneous Raman scattering projects the ion into one of the
    ground sublevels with a rate that does not depend on which qubit
    state it started in (Ozeri et al., PRA 75, 042329 (2007) Eqs.
    9--11). With equal branching to the two sublevels, half of the
    $\Gamma_\text{Ram}$ events change the qubit state, giving a
    **bidirectional** pair of collapse operators:

    $$
    L_\downarrow = \sqrt{\Gamma_\text{Ram}/2}\;\sigma_+,
    \qquad
    L_\uparrow = \sqrt{\Gamma_\text{Ram}/2}\;\sigma_-
    $$

    In this codebase $\sigma_+$ = `OperatorFactory.sigma_plus` =
    $|0\rangle\langle 1|$ is de-excitation and $\sigma_-$ =
    `OperatorFactory.sigma_minus` = $|1\rangle\langle 0|$ is
    excitation, so the two operators drive population in opposite
    directions at the same rate. The total state-changing rate is
    $\Gamma_\text{Ram}/2$ from *either* qubit state; events that
    return the ion to its original sublevel are not observable in
    this two-level model. This is population transfer (a depolarizing
    channel), not a $\sigma_x$ bit flip, and it is the dominant
    fundamental error source for laser-driven gates.

    Model scope: Raman events that leave the qubit manifold entirely
    (Ozeri's $\epsilon_D$ leakage into metastable $D$ levels) are not
    modeled -- there is no third level in this Hilbert space. Note
    also that $L_\downarrow$ has the same form as
    `tiqs.noise.qubit.spontaneous_emission_op`, so adding both
    channels for the same physical process double-counts decay.

    Parameters
    ----------
    ops : OperatorFactory
        Factory for constructing multi-body operators.
    ion : int
        Index of the target ion.
    rate : float
        Total spontaneous Raman scattering rate
        $\Gamma_\text{Ram}$ in events/second.

    Returns
    -------
    list[qutip.Qobj]
        Collapse operators $[L_\downarrow, L_\uparrow]$, or an empty
        list when ``rate`` is zero.

    Raises
    ------
    ValueError
        If ``rate`` is negative.
    """
    if rate < 0:
        raise ValueError(f"rate must be non-negative, got {rate}")
    if rate == 0:
        return []
    amp = np.sqrt(rate / 2)
    return [amp * ops.sigma_plus(ion), amp * ops.sigma_minus(ion)]

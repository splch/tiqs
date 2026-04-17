r"""QCCD transport: ion shuttling and crystal splitting.

.. include:: ../../docs/theory/transport.md
"""

import numpy as np
import qutip

from tiqs.hilbert_space.operators import OperatorFactory


def shuttle_motional_excitation(
    distance: float,
    duration: float,
    trap_frequency: float,
) -> float:
    r"""Estimate motional excitation from linear ion shuttling.

    For an optimized waveform, the residual excitation after
    shuttling scales as:

    $$
    \Delta\bar{n} \sim \left(\frac{d}{x_\text{zpf}}\right)^2
        \exp(-\pi\,\omega_\text{trap}\,T)
    $$

    This is the adiabaticity criterion: longer duration relative to
    the trap period means less excitation. Modern optimized waveforms
    achieve sub-quantum excitation for durations > 50 us at typical
    trap frequencies.

    The simplified model:

    $$
    \Delta\bar{n} \sim A\,\frac{d^2}{T^2\,\omega_\text{trap}^2}
    $$

    where $A$ is a geometry-dependent constant (order 1 for optimized
    waveforms).

    Parameters
    ----------
    distance : float
        Shuttling distance in meters.
    duration : float
        Shuttling time in seconds.
    trap_frequency : float
        Axial secular angular frequency (rad/s).

    Returns
    -------
    float
        Estimated number of added motional quanta.
    """
    # Currently only uses the adiabaticity-based model; `distance` is
    # accepted for API compatibility with a future distance-dependent model.
    n_periods = trap_frequency * duration / (2 * np.pi)
    # For optimized waveforms, excitation decays exponentially with
    # the number of trap oscillation periods during transport.
    # A ~ 5 sets the scale: ~5 quanta for a 1-period shuttle,
    # dropping exponentially toward a floor of 0.01 for very
    # adiabatic transport.
    return max(0.01, 5.0 * np.exp(-n_periods / 3.0))


def apply_shuttling_noise(
    rho: qutip.Qobj,
    ops: OperatorFactory,
    mode: int,
    added_quanta: float,
) -> qutip.Qobj:
    r"""Apply shuttling-induced motional excitation as thermal noise.

    Models the effect of shuttling as depositing phonons into the
    specified mode via a thermal channel. The rate is calibrated
    so that ``added_quanta`` phonons are deposited starting from
    vacuum. For non-vacuum initial states, the actual number of
    added phonons will be larger due to stimulated emission
    ($d\langle n\rangle/dt = \gamma(\langle n\rangle + 1)$).

    Parameters
    ----------
    rho : qutip.Qobj
        Input density matrix.
    ops : OperatorFactory
    mode : int
        Motional mode index.
    added_quanta : float
        Number of phonons added by the transport operation.

    Returns
    -------
    qutip.Qobj
        Density matrix after shuttling noise.
    """
    if added_quanta <= 0:
        return rho
    ad = ops.create(mode)
    # With L = sqrt(gamma) * a_dag, the master equation gives
    # d<n>/dt = gamma * (<n> + 1), so
    # <n>(t) = (<n>(0)+1)*exp(gamma*t) - 1.
    # To add exactly `added_quanta` phonons starting from vacuum:
    #   added_quanta = exp(gamma*t) - 1
    #   => gamma*t = ln(added_quanta + 1)
    t_evolve = 1e-6
    rate = np.log(added_quanta + 1) / t_evolve
    c_ops = [np.sqrt(rate) * ad]
    H = qutip.qzero(ops.hs.dims)
    tlist = [0, t_evolve]
    result = qutip.mesolve(H, rho, tlist, c_ops=c_ops)
    return result.states[-1]


def split_crystal_excitation(
    trap_frequency: float,
    split_duration: float,
) -> float:
    """Estimate motional excitation from splitting a two-ion crystal.

    Crystal splitting is more heating-prone than linear shuttling
    because the axial potential must be reshaped from a single well
    to a double well. The excitation depends on how adiabatically
    the potential is transformed.

    Parameters
    ----------
    trap_frequency : float
        Axial secular angular frequency (rad/s).
    split_duration : float
        Time for the splitting operation (s).

    Returns
    -------
    float
        Estimated added motional quanta.
    """
    adiabaticity = trap_frequency * split_duration
    if adiabaticity > 50:
        return 0.05
    return max(0.05, 2.0 * np.exp(-adiabaticity / 5))

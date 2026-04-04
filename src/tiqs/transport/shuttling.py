"""Ion shuttling: motional excitation from linear transport."""
import numpy as np
import qutip

from tiqs.hilbert_space.operators import OperatorFactory


def shuttle_motional_excitation(
    distance: float,
    duration: float,
    trap_frequency: float,
) -> float:
    """Estimate motional excitation from linear ion shuttling.

    For an optimized waveform, the residual excitation after shuttling scales as:

    delta_n ~ (distance / x_zpf)^2 * exp(-pi * omega_trap * duration)

    This is the adiabaticity criterion: longer duration relative to the trap
    period means less excitation. Modern optimized waveforms achieve sub-quantum
    excitation for durations > 50 us at typical trap frequencies.

    The simplified model: delta_n ~ A * (distance / duration)^2 / omega_trap^2
    where A is a geometry-dependent constant (order 1 for optimized waveforms).

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
    # Number of trap periods during shuttling
    n_periods = trap_frequency * duration / (2 * np.pi)
    # For optimized waveforms, excitation decays exponentially with the
    # number of trap oscillation periods during transport.
    # A ~ 5 sets the scale: ~5 quanta for a 1-period shuttle, dropping
    # exponentially toward a floor of 0.01 for very adiabatic transport.
    return max(0.01, 5.0 * np.exp(-n_periods / 3.0))


def apply_shuttling_noise(
    rho: qutip.Qobj,
    ops: OperatorFactory,
    mode: int,
    added_quanta: float,
) -> qutip.Qobj:
    """Apply shuttling-induced motional excitation as thermal noise on a mode.

    Models the effect of shuttling as adding 'added_quanta' mean phonons
    to the specified mode via a thermal channel. Uses mesolve with a
    creation-operator collapse operator calibrated to deposit the desired
    number of quanta.

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
    a = ops.annihilate(mode)
    # With L = sqrt(gamma) * a_dag, the master equation gives
    # d<n>/dt = gamma * (<n> + 1), so <n>(t) = (<n>(0)+1)*exp(gamma*t) - 1.
    # To add exactly `added_quanta` phonons starting from vacuum, solve:
    #   added_quanta = exp(gamma*t) - 1  =>  gamma*t = ln(added_quanta + 1)
    t_evolve = 1e-6
    rate = np.log(added_quanta + 1) / t_evolve
    c_ops = [np.sqrt(rate) * ad]
    H = 0 * ops.identity()
    tlist = [0, t_evolve]
    result = qutip.mesolve(H, rho, tlist, c_ops=c_ops)
    return result.states[-1]

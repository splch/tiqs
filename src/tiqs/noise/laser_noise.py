"""Laser phase and intensity noise models."""
import numpy as np
import qutip

from tiqs.hilbert_space.operators import OperatorFactory


def laser_phase_noise_op(
    ops: OperatorFactory,
    ion: int,
    rate: float,
) -> qutip.Qobj:
    """Effective collapse operator for laser phase noise: L = sqrt(rate/2) * sigma_z.

    Phase noise between Raman beams or on a direct optical drive appears as
    dephasing on the qubit. Rate is the effective linewidth of the
    beat note / laser.
    """
    return np.sqrt(rate / 2) * ops.sigma_z(ion)


def laser_intensity_noise_op(
    ops: OperatorFactory,
    ion: int,
    fractional_rms: float,
    rabi_frequency: float,
) -> qutip.Qobj:
    """Hamiltonian perturbation from laser intensity noise.

    Intensity fluctuations delta_I/I cause Rabi frequency errors:
    delta_Omega/Omega = (1/2) * delta_I/I

    This returns the systematic Hamiltonian shift; for stochastic modeling,
    use this operator as a collapse operator with appropriate rate.

    Parameters
    ----------
    ops : OperatorFactory
    ion : int
    fractional_rms : float
        RMS fractional intensity fluctuation (delta_I / I).
    rabi_frequency : float
        Nominal Rabi frequency (rad/s).

    Returns
    -------
    qutip.Qobj
        Hermitian operator representing the intensity-noise Hamiltonian perturbation.
    """
    delta_omega = (fractional_rms / 2) * rabi_frequency
    return (delta_omega / 2) * ops.sigma_x(ion)

r"""Laser phase and intensity noise models."""

import numpy as np
import qutip

from tiqs.hilbert_space.operators import OperatorFactory


def laser_phase_noise_op(
    ops: OperatorFactory,
    ion: int,
    rate: float,
) -> qutip.Qobj:
    r"""Effective collapse operator for laser phase noise.

    $$
    L = \sqrt{\gamma / 2}\;\sigma_z
    $$

    Phase noise between Raman beams or on a direct optical drive
    appears as dephasing on the qubit. Rate is the effective
    linewidth of the beat note / laser.

    Parameters
    ----------
    ops : OperatorFactory
        Factory for constructing multi-body operators.
    ion : int
        Index of the target ion.
    rate : float
        Effective linewidth of the laser or beat note in rad/s.

    Returns
    -------
    qutip.Qobj
        Collapse operator for laser phase noise.
    """
    return np.sqrt(rate / 2) * ops.sigma_z(ion)


def laser_intensity_noise_op(
    ops: OperatorFactory,
    ion: int,
    fractional_rms: float,
    rabi_frequency: float,
) -> qutip.Qobj:
    r"""Hamiltonian perturbation from laser intensity noise.

    Intensity fluctuations $\delta I / I$ cause Rabi frequency errors:

    $$
    \frac{\delta\Omega}{\Omega} = \frac{1}{2}\,\frac{\delta I}{I}
    $$

    This returns the systematic Hamiltonian shift; for stochastic modeling,
    use this operator as a collapse operator with appropriate rate.

    Parameters
    ----------
    ops : OperatorFactory
    ion : int
    fractional_rms : float
        RMS fractional intensity fluctuation ($\delta I / I$).
    rabi_frequency : float
        Nominal Rabi frequency (rad/s).

    Returns
    -------
    qutip.Qobj
        Hermitian operator representing the intensity-noise
        Hamiltonian perturbation.
    """
    delta_omega = (fractional_rms / 2) * rabi_frequency
    return (delta_omega / 2) * ops.sigma_x(ion)

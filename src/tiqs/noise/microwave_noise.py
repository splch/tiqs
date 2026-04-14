r"""Noise models specific to microwave-driven gradient gates.

Microwave gates eliminate photon scattering entirely but introduce
magnetic-field and gradient fluctuation noise as the dominant
coherence limits.
"""

import numpy as np
import qutip

from tiqs.hilbert_space.operators import OperatorFactory


def microwave_phase_noise_op(
    ops: OperatorFactory,
    ion: int,
    linewidth: float,
) -> qutip.Qobj:
    r"""Collapse operator for microwave oscillator phase noise.

    $$
    L = \sqrt{\gamma_\text{mw}/2}\;\sigma_z
    $$

    Functionally identical to laser phase noise but the rate is
    typically 3-6 orders of magnitude lower (microwave linewidth
    < 1 Hz vs laser beat-note linewidth 1-100 kHz).

    Parameters
    ----------
    ops : OperatorFactory
    ion : int
    linewidth : float
        Effective microwave linewidth in rad/s.
    """
    return np.sqrt(linewidth / 2) * ops.sigma_z(ion)


def microwave_amplitude_noise_op(
    ops: OperatorFactory,
    ion: int,
    fractional_rms: float,
    rabi_frequency: float,
) -> qutip.Qobj:
    r"""Hamiltonian perturbation from microwave amplitude noise.

    $$
    \delta H = \frac{\delta\Omega}{2}\,\sigma_x
    $$

    For microwave drives, $\delta\Omega/\Omega$ is the fractional
    amplitude fluctuation (no factor of 1/2, unlike laser intensity
    noise where $\delta\Omega/\Omega = \delta I / (2I)$).

    Parameters
    ----------
    ops : OperatorFactory
    ion : int
    fractional_rms : float
        RMS fractional amplitude fluctuation.
    rabi_frequency : float
        Nominal Rabi frequency (rad/s).
    """
    delta_omega = fractional_rms * rabi_frequency
    return (delta_omega / 2) * ops.sigma_x(ion)


def magnetic_field_t2(
    field_noise_spectral_density: float,
    qubit_zeeman_sensitivity: float,
) -> float:
    r"""Estimate T2 from ambient magnetic field noise.

    For white magnetic field noise with spectral density $S_B$
    (T${}^2$/Hz):

    $$
    1/T_2 = \left(\frac{\partial\omega_q}{\partial B}\right)^2 S_B
    $$

    Parameters
    ----------
    field_noise_spectral_density : float
        Single-sided PSD in T^2/Hz.
    qubit_zeeman_sensitivity : float
        $\partial\omega_q / \partial B$ in rad/s per Tesla.

    Returns
    -------
    float
        Estimated T2 in seconds.
    """
    gamma = qubit_zeeman_sensitivity**2 * field_noise_spectral_density
    if gamma <= 0:
        return float("inf")
    return 1.0 / gamma


def gradient_motional_dephasing_op(
    ops: OperatorFactory,
    mode: int,
    fractional_gradient_noise: float,
    gate_coupling: float,
    gate_detuning: float,
) -> qutip.Qobj:
    r"""Collapse operator for motional dephasing from gradient fluctuations.

    Gradient noise causes the spin-dependent force to fluctuate,
    leading to imperfect motional closure.  The effective dephasing
    rate is:

    $$
    \gamma = \left(\frac{\delta(dB/dz)}{dB/dz}\right)^2
    \frac{(\eta\,\Omega)^2}{\delta}
    $$

    The collapse operator is $L = \sqrt{\gamma}\,\hat{n}$.

    Parameters
    ----------
    ops : OperatorFactory
    mode : int
    fractional_gradient_noise : float
        RMS fractional gradient fluctuation.
    gate_coupling : float
        Product $\eta\,\Omega$ (rad/s).
    gate_detuning : float
        Gate detuning $\delta$ (rad/s).
    """
    rate = fractional_gradient_noise**2 * gate_coupling**2 / abs(gate_detuning)
    return np.sqrt(rate) * ops.number(mode)

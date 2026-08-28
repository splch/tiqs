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
    L = \sqrt{W / 4}\;\sigma_z
    $$

    Phase noise between Raman beams or on a direct optical drive
    appears as dephasing on the qubit. ``rate`` is the **FWHM
    linewidth** $W$ of the laser (or of the Raman beat note) in rad/s.

    A phase-diffusing (white-frequency-noise) laser has
    $\langle[\varphi(t) - \varphi(0)]^2\rangle = 2 D t$, first-order
    coherence $g_1(\tau) = e^{-D|\tau|}$ and hence a Lorentzian
    spectrum of half width at half maximum $D$, i.e. $W = 2 D$. The
    qubit coherence therefore decays at

    $$
    1 / T_2 = D = W / 2 = \pi\, W_\text{Hz}
    $$

    which is what $L = \sqrt{W/4}\,\sigma_z$ produces (a $\sigma_z$
    collapse operator of amplitude $\sqrt{\gamma/2}$ decays
    coherences at $\gamma$).

    Model scope: a Markovian $\sigma_z$ term reproduces white
    frequency noise only, giving exponential decay. Servo bumps and
    $1/f$ laser noise give non-exponential (Gaussian-like) decay and
    need explicit stochastic averaging instead.

    Parameters
    ----------
    ops : OperatorFactory
        Factory for constructing multi-body operators.
    ion : int
        Index of the target ion.
    rate : float
        FWHM linewidth of the laser or beat note in rad/s
        ($2\pi$ times the linewidth in Hz).

    Returns
    -------
    qutip.Qobj
        Collapse operator for laser phase noise.

    Raises
    ------
    ValueError
        If ``rate`` is negative.
    """
    if rate < 0:
        raise ValueError(f"rate must be non-negative, got {rate}")
    return np.sqrt(rate / 4) * ops.sigma_z(ion)


def laser_intensity_noise_op(
    ops: OperatorFactory,
    ion: int,
    fractional_rms: float,
    rabi_frequency: float,
) -> qutip.Qobj:
    r"""Hamiltonian perturbation from laser intensity noise.

    For a resonant single-photon drive the Rabi frequency follows the
    field amplitude, so intensity fluctuations give

    $$
    \frac{\delta\Omega}{\Omega}
        = \frac{1}{2}\,\frac{\delta I}{I}
    $$

    and the perturbation added to $H = (\Omega/2)\,\sigma_x$ is
    $(\delta\Omega/2)\,\sigma_x$, which is what this returns. For a
    two-photon Raman drive derived from a single laser
    $\Omega \propto I$, so $\delta\Omega/\Omega = \delta I / I$ --
    pass ``fractional_rms`` doubled in that case.

    Intensity noise produces **coherent over/under-rotation**, not
    decoherence, so this is a Hamiltonian term and must not be handed
    to a solver as a Lindblad collapse operator: it has units of
    rad/s rather than $\sqrt{1/\text{s}}$ (turning a Hamiltonian
    fluctuation into a jump operator needs a factor
    $\sqrt{\tau_c}$ for the noise correlation time), and used as
    written it dephases at $2(\delta\Omega/2)^2$ --
    $1.3 \times 10^{7}\;\text{s}^{-1}$ for 1% noise at
    $\Omega = 2\pi \times 1$ MHz, against a true $\pi$-pulse
    infidelity of $6.2\times 10^{-5}$. Model it by classical
    averaging instead: sample
    $\epsilon \sim \mathcal{N}(0, \texttt{fractional\_rms}/2)$, solve
    with `qutip.sesolve` using $H = (1 + \epsilon) H_0$, and average
    the resulting states or expectation values.

    Parameters
    ----------
    ops : OperatorFactory
        Factory for constructing multi-body operators.
    ion : int
        Index of the target ion.
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

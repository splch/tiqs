r"""QCCD transport: ion shuttling and crystal splitting.

.. include:: ../../docs/theory/transport.md
"""

import warnings

import numpy as np
import qutip

from tiqs.constants import HBAR, TWO_PI
from tiqs.hilbert_space.operators import OperatorFactory

SPLIT_REFERENCE_QUANTA = 2.0
"""Measured split excitation at `SPLIT_REFERENCE_ADIABATICITY`.

Bowler et al., *Phys. Rev. Lett.* **109**, 080502 (2012) separated a
two-ion crystal in 55 us and found "Fock state populations consistent
with coherent states of $\\bar{n} = 2.1 \\pm 0.1$ in zone A and
$\\bar{n} = 1.9 \\pm 0.1$ in zone B".
"""

SPLIT_REFERENCE_ADIABATICITY = TWO_PI * 0.7e6 * 55e-6
"""$\\omega_\\text{crit} T$ of the Bowler et al. separation, $\\approx 242$.

Their simulated minimum (critical-point) centre-of-mass frequency was
$2\\pi \\times 0.7$ MHz, a factor $\\sim 3$ below the single-well
frequency -- which is why the calibration is anchored to
$\\omega_\\text{crit}$ and not to the initial trap frequency.
"""

_TRUNCATION_SIGMAS = 4.0
"""Displaced-vacuum widths that must fit below the Fock cutoff."""


def shuttle_motional_excitation(
    distance: float,
    duration: float,
    trap_frequency: float,
    mass_kg: float,
    heating_rate: float = 0.0,
) -> float:
    r"""Motional excitation from linear ion shuttling.

    Translating a harmonic well of fixed frequency $\omega$ along the
    trap axis leaves the ion in a coherent state of the final well.
    The displacement amplitude is the Fourier component of the well's
    velocity at the secular frequency,

    $$
    \alpha = \frac{1}{2x_\text{zpf}}
        \int_0^T \dot{z}_0(t)\,e^{i\omega t}\,dt ,
    \qquad
    \Delta\bar{n} = |\alpha|^2
    $$

    with $x_\text{zpf} = \sqrt{\hbar / 2m\omega}$ (Reichle et al.,
    *Fortschr. Phys.* **54**, 666 (2006); Bowler et al., *Phys. Rev.
    Lett.* **109**, 080502 (2012) Eq. 1, credited there to Lau &
    James, *Phys. Rev. A* **83**, 062330 (2011)). The excitation is a
    property of the *waveform*, not of the initial motional state.

    TIQS evaluates this for one named reference waveform, the smooth
    $\sin^2$ velocity ramp $\dot{z}_0 = (2d/T)\sin^2(\pi t/T)$, which
    starts and stops with zero velocity and zero acceleration. Its
    integral is exact:

    $$
    \Delta\bar{n} = \left(\frac{d}{2x_\text{zpf}}\right)^2
        \frac{\text{sinc}^2(N - 1)}{N^2 (N + 1)^2},
    \qquad
    N = \frac{\omega T}{2\pi}
    $$

    with $\text{sinc}(x) = \sin(\pi x)/(\pi x)$ (`numpy.sinc`). Three
    features of this result matter in practice:

    - **Sudden limit.** As $N \to 0$ the expression tends to
      $(d/2x_\text{zpf})^2$: the ion never moves and is left displaced
      by the full distance $d$ from the new well centre.
    - **Power law, not exponential.** The envelope falls as $N^{-6}$
      for this waveform ($(\omega T)^{-2}$ for a hard-edged
      constant-velocity ramp, Bowler Eq. 2). Exponential suppression
      in $\omega T$ requires an analytic $C^\infty$ ramp and is not
      generic.
    - **Catch condition.** $\Delta\bar{n}$ vanishes whenever $N$ is an
      integer $\ge 2$: "an ion starting in its ground state of motion
      is caught back in the ground state" (Bowler et al.). These nulls
      are exact only for the idealized waveform and perfect timing, so
      values far below the trap's anomalous-heating contribution are
      not physical -- pass ``heating_rate`` to add that floor.

    Engineered waveforms beat the $\sin^2$ ramp by orders of magnitude
    in the diabatic regime (Walther et al., *Phys. Rev. Lett.* **109**,
    080501 (2012) measured 0.10(1) quanta for 280 um in 3.6 us at 1.41
    MHz, where this reference profile gives ~80), so treat the return
    value as the excitation of a specific unoptimized ramp rather than
    as a prediction for tuned hardware.

    Parameters
    ----------
    distance : float
        Shuttling distance in meters, $d \ge 0$.
    duration : float
        Shuttling time in seconds, $T > 0$.
    trap_frequency : float
        Axial secular angular frequency (rad/s), constant during
        transport.
    mass_kg : float
        Ion mass in kg (`IonSpecies.mass_kg`). Required: the
        excitation is measured in units of $x_\text{zpf}$, which
        depends on the mass.
    heating_rate : float, optional
        Anomalous heating rate in quanta/s (the ``heating_rate`` of
        `tiqs.noise.motional`). Adds $\dot{\bar{n}} T$, the
        contribution that *grows* with duration and sets the real
        floor on slow transport.

    Returns
    -------
    float
        Added motional quanta.

    Raises
    ------
    ValueError
        If ``distance`` or ``heating_rate`` is negative, or if
        ``duration``, ``trap_frequency`` or ``mass_kg`` is not
        strictly positive.
    """
    if distance < 0:
        raise ValueError(f"distance must be >= 0, got {distance}")
    if duration <= 0:
        raise ValueError(f"duration must be > 0, got {duration}")
    if trap_frequency <= 0:
        raise ValueError(f"trap_frequency must be > 0, got {trap_frequency}")
    if mass_kg <= 0:
        raise ValueError(f"mass_kg must be > 0, got {mass_kg}")
    if heating_rate < 0:
        raise ValueError(f"heating_rate must be >= 0, got {heating_rate}")

    n_periods = trap_frequency * duration / TWO_PI
    x_zpf = np.sqrt(HBAR / (2 * mass_kg * trap_frequency))
    # sinc(N - 1) keeps the N -> 1 resonance finite and reproduces the
    # sudden limit as N -> 0, where sinc(N - 1) -> N.
    envelope = np.sinc(n_periods - 1) / (n_periods * (n_periods + 1))
    coherent = (distance / (2 * x_zpf)) ** 2 * envelope**2
    return float(coherent + heating_rate * duration)


def apply_shuttling_noise(
    rho: qutip.Qobj,
    ops: OperatorFactory,
    mode: int,
    added_quanta: float,
    n_phases: int = 24,
) -> qutip.Qobj:
    r"""Apply transport excitation as a phase-averaged displacement.

    A translated harmonic well drives the mode with
    $H(t) = \hbar\omega a^\dagger a + f(t)(a + a^\dagger)$, whose exact
    propagator is a coherent displacement $D(\alpha)$ with
    $|\alpha|^2 = \Delta\bar{n}$. The phase $\arg\alpha$ depends on
    sub-nanosecond waveform timing and is not tracked, so TIQS applies
    the phase-averaged channel

    $$
    \rho \mapsto \frac{1}{P}\sum_{p=0}^{P-1}
        D(\alpha_p)\,\rho\,D^\dagger(\alpha_p),
    \qquad
    \alpha_p = \sqrt{\Delta\bar{n}}\;e^{2\pi i p / P}
    $$

    This is a mixture of unitaries, hence trace preserving and
    positive, and it acts only on ``mode``, so correlations with the
    qubits and other modes survive. Because
    $\langle D^\dagger n D\rangle = \langle n\rangle + \alpha^*\langle
    a\rangle + \alpha\langle a^\dagger\rangle + |\alpha|^2$ and the
    uniform phase grid cancels the linear terms exactly, the channel
    adds exactly ``added_quanta`` for *any* input state and leaves
    $\langle a\rangle$ unchanged.

    Both properties distinguish it from a $L = \sqrt{\gamma}\,
    a^\dagger$ amplifier channel, which would add
    $\Delta\bar{n}(1 + \bar{n}_0)$ and amplify a pre-existing coherent
    amplitude by $\sqrt{1 + \Delta\bar{n}}$. Transport excitation is
    state independent (Bowler et al., *Phys. Rev. Lett.* **109**,
    080502 (2012) Eq. 1 involves only the waveform), and the measured
    post-transport states are coherent, not thermal.

    From vacuum the result is a phase-averaged coherent state: the
    Fock populations are Poissonian with mean ``added_quanta``, not
    thermal. Coherences $\rho_{n n'}$ with $|n - n'|$ a nonzero
    multiple of $P$ survive the average, so use ``n_phases`` at least
    the mode's Fock dimension for an exactly diagonal output.

    Parameters
    ----------
    rho : qutip.Qobj
        Input density matrix (or ket) on the full space.
    ops : OperatorFactory
    mode : int
        Motional mode index.
    added_quanta : float
        Quanta added by the transport operation, $\Delta\bar{n} \ge 0$.
        Zero returns ``rho`` unchanged.
    n_phases : int, optional
        Number of displacement phases averaged over, $P \ge 2$.

    Returns
    -------
    qutip.Qobj
        Density matrix after transport excitation.

    Raises
    ------
    ValueError
        If ``added_quanta`` is negative or ``n_phases`` < 2.

    Warns
    -----
    UserWarning
        If the displaced state does not fit under the Fock cutoff, in
        which case the truncated displacement operator reflects
        population and under-adds energy.
    """
    if added_quanta < 0:
        raise ValueError(f"added_quanta must be >= 0, got {added_quanta}")
    if n_phases < 2:
        raise ValueError(f"n_phases must be >= 2, got {n_phases}")
    if added_quanta == 0:
        return rho

    fock_dim = ops.hs.fock_dim(mode)
    width = added_quanta + _TRUNCATION_SIGMAS * np.sqrt(added_quanta)
    if width > fock_dim - 1:
        warnings.warn(
            f"added_quanta={added_quanta:.3g} on mode {mode} needs a"
            f" Fock cutoff above {width:.3g} = <n> +"
            f" {_TRUNCATION_SIGMAS:g} sqrt(<n>), but n_fock ="
            f" {fock_dim}; the truncated displacement will add less"
            " than requested. Increase n_fock.",
            UserWarning,
            stacklevel=2,
        )

    if rho.isket:
        rho = qutip.ket2dm(rho)
    alpha = np.sqrt(added_quanta)
    out = None
    for phase in TWO_PI * np.arange(n_phases) / n_phases:
        displace = ops.embed_mode_operator(
            qutip.displace(fock_dim, alpha * np.exp(1j * phase)), mode
        )
        term = displace * rho * displace.dag()
        out = term if out is None else out + term
    return out / n_phases


def split_crystal_excitation(
    omega_crit: float,
    split_duration: float,
    heating_rate: float = 0.0,
) -> float:
    r"""Estimate motional excitation from splitting a two-ion crystal.

    Splitting reshapes a single well into a double well. On the way
    the axial confinement passes through a *critical point* where the
    external harmonic term vanishes and the potential is quartic; the
    minimum mode frequency there, $\omega_\text{crit}$, is a factor
    3-20 below the initial single-well frequency (Kaufmann et al.,
    *New J. Phys.* **16**, 073012 (2014) Table 1 lists
    $\omega_\text{crit}/2\pi = 0.11$-$0.29$ MHz for six segmented
    traps; Bowler et al. simulated 0.7 MHz for theirs). The
    critical-point frequency, not the initial one, sets adiabaticity,
    so it is what this function takes.

    In the impulsive (diabatic) regime the energy gain scales as the
    square of the rate at which the control parameter is swept through
    the critical point, i.e. $\delta E \propto T^{-2}$ (Kaufmann et
    al., Eq. 25). TIQS uses that scaling anchored to one measurement:

    $$
    \Delta\bar{n}_\text{split} \approx
        \Delta\bar{n}_\text{ref}
        \left(\frac{\omega_\text{ref} T_\text{ref}}
                   {\omega_\text{crit} T}\right)^2
        + \dot{\bar{n}}\,T
    $$

    with $\Delta\bar{n}_\text{ref} = 2$ quanta at
    $\omega_\text{ref} T_\text{ref} = 2\pi \times 0.7\,\text{MHz}
    \times 55\,\mu\text{s} \approx 242$ (Bowler et al., *Phys. Rev.
    Lett.* **109**, 080502 (2012)). Splitting is far more
    excitation-prone than shuttling at equal duration: the $T^{-2}$
    envelope decays much more slowly than the $T^{-6}$ of
    `shuttle_motional_excitation`, and there are no catch nulls.

    Accuracy and scope: this is a one-anchor power law, not a trap
    model. At $\omega_\text{crit}/2\pi = 0.18$ MHz and $T = 80$ us it
    returns 14 quanta where Ruster et al., *Phys. Rev. A* **90**,
    033410 (2014) measured $\bar{n} = 4.16(16)$ per ion with
    spectroscopically calibrated ramps, so expect factor-of-a-few
    over-estimates for optimized control. Optimized ramps can also
    cross into Kaufmann's adiabatic regime ($\chi < 1$, their Eq. 28),
    where the excitation is suppressed by an extra exponential this
    form does not contain.

    The second term is the anomalous-heating contribution
    $\Delta\bar{n}_\text{th} = \int_0^T \Gamma_h(\omega(t))\,dt
    \approx \dot{\bar{n}} T$ (Kaufmann et al., Sec. 3.3): it *grows*
    with duration, so with ``heating_rate`` > 0 the total has a
    minimum at $T^* = (2\Delta\bar{n}_\text{ref}
    \omega_\text{ref}^2 T_\text{ref}^2 /
    (\dot{\bar{n}}\omega_\text{crit}^2))^{1/3}$ and slower splitting
    beyond it is counterproductive -- "anomalous heating will strongly
    contribute to the energy gain at large splitting times". Heating
    is worst exactly where the confinement is weakest -- Kaufmann's
    trap followed $\Gamma_h \approx 6.3\,
    (\omega/2\pi\,\text{MHz})^{-1.81}$ quanta/ms -- so evaluate
    $\dot{\bar{n}}$ near $\omega_\text{crit}$, not at the single-well
    frequency.

    Parameters
    ----------
    omega_crit : float
        Minimum (critical-point) crystal mode angular frequency
        during the split, rad/s. Typically $2\pi \times 0.1$-$0.9$
        MHz -- *not* the initial single-well frequency.
    split_duration : float
        Total duration of the splitting ramp (s).
    heating_rate : float, optional
        Anomalous heating rate near the critical point in quanta/s.

    Returns
    -------
    float
        Estimated added motional quanta per ion.

    Raises
    ------
    ValueError
        If ``omega_crit`` or ``split_duration`` is not strictly
        positive, or ``heating_rate`` is negative.
    """
    if omega_crit <= 0:
        raise ValueError(f"omega_crit must be > 0, got {omega_crit}")
    if split_duration <= 0:
        raise ValueError(f"split_duration must be > 0, got {split_duration}")
    if heating_rate < 0:
        raise ValueError(f"heating_rate must be >= 0, got {heating_rate}")

    adiabaticity = omega_crit * split_duration
    impulsive = (
        SPLIT_REFERENCE_QUANTA
        * (SPLIT_REFERENCE_ADIABATICITY / adiabaticity) ** 2
    )
    return float(impulsive + heating_rate * split_duration)

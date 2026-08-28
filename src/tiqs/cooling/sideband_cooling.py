"""Resolved sideband cooling: analytical and simulated."""

import numpy as np
import qutip

from tiqs.hilbert_space.operators import OperatorFactory
from tiqs.hilbert_space.states import StateFactory


def sideband_cooling_nbar(
    gamma_eff: float,
    trap_frequency: float,
) -> float:
    r"""Analytical steady-state phonon number from resolved sideband cooling.

    $$
    \bar{n}_\text{final} \sim
        \left(\frac{\Gamma_\text{eff}}
             {2\omega_\text{trap}}\right)^2
    $$

    where $\Gamma_\text{eff}$ is the effective cooling
    rate (optical pumping rate or Raman transition
    linewidth).

    Model scope: this is the bracket-free form quoted by Wineland
    et al., J. Res. NIST **103**, 259 (1998) Sec. 3.1. Leibfried
    RMP **75**, 281 (2003) Eq. (112) carries an additional $O(1)$
    factor,

    $$
    \bar{n} \approx
      \left(\frac{\tilde{\Gamma}}{2\omega}\right)^2
      \left[\frac{\tilde{\eta}^2}{\eta^2} + \frac{1}{4}\right],
    $$

    with $\tilde{\eta}$ the Lamb-Dicke parameter of the
    spontaneously emitted photon and $\eta$ that of the cooling
    drive. The bracket is $1/4$ for
    $\tilde{\eta} \ll \eta$ and $\approx 5/4$ for
    $\tilde{\eta} = \eta$, so the value returned here is accurate
    only to a factor of a few.

    *trap_frequency* must be a **positive-energy** mode frequency.
    Feeding it a Penning magnetron frequency
    (`tiqs.trap.PenningTrap.omega_magnetron`) models the wrong
    direction: the magnetron mode carries negative total energy, so
    its sideband roles are exchanged and cooling requires the
    blue-detuned tone plus an axialization drive (Jain et al.,
    *Nature* **627**, 510 (2024)).

    Parameters
    ----------
    gamma_eff : float
        Effective linewidth of the cooling transition (rad/s).
    trap_frequency : float
        Motional mode frequency (rad/s), positive-energy mode only.

    Returns
    -------
    float
        Estimated final mean phonon number.

    Raises
    ------
    ValueError
        If either argument is not positive.
    """
    if gamma_eff <= 0:
        raise ValueError(f"gamma_eff must be > 0, got {gamma_eff}")
    if trap_frequency <= 0:
        raise ValueError(f"trap_frequency must be > 0, got {trap_frequency}")
    return (gamma_eff / (2 * trap_frequency)) ** 2


def sideband_cooling_simulate(
    ops: OperatorFactory,
    ion: int,
    mode: int,
    n_bar_initial: float,
    eta: float,
    rabi_frequency: float,
    optical_pumping_rate: float,
    n_cycles: int,
) -> float:
    r"""Simulate resolved sideband cooling as RSB pulses + optical pumping.

    Each cycle is simulated as two *sequential* segments:

    1. Red sideband $\pi$-pulse of duration
       $t_\pi = \pi/|\eta\Omega|$ with the RSB Hamiltonian only
       (no dissipation): $|0, n\rangle \to |1, n-1\rangle$, removing
       at most one phonon per cycle.
    2. Optical pumping for $5/\Gamma_\text{pump}$ with the collapse
       operator only ($H = 0$): $|1\rangle \to |0\rangle$ via
       spontaneous emission, resetting the spin without touching the
       phonon number.

    Operators (QuTiP conventions):

    - basis(2,0) = ground = bright state, basis(2,1) = excited = dark
      state. This is the *TIQS-internal* convention, not a universal
      one: it matches optical/shelving qubits but is inverted for
      direct-fluorescence hyperfine qubits (for $^{171}$Yb$^+$ the
      physically dark $|F{=}0\rangle$ maps onto TIQS $|0\rangle$). See
      `tiqs.spam.measurement.fluorescence_probabilities` for the full
      note.
    - sigmap() = $|0\rangle\langle 1|$ (de-excitation)
    - sigmam() = $|1\rangle\langle 0|$ (excitation)
    - RSB: $H = \frac{\eta\Omega}{2}
      (\sigma_-\,a + \sigma_+\,a^\dagger)$
    - Optical pumping: $\sqrt{\Gamma_\text{pump}}\,\sigma_+$

    Two exact properties of this operator pair make the pulsed
    bookkeeping meaningful: $[H, n + |1\rangle\langle 1|] = 0$, so a
    coherent RSB pulse removes at most one quantum, and the pumping
    dissipator's adjoint action on $n$ vanishes identically, so
    pumping changes no phonon populations.

    Model scope: an idealized RSB-only cooling channel. There is no
    off-resonant carrier or blue-sideband coupling (which would need
    the mode frequency, and whose Lorentzian suppression produces
    the cooling limit of Leibfried RMP Eqs. 109-112) and no
    $\tilde{\eta}$-scaled recoil heating on the repump. So
    $|0, n=0\rangle$ is an exact dark state, no heating channel
    exists, and the $(\Gamma_\text{eff}/2\omega)^2$ floor returned
    by ``sideband_cooling_nbar`` is *not* reproduced. Nor is the
    resolved-sideband condition
    $\Gamma_\text{pump}, \eta\Omega \ll \omega$ checked; the mode
    frequency does not enter this model at all.

    What does limit this model is the fixed pulse duration. The RSB
    coupling grows as $\eta\Omega\sqrt{n}$, so a $t_\pi$ calibrated
    for $|0,1\rangle \to |1,0\rangle$ transfers
    $\sin^2(\pi\sqrt{n}/2)$ out of $|0,n\rangle$ and leaves the
    Fock states with $\sqrt{n}$ even ($n = 4, 16, 36, \ldots$)
    completely dark - they are rotated by a multiple of $2\pi$ and
    never cool. Starting from $\bar{n} = 3$ the sequence therefore
    stalls near $\bar{n} = 1.4$, with $\approx 31\%$ of the
    population parked at $n = 4$. Real sequences avoid this by
    varying the pulse duration between cycles (or by pumping
    continuously during the drive, which is the distinct
    continuous-RSC technique).

    Parameters
    ----------
    ops : OperatorFactory
    ion : int
    mode : int
    n_bar_initial : float
        Initial mean phonon number (from Doppler cooling).
    eta : float
        Lamb-Dicke parameter. Only $|\eta|$ matters: the sign is a
        gauge choice (the parity transformation $a \to -a$ leaves
        every observable here invariant) and normal-mode
        eigenvectors carry an arbitrary sign.
    rabi_frequency : float
        Bare Rabi frequency (rad/s).
    optical_pumping_rate : float
        Optical pumping rate (1/s).
    n_cycles : int
        Number of cooling cycles.

    Returns
    -------
    float
        Final mean phonon number, clamped at 0 (nothing in this
        model heats, so integrator noise near a phonon-free state
        can otherwise return a small negative value).

    Raises
    ------
    ValueError
        If ``eta * rabi_frequency`` is zero (no coupling, hence no
        $\pi$-pulse), ``optical_pumping_rate`` is not positive, or
        ``n_cycles`` is less than 1.
    """
    rsb_rabi = eta * rabi_frequency
    if rsb_rabi == 0:
        raise ValueError(
            "eta * rabi_frequency is zero: a vanishing red-sideband"
            " coupling cannot perform a pi-pulse"
        )
    if optical_pumping_rate <= 0:
        raise ValueError(
            f"optical_pumping_rate must be > 0, got {optical_pumping_rate}"
        )
    if n_cycles < 1:
        raise ValueError(f"n_cycles must be >= 1, got {n_cycles}")

    sp = ops.sigma_plus(ion)  # sigmap = |0><1|
    sm = ops.sigma_minus(ion)  # sigmam = |1><0|
    a = ops.annihilate(mode)
    ad = ops.create(mode)
    n_op = ops.number(mode)

    # RSB Hamiltonian: |0,n> <-> |1,n-1>
    # sm*a takes |0,n> -> |1,n-1>, sp*ad is the hermitian conjugate
    H_rsb = (rsb_rabi / 2) * (sm * a + sp * ad)
    H_zero = qutip.qzero_like(H_rsb)
    # Optical pumping: dissipative |1> -> |0> using sigmap = |0><1|
    c_pump = [np.sqrt(optical_pumping_rate) * sp]

    sf = StateFactory(ops.hs)
    n_bar = [
        n_bar_initial if m == mode else 0.0 for m in range(ops.hs.n_modes)
    ]
    rho = sf.thermal_state(n_bar=n_bar)

    t_pi = np.pi / abs(rsb_rabi)
    t_pump = 5.0 / optical_pumping_rate  # e^-5 = 0.7% residual

    pulse_span = [0.0, t_pi]
    pump_span = [0.0, t_pump]
    for _ in range(n_cycles):
        rho = qutip.mesolve(H_rsb, rho, pulse_span).states[-1]
        pumped = qutip.mesolve(H_zero, rho, pump_span, c_ops=c_pump)
        rho = pumped.states[-1]

    return max(0.0, float(qutip.expect(n_op, rho)))

"""Molmer-Sorensen entangling gate Hamiltonian construction."""

from collections.abc import Callable

from tiqs.constants import TWO_PI
from tiqs.hilbert_space.operators import OperatorFactory


def _geometric_phase_hamiltonian(
    ops: OperatorFactory,
    ions: list[int],
    mode: int,
    eta: list[float],
    rabi_frequency: float,
    detuning: float,
    spin_op: Callable[[int], object],
) -> list:
    """Build a spin-dependent force Hamiltonian in the interaction picture.

    Shared implementation for MS ($\\sigma_x$) and light-shift
    ($\\sigma_z$) gates, which differ only in the spin operator::

        H(t) = sum_j eta_j Omega S_j
               (a^dag e^{+i delta t} + a e^{-i delta t})

    Pairing $a^\\dagger$ with $e^{+i\\delta t}$ fixes the sign of the
    geometric phase to $+\\mathrm{sign}(\\delta)$; see
    ``ms_gate_hamiltonian`` for the full convention statement.

    Parameters
    ----------
    ops : OperatorFactory
    ions : list[int]
    mode : int
    eta : list[float]
    rabi_frequency : float
    detuning : float
    spin_op : callable
        Function mapping ion index to the spin operator
        (e.g., ``ops.sigma_x`` or ``ops.sigma_z``).

    Returns
    -------
    list
        QuTiP list-format Hamiltonian.
    """
    a = ops.annihilate(mode)
    ad = ops.create(mode)
    H_terms = []
    for j, ion_idx in enumerate(ions):
        s_j = spin_op(ion_idx)
        coupling = eta[j] * rabi_frequency
        H_terms.append([coupling * ad * s_j, f"exp(1j*{detuning}*t)"])
        H_terms.append([coupling * a * s_j, f"exp(-1j*{detuning}*t)"])
    return H_terms


def ms_gate_duration(detuning: float, loops: int = 1) -> float:
    r"""Gate time for the MS gate:
    $\tau = 2\pi K / |\delta|$ where $K$ is the number of loops.

    The absolute value matters: the phase-space loop closes after
    $|\delta|\tau = 2\pi K$ regardless of which side of the sideband
    the tones sit on, so a negative ``detuning`` gives the same
    positive duration (only the sign of the geometric phase flips).

    Parameters
    ----------
    detuning : float
        Sideband detuning $\delta$ (rad/s). Must be nonzero.
    loops : int
        Number of phase-space loops ($K \ge 1$). More loops = slower
        but more robust.

    Returns
    -------
    float
        Gate duration in seconds.

    Raises
    ------
    ValueError
        If ``loops < 1`` (a zero-length gate is not a gate) or
        ``detuning == 0`` (the loop never closes).
    """
    if loops < 1:
        raise ValueError(f"loops must be >= 1, got {loops}")
    if detuning == 0:
        raise ValueError(
            "detuning must be nonzero: the phase-space loop never"
            " closes on the sideband resonance"
        )
    return TWO_PI * loops / abs(detuning)


def ms_gate_hamiltonian(
    ops: OperatorFactory,
    ions: list[int],
    mode: int,
    eta: list[float],
    rabi_frequency: float,
    detuning: float,
) -> list:
    r"""Construct the Molmer-Sorensen gate Hamiltonian in the interaction
    picture.

    The MS Hamiltonian for $N$ ions coupled to one motional mode:

    $$
    H_\mathrm{MS}(t) = \sum_j \eta_j \, \Omega \, \sigma_{x,j}
    \left( a^\dagger e^{i \delta t} + a \, e^{-i \delta t} \right)
    $$

    This is a spin-dependent force that displaces the motional state
    conditioned on the collective spin. After time
    $\tau = 2\pi K / |\delta|$, the motion returns to its initial
    state and the spins acquire a geometric phase proportional to the
    enclosed phase-space area.

    For two identically-coupled ions, the maximally entangling
    condition is $\eta \Omega = \delta / 4$ (single loop).

    Conventions
    -----------
    **Amplitude.** This implementation absorbs the factor of $1/2$
    from the theory-doc convention
    ($\frac{\hbar\eta\Omega}{2}\,\sigma_\phi\,[\ldots]$) into the
    definition of $\Omega$, so ``rabi_frequency`` is *half* the
    per-tone resonant carrier Rabi frequency of the bichromatic drive:
    $\Omega_\mathrm{code} = \Omega_\mathrm{carrier}/2$, and the
    spin-motion coupling is $\eta\,\Omega_\mathrm{code}$. Because the
    geometric phase goes as $\Omega^2$, feeding a measured per-tone
    carrier Rabi frequency in directly overshoots the entangling phase
    by $4\times$. ``SimulationRunner.run_ms_gate`` calibrates $\Omega$
    automatically.

    **Sign / tone placement.** Pairing $a^\dagger$ with
    $e^{+i\delta t}$ places the two tones *inside* the sidebands,
    $\omega_\pm = \omega_0 \pm (\omega_p - \delta)$, which is
    Sorensen & Molmer's own placement (PRA 62, 022311 (2000), Eqs. 4
    and 9). The second-order Magnus term is then
    $+i S^2 (\delta\tau - \sin\delta\tau)/\delta^2$, so the geometric
    phase is $\chi = +4\pi K \eta_i \eta_j \Omega^2 / \delta^2$ and

    $$
    U_\mathrm{MS} = e^{+i\chi\,\sigma_x^{(i)}\sigma_x^{(j)}},
    \qquad
    U_\mathrm{MS}|00\rangle
      = \bigl(|00\rangle + i\,s\,|11\rangle\bigr)/\sqrt{2}
    $$

    at $\chi = \pi/4$, where
    $s = \mathrm{sign}(\delta\,\eta_i\,\eta_j)$. A mode on which the
    two ions have opposite participation ($\eta_i \eta_j < 0$), or a
    negative ``detuning``, therefore produces the *conjugate* Bell
    state. ``analysis.fidelity.bell_state_fidelity`` targets $s = +1$
    by default.

    Model scope and approximations
    ------------------------------
    The returned Hamiltonian is the *idealized* spin-dependent force,
    so fidelities computed from it are upper bounds. Omitted:

    - **Off-resonant carrier.** The full first-order-in-$\eta$
      bichromatic Hamiltonian is
      $\Omega_\mathrm{carrier}\cos(\mu t)\sum_j[\sigma_x^{(j)}
      - \eta_j X(t)\,\sigma_y^{(j)}]$; only the slow half of the
      $\eta$ term is kept. The dropped zeroth-order ("1") term lies
      along the axis orthogonal to the force, so it does *not*
      commute with it. This function takes no mode frequency, so the
      result is exactly independent of $\omega_p$: the carrier error
      is inexpressible, not merely small. It is negligible for slow
      gates ($\sim 2\times 10^{-7}$ at $\delta = 2\pi\,$1 kHz) but
      reaches $10^{-4}$ to $10^{-2}$ for fast gates
      (arXiv:2501.02387), where hardware compensates it with pulse
      shaping.
    - **Spectator modes.** A single scalar ``mode`` is used. The
      propagator factorizes per mode (Sorensen & Molmer 2000,
      Sec. IV), so closure must hold for *every* mode; a two-ion
      chain with one uncompensated spectator costs $\sim 3\times
      10^{-4}$ Bell infidelity at $\delta = 2\pi\,$23 kHz. Callers
      who need this can concatenate the lists returned for several
      modes, but nothing here solves the simultaneous multi-mode
      closure and phase conditions, and no multi-mode Ising $J_{j,k}$
      is implemented.
    - **Pulse shaping.** Amplitude is constant (square pulse); there
      is no AM/FM/phase modulation and no Walsh basis, so ``loops``
      only rescales $\tau$ (and $\Omega \propto 1/\sqrt{K}$). The
      robust-gate techniques that suppress the two errors above
      cannot be represented.
    - **Debye-Waller factor.** The coupling is strictly linear in
      $\eta$ with unit Debye-Waller factor, i.e. the exact prefactor
      $e^{-\eta^2/2} L_n(\eta^2)$ is replaced by 1 and no
      spectator-mode occupation modulates it. Consequently the
      simulated gate is *exactly* insensitive to the initial motional
      state (the closed loop makes the propagator purely spin at
      $\tau$), and the validity condition $\eta\sqrt{\bar n} \ll 1$
      cannot be seen to fail. Realistic magnitude in this repo's
      axial geometry: $\delta\Omega/\Omega \sim 10^{-2}$ at
      $\bar n = 0.05$, i.e. MS infidelity $\sim 10^{-4}$
      (Wineland et al., J. Res. NIST 103, 259 (1998), Eq. 128).
    - **AC Stark shifts** from the off-resonant tones, and the
      counter-rotating sideband halves at $\pm(\mu + \omega_p)$.

    Parameters
    ----------
    ops : OperatorFactory
        Operator factory for the composite Hilbert space.
    ions : list[int]
        Indices of the ions to entangle.
    mode : int
        Index of the motional mode used as the bus.
    eta : list[float]
        Lamb-Dicke parameters for each ion on this mode.
    rabi_frequency : float
        Half the per-tone carrier Rabi frequency $\Omega$ (rad/s) of
        the bichromatic drive on each ion; see Conventions above.
    detuning : float
        Detuning $\delta$ (rad/s) from the motional sideband, with the
        tones placed inside the sidebands.

    Returns
    -------
    list
        QuTiP list-format Hamiltonian: ``[[H_op, coeff_string], ...]``.
    """
    return _geometric_phase_hamiltonian(
        ops, ions, mode, eta, rabi_frequency, detuning, ops.sigma_x
    )

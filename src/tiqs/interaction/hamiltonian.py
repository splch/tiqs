"""Laser-ion interaction Hamiltonians: carrier, sidebands, full."""

import numpy as np
import qutip

from tiqs.hilbert_space.operators import OperatorFactory


def carrier_hamiltonian(
    ops: OperatorFactory,
    ion: int,
    rabi_frequency: float,
    phase: float = 0.0,
) -> qutip.Qobj:
    r"""Carrier transition Hamiltonian.

    $$
    H = \frac{\Omega}{2}\bigl(\sigma_- e^{i\phi} + \sigma_+ e^{-i\phi}\bigr)
      = \frac{\Omega}{2}\bigl(\sigma_x\cos\phi + \sigma_y\sin\phi\bigr)
    $$

    Drives $|0\rangle \leftrightarrow |1\rangle$ without changing
    the motional state.

    TIQS convention: $|0\rangle$ is the ground state, so
    `ops.sigma_minus` $= |1\rangle\langle 0|$ is the *excitation*
    operator and is the one carrying $e^{+i\phi}$ - the same
    assignment used by the sideband Hamiltonians below and by
    `tiqs.gates.single_qubit`. The second form above uses QuTiP's
    $\sigma_y$ and makes $\phi$ the azimuth of the rotation axis in
    the $x$-$y$ plane.

    Parameters
    ----------
    ops : OperatorFactory
        Operator factory for the Hilbert space.
    ion : int
        Index of the target ion.
    rabi_frequency : float
        Rabi frequency $\Omega$ in rad/s.
    phase : float, optional
        Laser phase $\phi$ in radians.

    Returns
    -------
    qutip.Qobj
        The carrier Hamiltonian operator.
    """
    sp = ops.sigma_plus(ion)
    sm = ops.sigma_minus(ion)
    return (rabi_frequency / 2) * (
        sm * np.exp(1j * phase) + sp * np.exp(-1j * phase)
    )


def red_sideband_hamiltonian(
    ops: OperatorFactory,
    ion: int,
    mode: int,
    rabi_frequency: float,
    eta: float,
    phase: float = 0.0,
) -> qutip.Qobj:
    r"""Red sideband Hamiltonian.

    $$
    H = \frac{\eta\,\Omega}{2}\bigl(a\,\sigma_- e^{i\phi}
        + a^\dagger\,\sigma_+ e^{-i\phi}\bigr)
    $$

    Drives
    $|0,n\rangle \leftrightarrow |1,n-1\rangle$:
    excites the qubit while removing one phonon.

    In QuTiP's convention
    $\sigma_- = |1\rangle\langle 0|$ takes
    $|0\rangle \to |1\rangle$ (excitation), and
    $a$ removes a phonon, so the coupling term is
    $a\,\sigma_- + \text{h.c.}$

    This is the resonant (static) form obtained after dropping the
    off-resonant carrier and blue sideband. The factor $i$ that the
    Lamb-Dicke expansion attaches to every odd sideband order
    (Wineland et al., J. Res. NIST 103, 259 (1998), Eq. (23): the
    coupling phase of an order-$s$ sideband is $\phi + s\pi/2$) is
    absorbed into `phase` here, so `phase` is offset by $\pi/2$
    relative to the carrier phase. Use
    `full_interaction_hamiltonian` when carrier and sidebands must
    share one phase reference.

    Parameters
    ----------
    ops : OperatorFactory
        Operator factory for the Hilbert space.
    ion : int
        Index of the target ion.
    mode : int
        Index of the motional mode.
    rabi_frequency : float
        Rabi frequency $\Omega$ in rad/s.
    eta : float
        Lamb-Dicke parameter for this ion-mode pair.
    phase : float, optional
        Laser phase in radians.

    Returns
    -------
    qutip.Qobj
        The red-sideband Hamiltonian operator.
    """
    sp = ops.sigma_plus(ion)
    sm = ops.sigma_minus(ion)
    a = ops.annihilate(mode)
    ad = ops.create(mode)
    return (eta * rabi_frequency / 2) * (
        a * sm * np.exp(1j * phase) + ad * sp * np.exp(-1j * phase)
    )


def blue_sideband_hamiltonian(
    ops: OperatorFactory,
    ion: int,
    mode: int,
    rabi_frequency: float,
    eta: float,
    phase: float = 0.0,
) -> qutip.Qobj:
    r"""Blue sideband Hamiltonian.

    $$
    H = \frac{\eta\,\Omega}{2}\bigl(a^\dagger\,\sigma_- e^{i\phi}
        + a\,\sigma_+ e^{-i\phi}\bigr)
    $$

    Drives
    $|0,n\rangle \leftrightarrow |1,n+1\rangle$:
    excites the qubit while adding one phonon.

    In QuTiP's convention
    $\sigma_- = |1\rangle\langle 0|$ takes
    $|0\rangle \to |1\rangle$ (excitation), and
    $a^\dagger$ adds a phonon, so the coupling
    term is
    $a^\dagger\,\sigma_- + \text{h.c.}$

    As for the red sideband, this is the resonant (static) form and
    the Lamb-Dicke factor $i$ is absorbed into `phase`, which is
    therefore offset by $\pi/2$ from the carrier phase.

    Parameters
    ----------
    ops : OperatorFactory
        Operator factory for the Hilbert space.
    ion : int
        Index of the target ion.
    mode : int
        Index of the motional mode.
    rabi_frequency : float
        Rabi frequency $\Omega$ in rad/s.
    eta : float
        Lamb-Dicke parameter for this ion-mode pair.
    phase : float, optional
        Laser phase in radians.

    Returns
    -------
    qutip.Qobj
        The blue-sideband Hamiltonian operator.
    """
    sp = ops.sigma_plus(ion)
    sm = ops.sigma_minus(ion)
    a = ops.annihilate(mode)
    ad = ops.create(mode)
    return (eta * rabi_frequency / 2) * (
        ad * sm * np.exp(1j * phase) + a * sp * np.exp(-1j * phase)
    )


def _conjugate_pair(op: qutip.Qobj, offset: float) -> list:
    r"""QuTiP list entries for a term plus its Hermitian conjugate.

    The pair is $A\,e^{-i\Delta t} + A^\dagger e^{+i\Delta t}$ for
    operator $A$ and offset $\Delta$. Each half is listed separately
    because neither is Hermitian on its own; QuTiP sums the entries,
    so the pair is Hermitian at every $t$. When the offset vanishes
    the pair is time-independent and is collapsed into a single
    static operator.

    Parameters
    ----------
    op : qutip.Qobj
        The co-rotating half of the term.
    offset : float
        Oscillation frequency of the co-rotating half (rad/s); zero
        means the term is resonant.

    Returns
    -------
    list
        One static `qutip.Qobj` if ``offset == 0``, otherwise two
        ``[Qobj, coefficient_string]`` entries.
    """
    if offset == 0.0:
        return [op + op.dag()]
    return [
        [op, f"exp(-1j*{offset}*t)"],
        [op.dag(), f"exp(1j*{offset}*t)"],
    ]


def full_interaction_hamiltonian(
    ops: OperatorFactory,
    ion: int,
    mode: int,
    rabi_frequency: float,
    eta: float,
    detuning: float,
    mode_frequency: float,
    phase: float = 0.0,
    lamb_dicke_order: int = 1,
) -> list:
    r"""Full laser-ion interaction in the interaction picture.

    Returns the Hamiltonian in QuTiP list format. In the interaction
    picture with respect to the qubit and mode free evolution
    (Leibfried et al., Rev. Mod. Phys. 75, 281 (2003), Eq. (69);
    Wineland et al., J. Res. NIST 103, 259 (1998), Eq. (16)):

    $$
    H(t) = \frac{\Omega}{2}\,\sigma_-\,e^{i\phi}\,e^{-i\delta t}\,
        \exp\bigl[i\eta\bigl(a\,e^{-i\omega_m t}
            + a^\dagger e^{+i\omega_m t}\bigr)\bigr]
        + \text{h.c.}
    $$

    Expanding the displacement exponential to the requested
    Lamb-Dicke order (Leibfried Eq. (72)) gives the terms this
    function returns:

    $$
    \begin{aligned}
    H_\text{car} &= \frac{\Omega}{2}
        \bigl[\sigma_- e^{i\phi} e^{-i\delta t} + \text{h.c.}\bigr]\\
    H_\text{rsb} &= \frac{i\eta\Omega}{2}
        \bigl[a\,\sigma_- e^{i\phi} e^{-i(\delta + \omega_m)t}
        + \text{h.c.}\bigr]\\
    H_\text{bsb} &= \frac{i\eta\Omega}{2}
        \bigl[a^\dagger\sigma_- e^{i\phi} e^{-i(\delta - \omega_m)t}
        + \text{h.c.}\bigr]\\
    H_\text{dw} &= -\frac{\eta^2\Omega}{2}
        \bigl(a^\dagger a + \tfrac{1}{2}\bigr)
        \bigl[\sigma_- e^{i\phi} e^{-i\delta t} + \text{h.c.}\bigr]\\
    H_\text{2rsb} &= -\frac{\eta^2\Omega}{4}
        \bigl[a^2\sigma_- e^{i\phi} e^{-i(\delta + 2\omega_m)t}
        + \text{h.c.}\bigr]\\
    H_\text{2bsb} &= -\frac{\eta^2\Omega}{4}
        \bigl[(a^\dagger)^2\sigma_- e^{i\phi}
        e^{-i(\delta - 2\omega_m)t} + \text{h.c.}\bigr]
    \end{aligned}
    $$

    With $\delta = \omega_L - \omega_0$ the resonances sit at
    $\delta = 0$ (carrier), $\delta = -\omega_m$ / $+\omega_m$
    (first red / blue sideband) and $\delta = -2\omega_m$ /
    $+2\omega_m$ (second red / blue sideband). $H_\text{dw}$ is the
    Debye-Waller reduction of the carrier Rabi frequency,
    $\Omega_n \approx \Omega[1 - \eta^2(2n+1)/2]$.

    The factor $i$ on the odd orders is the $\pi/2$-per-order
    coupling phase of Wineland Eq. (23). It cannot be absorbed into
    `phase` here, because carrier and sidebands share one phase
    reference in this Hamiltonian.

    Model scope and approximations: one ion and one mode; the
    Lamb-Dicke expansion is truncated at `lamb_dicke_order` but no
    rotating-wave approximation is applied on top of it, so
    off-resonant carrier/sideband terms (and hence their AC Stark
    shifts) are retained. Not included: spectator modes and the
    multi-mode Debye-Waller product
    $\prod_q e^{-\eta_q^2(2n_q+1)/2}$, light shifts from levels
    outside the two-level qubit (see
    `tiqs.interaction.raman.RamanPair.ac_stark_shift`), spontaneous
    emission, and laser amplitude/phase noise.

    Parameters
    ----------
    ops : OperatorFactory
        Operator factory for the Hilbert space.
    ion : int
        Index of the target ion.
    mode : int
        Index of the motional mode.
    rabi_frequency : float
        Rabi frequency $\Omega$ (rad/s).
    eta : float
        Lamb-Dicke parameter for this ion-mode pair.
    detuning : float
        Laser detuning $\delta = \omega_L - \omega_0$ from the qubit
        resonance (rad/s). Negative is red-detuned:
        $\delta = -\omega_m$ is the first red sideband.
    mode_frequency : float
        Motional mode frequency $\omega_m$ (rad/s). Must be a
        **positive-energy** mode: the red sideband is placed at
        $\delta = -\omega_m$ and the blue at $+\omega_m$, which is
        exchanged for the negative-energy Penning magnetron mode
        (`tiqs.trap.PenningTrap.omega_magnetron`). Passing an unsigned
        magnetron frequency here drives the opposite sideband.
    phase : float
        Laser phase $\phi$ (radians), carried by the excitation
        operator $\sigma_-$ as $e^{+i\phi}$.
    lamb_dicke_order : int
        Order of the Lamb-Dicke expansion: 1 (carrier and first
        sidebands) or 2 (adds Debye-Waller and both second
        sidebands).

    Returns
    -------
    list
        QuTiP list-format Hamiltonian. Each of the 3
        (`lamb_dicke_order=1`) or 6 (`lamb_dicke_order=2`) terms
        above contributes a conjugate pair of entries
        ``[Qobj, coefficient_string]``, in the order carrier, red
        sideband, blue sideband, Debye-Waller, second red, second
        blue - except that a term whose oscillation frequency is
        exactly zero is collapsed into a single static `qutip.Qobj`.
        So a first-order Hamiltonian has 6 entries and a
        second-order one 12, minus one entry for every resonant term
        (5 and 10 on the carrier resonance, where carrier and
        Debye-Waller are both static).

    Raises
    ------
    ValueError
        If `lamb_dicke_order` is neither 1 nor 2.
    """
    if lamb_dicke_order not in (1, 2):
        raise ValueError(
            "lamb_dicke_order must be 1 or 2, got "
            f"{lamb_dicke_order}; higher orders are not implemented"
        )

    sm = ops.sigma_minus(ion)
    a = ops.annihilate(mode)
    ad = ops.create(mode)
    drive = (rabi_frequency / 2) * np.exp(1j * phase)

    # Carrier: zeroth order in eta, resonant at delta = 0.
    H_terms = _conjugate_pair(drive * sm, detuning)

    # First order: i*eta*(a e^{-i*wm*t} + ad e^{+i*wm*t}).
    # a * sm removes a phonon and excites the qubit (red sideband),
    # so it is resonant at delta = -wm.
    H_terms += _conjugate_pair(
        1j * eta * drive * a * sm, detuning + mode_frequency
    )
    H_terms += _conjugate_pair(
        1j * eta * drive * ad * sm, detuning - mode_frequency
    )

    if lamb_dicke_order == 2:
        # (i*eta)^2/2! * (a e^{-i*wm*t} + ad e^{+i*wm*t})^2
        #   = -eta^2/2 * (a^2 e^{-2i*wm*t} + ad^2 e^{+2i*wm*t} + 2n + 1)
        second_order = -(eta**2) / 2 * drive
        n_op = ops.number(mode)
        # Debye-Waller: carrier Rabi frequency reduced by eta^2*(n+1/2).
        H_terms += _conjugate_pair(
            second_order * (2 * n_op + ops.identity()) * sm, detuning
        )
        H_terms += _conjugate_pair(
            second_order * a * a * sm, detuning + 2 * mode_frequency
        )
        H_terms += _conjugate_pair(
            second_order * ad * ad * sm, detuning - 2 * mode_frequency
        )

    return H_terms

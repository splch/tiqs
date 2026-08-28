"""Single-qubit gate implementations: bare rotations and composite
pulse sequences."""

from dataclasses import dataclass

import numpy as np
import qutip

from tiqs.constants import TWO_PI
from tiqs.hilbert_space.operators import OperatorFactory


@dataclass
class GatePulse:
    """A gate operation defined by a Hamiltonian and a duration.

    For composite gates, `pulses` contains the sequential
    (Hamiltonian, duration) pairs to be applied in order.

    .. warning::
       The pair ``(hamiltonian, duration)`` describes the whole gate
       **only when** ``pulses is None``. For a composite gate
       ``hamiltonian`` is the *first segment only* while ``duration``
       is the *total* of all segments, so evolving under
       ``(hamiltonian, duration)`` yields a bare over-rotation rather
       than the composite sequence. Callers must branch on
       ``pulses``::

           if gate.pulses is None:
               segments = [(gate.hamiltonian, gate.duration)]
           else:
               segments = gate.pulses

    Attributes
    ----------
    hamiltonian : qutip.Qobj or list
        The Hamiltonian operator for a simple gate; for a composite
        gate, the Hamiltonian of the *first* segment only (kept as a
        representative for introspection, not for evolution).
    duration : float
        Total gate duration in seconds, summed over all segments.
    pulses : list of tuple or None
        For composite gates, a list of ``(Hamiltonian, duration)``
        pairs applied sequentially. ``None`` for simple gates, which
        is the only case where ``(hamiltonian, duration)`` is a
        complete description.
    """

    hamiltonian: qutip.Qobj | list
    duration: float
    pulses: list[tuple] | None = None


def rx_gate(
    ops: OperatorFactory,
    ion: int,
    theta: float,
    rabi_frequency: float = TWO_PI * 1e6,
) -> GatePulse:
    r"""Rotation about X by angle $\theta$.

    $R_x(\theta) = e^{-i \theta \sigma_x / 2}$,
    implemented as
    $H = \mathrm{sign}(\theta)\,\frac{\Omega}{2}\sigma_x$
    for time $t = |\theta| / \Omega$.

    Parameters
    ----------
    ops : OperatorFactory
        Operator factory for the Hilbert space.
    ion : int
        Index of the target ion.
    theta : float
        Rotation angle in radians.
    rabi_frequency : float, optional
        Rabi frequency $\Omega$ in rad/s.

    Returns
    -------
    GatePulse
        Gate with the X-rotation Hamiltonian and duration.
    """
    sign = 1 if theta >= 0 else -1
    H = sign * (rabi_frequency / 2) * ops.sigma_x(ion)
    duration = abs(theta) / rabi_frequency
    return GatePulse(hamiltonian=H, duration=duration)


def ry_gate(
    ops: OperatorFactory,
    ion: int,
    theta: float,
    rabi_frequency: float = TWO_PI * 1e6,
) -> GatePulse:
    r"""Rotation about Y by angle $\theta$.

    $R_y(\theta) = e^{-i \theta \sigma_y / 2}$.

    Parameters
    ----------
    ops : OperatorFactory
        Operator factory for the Hilbert space.
    ion : int
        Index of the target ion.
    theta : float
        Rotation angle in radians.
    rabi_frequency : float, optional
        Rabi frequency $\Omega$ in rad/s.

    Returns
    -------
    GatePulse
        Gate with the Y-rotation Hamiltonian and duration.
    """
    sign = 1 if theta >= 0 else -1
    H = sign * (rabi_frequency / 2) * ops.sigma_y(ion)
    duration = abs(theta) / rabi_frequency
    return GatePulse(hamiltonian=H, duration=duration)


def rz_gate(
    ops: OperatorFactory,
    ion: int,
    phi: float,
    rabi_frequency: float = TWO_PI * 1e6,
) -> GatePulse:
    r"""Rotation about Z by angle $\phi$.

    $R_z(\phi) = e^{-i \phi \sigma_z / 2}$, implemented as
    $H = \mathrm{sign}(\phi)\,\frac{\Omega}{2}\sigma_z$ for time
    $t = |\phi| / \Omega$.

    .. note::
       A resonant carrier drive generates only *equatorial* rotations
       (see ``_rotation_hamiltonian``), so no laser pulse produces
       this $\sigma_z$ generator directly. On hardware $R_z$ is
       normally **virtual**: a phase offset applied to every
       subsequent pulse, with zero duration and zero error. The
       physical alternative is an off-resonant AC-Stark-shift pulse,
       whose precession rate is the differential light shift, which is
       far smaller than a carrier Rabi frequency. This function models
       $H = \pm(\Omega/2)\sigma_z$ directly as a simulation
       convenience, so ``rabi_frequency`` should be read as "the
       $\sigma_z$ precession rate", not as a Rabi frequency, and the
       returned ``duration`` is not a hardware gate time.

    Parameters
    ----------
    ops : OperatorFactory
        Operator factory for the Hilbert space.
    ion : int
        Index of the target ion.
    phi : float
        Rotation angle in radians.
    rabi_frequency : float, optional
        $\sigma_z$ precession rate in rad/s (see the note above).

    Returns
    -------
    GatePulse
        Gate with the Z-rotation Hamiltonian and duration.
    """
    sign = 1 if phi >= 0 else -1
    H = sign * (rabi_frequency / 2) * ops.sigma_z(ion)
    duration = abs(phi) / rabi_frequency
    return GatePulse(hamiltonian=H, duration=duration)


def _rotation_hamiltonian(
    ops: OperatorFactory,
    ion: int,
    phase: float,
    rabi_frequency: float,
) -> qutip.Qobj:
    r"""Hamiltonian for a rotation about an axis in the x-y plane.

    The axis is set by the angle `phase`:

    $$
    H = \frac{\Omega}{2}
      \bigl(\sigma_- e^{i\varphi}
      + \sigma_+ e^{-i\varphi}\bigr)
      = \frac{\Omega}{2}
      \bigl(\sigma_x \cos\varphi
      + \sigma_y \sin\varphi\bigr)
    $$

    .. note::
       In this codebase ``ops.sigma_plus`` is ``qutip.sigmap()``
       $= |0\rangle\langle 1|$, i.e. the **de-excitation** operator
       (QuTiP's raising operator in matrix convention), while
       ``ops.sigma_minus`` $= |1\rangle\langle 0|$ is the
       **excitation** operator. The repo-wide convention pairs the
       excitation operator $\sigma_-$ with $e^{+i\varphi}$, which is
       what makes the equality above hold with QuTiP's
       $\sigma_y = \bigl(\begin{smallmatrix}0 & -i\\ i &
       0\end{smallmatrix}\bigr)$. Theory docs use the textbook
       labelling $\sigma_+ = |e\rangle\langle g|$, which is this
       code's ``sigma_minus``.

    Parameters
    ----------
    ops : OperatorFactory
        Operator factory for the Hilbert space.
    ion : int
        Index of the target ion.
    phase : float
        Angle of the rotation axis in the x-y plane
        (radians).
    rabi_frequency : float
        Rabi frequency $\Omega$ in rad/s.

    Returns
    -------
    qutip.Qobj
        The rotation Hamiltonian operator.
    """
    return (rabi_frequency / 2) * (
        ops.sigma_x(ion) * np.cos(phase) + ops.sigma_y(ion) * np.sin(phase)
    )


def sk1_composite_gate(
    ops: OperatorFactory,
    ion: int,
    theta: float,
    rabi_frequency: float = TWO_PI * 1e6,
) -> GatePulse:
    r"""SK1 composite pulse sequence robust against amplitude errors.

    SK1 cancels the **first-order** amplitude error: for a fractional
    Rabi-frequency error $\epsilon$ the residual propagator error is
    $O(\epsilon^2)$, so the average gate infidelity scales as
    $\epsilon^4$ instead of the bare gate's $\epsilon^2$
    (Brown, Harrow & Chuang, PRA 70, 052318 (2004)).

    $$
    \mathrm{SK1}(\theta) = R_{\beta}(|\theta|),\;
      R_{\beta + \phi_1}(2\pi),\;
      R_{\beta - \phi_1}(2\pi)
    $$

    where $\phi_1 = \arccos\!\left(-|\theta| / 4\pi\right)$ and the
    base phase $\beta$ carries the sign of the requested angle,
    $\beta = 0$ for $\theta \ge 0$ and $\beta = \pi$ for $\theta < 0$.
    Flipping the drive axis rather than the pulse area is what makes
    $R_\pi(|\theta|) = R_x(-|\theta|)$, so the sequence reproduces
    $R_x(\theta)$ exactly (not merely up to a global phase) for either
    sign. The three rotations are all about axes in the x-y plane but
    at different phases.

    Because the first-order error generator of the sequence is
    $\bigl[|\theta|/2 + 2\pi\cos\phi_1\bigr]\,\sigma_\beta$, $\phi_1$
    must be built from the *magnitude* actually driven; using signed
    $\theta$ there doubles the generator instead of cancelling it.

    The returned GatePulse has:
    - hamiltonian: the Hamiltonian for the first segment
      ($R_\beta$), used as the "representative" Hamiltonian. The
      actual composite sequence is stored in the `pulses` attribute.
    - duration: total duration of all three pulses.

    Parameters
    ----------
    ops : OperatorFactory
        Operator factory for the Hilbert space.
    ion : int
        Index of the target ion.
    theta : float
        Signed rotation angle about X in radians, $|\theta| \le 4\pi$.
    rabi_frequency : float, optional
        Rabi frequency $\Omega$ in rad/s.

    Returns
    -------
    GatePulse
        Composite gate whose `pulses` implement $R_x(\theta)$.

    Raises
    ------
    ValueError
        If $|\theta| > 4\pi$, where $\phi_1$ has no real solution.
    """
    magnitude = abs(theta)
    if magnitude > 4 * np.pi:
        raise ValueError(
            f"SK1 requires |theta| <= 4*pi, got theta={theta:.4f}"
        )
    base_phase = 0.0 if theta >= 0 else np.pi
    phi1 = np.arccos(-magnitude / (4 * np.pi))

    t_theta = magnitude / rabi_frequency
    t_2pi = TWO_PI / rabi_frequency
    total_duration = t_theta + 2 * t_2pi

    H0 = _rotation_hamiltonian(ops, ion, base_phase, rabi_frequency)
    H1 = _rotation_hamiltonian(ops, ion, base_phase + phi1, rabi_frequency)
    H2 = _rotation_hamiltonian(ops, ion, base_phase - phi1, rabi_frequency)

    return GatePulse(
        hamiltonian=H0,
        duration=total_duration,
        pulses=[(H0, t_theta), (H1, t_2pi), (H2, t_2pi)],
    )


def bb1_composite_gate(
    ops: OperatorFactory,
    ion: int,
    theta: float,
    rabi_frequency: float = TWO_PI * 1e6,
) -> GatePulse:
    r"""BB1 (Broadband-1) composite pulse robust against amplitude
    errors.

    BB1 cancels the **first- and second-order** amplitude errors: the
    residual propagator error is $O(\epsilon^3)$, so the average gate
    infidelity scales as $\epsilon^6$
    (Wimperis, J. Magn. Reson. A 109, 221 (1994);
    Merrill & Brown, Adv. Chem. Phys. 154, 241 (2014)).
    Stated in the same convention, ``sk1_composite_gate`` cancels only
    the first order ($O(\epsilon^2)$ propagator error, $\epsilon^4$
    infidelity).

    $$
    \mathrm{BB1}(\theta) = R_{\beta}(|\theta|),\;
      R_{\beta + \phi_1}(\pi),\;
      R_{\beta + 3\phi_1}(2\pi),\;
      R_{\beta + \phi_1}(\pi)
    $$

    i.e. the target rotation plus three correction pulses. Here
    $\phi_1 = \arccos\!\left(-|\theta| / 4\pi\right)$ and the base
    phase $\beta$ carries the sign of the requested angle,
    $\beta = 0$ for $\theta \ge 0$ and $\beta = \pi$ for $\theta < 0$
    (see ``sk1_composite_gate`` for why the sign must live in the axis
    and $\phi_1$ must be built from $|\theta|$).

    Parameters
    ----------
    ops : OperatorFactory
        Operator factory for the Hilbert space.
    ion : int
        Index of the target ion.
    theta : float
        Signed rotation angle about X in radians, $|\theta| \le 4\pi$.
    rabi_frequency : float, optional
        Rabi frequency $\Omega$ in rad/s.

    Returns
    -------
    GatePulse
        Composite gate whose `pulses` implement $R_x(\theta)$.

    Raises
    ------
    ValueError
        If $|\theta| > 4\pi$, where $\phi_1$ has no real solution.
    """
    magnitude = abs(theta)
    if magnitude > 4 * np.pi:
        raise ValueError(
            f"BB1 requires |theta| <= 4*pi, got theta={theta:.4f}"
        )
    base_phase = 0.0 if theta >= 0 else np.pi
    phi1 = np.arccos(-magnitude / (4 * np.pi))

    t_theta = magnitude / rabi_frequency
    t_pi = np.pi / rabi_frequency
    t_2pi = TWO_PI / rabi_frequency
    total_duration = t_theta + 2 * t_pi + t_2pi

    H0 = _rotation_hamiltonian(ops, ion, base_phase, rabi_frequency)
    H1 = _rotation_hamiltonian(ops, ion, base_phase + phi1, rabi_frequency)
    H2 = _rotation_hamiltonian(ops, ion, base_phase + 3 * phi1, rabi_frequency)

    return GatePulse(
        hamiltonian=H0,
        duration=total_duration,
        pulses=[(H0, t_theta), (H1, t_pi), (H2, t_2pi), (H1, t_pi)],
    )

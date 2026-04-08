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
    """Rotation about X by angle theta: R_x(theta) = exp(-i*theta*sigma_x/2).

    Implemented as H = (Omega/2) * sigma_x for time t = theta/Omega.
    """
    H = (rabi_frequency / 2) * ops.sigma_x(ion)
    duration = abs(theta) / rabi_frequency
    return GatePulse(hamiltonian=H, duration=duration)


def ry_gate(
    ops: OperatorFactory,
    ion: int,
    theta: float,
    rabi_frequency: float = TWO_PI * 1e6,
) -> GatePulse:
    """Rotation about Y by angle theta:
    R_y(theta) = exp(-i*theta*sigma_y/2)."""
    H = (rabi_frequency / 2) * ops.sigma_y(ion)
    duration = abs(theta) / rabi_frequency
    return GatePulse(hamiltonian=H, duration=duration)


def rz_gate(
    ops: OperatorFactory,
    ion: int,
    phi: float,
    rabi_frequency: float = TWO_PI * 1e6,
) -> GatePulse:
    """Rotation about Z by angle phi: R_z(phi) = exp(-i*phi*sigma_z/2)."""
    H = (rabi_frequency / 2) * ops.sigma_z(ion)
    duration = abs(phi) / rabi_frequency
    return GatePulse(hamiltonian=H, duration=duration)


def _rotation_hamiltonian(
    ops: OperatorFactory,
    ion: int,
    phase: float,
    rabi_frequency: float,
) -> qutip.Qobj:
    """Hamiltonian for a rotation about an axis in the x-y plane at
    angle `phase`.

    H = (Omega/2) * (sigma_+ * e^{i*phase} + sigma_- * e^{-i*phase})
      = (Omega/2) * (sigma_x * cos(phase) + sigma_y * sin(phase))
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
    """SK1 composite pulse sequence that compensates amplitude errors to
    first order.

    SK1(theta) = R_0(theta), then R_{phi1}(2*pi), then R_{-phi1}(2*pi)

    where phi1 = arccos(-theta/(4*pi)).
    The three rotations are all about axes in the x-y plane but at
    different phases.

    The returned GatePulse has:
    - hamiltonian: the Hamiltonian for the first segment (R_0), used as
      the "representative" Hamiltonian. The actual composite sequence is
      stored in the `pulses` attribute.
    - duration: total duration of all three pulses.
    """
    arg = -theta / (4 * np.pi)
    if abs(arg) > 1:
        raise ValueError(
            f"SK1 requires |theta| <= 4*pi, got theta={theta:.4f}"
        )
    phi1 = np.arccos(arg)

    t_theta = abs(theta) / rabi_frequency
    t_2pi = TWO_PI / rabi_frequency
    total_duration = t_theta + 2 * t_2pi

    H0 = _rotation_hamiltonian(ops, ion, 0.0, rabi_frequency)
    H1 = _rotation_hamiltonian(ops, ion, phi1, rabi_frequency)
    H2 = _rotation_hamiltonian(ops, ion, -phi1, rabi_frequency)

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
    """BB1 (Broadband-1) composite pulse: compensates amplitude errors to
    second order.

    BB1(theta) = R_0(theta), R_{phi1}(pi), R_{3*phi1}(2*pi), R_{phi1}(pi)
    where phi1 = arccos(-theta/(4*pi)).
    """
    arg = -theta / (4 * np.pi)
    if abs(arg) > 1:
        raise ValueError(
            f"BB1 requires |theta| <= 4*pi, got theta={theta:.4f}"
        )
    phi1 = np.arccos(arg)

    t_theta = abs(theta) / rabi_frequency
    t_pi = np.pi / rabi_frequency
    t_2pi = TWO_PI / rabi_frequency
    total_duration = t_theta + 2 * t_pi + t_2pi

    H0 = _rotation_hamiltonian(ops, ion, 0.0, rabi_frequency)
    H1 = _rotation_hamiltonian(ops, ion, phi1, rabi_frequency)
    H2 = _rotation_hamiltonian(ops, ion, 3 * phi1, rabi_frequency)

    return GatePulse(
        hamiltonian=H0,
        duration=total_duration,
        pulses=[(H0, t_theta), (H1, t_pi), (H2, t_2pi), (H1, t_pi)],
    )

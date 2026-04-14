r"""Time-varying pulse waveforms for trapped-ion gates.

Provides waveform types, a :class:`Pulse` dataclass, coefficient
rendering for QuTiP, preset constructors (smooth gate, AM gate),
and numerical loop-closure verification.

The smooth gate (arXiv:2510.17286) achieved 99.99% two-qubit
fidelity without ground-state cooling by ramping the detuning
adiabatically.  With time-varying detuning $\delta(t)$, the
oscillating phase in the interaction picture becomes

$$
\varphi(t) = \int_0^t \delta(t')\,dt'
$$

and the Hamiltonian coefficient is
$\Omega(t)\,\exp\!\bigl(\pm i\,\varphi(t)\bigr)$.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from scipy.integrate import cumulative_trapezoid, trapezoid

from tiqs.constants import TWO_PI


class Waveform(ABC):
    """A scalar function of time over a gate duration."""

    @abstractmethod
    def evaluate(
        self, t: float | np.ndarray, tau: float
    ) -> float | np.ndarray:
        """Evaluate at time(s) *t* for total duration *tau*."""
        ...

    def as_string(self, tau: float) -> str | None:
        """QuTiP string expression, or ``None`` if not expressible."""
        return None


class ConstantWaveform(Waveform):
    """A constant value (identity element for waveform composition)."""

    def __init__(self, value: float):
        self.value = value

    def evaluate(self, t, tau):
        if isinstance(t, np.ndarray):
            return np.full_like(t, self.value, dtype=float)
        return self.value

    def as_string(self, tau):
        return str(self.value)


class SinusoidalRamp(Waveform):
    r"""Sinusoidal detuning ramp for the smooth gate.

    $$\delta(t) = \delta_0 + A\,\cos(2\pi t/\tau)$$

    At $t=0$ and $t=\tau$: far-detuned ($\delta_0 + A$).
    At $t=\tau/2$: closest to resonance ($\delta_0 - A$).

    The integrated phase has a closed form:

    $$\varphi(t) = \delta_0\,t
      + \frac{A\,\tau}{2\pi}\,\sin(2\pi t/\tau)$$
    """

    def __init__(self, delta_0: float, amplitude: float):
        self.delta_0 = delta_0
        self.amplitude = amplitude

    def evaluate(self, t, tau):
        return self.delta_0 + self.amplitude * np.cos(TWO_PI * t / tau)

    def as_string(self, tau):
        omega_ramp = TWO_PI / tau
        return f"({self.delta_0} + {self.amplitude}*cos({omega_ramp}*t))"

    def integrated_phase_string(self, tau: float) -> str:
        """Closed-form string for the integrated phase."""
        omega_ramp = TWO_PI / tau
        coeff = self.amplitude / omega_ramp
        return f"({self.delta_0}*t + {coeff}*sin({omega_ramp}*t))"


class BlackmanWindow(Waveform):
    """Blackman window envelope for Rabi frequency."""

    def __init__(self, peak_value: float):
        self.peak_value = peak_value

    def evaluate(self, t, tau):
        x = t / tau
        return self.peak_value * (
            0.42 - 0.5 * np.cos(TWO_PI * x) + 0.08 * np.cos(2 * TWO_PI * x)
        )

    def as_string(self, tau):
        omega = TWO_PI / tau
        return (
            f"({self.peak_value}"
            f"*(0.42 - 0.5*cos({omega}*t) + 0.08*cos({2 * omega}*t)))"
        )


class PiecewiseConstant(Waveform):
    """Piecewise-constant waveform with *N* segments.

    Used for AM-optimized gates where each segment has an
    independently optimized amplitude.
    """

    def __init__(self, values: list[float]):
        self.values = np.asarray(values, dtype=float)

    def evaluate(self, t, tau):
        n_seg = len(self.values)
        seg_duration = tau / n_seg
        if isinstance(t, np.ndarray):
            indices = np.clip((t / seg_duration).astype(int), 0, n_seg - 1)
            return self.values[indices]
        idx = min(int(t / seg_duration), n_seg - 1)
        return float(self.values[idx])


@dataclass
class Pulse:
    r"""Time-varying gate drive combining amplitude and detuning profiles.

    Attributes
    ----------
    rabi_frequency : Waveform
        Rabi frequency profile $\Omega(t)$.
    detuning : Waveform
        Detuning profile $\delta(t)$.
    duration : float
        Gate duration in seconds.
    """

    rabi_frequency: Waveform
    detuning: Waveform
    duration: float

    @classmethod
    def constant(
        cls, rabi_frequency: float, detuning: float, duration: float
    ) -> Pulse:
        """Create a pulse with constant parameters."""
        return cls(
            rabi_frequency=ConstantWaveform(rabi_frequency),
            detuning=ConstantWaveform(detuning),
            duration=duration,
        )


def _integrated_phase_string(detuning_wf: Waveform, tau: float) -> str | None:
    """Integrated phase as a QuTiP string, or ``None``."""
    if isinstance(detuning_wf, ConstantWaveform):
        return f"{detuning_wf.value}*t"
    if isinstance(detuning_wf, SinusoidalRamp):
        return detuning_wf.integrated_phase_string(tau)
    return None


def _integrated_phase_array(
    detuning_wf: Waveform, tlist: np.ndarray, tau: float
) -> np.ndarray:
    """Numerically integrate the detuning to get the phase."""
    delta_values = detuning_wf.evaluate(tlist, tau)
    phase = np.zeros_like(tlist)
    phase[1:] = cumulative_trapezoid(delta_values, tlist)
    return phase


def build_pulsed_coefficient(
    rabi_wf: Waveform,
    detuning_wf: Waveform,
    tau: float,
    sign: int = 1,
    tlist: np.ndarray | None = None,
) -> str | np.ndarray:
    r"""Build a QuTiP coefficient for one pulsed Hamiltonian term.

    The coefficient is
    $\Omega(t)\,\exp\!\bigl(\mathrm{sign}\cdot i\,\varphi(t)\bigr)$
    where $\varphi(t) = \int_0^t \delta(t')\,dt'$.

    Returns a string when both waveforms have analytical forms,
    otherwise a complex NumPy array sampled at *tlist*.

    Parameters
    ----------
    rabi_wf : Waveform
        Rabi frequency profile.
    detuning_wf : Waveform
        Detuning profile.
    tau : float
        Gate duration (s).
    sign : int
        +1 for the $a^\dagger$ term, -1 for the $a$ term.
    tlist : np.ndarray or None
        Required when waveforms need array-format coefficients.
    """
    # Fast path: both constant -> same string as existing code
    if isinstance(rabi_wf, ConstantWaveform) and isinstance(
        detuning_wf, ConstantWaveform
    ):
        delta = detuning_wf.value
        omega = rabi_wf.value
        return f"{omega}*exp({sign}*1j*{delta}*t)"

    # Try the string path for known analytical forms
    phase_str = _integrated_phase_string(detuning_wf, tau)
    rabi_str = rabi_wf.as_string(tau)

    if phase_str is not None and rabi_str is not None:
        s = "+" if sign > 0 else "-"
        return f"({rabi_str})*exp({s}1j*({phase_str}))"

    # Fall back to array
    if tlist is None:
        raise ValueError(
            "tlist is required when waveforms cannot be expressed as strings"
        )
    phase = _integrated_phase_array(detuning_wf, tlist, tau)
    rabi_values = rabi_wf.evaluate(tlist, tau)
    return rabi_values * np.exp(sign * 1j * phase)


def smooth_ms_pulse(
    delta_0: float,
    ramp_amplitude: float,
    rabi_frequency: float,
    loops: int = 1,
) -> Pulse:
    r"""Smooth MS gate with sinusoidal detuning ramp.

    The detuning profile is
    $\delta(t) = \delta_0 + A\cos(2\pi t/\tau)$.
    Gate duration is $\tau = 2\pi K/\delta_0$ (mean detuning
    sets the period).

    Parameters
    ----------
    delta_0 : float
        Mean detuning (rad/s).
    ramp_amplitude : float
        Ramp amplitude *A* (rad/s).  Must be less than *delta_0*
        to keep $\delta(t) > 0$.
    rabi_frequency : float
        Constant Rabi frequency (rad/s).
    loops : int
        Phase-space loops *K*.
    """
    if ramp_amplitude >= abs(delta_0):
        raise ValueError(
            f"Ramp amplitude ({ramp_amplitude:.1f}) must be less "
            f"than |delta_0| ({abs(delta_0):.1f})"
        )
    tau = TWO_PI * loops / abs(delta_0)
    return Pulse(
        rabi_frequency=ConstantWaveform(rabi_frequency),
        detuning=SinusoidalRamp(delta_0, ramp_amplitude),
        duration=tau,
    )


def am_ms_pulse(
    segment_amplitudes: list[float],
    detuning: float,
    loops: int = 1,
) -> Pulse:
    """AM-optimized MS gate with piecewise-constant Rabi frequency.

    Parameters
    ----------
    segment_amplitudes : list[float]
        Rabi frequency per time segment (rad/s).
    detuning : float
        Constant detuning (rad/s).
    loops : int
        Phase-space loops.
    """
    tau = TWO_PI * loops / abs(detuning)
    return Pulse(
        rabi_frequency=PiecewiseConstant(segment_amplitudes),
        detuning=ConstantWaveform(detuning),
        duration=tau,
    )


def windowed_ms_pulse(
    rabi_frequency: float,
    detuning: float,
    window: str = "blackman",
    loops: int = 1,
) -> Pulse:
    """MS gate with a windowed Rabi frequency envelope.

    Parameters
    ----------
    rabi_frequency : float
        Peak Rabi frequency (rad/s).
    detuning : float
        Constant detuning (rad/s).
    window : str
        ``"blackman"`` or ``"constant"``.
    loops : int
        Phase-space loops.
    """
    tau = TWO_PI * loops / abs(detuning)
    if window == "blackman":
        rabi_wf = BlackmanWindow(rabi_frequency)
    elif window == "constant":
        rabi_wf = ConstantWaveform(rabi_frequency)
    else:
        raise ValueError(f"Unknown window type: {window}")
    return Pulse(
        rabi_frequency=rabi_wf,
        detuning=ConstantWaveform(detuning),
        duration=tau,
    )


class AdiabaticDetuningRamp(Waveform):
    r"""Adiabatic detuning sweep from arXiv:2510.17286.

    $$
    \delta(t) = \bigl(b + c\,g(t)\bigr)^{-1/j}
    $$

    where $g(t) = t/2 - (\tau_d / 4\pi)\sin(2\pi t/\tau_d)$,
    $b = \delta_\max^{-j}$, and
    $c = (2/\tau_d)(\delta_\min^{-j} - \delta_\max^{-j})$.

    The boundary conditions $d\delta/dt = 0$ at $t=0$ and
    $t=\tau_d$ ensure adiabatic closure.
    """

    def __init__(
        self,
        delta_max: float,
        delta_min: float,
        tau_d: float,
        j: int = 3,
    ):
        self.delta_max = abs(delta_max)
        self.delta_min = abs(delta_min)
        self.tau_d = tau_d
        self.j = j

    def evaluate(self, t, tau):
        j = self.j
        d_max = self.delta_max
        d_min = self.delta_min
        td = self.tau_d
        b = d_max ** (-j)
        c = (2 / td) * (d_min ** (-j) - d_max ** (-j))
        g = t / 2 - (td / (4 * np.pi)) * np.sin(TWO_PI * t / td)
        return (b + c * g) ** (-1.0 / j)


class Sin2Ramp(Waveform):
    r"""$\sin^2$ amplitude ramp: 0 to peak over duration $\tau_g$.

    $$
    \Omega(t) = \Omega_\max\,\sin^2(\pi t / 2\tau_g)
    $$
    """

    def __init__(self, peak: float, tau_g: float):
        self.peak = peak
        self.tau_g = tau_g

    def evaluate(self, t, tau):
        return self.peak * np.sin(np.pi * t / (2 * self.tau_g)) ** 2


def adiabatic_smooth_gate_pulse(
    omega_g: float,
    delta_max: float,
    delta_min: float,
    tau_g: float = 5e-6,
    tau_d: float = 100e-6,
    t_c: float = 0.0,
    j: int = 3,
    n_points: int = 2000,
) -> Pulse:
    r"""Adiabatic smooth gate from arXiv:2510.17286.

    Five-step sequence:

    1. Ramp $\Omega$: 0 to $\Omega_g$ over $\tau_g$ ($\sin^2$),
       $\delta = \delta_\max$.
    2. Sweep $\delta$: $\delta_\max \to \delta_\min$ over $\tau_d$,
       $\Omega = \Omega_g$.
    3. Hold at $\delta_\min$ for $t_c$.
    4. Sweep $\delta$: $\delta_\min \to \delta_\max$ over $\tau_d$.
    5. Ramp $\Omega$: $\Omega_g \to 0$ over $\tau_g$.

    Constructs explicit array waveforms for both $\Omega(t)$ and
    $\delta(t)$.

    Parameters
    ----------
    omega_g : float
        Gate Rabi frequency (rad/s).
    delta_max : float
        Far-detuned value (rad/s, positive).
    delta_min : float
        Near-resonance value (rad/s, positive).
    tau_g : float
        Omega ramp duration (s).
    tau_d : float
        Detuning sweep duration (s).
    t_c : float
        Hold time at delta_min (s).
    j : int
        Detuning profile exponent (default 3).
    n_points : int
        Total sample count.
    """
    tau_total = 2 * tau_g + 2 * tau_d + t_c
    tlist = np.linspace(0, tau_total, n_points)

    omega_arr = np.zeros(n_points)
    delta_arr = np.full(n_points, float(delta_max))

    ramp_wf = AdiabaticDetuningRamp(delta_max, delta_min, tau_d, j)

    for i, t in enumerate(tlist):
        if t <= tau_g:
            omega_arr[i] = omega_g * np.sin(np.pi * t / (2 * tau_g)) ** 2
            delta_arr[i] = delta_max
        elif t <= tau_g + tau_d:
            omega_arr[i] = omega_g
            t_local = t - tau_g
            delta_arr[i] = ramp_wf.evaluate(t_local, tau_d)
        elif t <= tau_g + tau_d + t_c:
            omega_arr[i] = omega_g
            delta_arr[i] = delta_min
        elif t <= tau_g + 2 * tau_d + t_c:
            omega_arr[i] = omega_g
            t_local = tau_d - (t - tau_g - tau_d - t_c)
            delta_arr[i] = ramp_wf.evaluate(t_local, tau_d)
        else:
            t_local = t - (tau_g + 2 * tau_d + t_c)
            omega_arr[i] = omega_g * np.cos(np.pi * t_local / (2 * tau_g)) ** 2
            delta_arr[i] = delta_max

    return Pulse(
        rabi_frequency=PiecewiseConstant(omega_arr.tolist()),
        detuning=PiecewiseConstant(delta_arr.tolist()),
        duration=tau_total,
    )


def adiabatic_geometric_phase(
    omega_g: float,
    delta_max: float,
    delta_min: float,
    tau_g: float = 5e-6,
    tau_d: float = 100e-6,
    t_c: float = 0.0,
    j: int = 3,
    n_points: int = 10000,
) -> float:
    r"""Compute the geometric phase for an adiabatic smooth gate.

    $$
    \theta_g \approx \int_0^{t_\text{total}}
      \frac{\Omega_g^2(t)}{\delta(t)}\,dt
    $$

    Parameters
    ----------
    omega_g, delta_max, delta_min, tau_g, tau_d, t_c, j : float
        Same as :func:`adiabatic_smooth_gate_pulse`.

    Returns
    -------
    float
        Geometric phase in radians.
    """
    pulse = adiabatic_smooth_gate_pulse(
        omega_g, delta_max, delta_min, tau_g, tau_d, t_c, j, n_points
    )
    tlist = np.linspace(0, pulse.duration, n_points)
    omega_t = pulse.rabi_frequency.evaluate(tlist, pulse.duration)
    delta_t = pulse.detuning.evaluate(tlist, pulse.duration)
    delta_t = np.maximum(delta_t, 1.0)
    integrand = omega_t**2 / delta_t
    return float(trapezoid(integrand, tlist))


def calibrate_adiabatic_smooth_gate(
    omega_g: float,
    delta_max: float,
    tau_g: float = 5e-6,
    tau_d: float = 100e-6,
    j: int = 3,
    target_phase: float = np.pi / 2,
) -> dict:
    r"""Find delta_min that gives $\theta_g$ = *target_phase*.

    The geometric phase is $\theta_g = \int \Omega_g^2/\delta\,dt$.
    Here ``omega_g`` is the **sideband Rabi frequency**, i.e. the
    value that appears directly in the gate Hamiltonian
    $H = \delta\,a^\dagger a + (\Omega_g/2)\,S_x\,(a+a^\dagger)$.

    When using ``adiabatic_gate_hamiltonian``, pass ``eta=[1, 1]``
    and the calibrated pulse directly.  When the physical Lamb-Dicke
    parameter $\eta$ is known, set
    ``omega_g = eta * Omega_carrier`` before calling this function.

    Returns
    -------
    dict
        ``delta_min``, ``geometric_phase``, ``gate_time``, ``pulse``.
    """
    from scipy.optimize import brentq

    def phase_error(log_delta_min):
        dm = np.exp(log_delta_min)
        phase = adiabatic_geometric_phase(
            omega_g, delta_max, dm, tau_g, tau_d, 0.0, j, 5000
        )
        return phase - target_phase

    log_max = np.log(delta_max * 0.99)
    log_min = np.log(delta_max * 0.001)

    p_lo = phase_error(log_min)
    p_hi = phase_error(log_max)
    if p_lo * p_hi > 0:
        raise ValueError(
            "Cannot find delta_min: phase does not cross "
            f"target. Phase at delta_min=delta_max*0.001: "
            f"{p_lo + target_phase:.4f}, at 0.99*delta_max: "
            f"{p_hi + target_phase:.4f}"
        )

    log_dm = brentq(phase_error, log_min, log_max, xtol=1e-10)
    dm = np.exp(log_dm)
    phase = adiabatic_geometric_phase(
        omega_g, delta_max, dm, tau_g, tau_d, 0.0, j
    )
    pulse = adiabatic_smooth_gate_pulse(
        omega_g, delta_max, dm, tau_g, tau_d, 0.0, j
    )

    return {
        "delta_min": dm,
        "geometric_phase": phase,
        "gate_time": pulse.duration,
        "pulse": pulse,
    }


def verify_loop_closure(
    pulse: Pulse,
    eta: float = 1.0,
    n_points: int = 10000,
) -> dict[str, float | bool]:
    r"""Numerically verify that a pulse closes the phase-space loop.

    Computes $\alpha(\tau) = \int_0^\tau \Omega(t)\,e^{i\varphi(t)}\,dt$
    and checks whether $|\alpha(\tau)| \approx 0$.

    Parameters
    ----------
    pulse : Pulse
    eta : float
        Lamb-Dicke parameter (scales the displacement).
    n_points : int
        Integration sample count.

    Returns
    -------
    dict
        ``closed`` (bool), ``residual`` (float),
        ``geometric_phase`` (float).
    """
    tau = pulse.duration
    tlist = np.linspace(0, tau, n_points)

    omega_t = pulse.rabi_frequency.evaluate(tlist, tau)
    phase = _integrated_phase_array(pulse.detuning, tlist, tau)

    integrand = eta * omega_t * np.exp(1j * phase)
    alpha_tau = trapezoid(integrand, tlist)

    # Geometric phase from enclosed area
    alpha_t = np.zeros(len(tlist), dtype=complex)
    alpha_cumul = cumulative_trapezoid(integrand, tlist, initial=0)
    alpha_t[:] = alpha_cumul

    d_alpha = np.gradient(alpha_t, tlist)
    geo_phase = abs(trapezoid(np.imag(np.conj(alpha_t) * d_alpha), tlist))

    return {
        "closed": bool(np.abs(alpha_tau) < 0.01),
        "residual": float(np.abs(alpha_tau)),
        "geometric_phase": float(geo_phase),
    }

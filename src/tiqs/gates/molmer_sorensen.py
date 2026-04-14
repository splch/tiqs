"""Molmer-Sorensen entangling gate Hamiltonian construction."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from tiqs.constants import PI, TWO_PI
from tiqs.hilbert_space.operators import OperatorFactory

if TYPE_CHECKING:
    from tiqs.pulses import Pulse


def ms_gate_duration(detuning: float, loops: int = 1) -> float:
    r"""Gate time for the MS gate: $\tau = 2\pi K / \delta$ where $K$ is the
    number of loops.

    Parameters
    ----------
    detuning : float
        Sideband detuning $\delta$ (rad/s).
    loops : int
        Number of phase-space loops ($K$). More loops = slower but more
        robust.

    Returns
    -------
    float
        Gate duration in seconds.
    """
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

    The MS Hamiltonian for two ions coupled to one motional mode:

    $$
    H_\mathrm{MS}(t) = \sum_j \eta_j \, \Omega \, \sigma_{x,j}
    \left( a^\dagger e^{i \delta t} + a \, e^{-i \delta t} \right)
    $$

    This is a spin-dependent force that displaces the motional state
    conditioned on the collective spin. After time
    $\tau = 2\pi K / \delta$, the motion returns to its initial state and
    the spins acquire a geometric phase proportional to the enclosed
    phase-space area.

    For two identically-coupled ions, the maximally entangling condition is
    $\eta \Omega = \delta / 4$ (single loop). The geometric phase scales as the
    square of the collective spin eigenvalue, so two ions need half the
    single-ion drive strength.

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
        Rabi frequency $\Omega$ (rad/s) of the bichromatic drive on each ion.
    detuning : float
        Detuning $\delta$ (rad/s) from the motional sideband.

    Returns
    -------
    list
        QuTiP list-format Hamiltonian: ``[[H_op, coeff_string], ...]``.
    """
    a = ops.annihilate(mode)
    ad = ops.create(mode)

    H_terms = []

    for j, ion_idx in enumerate(ions):
        sx_j = ops.sigma_x(ion_idx)
        coupling = eta[j] * rabi_frequency

        H_plus = coupling * ad * sx_j
        H_minus = coupling * a * sx_j

        H_terms.append([H_plus, f"exp(1j*{detuning}*t)"])
        H_terms.append([H_minus, f"exp(-1j*{detuning}*t)"])

    return H_terms


def ms_multimode_hamiltonian(
    ops: OperatorFactory,
    ions: list[int],
    modes: list[int],
    eta: np.ndarray,
    rabi_frequency: float | list[float],
    detunings: list[float],
) -> list:
    r"""Construct a multi-mode Molmer-Sorensen Hamiltonian.

    Sums contributions from all specified motional modes:

    $$
    H(t) = \sum_j \sum_p \eta_{j,p}\,\Omega_p\,\sigma_{x,j}
    \bigl(a_p^\dagger e^{i\delta_p t} + a_p\,e^{-i\delta_p t}\bigr)
    $$

    For a single-tone drive, all modes share the same Rabi frequency
    and the per-mode detunings are ``mu - omega_p``.  For a
    multi-tone drive, each mode has its own Rabi frequency and its
    own detuning chosen to close the phase-space loop.

    Parameters
    ----------
    ops : OperatorFactory
    ions : list[int]
        Ion indices participating in the gate.
    modes : list[int]
        Motional mode indices.
    eta : np.ndarray
        Lamb-Dicke matrix, shape ``(len(ions), len(modes))``.
        ``eta[j, p]`` is the parameter for ion ``ions[j]`` and
        mode ``modes[p]``.
    rabi_frequency : float or list[float]
        Rabi frequency in rad/s.  A scalar applies uniformly to
        all modes (single-tone).  A list provides per-mode
        amplitudes (multi-tone).
    detunings : list[float]
        Per-mode detuning ``delta_p`` in rad/s.

    Returns
    -------
    list
        QuTiP list-format Hamiltonian.
    """
    if isinstance(rabi_frequency, (int, float)):
        rabi_frequency = [float(rabi_frequency)] * len(modes)

    H_terms = []
    for p_idx, mode in enumerate(modes):
        a = ops.annihilate(mode)
        ad = ops.create(mode)
        delta_p = detunings[p_idx]
        omega_p = rabi_frequency[p_idx]

        for j_idx, ion_idx in enumerate(ions):
            sx_j = ops.sigma_x(ion_idx)
            coupling = eta[j_idx, p_idx] * omega_p

            H_terms.append([coupling * ad * sx_j, f"exp(1j*{delta_p}*t)"])
            H_terms.append([coupling * a * sx_j, f"exp(-1j*{delta_p}*t)"])

    return H_terms


def ms_multimode_hamiltonian_exact(
    ops: OperatorFactory,
    ions: list[int],
    modes: list[int],
    eta: np.ndarray,
    rabi_frequency: float | list[float],
    detunings: list[float],
) -> list:
    r"""Multi-mode MS Hamiltonian with exact Rabi frequencies.

    Replaces the first-order Lamb-Dicke coupling
    $\eta\,\Omega\,(a^\dagger + a)$ with the exact n-dependent
    sideband matrix elements from generalized Laguerre polynomials.
    This captures the Debye-Waller factor and makes the gate
    properly temperature-dependent.

    The operator for each (ion, mode) pair is built in the Fock
    basis using ``exact_sideband_operator``.

    Parameters
    ----------
    ops : OperatorFactory
    ions : list[int]
    modes : list[int]
    eta : np.ndarray
        Lamb-Dicke matrix, shape ``(len(ions), len(modes))``.
    rabi_frequency : float or list[float]
        Bare carrier Rabi frequency (rad/s).
    detunings : list[float]
        Per-mode detuning (rad/s).

    Returns
    -------
    list
        QuTiP list-format Hamiltonian.
    """
    import qutip

    from tiqs.interaction.exact_coupling import exact_sideband_operator

    if isinstance(rabi_frequency, (int, float)):
        rabi_frequency = [float(rabi_frequency)] * len(modes)

    H_terms = []
    for p_idx, mode in enumerate(modes):
        delta_p = detunings[p_idx]
        omega_p = rabi_frequency[p_idx]
        n_fock = ops.hs.fock_dim(mode)
        mode_subsys = ops.hs.n_ions + mode

        for j_idx, ion_idx in enumerate(ions):
            eta_jp = eta[j_idx, p_idx]
            sx_j = ops.sigma_x(ion_idx)

            # Blue sideband: exact matrix elements for |n> -> |n+1>
            M_bsb = exact_sideband_operator(eta_jp, n_fock, s=+1)
            bsb_local = qutip.Qobj(M_bsb * omega_p)
            bsb_full = ops._full_operator(bsb_local, mode_subsys)
            H_bsb = bsb_full * sx_j

            # Red sideband: exact matrix elements for |n> -> |n-1>
            M_rsb = exact_sideband_operator(eta_jp, n_fock, s=-1)
            rsb_local = qutip.Qobj(M_rsb * omega_p)
            rsb_full = ops._full_operator(rsb_local, mode_subsys)
            H_rsb = rsb_full * sx_j

            H_terms.append([H_bsb, f"exp(1j*{delta_p}*t)"])
            H_terms.append([H_rsb, f"exp(-1j*{delta_p}*t)"])

    return H_terms


def ms_single_tone_detunings(
    mu: float,
    mode_frequencies: np.ndarray,
) -> list[float]:
    r"""Per-mode detunings from a single bichromatic drive offset.

    Parameters
    ----------
    mu : float
        Bichromatic beat-note offset from the qubit frequency
        (rad/s).
    mode_frequencies : np.ndarray
        Mode angular frequencies ``omega_p``.

    Returns
    -------
    list[float]
        ``[mu - omega_p for each mode]``.
    """
    return [float(mu - omega_p) for omega_p in mode_frequencies]


def ms_geometric_phase(
    eta: np.ndarray,
    rabi_frequency: float | list[float],
    detunings: list[float],
    gate_time: float,
) -> np.ndarray:
    r"""Analytic pairwise geometric phase matrix.

    $$
    \chi_{j,k} = \sum_p
      \frac{4\pi\,K_p^\text{eff}\,\eta_{j,p}\,\eta_{k,p}
            \,\Omega_p^2}{\delta_p^2}
    $$

    where $K_p^\text{eff} = \delta_p\,\tau / (2\pi)$ is the
    (possibly non-integer) effective loop count.

    For two identically-coupled ions with the standard MS
    calibration $\eta\Omega = \delta/4$, the pairwise phase
    is $\chi = \pi/4$ (maximally entangling).

    Parameters
    ----------
    eta : np.ndarray
        Lamb-Dicke matrix, shape ``(N_ions, N_modes)``.
    rabi_frequency : float or list[float]
        Scalar or per-mode Rabi frequencies (rad/s).
    detunings : list[float]
        Per-mode detunings ``delta_p`` (rad/s).
    gate_time : float
        Total gate duration (s).

    Returns
    -------
    np.ndarray
        Symmetric phase matrix of shape ``(N_ions, N_ions)``.
    """
    n_ions, n_modes = eta.shape
    if isinstance(rabi_frequency, (int, float)):
        rabi_frequency = [float(rabi_frequency)] * n_modes

    chi = np.zeros((n_ions, n_ions))
    for p in range(n_modes):
        delta_p = detunings[p]
        if delta_p == 0:
            continue
        omega_p = rabi_frequency[p]
        k_eff = delta_p * gate_time / TWO_PI
        prefactor = 4 * PI * k_eff * omega_p**2 / delta_p**2
        for j in range(n_ions):
            for k in range(j, n_ions):
                contrib = prefactor * eta[j, p] * eta[k, p]
                chi[j, k] += contrib
                if k != j:
                    chi[k, j] += contrib
    return chi


def ms_residual_displacement(
    detunings: list[float],
    gate_time: float,
) -> list[float]:
    r"""Fractional residual phase-space displacement per mode.

    A value of 0.0 means the loop closes exactly.  Values near
    $\pm 0.5$ indicate maximum non-closure.

    Parameters
    ----------
    detunings : list[float]
        Per-mode detunings ``delta_p`` (rad/s).
    gate_time : float
        Gate duration (s).

    Returns
    -------
    list[float]
        Fractional residual for each mode: distance from the
        nearest integer loop count.
    """
    residuals = []
    for delta_p in detunings:
        k_eff = delta_p * gate_time / TWO_PI
        residuals.append(k_eff - round(k_eff))
    return residuals


def ms_gate_hamiltonian_pulsed(
    ops: OperatorFactory,
    ions: list[int],
    mode: int,
    eta: list[float],
    pulse: Pulse,
    tlist: np.ndarray | None = None,
) -> list:
    r"""Construct the MS gate Hamiltonian with time-varying parameters.

    Like :func:`ms_gate_hamiltonian` but accepts a
    :class:`~tiqs.pulses.Pulse` whose Rabi frequency and/or
    detuning may vary in time.

    The operator for each ion carries only the Lamb-Dicke factor
    $\eta_j$; the time-dependent $\Omega(t)$ is absorbed into the
    QuTiP coefficient.

    Parameters
    ----------
    ops : OperatorFactory
    ions : list[int]
    mode : int
    eta : list[float]
        Lamb-Dicke parameters per ion on this mode.
    pulse : Pulse
        Pulse specification with waveforms.
    tlist : np.ndarray or None
        Required when waveforms need array-format coefficients.

    Returns
    -------
    list
        QuTiP list-format Hamiltonian.
    """
    from tiqs.pulses import build_pulsed_coefficient

    a = ops.annihilate(mode)
    ad = ops.create(mode)
    tau = pulse.duration

    H_terms = []
    for j, ion_idx in enumerate(ions):
        sx_j = ops.sigma_x(ion_idx)

        H_plus = eta[j] * ad * sx_j
        H_minus = eta[j] * a * sx_j

        coeff_plus = build_pulsed_coefficient(
            pulse.rabi_frequency,
            pulse.detuning,
            tau,
            sign=+1,
            tlist=tlist,
        )
        coeff_minus = build_pulsed_coefficient(
            pulse.rabi_frequency,
            pulse.detuning,
            tau,
            sign=-1,
            tlist=tlist,
        )

        H_terms.append([H_plus, coeff_plus])
        H_terms.append([H_minus, coeff_minus])

    return H_terms


def ms_gate_hamiltonian_with_carrier(
    ops: OperatorFactory,
    ions: list[int],
    mode: int,
    eta: list[float],
    rabi_frequency: float,
    detuning: float,
) -> list:
    r"""MS gate Hamiltonian including off-resonant carrier excitation.

    The bichromatic drive that addresses the sidebands also drives
    the carrier transition off-resonantly.  This produces an AC
    Stark shift:

    $$
    \delta_\text{AC}^{(j)} = \frac{(\eta_j\,\Omega)^2}{\delta}
    $$

    which shifts each ion's effective qubit frequency during the
    gate.  Experiments compensate this by adjusting drive
    frequencies.

    The returned Hamiltonian includes:

    1. Standard MS sideband terms (same as ``ms_gate_hamiltonian``)
    2. Off-resonant carrier terms:
       $\frac{\Omega}{2}\,\sigma_{x,j}\,\cos(\delta\,t)$

    Parameters
    ----------
    ops : OperatorFactory
    ions : list[int]
    mode : int
    eta : list[float]
    rabi_frequency : float
    detuning : float

    Returns
    -------
    list
        QuTiP list-format Hamiltonian with sideband + carrier terms.
    """
    # Sideband terms (identical to ms_gate_hamiltonian)
    H_terms = ms_gate_hamiltonian(
        ops, ions, mode, eta, rabi_frequency, detuning
    )

    # Off-resonant carrier for each ion
    for ion_idx in ions:
        sp = ops.sigma_plus(ion_idx)
        sm = ops.sigma_minus(ion_idx)
        H_carrier = (rabi_frequency / 2) * (sp + sm)
        H_terms.append([H_carrier, f"cos({detuning}*t)"])

    return H_terms


def ms_ac_stark_shift(
    eta: list[float],
    rabi_frequency: float,
    detuning: float,
) -> list[float]:
    r"""AC Stark shift per ion from the MS bichromatic drive.

    $$
    \delta_\text{AC}^{(j)} = \frac{(\eta_j\,\Omega)^2}{\delta}
    $$

    This is the time-averaged frequency shift of ion $j$'s qubit
    from the off-resonant carrier excitation.  To compensate, the
    drive frequencies must be shifted by this amount.

    Parameters
    ----------
    eta : list[float]
    rabi_frequency : float
    detuning : float

    Returns
    -------
    list[float]
        AC Stark shift per ion in rad/s.
    """
    return [(e * rabi_frequency) ** 2 / detuning for e in eta]

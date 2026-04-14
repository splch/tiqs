"""Light-shift (geometric phase) gate: sigma_z-dependent force."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from tiqs.hilbert_space.operators import OperatorFactory

if TYPE_CHECKING:
    from tiqs.pulses import Pulse


def light_shift_gate_hamiltonian(
    ops: OperatorFactory,
    ions: list[int],
    mode: int,
    eta: list[float],
    rabi_frequency: float,
    detuning: float,
) -> list:
    r"""Construct the light-shift gate Hamiltonian.

    Uses a $\sigma_z \otimes \sigma_z$ interaction.

    Uses a state-dependent optical dipole force from off-resonant Raman beams.
    The AC Stark shift creates a $\sigma_z$-dependent force:

    $$
    H_\mathrm{LS}(t) = \sum_j \eta_j \, F_j \, \sigma_{z,j}
    \left( a^\dagger e^{i \delta t} + a \, e^{-i \delta t} \right)
    $$

    where $F_j = \eta_j \, \Omega$ is the effective force
    strength. This generates a
    $\sigma_z \otimes \sigma_z$ interaction (ZZ coupling),
    which is inherently
    insensitive to the optical phase of the laser beams.

    Parameters
    ----------
    ops : OperatorFactory
    ions : list[int]
    mode : int
    eta : list[float]
        Lamb-Dicke parameters for each ion.
    rabi_frequency : float
        Effective Rabi frequency from the light shift (rad/s).
    detuning : float
        Detuning from motional sideband (rad/s).

    Returns
    -------
    list
        QuTiP list-format Hamiltonian.
    """
    a = ops.annihilate(mode)
    ad = ops.create(mode)

    H_terms = []
    for j, ion_idx in enumerate(ions):
        sz_j = ops.sigma_z(ion_idx)
        coupling = eta[j] * rabi_frequency

        H_plus = coupling * ad * sz_j
        H_minus = coupling * a * sz_j

        H_terms.append([H_plus, f"exp(1j*{detuning}*t)"])
        H_terms.append([H_minus, f"exp(-1j*{detuning}*t)"])

    return H_terms


def light_shift_multimode_hamiltonian(
    ops: OperatorFactory,
    ions: list[int],
    modes: list[int],
    eta: np.ndarray,
    rabi_frequency: float | list[float],
    detunings: list[float],
) -> list:
    r"""Multi-mode light-shift gate Hamiltonian.

    Same structure as
    :func:`~tiqs.gates.molmer_sorensen.ms_multimode_hamiltonian`
    but with $\sigma_z$ replacing $\sigma_x$:

    $$
    H(t) = \sum_j \sum_p \eta_{j,p}\,\Omega_p\,\sigma_{z,j}
    \bigl(a_p^\dagger e^{i\delta_p t} + a_p\,e^{-i\delta_p t}\bigr)
    $$

    Parameters
    ----------
    ops : OperatorFactory
    ions : list[int]
        Ion indices participating in the gate.
    modes : list[int]
        Motional mode indices.
    eta : np.ndarray
        Lamb-Dicke matrix, shape ``(len(ions), len(modes))``.
    rabi_frequency : float or list[float]
        Scalar (single-tone) or per-mode list (multi-tone) in
        rad/s.
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
            sz_j = ops.sigma_z(ion_idx)
            coupling = eta[j_idx, p_idx] * omega_p

            H_terms.append([coupling * ad * sz_j, f"exp(1j*{delta_p}*t)"])
            H_terms.append([coupling * a * sz_j, f"exp(-1j*{delta_p}*t)"])

    return H_terms


def light_shift_gate_hamiltonian_pulsed(
    ops: OperatorFactory,
    ions: list[int],
    mode: int,
    eta: list[float],
    pulse: Pulse,
    tlist: np.ndarray | None = None,
) -> list:
    r"""Light-shift gate Hamiltonian with time-varying parameters.

    Same structure as
    :func:`~tiqs.gates.molmer_sorensen.ms_gate_hamiltonian_pulsed`
    but uses $\sigma_z$ instead of $\sigma_x$.

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
        sz_j = ops.sigma_z(ion_idx)

        H_plus = eta[j] * ad * sz_j
        H_minus = eta[j] * a * sz_j

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

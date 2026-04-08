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
    """Carrier transition Hamiltonian.

    H = (Omega/2)(sigma+ e^{i*phi} + sigma- e^{-i*phi})

    Drives |0> <-> |1> without changing the motional state.
    """
    sp = ops.sigma_plus(ion)
    sm = ops.sigma_minus(ion)
    return (rabi_frequency / 2) * (
        sp * np.exp(1j * phase) + sm * np.exp(-1j * phase)
    )


def red_sideband_hamiltonian(
    ops: OperatorFactory,
    ion: int,
    mode: int,
    rabi_frequency: float,
    eta: float,
    phase: float = 0.0,
) -> qutip.Qobj:
    """Red sideband Hamiltonian.

    H = (eta*Omega/2)(a * sigma_- + a_dag * sigma_+)

    Drives |0,n> <-> |1,n-1>: excites the qubit while removing one
    phonon.

    In QuTiP's convention sigma_- = |1><0| takes |0> -> |1>
    (excitation), and a removes a phonon, so the coupling term is
    a * sm + h.c.
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
    """Blue sideband Hamiltonian.

    H = (eta*Omega/2)(a_dag * sigma_- + a * sigma_+)

    Drives |0,n> <-> |1,n+1>: excites the qubit while adding one phonon.

    In QuTiP's convention sigma_- = |1><0| takes |0> -> |1>
    (excitation), and a_dag adds a phonon, so the coupling term is
    ad * sm + h.c.
    """
    sp = ops.sigma_plus(ion)
    sm = ops.sigma_minus(ion)
    a = ops.annihilate(mode)
    ad = ops.create(mode)
    return (eta * rabi_frequency / 2) * (
        ad * sm * np.exp(1j * phase) + a * sp * np.exp(-1j * phase)
    )


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
    """Full laser-ion interaction in the interaction picture.

    Returns the Hamiltonian in QuTiP list format.

    In the interaction picture with respect to the qubit and mode
    free evolution, the Hamiltonian becomes time-dependent. For
    first-order Lamb-Dicke:

    H(t) = (Omega/2) * sigma_x * cos(delta*t + phi)  [carrier]
        + (eta*Omega/2) * (a*sigma+ + a_dag*sigma-)
          * e^{-i*omega_m*t}                          [RSB]
        + (eta*Omega/2) * (a_dag*sigma+ + a*sigma-)
          * e^{+i*omega_m*t}                          [BSB]

    For the interaction picture w.r.t. the free Hamiltonian,
    returns:

    H = [H_drift, [H_rsb, coeff_rsb(t)],
         [H_bsb, coeff_bsb(t)], [H_carrier, coeff_carrier(t)]]

    Parameters
    ----------
    ops : OperatorFactory
    ion : int
    mode : int
    rabi_frequency : float
        Rabi frequency Omega (rad/s).
    eta : float
        Lamb-Dicke parameter for this ion-mode pair.
    detuning : float
        Laser detuning from qubit resonance delta (rad/s).
    mode_frequency : float
        Motional mode frequency omega_m (rad/s).
    phase : float
        Laser phase.
    lamb_dicke_order : int
        Order of Lamb-Dicke expansion (1 = standard, 2 = includes
        second-order corrections).

    Returns
    -------
    list
        QuTiP list-format Hamiltonian [[H_op, coeff_string], ...].
    """
    sp = ops.sigma_plus(ion)
    sm = ops.sigma_minus(ion)
    a = ops.annihilate(mode)
    ad = ops.create(mode)

    H_terms = []

    # Carrier: (Omega/2) * (sigma+ * e^{i*phi} + sigma- * e^{-i*phi})
    # * cos(delta*t)
    H_carrier = (rabi_frequency / 2) * (
        sp * np.exp(1j * phase) + sm * np.exp(-1j * phase)
    )
    if detuning == 0.0:
        # When on resonance, cos(0*t) = 1, so the carrier is time-independent.
        H_terms.append(H_carrier)
    else:
        H_terms.append([H_carrier, f"cos({detuning}*t)"])

    # Red sideband: (eta*Omega/2) * (a * sm + h.c.)
    # at detuning (delta - omega_m)
    # sm = |1><0| excites the qubit, a removes a phonon
    H_rsb_op = (eta * rabi_frequency / 2) * a * sm * np.exp(1j * phase)
    H_rsb_hc = (eta * rabi_frequency / 2) * ad * sp * np.exp(-1j * phase)
    rsb_det = detuning - mode_frequency
    H_terms.append([H_rsb_op, f"exp(1j*{rsb_det}*t)"])
    H_terms.append([H_rsb_hc, f"exp(-1j*{rsb_det}*t)"])

    # Blue sideband: (eta*Omega/2) * (ad * sm + h.c.)
    # at detuning (delta + omega_m)
    # sm = |1><0| excites the qubit, ad adds a phonon
    H_bsb_op = (eta * rabi_frequency / 2) * ad * sm * np.exp(1j * phase)
    H_bsb_hc = (eta * rabi_frequency / 2) * a * sp * np.exp(-1j * phase)
    bsb_det = detuning + mode_frequency
    H_terms.append([H_bsb_op, f"exp(1j*{bsb_det}*t)"])
    H_terms.append([H_bsb_hc, f"exp(-1j*{bsb_det}*t)"])

    if lamb_dicke_order >= 2:
        # Second-order Lamb-Dicke corrections from expansion of
        # exp(i*eta*(a+a_dag)):
        # (i*eta)^2/2! * (a+a_dag)^2 = -eta^2/2 * (a^2 + a_dag^2 + 2n + 1)
        n_op = ops.number(mode)
        # Debye-Waller factor: carrier Rabi freq is modified by -eta^2*(n+1/2)
        sigma_phi = sp * np.exp(1j * phase) + sm * np.exp(-1j * phase)
        dw_correction = -(eta**2) * rabi_frequency / 2
        H_dw = dw_correction * sigma_phi * (n_op + 0.5 * ops.identity())
        H_terms.append([H_dw, f"cos({detuning}*t)"])

        # Second red sideband: eta^2 * Omega / 4 * a^2 * sm
        # From: -eta^2/2 (from expansion) * Omega/2 (from Rabi) = eta^2*Omega/4
        H_2rsb_op = (
            (eta**2 * rabi_frequency / 4) * a * a * sm * np.exp(1j * phase)
        )
        H_2rsb_hc = (
            (eta**2 * rabi_frequency / 4) * ad * ad * sp * np.exp(-1j * phase)
        )
        rsb2_det = detuning - 2 * mode_frequency
        H_terms.append([H_2rsb_op, f"exp(1j*{rsb2_det}*t)"])
        H_terms.append([H_2rsb_hc, f"exp(-1j*{rsb2_det}*t)"])

    return H_terms

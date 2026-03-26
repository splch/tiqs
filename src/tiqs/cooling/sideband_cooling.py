"""Resolved sideband cooling: analytical and simulated."""
import numpy as np
import qutip

from tiqs.hilbert_space.operators import OperatorFactory


def sideband_cooling_nbar(
    gamma_eff: float,
    trap_frequency: float,
) -> float:
    """Analytical steady-state phonon number from resolved sideband cooling.

    n_bar_final ~ (Gamma_eff / (2 * omega_trap))^2

    where Gamma_eff is the effective cooling rate (optical pumping rate
    or Raman transition linewidth).

    Parameters
    ----------
    gamma_eff : float
        Effective linewidth of the cooling transition (rad/s).
    trap_frequency : float
        Motional mode frequency (rad/s).

    Returns
    -------
    float
        Estimated final mean phonon number.
    """
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
    """Simulate resolved sideband cooling as a sequence of RSB pulses + optical pumping.

    Each cycle:
    1. Red sideband pi-pulse: |0, n> -> |1, n-1> (removes one phonon)
    2. Optical pumping: |1> -> |0> via spontaneous emission (resets spin)

    We model this as a master equation with:
    - H = RSB Hamiltonian
    - Collapse: optical pumping at the given rate

    QuTiP conventions:
    - basis(2,0) = ground = bright state, basis(2,1) = excited = dark state
    - sigmap() = |0><1| (decay from excited to ground)
    - sigmam() = |1><0| (excitation from ground to excited)
    - RSB: sigmam * a + sigmap * a_dag  (|0,n> -> |1,n-1>)
    - Optical pumping to |0>: collapse = sqrt(rate) * sigmap  (|1> -> |0>)

    Parameters
    ----------
    ops : OperatorFactory
    ion : int
    mode : int
    n_bar_initial : float
        Initial mean phonon number (from Doppler cooling).
    eta : float
        Lamb-Dicke parameter.
    rabi_frequency : float
        Bare Rabi frequency (rad/s).
    optical_pumping_rate : float
        Optical pumping rate (rad/s).
    n_cycles : int
        Number of cooling cycles.

    Returns
    -------
    float
        Final mean phonon number.
    """
    sp = ops.sigma_plus(ion)   # sigmap = |0><1|
    sm = ops.sigma_minus(ion)  # sigmam = |1><0|
    a = ops.annihilate(mode)
    ad = ops.create(mode)
    n_op = ops.number(mode)

    rsb_rabi = eta * rabi_frequency
    # RSB Hamiltonian: |0,n> <-> |1,n-1>
    # sm*a takes |0,n> -> |1,n-1>, sp*ad is the hermitian conjugate
    H_rsb = (rsb_rabi / 2) * (sm * a + sp * ad)

    # Optical pumping: dissipative |1> -> |0> using sigmap = |0><1|
    c_ops = [np.sqrt(optical_pumping_rate) * sp]

    hs = ops.hs
    fock_dim = hs.fock_dim(mode)
    rho0 = qutip.tensor(
        qutip.ket2dm(qutip.basis(2, 0)),
        qutip.thermal_dm(fock_dim, n_bar_initial),
    )

    t_pi = np.pi / rsb_rabi if rsb_rabi > 0 else 1e-6
    t_pump = 5.0 / optical_pumping_rate
    t_cycle = t_pi + t_pump
    t_total = n_cycles * t_cycle

    tlist = np.linspace(0, t_total, n_cycles * 20)
    result = qutip.mesolve(H_rsb, rho0, tlist, c_ops=c_ops, e_ops=[n_op])

    return float(result.expect[0][-1])

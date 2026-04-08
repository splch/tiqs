"""Cirac-Zoller gate: the original trapped-ion entangling gate (1995)."""

import numpy as np

from tiqs.constants import TWO_PI
from tiqs.gates.single_qubit import GatePulse
from tiqs.hilbert_space.operators import OperatorFactory


def cirac_zoller_gate(
    ops: OperatorFactory,
    ion_a: int,
    ion_b: int,
    mode: int,
    eta: list[float],
    rabi_frequency: float = TWO_PI * 100e3,
) -> list[GatePulse]:
    r"""Cirac-Zoller controlled-phase gate using sequential
    red sideband pulses.

    Three-step sequence:
    1. $\pi$ pulse on RSB of ion A: maps
       $|\!\uparrow_A, 0\rangle
       \to -i|\!\downarrow_A, 1\rangle$
    2. $2\pi$ pulse on RSB of ion B (to auxiliary level):
       $|\!\downarrow_B, 1\rangle \to -|\!\downarrow_B, 1\rangle$
    3. Reverse $\pi$ pulse on RSB of ion A: unmaps motion back to ion A

    REQUIRES the motional mode to be in the ground state $|n=0\rangle$.

    Parameters
    ----------
    ops : OperatorFactory
    ion_a, ion_b : int
        Ion indices.
    mode : int
        Motional mode index (must be in ground state).
    eta : list[float]
        Lamb-Dicke parameters $[\eta_a, \eta_b]$.
    rabi_frequency : float
        Bare Rabi frequency.

    Returns
    -------
    list[GatePulse]
        Three sequential pulses to be applied in order.
    """
    sp_a = ops.sigma_plus(ion_a)
    sm_a = ops.sigma_minus(ion_a)
    sp_b = ops.sigma_plus(ion_b)
    sm_b = ops.sigma_minus(ion_b)
    a = ops.annihilate(mode)
    ad = ops.create(mode)

    # RSB coupling: sm*a + sp*ad drives |0,n> <-> |1,n-1>
    # sm = sigmam = |1><0| takes |0> -> |1>, a removes a phonon
    # sp = sigmap = |0><1| takes |1> -> |0>, ad adds a phonon
    # (hermitian conjugate)

    # Step 1: RSB pi-pulse on ion A: maps |0_A, n=1> -> |1_A, n=0>
    rsb_rabi_a = eta[0] * rabi_frequency
    H1 = (rsb_rabi_a / 2) * (sm_a * a + sp_a * ad)
    t1 = np.pi / rsb_rabi_a

    # Step 2: RSB 2*pi-pulse on ion B (conditional phase)
    rsb_rabi_b = eta[1] * rabi_frequency
    H2 = (rsb_rabi_b / 2) * (sm_b * a + sp_b * ad)
    t2 = TWO_PI / rsb_rabi_b

    # Step 3: Reverse RSB pi-pulse on ion A (with phase shift pi)
    H3 = (rsb_rabi_a / 2) * (
        sm_a * a * np.exp(1j * np.pi) + sp_a * ad * np.exp(-1j * np.pi)
    )
    t3 = np.pi / rsb_rabi_a

    return [
        GatePulse(hamiltonian=H1, duration=t1),
        GatePulse(hamiltonian=H2, duration=t2),
        GatePulse(hamiltonian=H3, duration=t3),
    ]

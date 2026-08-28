"""Two-level analogue of the Cirac-Zoller sequence (PRL 74, 4091).

The original 1995 protocol needs a third, auxiliary internal level.
The pulse sequence built here omits it, so it is *not* an entangling
gate - see ``cirac_zoller_gate`` for the map it actually realises.
"""

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
    r"""Two-level red-sideband analogue of the Cirac-Zoller sequence.

    .. warning::
       **This is not a controlled-phase gate and not an entangling
       gate.** The auxiliary internal level of Cirac & Zoller,
       PRL 74, 4091 (1995) is required for the real gate; without it
       the sequence below reduces to a local operator plus leakage.
       Use ``ms_gate_hamiltonian`` for entangling operations. The
       function is kept because the three-pulse structure is
       instructive, not because the map is usable.

    Three-step sequence, driven entirely on the *qubit* red sideband:

    1. $\pi$ pulse on the RSB of ion A: maps
       $|1_A, 0\rangle \to -i\,|0_A, 1\rangle$
    2. $2\pi$ pulse on the RSB of ion B, intended to phase only the
       "phonon present" branch $|0_B, 1\rangle$
    3. Reverse $\pi$ pulse on the RSB of ion A ($-H_1$), unmapping the
       motion back onto ion A

    REQUIRES the motional mode to start in $|n = 0\rangle$.

    Measured truth of the implemented map
    -------------------------------------
    On the computational subspace tensored with $|n = 0\rangle$, the
    product of the three exact propagators is diagonal:

    $$
    \mathrm{diag}\bigl(1,\, -1,\, -1,\, \cos(\sqrt{2}\pi)\bigr),
    \qquad \cos(\sqrt{2}\pi) = -0.26626
    $$

    with $\sin^2(\sqrt{2}\pi) = 92.91\%$ of the $|11\rangle$
    population leaking out of $|n = 0\rangle$ (outgoing
    $\bar n = 1.27$). Both numbers are independent of $\eta$,
    $\Omega$, and the Fock truncation. Two distinct defects produce
    this:

    - **Conditionality is lost.** Step 2 drives ion B's own qubit
      sideband, so the resonant doublet is
      $\{|1_B, 0\rangle, |0_B, 1\rangle\}$ - one state from *each*
      logical branch. The same $2\pi$ rotation that correctly phases
      the phonon-present branch also phases $|0_A 1_B, 0\rangle$, so
      $|01\rangle$ picks up the same $-1$ as $|10\rangle$. The
      leakage-free $3\times 3$ block $\mathrm{diag}(1, -1, -1)$ is
      therefore the *local* operator $\sigma_z \otimes \sigma_z$
      restricted to those states, which is separable and generates no
      entanglement. The auxiliary level exists to leave ion B's
      excited qubit state dark, i.e. to supply the conditionality -
      not merely to suppress Fock leakage.
    - **The $|11\rangle$ pulse area is wrong by $\sqrt{2}$.** After
      step 1 that branch sits on $|1_B, 1\rangle$, whose coupling to
      $|0_B, 2\rangle$ carries the $\sqrt{n}$ enhancement, so the
      intended $2\pi$ pulse becomes $2\sqrt{2}\pi$ and leaves the
      surviving amplitude $\cos(\sqrt{2}\pi)$.

    Note that the previous docstring claim "exact for input states
    $|00\rangle$, $|01\rangle$, $|10\rangle$" holds only in the narrow
    sense that those inputs do not leak; the $|01\rangle$ *phase* is
    not the protocol's.

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

    # Step 1: RSB pi-pulse on ion A: maps |1_A, n=0> -> -i|0_A, n=1>
    rsb_rabi_a = eta[0] * rabi_frequency
    H1 = (rsb_rabi_a / 2) * (sm_a * a + sp_a * ad)
    t1 = np.pi / rsb_rabi_a

    # Step 2: RSB 2*pi-pulse on ion B. In the real protocol this runs
    # on a |0_B> <-> |aux_B> sideband so that |1_B> stays dark. Driving
    # ion B's own qubit sideband instead makes |1_B, 0> resonant too,
    # which is what costs the gate its conditionality (see docstring).
    rsb_rabi_b = eta[1] * rabi_frequency
    H2 = (rsb_rabi_b / 2) * (sm_b * a + sp_b * ad)
    t2 = TWO_PI / rsb_rabi_b

    # Step 3: Reverse RSB pi-pulse on ion A (phase shift pi negates H1)
    t3 = np.pi / rsb_rabi_a

    return [
        GatePulse(hamiltonian=H1, duration=t1),
        GatePulse(hamiltonian=H2, duration=t2),
        GatePulse(hamiltonian=-H1, duration=t3),
    ]

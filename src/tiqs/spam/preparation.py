r"""State preparation via optical pumping."""

import numpy as np
import qutip

from tiqs.hilbert_space.operators import OperatorFactory


def optical_pumping_ops(
    ops: OperatorFactory,
    ion: int,
    pumping_rate: float,
) -> list[qutip.Qobj]:
    r"""Collapse operators modeling optical pumping to $|0\rangle$.

    Optical pumping drives $|1\rangle \to |0\rangle$ dissipatively via a
    cycling transition. Modeled as spontaneous decay from $|1\rangle$ to
    $|0\rangle$ at the pumping rate, so an ion starting in $|1\rangle$
    reaches $p_0(t) = 1 - e^{-\Gamma_p t}$.

    $|0\rangle$ = ``basis(2, 0)`` = ground is the TIQS-internal target
    of optical pumping; see `tiqs.spam.measurement` for the
    corresponding bright/dark labeling.

    QuTiP convention: ``sigmap()`` $= |0\rangle\langle 1|$ drives
    $|1\rangle \to |0\rangle$ (de-excitation).

    Parameters
    ----------
    ops : OperatorFactory
    ion : int
    pumping_rate : float
        Effective optical pumping rate $\Gamma_p$ in s$^{-1}$. This is
        an incoherent population-transfer rate, not an angular
        frequency, so no factor of $2\pi$ applies.

    Returns
    -------
    list[qutip.Qobj]
        Collapse operators for optical pumping.

    Raises
    ------
    ValueError
        If ``pumping_rate`` is not positive.

    Notes
    -----
    Idealizations - the channel acts on the qubit alone:

    - **No photon recoil.** Each pumping cycle scatters several
      photons, every one depositing of order $\eta^2$ quanta into the
      motional modes; this model leaves the motion untouched. Compose
      with `tiqs.noise.motional.motional_heating_ops` when recoil
      matters for a ground-state-cooled ion.
    - **No leakage.** Real pumping proceeds via $^2P_{1/2}$ and
      branches into the metastable $D$ states, which is why a repumper
      is required (935 nm for Yb$^+$, 866 nm for Ca$^+$, 650 nm for
      Ba$^+$). No $D$-state population is tracked here.
    - **No error floor.** $p_0 \to 1$ as $\Gamma_p t \to \infty$,
      whereas measured preparation errors saturate near $10^{-4}$
      (Pino et al., *Nature* **592**, 209 (2021)). Adding a reverse
      operator $\sqrt{\Gamma_d}\,\sigma_-$ imposes a floor
      $\Gamma_d / (\Gamma_p + \Gamma_d)$ if one is wanted.
    """
    if pumping_rate <= 0.0:
        raise ValueError(f"pumping_rate must be > 0, got {pumping_rate}")
    # sigma_plus maps to sigmap() = |0><1|, which takes |1> -> |0>
    return [np.sqrt(pumping_rate) * ops.sigma_plus(ion)]


def prepare_qubit(
    ops: OperatorFactory,
    ion: int,
    initial_state: qutip.Qobj,
    pumping_rate: float,
    duration: float,
) -> qutip.Qobj:
    r"""Simulate optical pumping to prepare a qubit in $|0\rangle$.

    Integrates the Lindblad equation with the single collapse operator
    of `optical_pumping_ops` and no Hamiltonian, so the qubit
    population follows $p_1(t) = p_1(0)\, e^{-\Gamma_p t}$ exactly and
    the motional state is unchanged. See `optical_pumping_ops` for the
    idealizations this inherits.

    Parameters
    ----------
    ops : OperatorFactory
    ion : int
    initial_state : qutip.Qobj
        Initial density matrix.
    pumping_rate : float
        Optical pumping rate $\Gamma_p$ in s$^{-1}$.
    duration : float
        Pumping duration in seconds.

    Returns
    -------
    qutip.Qobj
        Final density matrix after pumping.

    Raises
    ------
    ValueError
        If ``pumping_rate`` is not positive or ``duration`` is
        negative.
    """
    if duration < 0.0:
        raise ValueError(f"duration must be >= 0, got {duration}")
    c_ops = optical_pumping_ops(ops, ion, pumping_rate)
    H = qutip.qzero(ops.hs.dims)
    tlist = np.linspace(0, duration, 20)
    result = qutip.mesolve(H, initial_state, tlist, c_ops=c_ops)
    return result.states[-1]

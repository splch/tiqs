"""Light-shift (geometric phase) gate: sigma_z-dependent force."""

from tiqs.gates.molmer_sorensen import _geometric_phase_hamiltonian
from tiqs.hilbert_space.operators import OperatorFactory


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
    H_\mathrm{LS}(t) = \sum_j \eta_j \, \Omega \, \sigma_{z,j}
    \left( a^\dagger e^{i \delta t} + a \, e^{-i \delta t} \right)
    $$

    As with ``ms_gate_hamiltonian``, the factor of $1/2$ from the
    theory-doc convention is absorbed into $\Omega$, so
    ``rabi_frequency`` is half the per-tone value and the
    spin-motion coupling is $\eta\,\Omega$.
    This generates a $\sigma_z \otimes \sigma_z$ interaction
    (ZZ coupling), which is inherently insensitive to the optical
    phase of the laser beams.

    Sign convention: pairing $a^\dagger$ with $e^{+i\delta t}$ gives
    $U_\mathrm{LS} = e^{+i\chi\,\sigma_z^{(i)}\sigma_z^{(j)}}$ with
    $\chi = 4\pi K \eta_i \eta_j \Omega^2 / \delta^2$, maximally
    entangling at $\chi = \pi/4$, i.e. $\eta\Omega = \delta/4$ for two
    identically-coupled ions. Since all the $\sigma_z$ commute, the
    Magnus expansion terminates at second order and this unitary is
    *exact* at $\tau = 2\pi K / |\delta|$.

    Model scope and approximations
    ------------------------------
    Same single-mode, square-pulse, unit-Debye-Waller idealization as
    ``ms_gate_hamiltonian`` (see its scope section for magnitudes),
    with one important difference: the **carrier term is not a
    defect here.** In the far-detuned optical-dipole-force
    configuration the zeroth-order term is the $\sigma_z$ AC Stark
    shift, which commutes exactly with the $\sigma_z$ force and so
    contributes only a calibratable single-qubit phase; there is no
    spin-flip carrier at amplitude $\Omega$. What remains omitted is
    spectator-mode coupling, pulse shaping, and the operator-valued
    Debye-Waller factor.

    Parameters
    ----------
    ops : OperatorFactory
    ions : list[int]
    mode : int
    eta : list[float]
        Lamb-Dicke parameters for each ion.
    rabi_frequency : float
        Effective Rabi frequency from the light shift (rad/s), in the
        same half-amplitude convention as ``ms_gate_hamiltonian``.
    detuning : float
        Detuning from motional sideband (rad/s).

    Returns
    -------
    list
        QuTiP list-format Hamiltonian.
    """
    return _geometric_phase_hamiltonian(
        ops, ions, mode, eta, rabi_frequency, detuning, ops.sigma_z
    )

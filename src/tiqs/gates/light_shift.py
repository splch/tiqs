"""Light-shift (geometric phase) gate: sigma_z-dependent force."""

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
    theory-doc convention is absorbed into $\Omega$.
    This generates a $\sigma_z \otimes \sigma_z$ interaction
    (ZZ coupling), which is inherently insensitive to the optical
    phase of the laser beams.

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

r"""Correlated noise operators across multiple ions.

Real experiments have noise sources that affect all ions
collectively (e.g., one laser drives all ions, one magnetic
field permeates the trap).  Correlated noise gives different
error scaling than independent per-ion noise:

- A Bell state $(|00\rangle + i|11\rangle)/\sqrt{2}$ is
  **immune** to common-mode dephasing (both components shift
  together).
- Independent dephasing overestimates the error on such states
  by roughly 2x.
"""

import numpy as np
import qutip

from tiqs.hilbert_space.operators import OperatorFactory


def correlated_dephasing_op(
    ops: OperatorFactory,
    ions: list[int],
    t2: float,
) -> qutip.Qobj:
    r"""Collapse operator for correlated (common-mode) dephasing.

    $$
    L = \sqrt{\gamma_\phi / 2}\;\sum_j \sigma_z^{(j)}
    $$

    A single noise source (magnetic field, laser phase) shifts
    all ions' frequencies together.  The collective $\sigma_z$
    preserves states where all qubits are in the same
    computational basis state (e.g., $|00\rangle + i|11\rangle$).

    Parameters
    ----------
    ops : OperatorFactory
    ions : list[int]
        Ion indices affected by the common noise source.
    t2 : float
        Collective dephasing time in seconds.

    Returns
    -------
    qutip.Qobj
        Collective collapse operator.
    """
    gamma_phi = 1.0 / t2
    sz_total = sum(ops.sigma_z(i) for i in ions)
    return np.sqrt(gamma_phi / 2) * sz_total


def correlated_phase_noise_op(
    ops: OperatorFactory,
    ions: list[int],
    linewidth: float,
) -> qutip.Qobj:
    r"""Collapse operator for correlated laser/microwave phase noise.

    $$
    L = \sqrt{\gamma / 2}\;\sum_j \sigma_z^{(j)}
    $$

    Laser phase fluctuations imprint the same phase error on every
    ion driven by the same beam.

    Parameters
    ----------
    ops : OperatorFactory
    ions : list[int]
        Ion indices illuminated by the same laser/microwave source.
    linewidth : float
        Effective linewidth of the source in rad/s.

    Returns
    -------
    qutip.Qobj
        Collective phase-noise collapse operator.
    """
    sz_total = sum(ops.sigma_z(i) for i in ions)
    return np.sqrt(linewidth / 2) * sz_total


def correlated_intensity_noise_hamiltonian(
    ops: OperatorFactory,
    ions: list[int],
    fractional_rms: float,
    rabi_frequency: float,
) -> qutip.Qobj:
    r"""Hamiltonian perturbation from correlated intensity noise.

    A single laser's intensity fluctuation causes the same Rabi
    frequency error on all illuminated ions:

    $$
    \delta H = \frac{\delta\Omega}{2}\,\sum_j \sigma_x^{(j)}
    $$

    Parameters
    ----------
    ops : OperatorFactory
    ions : list[int]
    fractional_rms : float
        RMS fractional intensity fluctuation $\delta I / I$.
    rabi_frequency : float
        Nominal Rabi frequency (rad/s).

    Returns
    -------
    qutip.Qobj
        Hermitian Hamiltonian perturbation.
    """
    delta_omega = (fractional_rms / 2) * rabi_frequency
    sx_total = sum(ops.sigma_x(i) for i in ions)
    return (delta_omega / 2) * sx_total

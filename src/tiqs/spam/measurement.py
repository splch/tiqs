r"""State measurement via fluorescence detection."""

import numpy as np
import qutip

from tiqs.hilbert_space.operators import OperatorFactory


def fluorescence_probabilities(
    state: qutip.Qobj,
    ions: list[int],
) -> list[float]:
    r"""Compute probability of each ion being bright.

    Bright state is $|0\rangle$.

    In the standard convention: $|0\rangle$ (``basis(2,0)``) is the bright
    state that fluoresces, $|1\rangle$ (``basis(2,1)``) is the dark state.

    Parameters
    ----------
    state : qutip.Qobj
        Current quantum state (ket or density matrix).
    ions : list[int]
        Ion indices to measure.

    Returns
    -------
    list[float]
        Probability of bright ($|0\rangle$) for each ion.
    """
    rho = qutip.ket2dm(state) if state.isket else state
    return [rho.ptrace(ion)[0, 0].real for ion in ions]


def sample_measurement(
    state: qutip.Qobj,
    ions: list[int],
    rng: np.random.Generator,
    spam_error: float = 0.0,
) -> list[int]:
    r"""Sample a projective measurement outcome from the joint qubit
    distribution.

    Samples from the full joint probability distribution over all
    $2^N$ computational basis states of the measured ions, correctly
    preserving quantum correlations. For entangled states (e.g.,
    Bell states), correlated outcomes are produced.

    Parameters
    ----------
    state : qutip.Qobj
        Full system state (ket or density matrix).
    ions : list[int]
        Ion subsystem indices to measure.
    rng : np.random.Generator
    spam_error : float
        Probability of misidentifying each bit independently.

    Returns
    -------
    list[int]
        Measurement outcomes (0 or 1) for each ion.
    """
    if state.isket:
        rho = qutip.ket2dm(state)
    else:
        rho = state

    rho_ions = rho.ptrace(ions)
    n = len(ions)
    dim = 2**n

    # Extract diagonal of the density matrix in the computational
    # basis. This gives P(bitstring) for each bitstring.
    probs = np.maximum(rho_ions.diag().real, 0.0)
    probs /= probs.sum()

    # Sample one bitstring from the joint distribution
    outcome_idx = rng.choice(dim, p=probs)

    # Convert index to bit list:
    # index 0 -> [0,0,...], index 1 -> [0,0,...,1], etc.
    bits = [(outcome_idx >> (n - 1 - k)) & 1 for k in range(n)]

    # Apply SPAM error independently to each bit
    if spam_error > 0:
        bits = [b if rng.random() > spam_error else (1 - b) for b in bits]

    return bits


def measurement_fidelity(
    bright_photon_rate: float,
    dark_photon_rate: float,
    detection_window: float,
    collection_efficiency: float,
) -> float:
    """Estimate single-shot readout fidelity from photon counting parameters.

    Models the bright/dark discrimination with Poisson photon
    statistics. Optimal threshold minimizes the sum of false-bright
    and false-dark errors.

    Parameters
    ----------
    bright_photon_rate : float
        Photon scattering rate for bright state (photons/s).
    dark_photon_rate : float
        Background/dark count rate (photons/s).
    detection_window : float
        Detection time window (s).
    collection_efficiency : float
        Fraction of emitted photons collected
        (typically 0.02-0.05).

    Returns
    -------
    float
        Estimated readout fidelity.
    """
    from scipy.stats import poisson

    n_bright = bright_photon_rate * detection_window * collection_efficiency
    n_dark = dark_photon_rate * detection_window * collection_efficiency

    thresholds = np.arange(1, max(2, int(n_bright) + 5))
    p_correct_bright = 1 - poisson.cdf(thresholds - 1, n_bright)
    p_correct_dark = poisson.cdf(thresholds - 1, n_dark)
    return float(np.max(0.5 * (p_correct_bright + p_correct_dark)))


def mid_circuit_measurement(
    rho: qutip.Qobj,
    ops: OperatorFactory,
    ion: int,
    rng: np.random.Generator,
) -> tuple[qutip.Qobj, int]:
    r"""Perform a mid-circuit measurement on one ion, projecting and
    renormalizing.

    This models the measurement backaction: the state is projected
    onto $|0\rangle$ or $|1\rangle$ for the measured ion while preserving
    the rest of the system.

    Parameters
    ----------
    rho : qutip.Qobj
        Current density matrix.
    ops : OperatorFactory
    ion : int
        Ion to measure.
    rng : np.random.Generator

    Returns
    -------
    tuple[qutip.Qobj, int]
        (post-measurement state, measurement outcome 0 or 1).
    """
    dims = ops.hs.dims

    # Projectors: P_0 = |0><0|, P_1 = |1><1| on the measured ion
    P0 = qutip.expand_operator(qutip.ket2dm(qutip.basis(2, 0)), dims, ion)
    P1 = qutip.expand_operator(qutip.ket2dm(qutip.basis(2, 1)), dims, ion)

    p0 = (P0 * rho).tr().real
    p1 = (P1 * rho).tr().real

    if rng.random() < p0 / (p0 + p1):
        outcome = 0
        rho_post = P0 * rho * P0 / p0
    else:
        outcome = 1
        rho_post = P1 * rho * P1 / p1

    return rho_post, outcome

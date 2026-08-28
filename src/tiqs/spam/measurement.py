r"""State measurement via fluorescence detection."""

import numpy as np
import qutip

from tiqs.hilbert_space.operators import OperatorFactory


def _validate_ions(state: qutip.Qobj, ions: list[int]) -> None:
    """Check that ``ions`` names distinct two-level subsystems.

    Parameters
    ----------
    state : qutip.Qobj
        State whose subsystem dimensions are inspected.
    ions : list[int]
        Candidate ion subsystem indices.

    Raises
    ------
    ValueError
        If ``ions`` is empty or repeats an index.
    IndexError
        If an index is out of range, or names a subsystem whose
        dimension is not 2 (i.e. a motional mode rather than a qubit).
    """
    dims = state.dims[0]
    if not ions:
        raise ValueError("ions must name at least one ion")
    if len(set(ions)) != len(ions):
        raise ValueError(f"ions must be distinct, got {ions}")
    for ion in ions:
        if ion < 0 or ion >= len(dims):
            raise IndexError(f"Ion index {ion} out of range [0, {len(dims)})")
        if dims[ion] != 2:
            raise IndexError(
                f"Subsystem {ion} has dimension {dims[ion]}, not 2:"
                " it is a motional mode, not an ion qubit"
            )


def fluorescence_probabilities(
    state: qutip.Qobj,
    ions: list[int],
) -> list[float]:
    r"""Compute probability of each ion being bright.

    Bright state is $|0\rangle$.

    TIQS-internal convention: $|0\rangle$ (``basis(2,0)``) is the
    ground state, the state optical pumping prepares, and the state
    labeled bright; $|1\rangle$ (``basis(2,1)``) is dark. This matches
    optical/shelving qubits (e.g. $^{40}$Ca$^+$ $S_{1/2}$ bright vs
    $D_{5/2}$ dark, Myerson et al., PRL **100**, 200502 (2008)) but is
    inverted relative to the usual labeling of direct-fluorescence
    hyperfine qubits, where $|F{=}0, m_F{=}0\rangle$ is dark and
    $|F{=}1, m_F{=}0\rangle$ is bright (Noek et al., Opt. Lett. **38**,
    4735 (2013)). For $^{171}$Yb$^+$ the physical (dark) $|F{=}0\rangle$
    level therefore maps onto TIQS $|0\rangle$, and the probabilities
    returned here are its populations; relabel on output if the
    hyperfine convention is wanted.

    Parameters
    ----------
    state : qutip.Qobj
        Current quantum state (ket or density matrix).
    ions : list[int]
        Ion indices to measure. Positional correspondence with the
        returned list is honored for any order.

    Returns
    -------
    list[float]
        Probability of bright ($|0\rangle$) for each ion.

    Raises
    ------
    ValueError
        If ``ions`` is empty or repeats an index.
    IndexError
        If an index is out of range or names a motional mode.
    """
    rho = qutip.ket2dm(state) if state.isket else state
    _validate_ions(rho, ions)
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
        Ion subsystem indices to measure. Any order is accepted;
        ``result[k]`` is the outcome for ``ions[k]``.
    rng : np.random.Generator
    spam_error : float
        Probability of misidentifying each bit independently.

    Returns
    -------
    list[int]
        Measurement outcomes (0 or 1) for each ion, in the order the
        ions were requested.

    Raises
    ------
    ValueError
        If ``ions`` is empty or repeats an index.
    IndexError
        If an index is out of range or names a motional mode.

    Notes
    -----
    ``Qobj.ptrace`` sorts its argument and never reorders subsystems,
    so the reduced density matrix is always in ascending subsystem
    order. The bits are permuted back onto the caller's ``ions`` order
    before returning.
    """
    if state.isket:
        rho = qutip.ket2dm(state)
    else:
        rho = state

    _validate_ions(rho, ions)

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

    # ptrace returned sorted(ions); ranks[k] is the position of
    # ions[k] within sorted(ions), so this restores caller order.
    ranks = np.argsort(np.argsort(ions))
    bits = [bits[r] for r in ranks]

    # Apply SPAM error independently to each bit
    if spam_error > 0:
        bits = [b if rng.random() > spam_error else (1 - b) for b in bits]

    return bits


def measurement_fidelity(
    bright_photon_rate: float,
    dark_photon_rate: float,
    detection_window: float,
    collection_efficiency: float,
    background_rate: float = 0.0,
) -> float:
    r"""Estimate single-shot readout fidelity from photon counting
    parameters.

    Models the bright/dark discrimination with Poisson photon
    statistics. The two count means are

    $$
    \mu_b = R_b\, t_\text{det}\, \eta_c + R_\text{bg}\, t_\text{det},
    \qquad
    \mu_d = R_d\, t_\text{det}\, \eta_c + R_\text{bg}\, t_\text{det},
    $$

    and the threshold $n^*$ is the Bayes-optimal one for equal priors,
    where the two Poisson likelihoods cross:
    $n^* = (\mu_b - \mu_d)/\ln(\mu_b/\mu_d)$. The returned fidelity is
    $\tfrac12[P(n \geq n^* \mid \mu_b) + P(n < n^* \mid \mu_d)]$
    maximized over integer thresholds bracketing $n^*$.

    Parameters
    ----------
    bright_photon_rate : float
        Photon scattering rate for bright state (photons/s emitted by
        the ion, so scaled by ``collection_efficiency``).
    dark_photon_rate : float
        Off-resonant photon scattering rate of the dark state
        (photons/s emitted by the ion, so also scaled by
        ``collection_efficiency``). This is an ion-side rate, not a
        detector background; use ``background_rate`` for that.
    detection_window : float
        Detection time window (s).
    collection_efficiency : float
        Fraction of emitted photons collected and detected. Measured
        values: 0.001 for an NA $\approx 0.3$ objective (Olmschenk
        et al., PRA **76**, 052314 (2007)), 0.0019 (Myerson et al.,
        PRL **100**, 200502 (2008)), 0.017 (Gaebler et al., PRA
        **104**, 062440 (2021)), up to 0.022 with a dedicated NA = 0.6
        objective (Noek et al., Opt. Lett. **38**, 4735 (2013)).
    background_rate : float, optional
        Detector-side background count rate (counts/s): stray light
        plus detector dark counts. This is already a *detected* rate,
        so it is not attenuated by ``collection_efficiency``, and it
        adds to both count means. Default 0.

    Returns
    -------
    float
        Estimated readout fidelity.

    Raises
    ------
    ValueError
        If any rate is negative, ``detection_window`` is not positive,
        ``collection_efficiency`` is outside $[0, 1]$, or the bright
        count mean does not exceed the dark count mean (no
        discrimination is possible then).

    Notes
    -----
    Photon statistics only: this is an **upper bound** on the
    achievable fidelity. Loss of the qubit state *during* the
    detection window - metastable-state decay for a shelved qubit,
    off-resonant repumping for a hyperfine qubit - is not modeled,
    and that term dominates real experiments. At Myerson's published
    conditions ($R_b = 55800$/s, $R_d = 442$/s, $t_b = 420\;\mu$s,
    both rates already detected so $\eta_c = 1$) this function returns
    an error of $1.3\times 10^{-6}$, whereas the measured
    threshold-method error is $1.8(1)\times 10^{-4}$, dominated by
    $D_{5/2}$ decay at $t_b/\tau_D = 420\;\mu\text{s}/1.168\;\text{s}
    = 3.6\times 10^{-4}$. Add such terms externally; TIQS qubits are
    strictly two-level, so no shelved level exists to decay here.
    """
    from scipy.stats import poisson

    if bright_photon_rate < 0.0:
        raise ValueError(
            f"bright_photon_rate must be >= 0, got {bright_photon_rate}"
        )
    if dark_photon_rate < 0.0:
        raise ValueError(
            f"dark_photon_rate must be >= 0, got {dark_photon_rate}"
        )
    if background_rate < 0.0:
        raise ValueError(
            f"background_rate must be >= 0, got {background_rate}"
        )
    if detection_window <= 0.0:
        raise ValueError(
            f"detection_window must be > 0, got {detection_window}"
        )
    if not 0.0 <= collection_efficiency <= 1.0:
        raise ValueError(
            "collection_efficiency must be in [0, 1], got"
            f" {collection_efficiency}"
        )

    n_background = background_rate * detection_window
    n_bright = (
        bright_photon_rate * detection_window * collection_efficiency
        + n_background
    )
    n_dark = (
        dark_photon_rate * detection_window * collection_efficiency
        + n_background
    )
    if n_bright <= n_dark:
        raise ValueError(
            f"bright count mean {n_bright} must exceed dark count mean"
            f" {n_dark} for the states to be distinguishable"
        )

    # Bayes-optimal threshold for equal priors: the Poisson likelihood
    # ratio crosses unity at n* = (mu_b - mu_d)/ln(mu_b/mu_d), and the
    # fidelity is unimodal in the integer threshold, so a few counts on
    # either side of n* bracket the optimum (thresholds are >= 1).
    if n_dark <= 0.0:
        n_star = 0.0
    else:
        n_star = (n_bright - n_dark) / np.log(n_bright / n_dark)
    lo = max(1, int(np.floor(n_star)) - 2)
    thresholds = np.arange(lo, int(np.ceil(n_star)) + 4)

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
    the rest of the system. Outcome $i$ is drawn with probability
    $\mathrm{tr}(P_i \rho)/\mathrm{tr}(\rho)$.

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

    Raises
    ------
    IndexError
        If ``ion`` is not a valid ion index for ``ops``.
    ValueError
        If ``rho`` has no population on the measured ion (zero trace).

    Notes
    -----
    An ideal, instantaneous projection. None of the detection-side
    physics is included: no photon recoil into the motional modes, no
    scattered-light AC Stark shift or depumping of neighboring ions
    (measured at $\sim 2\times 10^{-5}$ crosstalk in QCCD
    architectures, Pino et al., *Nature* **592**, 209 (2021)), and no
    state decay during the detection window. Ions are strictly
    two-level here, so shelving cannot be represented either.
    """
    n_ions = ops.hs.n_ions
    if ion < 0 or ion >= n_ions:
        raise IndexError(f"Ion index {ion} out of range [0, {n_ions})")
    dims = ops.hs.dims

    # Projectors: P_0 = |0><0|, P_1 = |1><1| on the measured ion
    P0 = qutip.expand_operator(qutip.ket2dm(qutip.basis(2, 0)), dims, ion)
    P1 = qutip.expand_operator(qutip.ket2dm(qutip.basis(2, 1)), dims, ion)

    p0 = (P0 * rho).tr().real
    p1 = (P1 * rho).tr().real

    if p0 + p1 <= 0.0:
        raise ValueError(
            f"State has zero trace on ion {ion} (p0={p0}, p1={p1});"
            " cannot draw a measurement outcome"
        )

    if rng.random() < p0 / (p0 + p1):
        outcome = 0
        rho_post = P0 * rho * P0 / p0
    else:
        outcome = 1
        rho_post = P1 * rho * P1 / p1

    return rho_post, outcome

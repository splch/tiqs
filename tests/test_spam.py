import math

import numpy as np
import pytest
import qutip

from tiqs.hilbert_space.builder import HilbertSpace
from tiqs.hilbert_space.operators import OperatorFactory
from tiqs.hilbert_space.states import StateFactory
from tiqs.spam.measurement import (
    fluorescence_probabilities,
    measurement_fidelity,
    mid_circuit_measurement,
    sample_measurement,
)
from tiqs.spam.preparation import optical_pumping_ops, prepare_qubit


@pytest.fixture
def system():
    hs = HilbertSpace(n_ions=2, n_modes=1, n_fock=5)
    ops = OperatorFactory(hs)
    sf = StateFactory(hs)
    return hs, ops, sf


def _poisson_cdf_table(mu: float, n_max: int) -> list[float]:
    """P(n <= k) for k = 0..n_max, from the pmf series in log space.

    Independent of ``scipy.stats.poisson``: each term is evaluated as
    exp(-mu + n ln mu - ln n!) so the reference cannot inherit a bug
    from the implementation's CDF routine. Terms are positive and sum
    to at most 1, and Neumaier compensation keeps the accumulation
    exact to well under the 1e-12 tolerance the tests use.
    """
    if mu == 0.0:
        return [1.0] * (n_max + 1)
    log_mu = math.log(mu)
    total = 0.0
    compensation = 0.0
    cdf = []
    for n in range(n_max + 1):
        term = math.exp(-mu + n * log_mu - math.lgamma(n + 1))
        updated = total + term
        if abs(total) >= abs(term):
            compensation += (total - updated) + term
        else:
            compensation += (term - updated) + total
        total = updated
        cdf.append(total + compensation)
    return cdf


def _threshold_fidelity_reference(mu_bright: float, mu_dark: float) -> float:
    """Brute-force maximum of the Poisson threshold fidelity.

    Scans every integer threshold out to well beyond the bright mean,
    so it does not depend on the closed-form optimal threshold used by
    ``measurement_fidelity``.
    """
    n_max = math.ceil(mu_bright + 10.0 * math.sqrt(mu_bright) + 10.0)
    cdf_bright = _poisson_cdf_table(mu_bright, n_max)
    cdf_dark = _poisson_cdf_table(mu_dark, n_max)
    return max(
        0.5 * ((1.0 - cdf_bright[t - 1]) + cdf_dark[t - 1])
        for t in range(1, n_max + 1)
    )


class TestPreparation:
    def test_pumping_operator_drives_excited_to_ground(self, system):
        r"""The collapse operator must take $|1\rangle \to |0\rangle$.

        A sigma_plus/sigma_minus swap would pump the wrong way while
        still producing a single, correctly normalized operator.
        """
        hs, ops, sf = system
        rate = 4e6
        (c_op,) = optical_pumping_ops(ops, ion=0, pumping_rate=rate)

        excited = sf.product_state([1, 0], [0])
        ground = sf.product_state([0, 0], [0])
        assert (c_op * excited - math.sqrt(rate) * ground).norm() < 1e-9
        assert (c_op * ground).norm() < 1e-12

    @pytest.mark.parametrize("gamma_t", [0.2, 0.5, 1.0, 2.0, 3.0, 5.0])
    def test_prepare_qubit_follows_exponential_law(self, system, gamma_t):
        r"""Pumping an ion from $|1\rangle$ obeys $p_0 = 1 - e^{-\Gamma t}$.

        Exact solution of the Lindblad equation for a single decay
        channel; pins the rate so a factor error cannot hide behind a
        saturated population.
        """
        hs, ops, sf = system
        rate = 1e5
        rho0 = qutip.ket2dm(sf.product_state([1, 0], [0]))
        rho = prepare_qubit(
            ops,
            ion=0,
            initial_state=rho0,
            pumping_rate=rate,
            duration=gamma_t / rate,
        )
        p_ground = rho.ptrace(0)[0, 0].real
        assert p_ground == pytest.approx(1.0 - math.exp(-gamma_t), abs=1e-5)

    def test_prepare_qubit_leaves_motion_untouched(self, system):
        """No photon recoil: the documented idealization, pinned.

        The channel is qubit-local, so a Fock state survives pumping
        exactly. Guards the ``optical_pumping_ops`` Notes section.
        """
        hs, ops, sf = system
        rho0 = qutip.ket2dm(sf.product_state([1, 0], [1]))
        rho = prepare_qubit(
            ops,
            ion=0,
            initial_state=rho0,
            pumping_rate=1e7,
            duration=1e-6,
        )
        n_bar = qutip.expect(ops.number(0), rho)
        assert n_bar == pytest.approx(1.0, abs=1e-9)

    @pytest.mark.parametrize("rate", [0.0, -1e6])
    def test_pumping_rate_must_be_positive(self, system, rate):
        """A non-positive rate raised no error and produced NaNs."""
        hs, ops, sf = system
        with pytest.raises(ValueError, match="pumping_rate must be > 0"):
            optical_pumping_ops(ops, ion=0, pumping_rate=rate)

    def test_prepare_qubit_rejects_negative_duration(self, system):
        hs, ops, sf = system
        rho0 = qutip.ket2dm(sf.ground_state())
        with pytest.raises(ValueError, match="duration must be >= 0"):
            prepare_qubit(
                ops,
                ion=0,
                initial_state=rho0,
                pumping_rate=1e6,
                duration=-1e-6,
            )


class TestFluorescence:
    def test_fluorescence_ground_state_bright(self, system):
        """Ground state |0> is bright in the TIQS-internal convention."""
        hs, ops, sf = system
        psi = sf.ground_state()
        probs = fluorescence_probabilities(psi, ions=[0, 1])
        assert probs[0] == pytest.approx(1.0, abs=1e-12)
        assert probs[1] == pytest.approx(1.0, abs=1e-12)

    def test_fluorescence_excited_dark(self, system):
        """Excited state |1> is dark."""
        hs, ops, sf = system
        psi = sf.product_state([1, 0], [0])
        probs = fluorescence_probabilities(psi, ions=[0, 1])
        assert probs[0] == pytest.approx(0.0, abs=1e-12)
        assert probs[1] == pytest.approx(1.0, abs=1e-12)

    def test_fluorescence_honors_ion_order(self, system):
        """probs[k] belongs to ions[k], for any order."""
        hs, ops, sf = system
        psi = sf.product_state([1, 0], [0])
        assert fluorescence_probabilities(psi, ions=[1, 0]) == pytest.approx(
            [1.0, 0.0], abs=1e-12
        )

    def test_fluorescence_rejects_mode_index(self, system):
        """A motional index is not an ion; |n=0> is not a bright state."""
        hs, ops, sf = system
        psi = sf.ground_state()
        with pytest.raises(IndexError, match="not 2"):
            fluorescence_probabilities(psi, ions=[2])

    def test_fluorescence_rejects_out_of_range(self, system):
        hs, ops, sf = system
        psi = sf.ground_state()
        with pytest.raises(IndexError, match="out of range"):
            fluorescence_probabilities(psi, ions=[7])


class TestSampleMeasurement:
    def test_sample_measurement_returns_bits(self, system):
        hs, ops, sf = system
        psi = sf.ground_state()
        bits = sample_measurement(
            psi, ions=[0, 1], rng=np.random.default_rng(42)
        )
        assert len(bits) == 2
        assert all(b in [0, 1] for b in bits)

    @pytest.mark.parametrize(
        ("ions", "expected"),
        [
            ([0, 1, 2], [0, 1, 1]),
            ([2, 1, 0], [1, 1, 0]),
            ([1, 0], [1, 0]),
            ([2, 0], [1, 0]),
            ([1, 2, 0], [1, 1, 0]),
        ],
    )
    def test_sample_measurement_honors_ion_order(self, ions, expected):
        """bits[k] must label ions[k] even for non-ascending lists.

        ``Qobj.ptrace`` sorts its selection, so the outcome bits used
        to come back in ascending subsystem order regardless of the
        requested order. Deterministic product state |0,1,1>.
        """
        hs = HilbertSpace(n_ions=3, n_modes=1, n_fock=4)
        sf = StateFactory(hs)
        psi = sf.product_state([0, 1, 1], [0])
        rng = np.random.default_rng(7)
        assert sample_measurement(psi, ions, rng) == expected

    def test_sample_measurement_marginals_follow_ion_order(self, system):
        """Permuted marginals must swap, not stay put."""
        hs, ops, sf = system
        p1_ion0, p1_ion1 = 0.1, 0.9
        psi = qutip.tensor(
            math.sqrt(1 - p1_ion0) * qutip.basis(2, 0)
            + math.sqrt(p1_ion0) * qutip.basis(2, 1),
            math.sqrt(1 - p1_ion1) * qutip.basis(2, 0)
            + math.sqrt(p1_ion1) * qutip.basis(2, 1),
            qutip.basis(5, 0),
        )
        rng = np.random.default_rng(2024)
        n_shots = 800
        counts = np.zeros(2)
        for _ in range(n_shots):
            counts += sample_measurement(psi, [1, 0], rng)
        freq = counts / n_shots
        tol = 4.0 * math.sqrt(0.25 / n_shots)
        assert freq[0] == pytest.approx(p1_ion1, abs=tol)
        assert freq[1] == pytest.approx(p1_ion0, abs=tol)

    def test_sample_measurement_permutation_keeps_correlations(self, system):
        """Reordering must not break the joint distribution.

        The state sqrt(0.8)|01> + sqrt(0.2)|10> is perfectly
        anti-correlated, so every shot must sum to 1 whatever the
        requested order, and the ion-1 marginal must be 0.8.
        """
        hs, ops, sf = system
        psi = (
            math.sqrt(0.8) * sf.product_state([0, 1], [0])
            + math.sqrt(0.2) * sf.product_state([1, 0], [0])
        ).unit()
        rng = np.random.default_rng(11)
        n_shots = 800
        ones_ion1 = 0
        for _ in range(n_shots):
            bits = sample_measurement(psi, [1, 0], rng)
            assert sum(bits) == 1
            ones_ion1 += bits[0]
        tol = 4.0 * math.sqrt(0.8 * 0.2 / n_shots)
        assert ones_ion1 / n_shots == pytest.approx(0.8, abs=tol)

    @pytest.mark.parametrize("spam_error", [0.1, 0.5])
    def test_spam_error_flip_rate(self, system, spam_error):
        """Flip frequency must match spam_error within 4 sigma.

        The old +/-7-sigma window also accepted an effective error of
        0.3 or 0.65.
        """
        hs, ops, sf = system
        psi = sf.ground_state()
        rng = np.random.default_rng(42)
        n_shots = 800
        flipped = 0
        for _ in range(n_shots):
            bits = sample_measurement(
                psi, ions=[0], rng=rng, spam_error=spam_error
            )
            flipped += bits[0]
        tol = 4.0 * math.sqrt(spam_error * (1 - spam_error) / n_shots)
        assert flipped / n_shots == pytest.approx(spam_error, abs=tol)

    def test_sample_measurement_rejects_duplicate_ions(self, system):
        hs, ops, sf = system
        psi = sf.ground_state()
        with pytest.raises(ValueError, match="distinct"):
            sample_measurement(psi, [0, 0], np.random.default_rng(0))

    def test_sample_measurement_rejects_mode_index(self, system):
        hs, ops, sf = system
        psi = sf.ground_state()
        with pytest.raises(IndexError, match="not 2"):
            sample_measurement(psi, [0, 2], np.random.default_rng(0))


class TestMeasurementFidelity:
    def test_single_photon_threshold_closed_form(self):
        r"""Pin the fidelity where it is analytic.

        For $\mu_b = 2$, $\mu_d = 2\times 10^{-3}$ the Bayes-optimal
        threshold is $n^* = (\mu_b-\mu_d)/\ln(\mu_b/\mu_d) = 0.289$, so
        one count suffices and
        $F = \tfrac12[(1-e^{-\mu_b}) + e^{-\mu_d}]$ exactly.
        """
        expected = 0.5 * ((1 - math.exp(-2.0)) + math.exp(-2e-3))
        assert measurement_fidelity(2.0, 2e-3, 1.0, 1.0) == pytest.approx(
            expected, abs=1e-14
        )
        # mu_d = 0 collapses the dark term to unity.
        assert measurement_fidelity(2.0, 0.0, 1.0, 1.0) == pytest.approx(
            0.5 * (1 - math.exp(-2.0)) + 0.5, abs=1e-14
        )

    def test_collection_efficiency_scales_both_means(self):
        """Both ion-side rates carry eta; dropping either changes F.

        Same (mu_b, mu_d) reached three ways, so the eta factors are
        pinned rather than merely present.
        """
        expected = 0.5 * ((1 - math.exp(-2.0)) + math.exp(-2e-3))
        assert measurement_fidelity(200.0, 0.2, 1.0, 0.01) == pytest.approx(
            expected, abs=1e-14
        )
        assert measurement_fidelity(2e5, 200.0, 1e-3, 0.01) == pytest.approx(
            expected, abs=1e-14
        )

    @pytest.mark.parametrize(
        ("mu_bright", "mu_dark"),
        [
            (0.5, 1e-3),
            (2.0, 2e-3),
            (23.436, 0.18564),
            (90.0, 9e-4),
            (200.0, 5.0),
            (1e4, 1.0),
        ],
    )
    def test_matches_brute_force_threshold_scan(self, mu_bright, mu_dark):
        """Closed-form threshold must find the brute-force optimum.

        Reference sums the Poisson pmf series directly and scans every
        integer threshold, so a mis-centered or too-narrow bracket
        shows up immediately.
        """
        got = measurement_fidelity(mu_bright, mu_dark, 1.0, 1.0)
        assert got == pytest.approx(
            _threshold_fidelity_reference(mu_bright, mu_dark), abs=1e-12
        )

    def test_background_rate_is_not_scaled_by_efficiency(self):
        """Detector background is an already-detected rate.

        Myerson et al., PRL 100, 200502 (2008) quote R_D = 442/s at a
        net efficiency of 0.19%, i.e. R_D is measured at the detector.
        Passing it as ``background_rate`` must equal folding R_D/eta
        into both ion-side rates, and must be far worse than
        eta-scaling it (which the old single-argument model did).
        """
        r_bg, window, eta = 442.0, 420e-6, 0.0019
        r_bright = 55800.0 / eta

        detector_side = measurement_fidelity(
            r_bright, 0.0, window, eta, background_rate=r_bg
        )
        folded = measurement_fidelity(
            r_bright + r_bg / eta, r_bg / eta, window, eta
        )
        assert detector_side == pytest.approx(folded, abs=1e-15)

        # mu_d = R_bg * t = 0.18564 exactly, not R_bg * t * eta.
        eta_scaled = measurement_fidelity(r_bright, r_bg, window, eta)
        assert (1 - detector_side) > 100 * (1 - eta_scaled)

    def test_myerson_conditions_are_an_upper_bound(self):
        """Photon statistics alone cannot reach the measured error.

        Myerson et al., PRL 100, 200502 (2008): R_B = 55800/s,
        R_D = 442/s, t_b = 420 us (both rates already detected) with a
        measured threshold-method error of 1.8(1)e-4, dominated by
        D_5/2 decay during t_b (420 us / 1.168 s = 3.6e-4). Decay
        during detection is deliberately not modeled, so the error
        here must sit far below the published one.
        """
        fid = measurement_fidelity(55800.0, 442.0, 420e-6, 1.0)
        error = 1 - fid
        assert error == pytest.approx(1.285e-6, rel=1e-3)
        assert error < 1.8e-4 / 100

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"collection_efficiency": 5.0}, "collection_efficiency"),
            ({"collection_efficiency": -0.03}, "collection_efficiency"),
            ({"detection_window": 0.0}, "detection_window"),
            ({"detection_window": -3e-4}, "detection_window"),
            ({"bright_photon_rate": -1.0}, "bright_photon_rate"),
            ({"dark_photon_rate": -1.0}, "dark_photon_rate"),
            ({"background_rate": -1.0}, "background_rate"),
        ],
    )
    def test_rejects_unphysical_inputs(self, kwargs, match):
        """These used to return NaN or an over-unity-efficiency value."""
        args = {
            "bright_photon_rate": 1e7,
            "dark_photon_rate": 100.0,
            "detection_window": 300e-6,
            "collection_efficiency": 0.03,
        }
        args.update(kwargs)
        with pytest.raises(ValueError, match=match):
            measurement_fidelity(**args)

    def test_rejects_indistinguishable_means(self):
        """No threshold can separate equal Poisson means."""
        with pytest.raises(ValueError, match="must exceed dark count mean"):
            measurement_fidelity(100.0, 100.0, 1e-3, 0.5)


class TestMidCircuitMeasurement:
    def test_mid_circuit_projects(self, system):
        hs, ops, sf = system
        plus = (
            sf.product_state([0, 0], [0]) + sf.product_state([1, 0], [0])
        ).unit()
        rho_out, outcome = mid_circuit_measurement(
            qutip.ket2dm(plus),
            ops,
            ion=0,
            rng=np.random.default_rng(42),
        )
        # After measurement, ion 0 should be in a definite state
        rho_q = rho_out.ptrace(0)
        assert max(rho_q.eigenenergies()) == pytest.approx(1.0, abs=1e-12)
        assert rho_out.tr() == pytest.approx(1.0, abs=1e-12)
        assert rho_q[outcome, outcome].real == pytest.approx(1.0, abs=1e-12)

    def test_mid_circuit_outcome_frequencies_obey_born_rule(self, system):
        """Outcome 0 must occur with probability tr(P_0 rho).

        Amplitudes sqrt(0.8)/sqrt(0.2) give p0 = 0.8; checked to 4
        binomial sigma.
        """
        hs, ops, sf = system
        p0 = 0.8
        psi = (
            math.sqrt(p0) * sf.product_state([0, 0], [0])
            + math.sqrt(1 - p0) * sf.product_state([1, 0], [0])
        ).unit()
        rho = qutip.ket2dm(psi)
        rng = np.random.default_rng(5)
        n_shots = 800
        zeros = 0
        for _ in range(n_shots):
            _, outcome = mid_circuit_measurement(rho, ops, 0, rng)
            zeros += 1 - outcome
        tol = 4.0 * math.sqrt(p0 * (1 - p0) / n_shots)
        assert zeros / n_shots == pytest.approx(p0, abs=tol)

    def test_mid_circuit_preserves_spectator_ion(self, system):
        """Projecting ion 0 must leave ion 1 exactly where it was."""
        hs, ops, sf = system
        spectator = math.sqrt(0.7) * qutip.basis(2, 0) + math.sqrt(
            0.3
        ) * qutip.basis(2, 1)
        psi = qutip.tensor(
            (qutip.basis(2, 0) + qutip.basis(2, 1)).unit(),
            spectator,
            qutip.basis(5, 0),
        )
        rho_out, _ = mid_circuit_measurement(
            qutip.ket2dm(psi), ops, 0, np.random.default_rng(3)
        )
        assert (rho_out.ptrace(1) - qutip.ket2dm(spectator)).norm() < 1e-12

    @pytest.mark.parametrize("ion", [2, 3, -1])
    def test_mid_circuit_rejects_bad_ion_index(self, system, ion):
        """Mode and out-of-range indices used to reach QuTiP."""
        hs, ops, sf = system
        rho = qutip.ket2dm(sf.ground_state())
        with pytest.raises(IndexError, match="out of range"):
            mid_circuit_measurement(rho, ops, ion, np.random.default_rng(0))

    def test_mid_circuit_rejects_zero_trace(self, system):
        """Used to raise ZeroDivisionError."""
        hs, ops, sf = system
        rho = qutip.qzero(ops.hs.dims)
        with pytest.raises(ValueError, match="zero trace"):
            mid_circuit_measurement(rho, ops, 0, np.random.default_rng(0))

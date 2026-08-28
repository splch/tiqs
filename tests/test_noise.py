import numpy as np
import pytest
import qutip

from tiqs.constants import ELECTRON_CHARGE, HBAR, TWO_PI
from tiqs.hilbert_space.builder import HilbertSpace
from tiqs.hilbert_space.operators import OperatorFactory
from tiqs.hilbert_space.states import StateFactory
from tiqs.noise.crosstalk import crosstalk_hamiltonian
from tiqs.noise.laser_noise import (
    laser_intensity_noise_op,
    laser_phase_noise_op,
)
from tiqs.noise.motional import (
    heating_rate_from_noise,
    motional_dephasing_op,
    motional_heating_ops,
)
from tiqs.noise.photon_scattering import (
    raman_scattering_ops,
    rayleigh_scattering_op,
)
from tiqs.noise.qubit import qubit_dephasing_op, spontaneous_emission_op
from tiqs.species.ion import get_species

TIGHT = {"atol": 1e-12, "rtol": 1e-10, "nsteps": 100_000}
"""Solver options for tests that compare against exact analytic laws."""


@pytest.fixture
def system():
    """Two qubits plus one (nearly trivial) mode for spin channels."""
    hs = HilbertSpace(n_ions=2, n_modes=1, n_fock=2)
    ops = OperatorFactory(hs)
    sf = StateFactory(hs)
    return hs, ops, sf


@pytest.fixture
def motion():
    """One ion with a generously truncated mode for motional channels."""
    hs = HilbertSpace(n_ions=1, n_modes=1, n_fock=40)
    ops = OperatorFactory(hs)
    sf = StateFactory(hs)
    return hs, ops, sf


def _lindblad_rate(c_ops, observable, state):
    r"""Exact $d\langle O\rangle/dt$ from the dissipator alone."""
    return sum(
        qutip.expect(
            L.dag() * observable * L
            - 0.5 * (L.dag() * L * observable + observable * L.dag() * L),
            state,
        )
        for L in c_ops
    )


def _plus_state(ops, sf, ion):
    """Equal superposition on one ion, everything else in the ground state."""
    hs = ops.hs
    qubits = [0] * hs.n_ions
    ket0 = sf.product_state(qubits, [0] * hs.n_modes)
    qubits[ion] = 1
    ket1 = sf.product_state(qubits, [0] * hs.n_modes)
    return (ket0 + ket1).unit()


class TestMotionalHeating:
    def test_infinite_temperature_bath_heats_linearly(self, motion):
        """<n>(t) = ndot*t exactly.

        Anomalous heating from the ground state is linear in time
        (Turchette et al., PRA 62, 053807 (2000) Sec. III.A.3); the
        infinite-temperature bath (n_bar_env=None) must reproduce that
        law, not exponential growth.
        """
        _, ops, sf = motion
        n_dot = 1e4
        c_ops = motional_heating_ops(ops, mode=0, heating_rate=n_dot)
        assert len(c_ops) == 2
        tlist = np.linspace(0, 2e-4, 9)
        result = qutip.mesolve(
            0 * ops.identity(),
            sf.ground_state(),
            tlist,
            c_ops=c_ops,
            e_ops=[ops.number(0)],
            options=TIGHT,
        )
        np.testing.assert_allclose(result.expect[0], n_dot * tlist, atol=1e-6)

    def test_finite_bath_equilibrates_at_n_bar_env(self, motion):
        """<n>(t) = n_bar_env * (1 - exp(-Gamma t)).

        Turchette et al., PRA 62, 053807 (2000) Eq. (8) with
        Gamma = ndot / n_bar_env, so the mode equilibrates at
        n_bar_env instead of growing without bound.
        """
        _, ops, sf = motion
        n_dot, n_bar = 1e4, 3.0
        gamma = n_dot / n_bar
        c_ops = motional_heating_ops(
            ops, mode=0, heating_rate=n_dot, n_bar_env=n_bar
        )
        tlist = np.linspace(0, 2e-3, 9)
        result = qutip.mesolve(
            0 * ops.identity(),
            sf.ground_state(),
            tlist,
            c_ops=c_ops,
            e_ops=[ops.number(0)],
            options=TIGHT,
        )
        expected = n_bar * (1.0 - np.exp(-gamma * tlist))
        np.testing.assert_allclose(result.expect[0], expected, atol=1e-3)

    @pytest.mark.parametrize("n_bar_env", [None, 0.5, 3.0, 100.0])
    def test_ground_state_slope_is_the_heating_rate(self, motion, n_bar_env):
        """d<n>/dt from vacuum equals ndot for every bath temperature.

        Brownnutt et al., RMP 87, 1419 (2015) Eq. (18): close to the
        ground state ndot = Gamma * N_bar, independent of N_bar.
        """
        _, ops, sf = motion
        n_dot = 1e4
        c_ops = motional_heating_ops(
            ops, mode=0, heating_rate=n_dot, n_bar_env=n_bar_env
        )
        slope = _lindblad_rate(c_ops, ops.number(0), sf.ground_state())
        assert slope == pytest.approx(n_dot, rel=1e-12)

    @pytest.mark.parametrize("n_bar_env", [None, 0.5, 3.0])
    def test_damping_coefficient_has_the_thermal_sign(self, motion, n_bar_env):
        """d<n>/dt decreases with <n>: the bath damps, never amplifies.

        The inverted assignment (n_bar+1 on a^dag) gives
        d<n>/dt = ndot(<n> + n_bar + 1), which grows with <n> and has
        no fixed point.
        """
        _, ops, sf = motion
        n_dot = 1e4
        c_ops = motional_heating_ops(
            ops, mode=0, heating_rate=n_dot, n_bar_env=n_bar_env
        )
        n_op = ops.number(0)
        slope_0 = _lindblad_rate(c_ops, n_op, sf.ground_state())
        slope_5 = _lindblad_rate(c_ops, n_op, sf.product_state([0], [5]))
        assert slope_5 <= slope_0
        if n_bar_env is None:
            assert slope_5 == pytest.approx(slope_0, rel=1e-12)
        else:
            gamma = n_dot / n_bar_env
            assert slope_5 == pytest.approx(n_dot - 5 * gamma, rel=1e-12)

    def test_zero_temperature_bath_is_rejected(self, motion):
        """A T=0 bath cannot heat, so n_bar_env=0 must not be accepted."""
        _, ops, _ = motion
        with pytest.raises(ValueError, match="only cools"):
            motional_heating_ops(ops, mode=0, heating_rate=1e4, n_bar_env=0.0)

    def test_negative_inputs_are_rejected(self, motion):
        _, ops, _ = motion
        with pytest.raises(ValueError, match="heating_rate"):
            motional_heating_ops(ops, mode=0, heating_rate=-1.0)
        with pytest.raises(ValueError, match="n_bar_env"):
            motional_heating_ops(ops, mode=0, heating_rate=1e4, n_bar_env=-1.0)

    def test_zero_rate_gives_no_operators(self, motion):
        _, ops, _ = motion
        assert motional_heating_ops(ops, mode=0, heating_rate=0.0) == []


class TestMotionalDephasing:
    @pytest.mark.parametrize("n_hi", [1, 2, 3])
    def test_fock_coherence_decays_as_delta_n_squared(self, motion, n_hi):
        """<n|rho|n'> decays at rate*(n - n')^2/2.

        Exact solution of the L = sqrt(gamma) * n dissipator.
        """
        _, ops, sf = motion
        rate = 1e4
        c_op = motional_dephasing_op(ops, mode=0, rate=rate)
        ket_lo = sf.product_state([0], [0])
        ket_hi = sf.product_state([0], [n_hi])
        psi0 = (ket_lo + ket_hi).unit()
        tlist = np.linspace(0, 3e-4, 9)
        result = qutip.mesolve(
            0 * ops.identity(),
            psi0,
            tlist,
            c_ops=[c_op],
            options=TIGHT,
        )
        coherence = np.array([
            abs(ket_lo.dag() * rho * ket_hi) for rho in result.states
        ])
        expected = 0.5 * np.exp(-rate * n_hi**2 * tlist / 2)
        np.testing.assert_allclose(coherence, expected, atol=1e-9)

    def test_negative_rate_is_rejected(self, motion):
        _, ops, _ = motion
        with pytest.raises(ValueError, match="rate"):
            motional_dephasing_op(ops, mode=0, rate=-1.0)


class TestHeatingRateFromNoise:
    def test_matches_brownnutt_eq12(self):
        """ndot = e^2 S_E / (4 m hbar omega) with omega = 2 pi f.

        Brownnutt et al., RMP 87, 1419 (2015) Eq. (12), evaluated at
        the reference distance and frequency where both scalings are 1.
        """
        s_e, f_hz = 1e-11, 1e6
        mass = get_species("Ca40").mass_kg
        expected = ELECTRON_CHARGE**2 * s_e / (4 * mass * HBAR * TWO_PI * f_hz)
        got = heating_rate_from_noise(s_e, 100e-6, f_hz)
        assert got == pytest.approx(expected, rel=1e-12)
        assert got == pytest.approx(1459.5, rel=1e-3)

    @pytest.mark.parametrize("beta", [4.0, 3.5, 2.0])
    def test_distance_exponent_is_exact(self, beta):
        """Fitted distance exponent equals beta to machine precision.

        The default 4.0 is the planar patch-potential prediction
        (Brownnutt et al., RMP 87, 1419 (2015) Sec. IV); the fit must
        be tight enough to reject the nearby measured value 3.5.
        """
        distances = np.array([50e-6, 100e-6, 200e-6])
        rates = np.array([
            heating_rate_from_noise(1e-11, d, 1e6, beta=beta)
            for d in distances
        ])
        fitted = np.log(rates[:-1] / rates[1:]) / np.log(
            distances[1:] / distances[:-1]
        )
        np.testing.assert_allclose(fitted, beta, atol=1e-12)

    @pytest.mark.parametrize("alpha", [0.0, 1.0, 2.0])
    def test_frequency_scaling_includes_the_explicit_omega(self, alpha):
        """ndot ~ f^-(1+alpha): S_E ~ f^-alpha times the 1/omega in Eq. 12."""
        r_low = heating_rate_from_noise(1e-11, 100e-6, 1e6, alpha=alpha)
        r_high = heating_rate_from_noise(1e-11, 100e-6, 2e6, alpha=alpha)
        assert r_low / r_high == pytest.approx(2.0 ** (1 + alpha), rel=1e-12)

    def test_rate_is_inversely_proportional_to_mass(self):
        """ndot ~ 1/m, so the Ca-40 default overestimates Yb-171."""
        m_ca = get_species("Ca40").mass_kg
        m_yb = get_species("Yb171").mass_kg
        r_default = heating_rate_from_noise(1e-11, 100e-6, 1e6)
        r_yb = heating_rate_from_noise(1e-11, 100e-6, 1e6, mass_kg=m_yb)
        assert r_default / r_yb == pytest.approx(m_yb / m_ca, rel=1e-12)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"spectral_density": -1e-11},
            {"distance": 0.0},
            {"distance": -100e-6},
            {"frequency_hz": 0.0},
            {"mass_kg": -1.0},
            {"reference_distance": 0.0},
            {"reference_frequency_hz": -1e6},
        ],
    )
    def test_unphysical_inputs_are_rejected(self, kwargs):
        call = {
            "spectral_density": 1e-11,
            "distance": 100e-6,
            "frequency_hz": 1e6,
            **kwargs,
        }
        with pytest.raises(ValueError):
            heating_rate_from_noise(**call)


class TestQubitNoise:
    def test_pure_dephasing_decays_coherence_at_one_over_t2(self, system):
        """<sigma_x> = exp(-t/T2) for a pure-dephasing qubit."""
        _, ops, sf = system
        t2 = 1e-4
        tlist = np.linspace(0, 3e-4, 9)
        result = qutip.mesolve(
            0 * ops.identity(),
            _plus_state(ops, sf, 0),
            tlist,
            c_ops=[qubit_dephasing_op(ops, ion=0, t2=t2)],
            e_ops=[ops.sigma_x(0)],
            options=TIGHT,
        )
        np.testing.assert_allclose(
            result.expect[0], np.exp(-tlist / t2), atol=1e-9
        )

    def test_dephasing_plus_relaxation_gives_the_requested_t2(self, system):
        """gamma_phi = 1/T2 - 1/(2 T1) is calibrated so that adding
        spontaneous emission leaves the coherence decaying at 1/T2."""
        _, ops, sf = system
        t1, t2 = 2e-4, 3e-4
        c_ops = [
            qubit_dephasing_op(ops, ion=0, t2=t2, t1=t1),
            spontaneous_emission_op(ops, ion=0, t1=t1),
        ]
        tlist = np.linspace(0, 3e-4, 9)
        result = qutip.mesolve(
            0 * ops.identity(),
            _plus_state(ops, sf, 0),
            tlist,
            c_ops=c_ops,
            e_ops=[ops.sigma_x(0)],
            options=TIGHT,
        )
        np.testing.assert_allclose(
            result.expect[0], np.exp(-tlist / t2), atol=1e-9
        )

    def test_spontaneous_emission_population_decays_exponentially(
        self, system
    ):
        """P_excited(t) = exp(-t/T1)."""
        _, ops, sf = system
        t1 = 1e-4
        psi0 = sf.product_state([1, 0], [0])
        tlist = np.linspace(0, 3e-4, 9)
        result = qutip.mesolve(
            0 * ops.identity(),
            psi0,
            tlist,
            c_ops=[spontaneous_emission_op(ops, ion=0, t1=t1)],
            e_ops=[ops.sigma_z(0)],
            options=TIGHT,
        )
        p_excited = (1.0 - np.array(result.expect[0])) / 2
        np.testing.assert_allclose(p_excited, np.exp(-tlist / t1), atol=1e-9)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"t2": 0.0},
            {"t2": -1e-4},
            {"t2": 1e-4, "t1": 0.0},
            {"t2": 1e-4, "t1": -1e-4},
            {"t2": 10.0, "t1": 1.0},
        ],
    )
    def test_dephasing_rejects_unphysical_times(self, system, kwargs):
        _, ops, _ = system
        with pytest.raises(ValueError):
            qubit_dephasing_op(ops, ion=0, **kwargs)

    def test_spontaneous_emission_rejects_unphysical_t1(self, system):
        _, ops, _ = system
        with pytest.raises(ValueError):
            spontaneous_emission_op(ops, ion=0, t1=0.0)


class TestPhotonScattering:
    def test_rayleigh_dephases_at_half_gamma_el(self, system):
        """L = sqrt(Gamma_el/4) sigma_z decays coherence at Gamma_el/2.

        Uys et al., PRL 105, 200401 (2010) Eqs. (6) and (8).
        """
        _, ops, sf = system
        gamma_el = 1e4
        tlist = np.linspace(0, 3e-4, 9)
        result = qutip.mesolve(
            0 * ops.identity(),
            _plus_state(ops, sf, 0),
            tlist,
            c_ops=[rayleigh_scattering_op(ops, ion=0, rate=gamma_el)],
            e_ops=[ops.sigma_x(0)],
            options=TIGHT,
        )
        np.testing.assert_allclose(
            result.expect[0], np.exp(-gamma_el * tlist / 2), atol=1e-9
        )

    def test_raman_transfers_population_out_of_the_ground_state(self, system):
        """P_1(t) = (1 - exp(-R t))/2 starting from |0>.

        Ozeri et al., PRA 75, 042329 (2007) Eqs. (9)-(11): a Raman
        event projects the ion into either ground sublevel, so the
        channel must act on the prepared |0> state too.
        """
        _, ops, sf = system
        rate = 1e4
        c_ops = raman_scattering_ops(ops, ion=0, rate=rate)
        assert len(c_ops) == 2
        tlist = np.linspace(0, 3e-4, 9)
        result = qutip.mesolve(
            0 * ops.identity(),
            sf.ground_state(),
            tlist,
            c_ops=c_ops,
            e_ops=[ops.sigma_z(0)],
            options=TIGHT,
        )
        p_excited = (1.0 - np.array(result.expect[0])) / 2
        expected = 0.5 * (1.0 - np.exp(-rate * tlist))
        np.testing.assert_allclose(p_excited, expected, atol=1e-9)

    @pytest.mark.parametrize("qubit", [[0, 0], [1, 0]])
    def test_raman_event_rate_is_state_independent(self, system, qubit):
        """Total jump rate sum <L^dag L> = R/2 from either qubit state.

        Ozeri et al., PRA 75, 042329 (2007): the scattering rate does
        not depend on which qubit state the ion occupies.
        """
        _, ops, sf = system
        rate = 1e4
        c_ops = raman_scattering_ops(ops, ion=0, rate=rate)
        psi = sf.product_state(qubit, [0])
        total = sum(qutip.expect(L.dag() * L, psi) for L in c_ops)
        assert total == pytest.approx(rate / 2, rel=1e-12)

    def test_raman_superposition_event_rate(self, system):
        """The same total rate holds for an equatorial state."""
        _, ops, sf = system
        rate = 1e4
        c_ops = raman_scattering_ops(ops, ion=0, rate=rate)
        psi = _plus_state(ops, sf, 0)
        total = sum(qutip.expect(L.dag() * L, psi) for L in c_ops)
        assert total == pytest.approx(rate / 2, rel=1e-12)

    def test_negative_and_zero_rates(self, system):
        _, ops, _ = system
        assert raman_scattering_ops(ops, ion=0, rate=0.0) == []
        with pytest.raises(ValueError, match="rate"):
            raman_scattering_ops(ops, ion=0, rate=-1.0)
        with pytest.raises(ValueError, match="rate"):
            rayleigh_scattering_op(ops, ion=0, rate=-1.0)


class TestLaserNoise:
    def test_phase_noise_dephases_at_half_the_linewidth(self, system):
        """A Lorentzian laser of FWHM W (rad/s) gives 1/T2 = W/2.

        Phase diffusion with <dphi^2> = 2Dt has g1 = exp(-D t) and a
        Lorentzian spectrum of HWHM D, so W = 2D and the Ramsey decay
        rate is W/2 = pi * FWHM_Hz.
        """
        _, ops, sf = system
        fwhm_hz = 1e3
        linewidth = TWO_PI * fwhm_hz
        tlist = np.linspace(0, 1e-3, 9)
        result = qutip.mesolve(
            0 * ops.identity(),
            _plus_state(ops, sf, 0),
            tlist,
            c_ops=[laser_phase_noise_op(ops, ion=0, rate=linewidth)],
            e_ops=[ops.sigma_x(0)],
            options=TIGHT,
        )
        np.testing.assert_allclose(
            result.expect[0], np.exp(-np.pi * fwhm_hz * tlist), atol=1e-9
        )

    def test_phase_noise_rejects_negative_linewidth(self, system):
        _, ops, _ = system
        with pytest.raises(ValueError, match="rate"):
            laser_phase_noise_op(ops, ion=0, rate=-1.0)

    def test_intensity_noise_is_a_coherent_over_rotation(self, system):
        """A pi pulse with dI/I over-rotates by pi*dI/(4I).

        The operator is a Hamiltonian perturbation, so the error is
        coherent: infidelity = sin^2(pi * dI/I / 4), which is
        6.2e-5 for 1% intensity noise - not the ~1e7 s^-1 decoherence
        it would produce if misused as a collapse operator.
        """
        _, ops, sf = system
        rabi = TWO_PI * 1e6
        rms = 0.01
        h_noise = laser_intensity_noise_op(
            ops, ion=0, fractional_rms=rms, rabi_frequency=rabi
        )
        assert h_noise.isherm
        h_total = (rabi / 2) * ops.sigma_x(0) + h_noise
        t_pi = np.pi / rabi
        result = qutip.sesolve(
            h_total, sf.ground_state(), [0.0, t_pi], options=TIGHT
        )
        p_excited = (1.0 - qutip.expect(ops.sigma_z(0), result.states[-1])) / 2
        infidelity = 1.0 - p_excited
        assert infidelity == pytest.approx(
            np.sin(np.pi * rms / 4) ** 2, rel=1e-4
        )
        assert infidelity == pytest.approx(6.17e-5, rel=1e-2)

    def test_intensity_noise_errors_add_coherently(self, system):
        """Two pi pulses quadruple the error, as amplitude errors do.

        After 2 pi pulses the residual excited population is
        sin^2(pi * dI/I / 2) ~ 4x the single-pulse error, because a
        coherent over-rotation accumulates in amplitude. An
        incoherent (Lindblad) channel of the same single-pulse size
        would merely double it.
        """
        _, ops, sf = system
        rabi = TWO_PI * 1e6
        rms = 0.01
        h_total = (rabi / 2) * ops.sigma_x(0) + laser_intensity_noise_op(
            ops, ion=0, fractional_rms=rms, rabi_frequency=rabi
        )
        t_pi = np.pi / rabi
        result = qutip.sesolve(
            h_total, sf.ground_state(), [0.0, 2 * t_pi], options=TIGHT
        )
        p_excited = (1.0 - qutip.expect(ops.sigma_z(0), result.states[-1])) / 2
        single = np.sin(np.pi * rms / 4) ** 2
        assert p_excited == pytest.approx(
            np.sin(np.pi * rms / 2) ** 2, rel=1e-4
        )
        assert p_excited / single == pytest.approx(4.0, rel=1e-3)


class TestCrosstalk:
    def test_neighbor_rotates_by_the_crosstalk_fraction(self, system):
        """After a target pi pulse the neighbor sits at
        sin^2(eps * pi / 2), and the target ion is untouched by this
        term."""
        _, ops, sf = system
        eps, rabi = 0.05, TWO_PI * 1e6
        h = crosstalk_hamiltonian(
            ops,
            target_ion=0,
            neighbor_ion=1,
            crosstalk_fraction=eps,
            rabi_frequency=rabi,
        )
        t_pi = np.pi / rabi
        result = qutip.sesolve(h, sf.ground_state(), [0.0, t_pi])
        final = result.states[-1]
        p_neighbor = (1.0 - qutip.expect(ops.sigma_z(1), final)) / 2
        assert p_neighbor == pytest.approx(
            np.sin(eps * np.pi / 2) ** 2, rel=1e-6
        )
        assert qutip.expect(ops.sigma_z(0), final) == pytest.approx(
            1.0, abs=1e-9
        )

    @pytest.mark.parametrize("phase", [0.0, 0.7, np.pi / 2, -1.1])
    def test_drive_axis_follows_the_repo_phase_convention(self, system, phase):
        """H = (eps Omega / 2)(sigma_x cos phi + sigma_y sin phi).

        The excitation operator sigma_minus = |1><0| carries
        exp(+i phi); mirroring the phase would flip the sigma_y term.
        """
        _, ops, _ = system
        eps, rabi = 0.01, 1e6
        h = crosstalk_hamiltonian(
            ops,
            target_ion=0,
            neighbor_ion=1,
            crosstalk_fraction=eps,
            rabi_frequency=rabi,
            phase=phase,
        )
        expected = (eps * rabi / 2) * (
            ops.sigma_x(1) * np.cos(phase) + ops.sigma_y(1) * np.sin(phase)
        )
        assert (h - expected).norm() == pytest.approx(
            0.0, abs=1e-9 * eps * rabi
        )

    def test_phase_pi_over_two_rotates_toward_plus_x(self, system):
        """phase = pi/2 drives about +sigma_y, taking |0> (+z) to +x."""
        _, ops, sf = system
        rabi = TWO_PI * 1e6
        h = crosstalk_hamiltonian(
            ops,
            target_ion=0,
            neighbor_ion=1,
            crosstalk_fraction=1.0,
            rabi_frequency=rabi,
            phase=np.pi / 2,
        )
        result = qutip.sesolve(h, sf.ground_state(), [0.0, np.pi / (2 * rabi)])
        assert qutip.expect(
            ops.sigma_x(1), result.states[-1]
        ) == pytest.approx(1.0, abs=1e-6)

    @pytest.mark.parametrize(
        ("target", "neighbor"), [(2, 1), (-1, 1), (0, 2), (0, -1)]
    )
    def test_ion_indices_are_validated(self, system, target, neighbor):
        _, ops, _ = system
        with pytest.raises(IndexError):
            crosstalk_hamiltonian(
                ops,
                target_ion=target,
                neighbor_ion=neighbor,
                crosstalk_fraction=0.01,
                rabi_frequency=1e6,
            )

import math

import numpy as np
import pytest
import qutip

from tiqs.constants import AMU, HBAR, TWO_PI
from tiqs.hilbert_space.builder import HilbertSpace
from tiqs.hilbert_space.operators import OperatorFactory
from tiqs.hilbert_space.states import StateFactory
from tiqs.transport import (
    apply_shuttling_noise,
    shuttle_motional_excitation,
    split_crystal_excitation,
)

MASS_BE9 = 9.0 * AMU
MASS_CA40 = 40.0 * AMU
MASS_YB171 = 171.0 * AMU


@pytest.fixture
def system():
    hs = HilbertSpace(n_ions=1, n_modes=1, n_fock=40)
    ops = OperatorFactory(hs)
    sf = StateFactory(hs)
    return hs, ops, sf


def _sin2_velocity_profile(distance, duration, n_points):
    r"""Sample the reference $\sin^2$ velocity ramp and its times."""
    t = np.linspace(0.0, duration, n_points)
    return t, (2 * distance / duration) * np.sin(np.pi * t / duration) ** 2


def _fourier_excitation(distance, duration, trap_frequency, mass_kg):
    r"""$|\alpha|^2$ by direct quadrature of Bowler et al. Eq. (1).

    Independent numerical evaluation of
    $\alpha = (1/2x_\text{zpf}) \int_0^T \dot{z}_0 e^{i\omega t} dt$,
    with no reference to the closed form used by the module.
    """
    x_zpf = np.sqrt(HBAR / (2 * mass_kg * trap_frequency))
    t, velocity = _sin2_velocity_profile(distance, duration, 400001)
    integral = np.trapezoid(velocity * np.exp(1j * trap_frequency * t), t)
    return abs(integral / (2 * x_zpf)) ** 2


def _sesolve_excitation(
    distance, duration, trap_frequency, mass_kg, n_fock=60
):
    r"""Exact quantum excitation of an ion in a moving harmonic well.

    Integrates the lab-frame Hamiltonian
    $H/\hbar = \omega a^\dagger a
    - (\omega z_0(t) / 2x_\text{zpf})(a + a^\dagger)$, obtained from
    $p^2/2m + \tfrac{1}{2}m\omega^2 (x - z_0(t))^2$, where $z_0$ is
    the integral of the $\sin^2$ velocity ramp. The excitation is
    $\langle n \rangle$ measured about the *final* well centre
    $z_0(T) = d$, i.e. with $a \to a - \beta$,
    $\beta = d / 2x_\text{zpf}$.
    """
    x_zpf = np.sqrt(HBAR / (2 * mass_kg * trap_frequency))
    beta = distance / (2 * x_zpf)
    a = qutip.destroy(n_fock)

    def z0(t, _args=None):
        phase = TWO_PI * t / duration
        return distance * (t / duration - np.sin(phase) / TWO_PI)

    hamiltonian = [
        trap_frequency * a.dag() * a,
        [-(trap_frequency / (2 * x_zpf)) * (a + a.dag()), z0],
    ]
    result = qutip.sesolve(
        hamiltonian,
        qutip.basis(n_fock, 0),
        np.linspace(0.0, duration, 1001),
        options={"atol": 1e-12, "rtol": 1e-10, "nsteps": 100000},
    )
    psi = result.states[-1]
    n_lab = qutip.expect(a.dag() * a, psi)
    amplitude = qutip.expect(a, psi)
    return n_lab - 2 * beta * amplitude.real + beta**2


class TestShuttling:
    @pytest.mark.parametrize("n_periods", [0.5, 1.1, 2.7])
    def test_matches_moving_well_schrodinger_solution(self, n_periods):
        r"""Exact quantum reference: ion in a translated harmonic well.

        Anchors the closed form against a `qutip.sesolve` integration
        of $p^2/2m + \tfrac{1}{2}m\omega^2(x - z_0(t))^2$. The distance
        is kept at $4x_\text{zpf}$ so the displaced state fits under
        the Fock cutoff; the $d^2$ scaling law is checked separately.
        """
        trap_frequency = TWO_PI * 1e6
        duration = n_periods * TWO_PI / trap_frequency
        x_zpf = np.sqrt(HBAR / (2 * MASS_CA40 * trap_frequency))
        distance = 4 * x_zpf

        reference = _sesolve_excitation(
            distance, duration, trap_frequency, MASS_CA40
        )
        assert shuttle_motional_excitation(
            distance, duration, trap_frequency, MASS_CA40
        ) == pytest.approx(reference, rel=1e-6)

    @pytest.mark.parametrize("n_periods", [3.3, 20.4])
    def test_matches_fourier_integral_quadrature(self, n_periods):
        """Reproduces Bowler et al. Eq. (1) by direct quadrature."""
        trap_frequency = TWO_PI * 1.5e6
        duration = n_periods * TWO_PI / trap_frequency
        distance = 200e-6

        reference = _fourier_excitation(
            distance, duration, trap_frequency, MASS_CA40
        )
        assert shuttle_motional_excitation(
            distance, duration, trap_frequency, MASS_CA40
        ) == pytest.approx(reference, rel=1e-6)

    def test_sudden_limit_is_the_full_displacement(self):
        r"""$\omega T \to 0$ leaves the ion displaced by the whole $d$.

        An ion that cannot respond ends up at distance $d$ from the
        new well centre, i.e. in the coherent state
        $\alpha = d / 2x_\text{zpf}$.
        """
        trap_frequency = TWO_PI * 1e6
        x_zpf = np.sqrt(HBAR / (2 * MASS_CA40 * trap_frequency))
        distance = 500e-9
        expected = (distance / (2 * x_zpf)) ** 2

        sudden = shuttle_motional_excitation(
            distance,
            1e-4 * TWO_PI / trap_frequency,
            trap_frequency,
            MASS_CA40,
        )
        assert sudden == pytest.approx(expected, rel=1e-6)

    @pytest.mark.parametrize("n_periods", [2, 3, 5, 17])
    def test_catch_condition_nulls(self, n_periods):
        r"""Bowler et al.: at $\omega T = 2\pi N$ the ion is recaught.

        "For $\omega t_T = 2\pi N$, with $N$ an integer, an ion
        starting in its ground state of motion is caught back in the
        ground state." Half a period longer, the same shuttle deposits
        a measurable excitation, so the model must be non-monotonic in
        duration rather than a decaying exponential.
        """
        trap_frequency = TWO_PI * 1e6
        distance = 200e-6
        period = TWO_PI / trap_frequency

        null = shuttle_motional_excitation(
            distance, n_periods * period, trap_frequency, MASS_CA40
        )
        between = shuttle_motional_excitation(
            distance, (n_periods + 0.5) * period, trap_frequency, MASS_CA40
        )
        assert between > 0.1
        assert null < 1e-12 * between

    def test_scales_as_distance_squared(self):
        r"""$\Delta\bar{n} \propto d^2$ at fixed $\omega T$.

        The excitation is $|\alpha|^2$ with $\alpha$ linear in the
        waveform amplitude, so it must scale exactly as $d^2$ -- and
        vanish for a zero-length shuttle.
        """
        args = (10.5e-6, TWO_PI * 1e6, MASS_CA40)
        base = shuttle_motional_excitation(100e-6, *args)

        assert shuttle_motional_excitation(300e-6, *args) == pytest.approx(
            9.0 * base, rel=1e-12
        )
        assert shuttle_motional_excitation(0.0, *args) == 0.0
        assert base > 0.0

    def test_scales_linearly_with_ion_mass(self):
        r"""$\Delta\bar{n} \propto m$ through $x_\text{zpf}^{-2}$.

        $(d / x_\text{zpf})^2 = 2 m \omega d^2 / \hbar$, so a
        $^{171}$Yb$^+$ ion is excited exactly $171/9$ times as much
        as a $^{9}$Be$^+$ ion by the same waveform.
        """
        args = (200e-6, 10.5e-6, TWO_PI * 1e6)
        light = shuttle_motional_excitation(*args, MASS_BE9)
        heavy = shuttle_motional_excitation(*args, MASS_YB171)
        assert heavy / light == pytest.approx(171.0 / 9.0, rel=1e-12)

    def test_anomalous_heating_dominates_slow_transport(self):
        r"""The long-duration floor is $\dot{\bar{n}} T$, not a constant.

        Sterk et al., *npj Quantum Inf.* **8**, 68 (2022) measured a
        295(24) quanta/s background heating rate; integrated over the
        transport it grows with duration, so very slow shuttling is
        not free.
        """
        heating_rate = 295.0

        def excitation(duration):
            return shuttle_motional_excitation(
                distance=200e-6,
                duration=duration,
                trap_frequency=TWO_PI * 1e6,
                mass_kg=MASS_CA40,
                heating_rate=heating_rate,
            )

        slow = excitation(1e-3)
        slower = excitation(10e-3)
        assert slow == pytest.approx(heating_rate * 1e-3, rel=1e-6)
        assert slower == pytest.approx(heating_rate * 10e-3, rel=1e-6)
        assert slower > slow

    def test_reference_ramp_is_not_an_optimized_waveform(self):
        r"""Scope guard on the absolute normalization.

        Walther et al., *Phys. Rev. Lett.* **109**, 080501 (2012) moved
        a $^{40}$Ca$^+$ ion 280 um in 3.6 us at $2\pi \times 1.41$ MHz
        with 0.10(1) quanta of residual excitation, using a
        numerically optimized waveform. The unoptimized $\sin^2$ ramp
        needs many more trap periods to get there, so the estimate must
        sit orders of magnitude above that measurement -- pinning the
        prefactor at real experimental parameters and keeping the
        docstring's "engineered waveforms do far better" caveat honest.
        """
        walther = shuttle_motional_excitation(
            280e-6, 3.6e-6, TWO_PI * 1.41e6, MASS_CA40
        )
        assert 20.0 < walther < 300.0

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"distance": -1e-6},
            {"duration": 0.0},
            {"duration": -1e-5},
            {"trap_frequency": 0.0},
            {"mass_kg": 0.0},
            {"heating_rate": -1.0},
        ],
    )
    def test_rejects_unphysical_inputs(self, kwargs):
        args = {
            "distance": 200e-6,
            "duration": 50e-6,
            "trap_frequency": TWO_PI * 1e6,
            "mass_kg": MASS_CA40,
        }
        with pytest.raises(ValueError):
            shuttle_motional_excitation(**(args | kwargs))


class TestShuttlingNoise:
    def test_adds_requested_quanta_from_vacuum(self, system):
        _, ops, sf = system
        rho = qutip.ket2dm(sf.ground_state())
        after = apply_shuttling_noise(rho, ops, mode=0, added_quanta=0.5)
        assert qutip.expect(ops.number(0), after) == pytest.approx(
            0.5, abs=1e-9
        )

    @pytest.mark.parametrize("n_bar_0", [0.0, 0.5, 2.0])
    def test_added_quanta_independent_of_initial_occupation(
        self, system, n_bar_0
    ):
        r"""Transport excitation does not depend on the initial state.

        Bowler et al. Eq. (1) gives $\alpha$ from the waveform and
        $\omega$ alone, so the deposited energy is the same from
        vacuum as from a warm mode. An
        $L = \sqrt{\gamma}a^\dagger$ amplifier channel would instead
        add $\Delta\bar{n}(1 + \bar{n}_0)$.
        """
        _, ops, sf = system
        rho = (
            qutip.ket2dm(sf.ground_state())
            if n_bar_0 == 0.0
            else sf.thermal_state([n_bar_0])
        )
        number = ops.number(0)
        before = qutip.expect(number, rho)
        after = apply_shuttling_noise(rho, ops, 0, added_quanta=0.5)
        assert qutip.expect(number, after) - before == pytest.approx(
            0.5, abs=1e-5
        )

    def test_coherent_amplitude_is_not_amplified(self, system):
        r"""A phase-averaged kick adds energy without gain.

        Phase averaging cancels the cross term, so
        $\langle a \rangle$ is unchanged and
        $\langle n \rangle \to \langle n \rangle
        + \Delta\bar{n}$. The amplifier channel gave
        $\langle a\rangle \to \sqrt{1 + \Delta\bar{n}}
        \langle a \rangle$ and $\langle n \rangle = 3$ here.
        """
        hs, ops, _ = system
        rho = qutip.tensor(
            qutip.ket2dm(qutip.basis(2, 0)),
            qutip.coherent_dm(hs.fock_dim(0), 1.0),
        )
        after = apply_shuttling_noise(rho, ops, 0, added_quanta=1.0)
        assert qutip.expect(ops.annihilate(0), after) == pytest.approx(
            1.0, abs=1e-9
        )
        assert qutip.expect(ops.number(0), after) == pytest.approx(
            2.0, abs=1e-9
        )

    def test_preserves_qubit_coherence(self, system):
        """The channel acts only on the addressed mode."""
        hs, ops, _ = system
        psi = qutip.tensor(
            (qutip.basis(2, 0) + qutip.basis(2, 1)).unit(),
            qutip.basis(hs.fock_dim(0), 0),
        )
        after = apply_shuttling_noise(
            qutip.ket2dm(psi), ops, 0, added_quanta=0.7
        )
        qubit = after.ptrace(0)
        assert qubit.full() == pytest.approx(np.full((2, 2), 0.5), abs=1e-9)

    def test_vacuum_output_is_poissonian(self, system):
        r"""A phase-averaged coherent state, not a thermal one.

        Bowler et al. found "Fock state populations consistent with
        coherent states" after separation, i.e.
        $p_n = e^{-\bar{n}}\bar{n}^n/n!$ -- distinct from the
        geometric populations an amplifier channel produces.
        """
        hs, ops, sf = system
        dim = hs.fock_dim(0)
        added = 1.5
        after = apply_shuttling_noise(
            qutip.ket2dm(sf.ground_state()),
            ops,
            0,
            added,
            n_phases=dim,
        )
        populations = np.real(np.diag(after.ptrace(1).full()))[:8]
        poisson = [
            math.exp(-added) * added**n / math.factorial(n) for n in range(8)
        ]
        assert populations == pytest.approx(poisson, abs=1e-9)

    def test_channel_is_trace_preserving_and_positive(self, system):
        """A mixture of unitaries is a valid quantum channel."""
        _, ops, sf = system
        after = apply_shuttling_noise(
            sf.thermal_state([0.3]), ops, 0, added_quanta=1.0
        )
        assert after.tr().real == pytest.approx(1.0, abs=1e-12)
        assert min(after.eigenenergies()) > -1e-12

    def test_zero_quanta_leaves_state_untouched(self, system):
        _, ops, sf = system
        rho = qutip.ket2dm(sf.ground_state())
        assert apply_shuttling_noise(rho, ops, 0, 0.0) == rho

    def test_warns_when_displacement_exceeds_fock_cutoff(self, system):
        """Truncated displacement silently under-adds energy."""
        hs, ops, sf = system
        rho = qutip.ket2dm(sf.ground_state())
        with pytest.warns(UserWarning, match="Fock cutoff"):
            after = apply_shuttling_noise(rho, ops, 0, 30.0)
        assert qutip.expect(ops.number(0), after) < 30.0

    @pytest.mark.parametrize(
        ("kwargs", "error"),
        [
            ({"added_quanta": -0.1}, ValueError),
            ({"n_phases": 1}, ValueError),
            ({"mode": 3}, IndexError),
        ],
    )
    def test_rejects_unphysical_inputs(self, system, kwargs, error):
        _, ops, sf = system
        args = {"mode": 0, "added_quanta": 0.5}
        with pytest.raises(error):
            apply_shuttling_noise(
                qutip.ket2dm(sf.ground_state()), ops, **(args | kwargs)
            )


class TestCrystalSplitting:
    def test_bowler_separation_anchor(self):
        r"""55 us at $\omega_\text{crit} = 2\pi \times 0.7$ MHz -> 2 quanta.

        Bowler et al., *Phys. Rev. Lett.* **109**, 080502 (2012)
        measured coherent states of $\bar{n} = 2.1(1)$ and
        $\bar{n} = 1.9(1)$ in the two zones after a 55 us separation.
        """
        assert split_crystal_excitation(
            TWO_PI * 0.7e6, 55e-6
        ) == pytest.approx(2.0, rel=1e-9)

    def test_ruster_order_of_magnitude(self):
        r"""80 us at $\omega_\text{crit} = 2\pi \times 0.18$ MHz.

        Ruster et al., *Phys. Rev. A* **90**, 033410 (2014) measured
        $\bar{n} = 4.16(16)$ quanta per ion; Kaufmann et al.,
        *New J. Phys.* **16**, 073012 (2014) Table 1 gives
        $\omega_\text{crit}/2\pi = 0.11$-$0.29$ MHz for traps of that
        class. A one-anchor power law should land within a factor of
        a few, and nowhere near the sub-quantum values the old
        constant floor returned.
        """
        estimate = split_crystal_excitation(TWO_PI * 0.18e6, 80e-6)
        assert 4.16 / 5.0 < estimate < 4.16 * 5.0

    def test_impulsive_scaling_is_inverse_square(self):
        r"""$\delta E \propto \dot{\alpha}_\text{CP}^2 \propto T^{-2}$.

        Kaufmann et al., Eq. (25): in the impulsive regime the energy
        gain is quadratic in the sweep rate through the critical
        point, so halving the duration quadruples the excitation.
        """
        omega_crit = TWO_PI * 0.5e6
        slow = split_crystal_excitation(omega_crit, 100e-6)
        fast = split_crystal_excitation(omega_crit, 50e-6)
        assert fast == pytest.approx(4.0 * slow, rel=1e-12)

    def test_critical_point_frequency_controls_excitation(self):
        r"""Adiabaticity is set by $\omega_\text{crit}$, not $\omega_0$.

        A trap whose critical-point frequency is four times lower is
        sixteen times worse at fixed duration. Feeding the initial
        single-well frequency (several times higher) therefore
        under-estimates the excitation badly.
        """
        duration = 55e-6
        soft = split_crystal_excitation(TWO_PI * 0.175e6, duration)
        stiff = split_crystal_excitation(TWO_PI * 0.7e6, duration)
        assert soft == pytest.approx(16.0 * stiff, rel=1e-12)
        assert soft > 30.0

    def test_no_floor_at_long_durations(self):
        """Without anomalous heating nothing pins the estimate up.

        The old model returned a hard 0.05 for every duration beyond
        3 us, which contradicted its own 2-quanta anchor at 55 us.
        """
        assert split_crystal_excitation(TWO_PI * 0.7e6, 10e-3) < 1e-4

    def test_anomalous_heating_penalizes_slow_splits(self):
        r"""Slower is not always better.

        Kaufmann et al., Sec. 3.3: "anomalous heating will strongly
        contribute to the energy gain at large splitting times". With
        $\dot{\bar{n}} > 0$ the total has a minimum at
        $T^* = (2 \Delta\bar{n}_\text{ref} (\omega_\text{ref}
        T_\text{ref})^2 / \dot{\bar{n}}
        \omega_\text{crit}^2)^{1/3}$.
        """
        omega_crit = TWO_PI * 0.7e6
        heating_rate = 100.0
        reference = TWO_PI * 0.7e6 * 55e-6
        t_min = (2 * 2.0 * reference**2 / (heating_rate * omega_crit**2)) ** (
            1 / 3
        )

        at_min = split_crystal_excitation(omega_crit, t_min, heating_rate)
        faster = split_crystal_excitation(omega_crit, t_min / 3, heating_rate)
        slower = split_crystal_excitation(omega_crit, t_min * 3, heating_rate)
        assert at_min < faster
        assert at_min < slower
        assert slower == pytest.approx(heating_rate * t_min * 3, rel=0.05)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"omega_crit": 0.0},
            {"omega_crit": -TWO_PI * 1e6},
            {"split_duration": 0.0},
            {"split_duration": -55e-6},
            {"heating_rate": -1.0},
        ],
    )
    def test_rejects_unphysical_inputs(self, kwargs):
        args = {
            "omega_crit": TWO_PI * 0.7e6,
            "split_duration": 55e-6,
        }
        with pytest.raises(ValueError):
            split_crystal_excitation(**(args | kwargs))

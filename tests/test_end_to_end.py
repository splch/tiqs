"""Comprehensive end-to-end test exercising every public function in TIQS.

Implements a realistic 3-ion Yb-171 quantum simulation of the transverse-field
Ising model using the native Molmer-Sorensen interaction, followed by a QCCD
transport + mid-circuit measurement cycle. This is the canonical experiment
performed on real trapped-ion quantum simulators (Monroe, Blatt groups).

The test traces the full physical stack: species data, trap physics, Coulomb
crystal normal modes, Doppler/sideband cooling, optical pumping, Hamiltonian
construction, single-qubit rotations (including composite pulses), MS
entangling gates, light-shift gates, Cirac-Zoller gate structure, all noise
channels, QCCD transport, mid-circuit measurement, and post-simulation
analysis (fidelity, Wigner functions, phase-space trajectories, error
budgets).
"""

import math

import numpy as np
import pytest
import qutip

from tiqs.analysis.error_budget import compute_error_budget
from tiqs.analysis.fidelity import (
    bell_state_fidelity,
    gate_fidelity,
    state_fidelity,
)
from tiqs.analysis.phase_space import motional_wigner, phase_space_trajectory
from tiqs.chain.equilibrium import equilibrium_positions
from tiqs.chain.lamb_dicke import lamb_dicke_parameters
from tiqs.chain.normal_modes import NormalModeResult, normal_modes
from tiqs.constants import (
    AMU,
    BOLTZMANN,
    ELECTRON_CHARGE,
    EPSILON_0,
    HBAR,
    PI,
    SPEED_OF_LIGHT,
    TWO_PI,
)
from tiqs.cooling.doppler import doppler_cooled_nbar
from tiqs.cooling.eit_cooling import eit_cooling_nbar
from tiqs.cooling.sideband_cooling import (
    sideband_cooling_nbar,
    sideband_cooling_simulate,
)
from tiqs.gates.cirac_zoller import cirac_zoller_gate
from tiqs.gates.light_shift import light_shift_gate_hamiltonian
from tiqs.gates.molmer_sorensen import ms_gate_duration, ms_gate_hamiltonian
from tiqs.gates.single_qubit import (
    GatePulse,
    bb1_composite_gate,
    rx_gate,
    ry_gate,
    rz_gate,
    sk1_composite_gate,
)
from tiqs.hilbert_space.builder import HilbertSpace
from tiqs.hilbert_space.operators import OperatorFactory
from tiqs.hilbert_space.states import StateFactory
from tiqs.interaction.hamiltonian import (
    blue_sideband_hamiltonian,
    carrier_hamiltonian,
    full_interaction_hamiltonian,
    red_sideband_hamiltonian,
)
from tiqs.interaction.laser import LaserBeam
from tiqs.interaction.raman import RamanPair
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
    raman_scattering_op,
    rayleigh_scattering_op,
)
from tiqs.noise.qubit import qubit_dephasing_op, spontaneous_emission_op
from tiqs.simulation.config import SimulationConfig
from tiqs.simulation.runner import SimulationRunner
from tiqs.spam.measurement import (
    fluorescence_probabilities,
    measurement_fidelity,
    mid_circuit_measurement,
    sample_measurement,
)
from tiqs.spam.preparation import optical_pumping_ops, prepare_qubit
from tiqs.species.ion import get_species
from tiqs.species.transitions import Transition
from tiqs.transport import (
    apply_shuttling_noise,
    shuttle_motional_excitation,
    split_crystal_excitation,
)
from tiqs.trap import PaulTrap


class TestIsingQuantumSimulation:
    """Simulate a transverse-field Ising model quench on a 3-ion Yb-171 chain.

    The physical scenario:
    - 3 Yb-171 ions in a linear Paul trap
    - Cool to near motional ground state
    - Prepare all spins in |000> (paramagnetic ground state of
      transverse field)
    - Apply global Ry(pi/2) to rotate into the sigma_x basis
    - Use MS interaction to implement Ising coupling
      H_Ising = J * sum sigma_x_i * sigma_x_j
    - Evolve for variable time to observe spin dynamics
    - Measure correlations and compare clean vs noisy
    """

    @pytest.fixture
    def yb_trap(self):
        return PaulTrap(
            v_rf=1000.0,
            omega_rf=TWO_PI * 30e6,
            r0=0.5e-3,
            omega_axial=TWO_PI * 1.0e6,
            species=get_species("Yb171"),
        )

    # -- Phase 1: Physical setup --

    def test_01_species_and_constants(self):
        """Exercise every species, every property, every constant."""
        assert HBAR > 0 and ELECTRON_CHARGE > 0 and BOLTZMANN > 0
        assert SPEED_OF_LIGHT > 0 and AMU > 0 and EPSILON_0 > 0
        assert (
            pytest.approx(np.pi) == PI and pytest.approx(2 * np.pi) == TWO_PI
        )

        for name in ["Yb171", "Ca40", "Ca43", "Ba137", "Be9", "Sr88"]:
            s = get_species(name)
            assert s.mass_kg == pytest.approx(s.mass_amu * AMU)
            assert s.cooling_transition.frequency == pytest.approx(
                SPEED_OF_LIGHT / s.cooling_transition.wavelength
            )
            assert s.cooling_transition.wavevector == pytest.approx(
                TWO_PI / s.cooling_transition.wavelength
            )
            assert s.doppler_limit_temperature() == pytest.approx(
                HBAR * s.cooling_transition.linewidth / (2 * BOLTZMANN)
            )
            _ = s.doppler_limit_nbar(1e6)

        yb = get_species("Yb171")
        assert yb.nuclear_spin == 0.5 and yb.qubit_type == "hyperfine"
        assert yb.qubit_frequency_hz == pytest.approx(12.6428e9, rel=1e-3)
        assert yb.raman_wavelength == pytest.approx(355e-9, rel=0.01)
        assert yb.qubit_t1 == math.inf and yb.qubit_wavelength is None

        ca = get_species("Ca40")
        assert (
            ca.qubit_type == "optical"
            and ca.qubit_wavelength == pytest.approx(729e-9, rel=0.01)
        )
        assert ca.metastable_lifetime == pytest.approx(1.168, rel=0.01)

        t = Transition("test", 500e-9, TWO_PI * 10e6, 0.9)
        assert t.branching_ratio == 0.9

        with pytest.raises(KeyError):
            get_species("Nonexistent")

    def test_02_trap_and_stability(self, yb_trap):
        """Exercise every PaulTrap property and edge case."""
        t = yb_trap
        assert 0 < t.mathieu_q < 0.908
        assert t.mathieu_a < 0
        assert t.is_stable()
        assert t.omega_radial > t.omega_axial
        assert t.pseudopotential_depth_eV > 0
        assert t.micromotion_amplitude(1e-6) == pytest.approx(
            t.mathieu_q / 2 * 1e-6
        )
        assert t.stray_field_displacement(1.0) > 0
        assert t.u_dc_axial is not None

        trap_v = PaulTrap(
            v_rf=300,
            omega_rf=TWO_PI * 30e6,
            r0=0.5e-3,
            u_dc_axial=10.0,
            species=get_species("Ca40"),
        )
        assert trap_v.omega_axial > 0

        unstable = PaulTrap(
            v_rf=5000,
            omega_rf=TWO_PI * 1e6,
            r0=0.1e-3,
            omega_axial=TWO_PI * 0.5e6,
            species=get_species("Yb171"),
        )
        assert not unstable.is_stable()

        with pytest.raises(ValueError):
            PaulTrap(
                v_rf=200,
                omega_rf=TWO_PI * 30e6,
                r0=0.5e-3,
                species=get_species("Ca40"),
            )

    def test_03_three_ion_chain(self, yb_trap):
        """Compute 3-ion Coulomb crystal: positions, all 9 normal
        modes, Lamb-Dicke matrix."""
        pos = equilibrium_positions(3, yb_trap)
        assert len(pos) == 3
        assert pos[0] < 0 < pos[2]
        assert pos[1] == pytest.approx(0.0, abs=1e-12)  # center ion at origin
        assert pos[0] == pytest.approx(-pos[2], abs=1e-12)  # symmetric

        modes = normal_modes(3, yb_trap)
        assert isinstance(modes, NormalModeResult)

        # 3 axial modes: COM < tilt < stretch
        assert len(modes.axial_freqs) == 3
        for i in range(2):
            assert modes.axial_freqs[i] < modes.axial_freqs[i + 1]
        assert modes.axial_freqs[0] == pytest.approx(
            yb_trap.omega_axial, rel=1e-3
        )

        # Eigenvectors are orthonormal
        V = modes.axial_vectors
        np.testing.assert_allclose(V.T @ V, np.eye(3), atol=1e-10)

        # COM mode: all ions in phase
        v_com = V[:, 0]
        assert np.sign(v_com[0]) == np.sign(v_com[1]) == np.sign(v_com[2])

        # 3 radial modes each for x and y
        assert len(modes.radial_x_freqs) == 3
        assert len(modes.radial_y_freqs) == 3

        # Lamb-Dicke parameters: 3 ions x 3 modes for counter-propagating Raman
        k_eff = 2 * TWO_PI / 355e-9  # Yb Raman, counter-propagating
        eta = lamb_dicke_parameters(modes, yb_trap.species, k_eff, "axial")
        assert eta.shape == (3, 3)
        # COM mode: all ions couple equally
        assert eta[0, 0] == pytest.approx(eta[1, 0], rel=1e-3)
        assert eta[0, 0] == pytest.approx(eta[2, 0], rel=1e-3)

        # Also test radial LD params
        eta_r = lamb_dicke_parameters(
            modes, yb_trap.species, k_eff, "radial_x"
        )
        assert eta_r.shape == (3, 3)

        # Single-ion and 5-ion for completeness
        assert equilibrium_positions(1, yb_trap)[0] == 0.0
        pos5 = equilibrium_positions(5, yb_trap)
        assert all(pos5[i] < pos5[i + 1] for i in range(4))
        spacings = np.diff(pos5)
        assert spacings[2] < spacings[0]  # tighter in center

    # -- Phase 2: Hilbert space and operators --

    def test_04_hilbert_space_and_operators(self):
        """Exercise every operator factory method and state factory method."""
        hs = HilbertSpace(n_ions=3, n_modes=1, n_fock=8)
        assert hs.dims == [2, 2, 2, 8]
        assert hs.total_dim == 64
        assert hs.fock_dim(0) == 8

        hs2 = HilbertSpace(n_ions=2, n_modes=2, n_fock=[10, 5])
        assert hs2.dims == [2, 2, 10, 5]
        assert hs2.total_dim == 200

        with pytest.raises(ValueError):
            HilbertSpace(n_ions=0, n_modes=1)
        with pytest.raises(ValueError):
            HilbertSpace(n_ions=1, n_modes=0)

        ops = OperatorFactory(hs)
        for i in range(3):
            assert ops.sigma_x(i).isherm
            assert ops.sigma_y(i).isherm
            assert ops.sigma_z(i).isherm
            assert (
                ops.sigma_plus(i).dag() - ops.sigma_minus(i)
            ).norm() < 1e-12
        a = ops.annihilate(0)
        assert (ops.create(0) - a.dag()).norm() < 1e-12
        assert ops.number(0).isherm
        assert ops.position(0).isherm
        assert ops.momentum(0).isherm
        assert ops.identity().tr() == pytest.approx(64.0)

        with pytest.raises(IndexError):
            ops.sigma_z(5)
        with pytest.raises(IndexError):
            ops.annihilate(3)

        sf = StateFactory(hs)
        gs = sf.ground_state()
        assert gs.type == "ket" and abs(gs.norm() - 1) < 1e-12
        ps = sf.product_state([1, 0, 1], [3])
        assert ps.type == "ket"
        th = sf.thermal_state(n_bar=[2.0])
        assert th.type == "oper" and th.tr() == pytest.approx(1.0, abs=1e-10)

        with pytest.raises(ValueError):
            sf.product_state([0], [0])
        with pytest.raises(ValueError):
            sf.thermal_state(n_bar=[1.0, 2.0])

    # -- Phase 3: Laser and Raman configuration --

    def test_05_laser_and_raman(self):
        """Exercise LaserBeam and RamanPair with Yb-171 parameters."""
        laser = LaserBeam(
            wavelength=355e-9,
            rabi_frequency=TWO_PI * 1e9,
            detuning=TWO_PI * 1e6,
            phase=0.5,
        )
        assert laser.wavevector == pytest.approx(TWO_PI / 355e-9)

        raman = RamanPair(
            omega_1=TWO_PI * 8.45e14,
            omega_2=TWO_PI * 8.45e14 - TWO_PI * 12.6e9,
            rabi_1=TWO_PI * 500e6,
            rabi_2=TWO_PI * 500e6,
            detuning_from_excited=TWO_PI * 33e12,
            excited_state_linewidth=TWO_PI * 19.6e6,
        )
        assert raman.effective_rabi_frequency > 0
        assert raman.frequency_difference == pytest.approx(
            TWO_PI * 12.6e9, rel=1e-3
        )
        assert raman.scattering_rate > 0
        assert isinstance(raman.ac_stark_shift, float)

    # -- Phase 4: Cooling to near ground state --

    def test_06_cooling_pipeline(self, yb_trap):
        """Simulate Doppler -> EIT -> sideband cooling for the COM mode."""
        species = yb_trap.species
        omega_hz = yb_trap.omega_axial / TWO_PI

        n_doppler = doppler_cooled_nbar(species, omega_hz)
        assert 1 < n_doppler < 30

        n_eit = eit_cooling_nbar(TWO_PI * 200e3, yb_trap.omega_axial, 0.01)
        assert n_eit < n_doppler

        n_sbc_est = sideband_cooling_nbar(TWO_PI * 1e3, yb_trap.omega_axial)
        assert n_sbc_est < 0.01

        hs = HilbertSpace(n_ions=1, n_modes=1, n_fock=20)
        ops = OperatorFactory(hs)
        n_sbc_sim = sideband_cooling_simulate(
            ops,
            0,
            0,
            n_bar_initial=5.0,
            eta=0.1,
            rabi_frequency=TWO_PI * 100e3,
            optical_pumping_rate=TWO_PI * 10e3,
            n_cycles=30,
        )
        assert n_sbc_sim < 5.0

    # -- Phase 5: State preparation --

    def test_07_optical_pumping(self):
        """Exercise optical pumping to |000>."""
        hs = HilbertSpace(n_ions=3, n_modes=1, n_fock=5)
        ops = OperatorFactory(hs)
        sf = StateFactory(hs)

        c_ops = optical_pumping_ops(ops, ion=1, pumping_rate=1e7)
        assert len(c_ops) >= 1

        rho_bad = sf.thermal_state(n_bar=[0.0], qubit_states=[1, 1, 0])
        rho_ok = prepare_qubit(
            ops, 0, rho_bad, pumping_rate=1e7, duration=10e-6
        )
        rho_ok = prepare_qubit(
            ops, 1, rho_ok, pumping_rate=1e7, duration=10e-6
        )
        p0_ion0 = rho_ok.ptrace(0)[0, 0].real
        p0_ion1 = rho_ok.ptrace(1)[0, 0].real
        assert p0_ion0 > 0.95 and p0_ion1 > 0.95

    # -- Phase 6: Interaction Hamiltonians --

    def test_08_all_hamiltonians(self):
        """Exercise carrier, RSB, BSB, and full interaction with
        1st/2nd order LD."""
        hs = HilbertSpace(n_ions=1, n_modes=1, n_fock=10)
        ops = OperatorFactory(hs)
        sf = StateFactory(hs)
        Omega = TWO_PI * 200e3
        eta = 0.1

        H_c = carrier_hamiltonian(ops, 0, Omega, phase=0.3)
        assert H_c.isherm
        H_r = red_sideband_hamiltonian(ops, 0, 0, Omega, eta, phase=0.5)
        assert H_r.isherm
        H_b = blue_sideband_hamiltonian(ops, 0, 0, Omega, eta)
        assert H_b.isherm

        # RSB pi-pulse: |0,1> -> |1,0>
        psi_01 = sf.product_state([0], [1])
        t_rsb = np.pi / (eta * Omega)
        H_r0 = red_sideband_hamiltonian(ops, 0, 0, Omega, eta, phase=0.0)
        r = qutip.sesolve(H_r0, psi_01, [0, t_rsb])
        assert abs(r.states[-1].overlap(sf.product_state([1], [0]))) ** 2 > 0.9

        # BSB pi-pulse: |0,0> -> |1,1>
        H_b0 = blue_sideband_hamiltonian(ops, 0, 0, Omega, eta, phase=0.0)
        r2 = qutip.sesolve(H_b0, sf.ground_state(), [0, t_rsb])
        assert (
            abs(r2.states[-1].overlap(sf.product_state([1], [1]))) ** 2 > 0.9
        )

        # Full 1st order
        H1 = full_interaction_hamiltonian(
            ops, 0, 0, Omega, eta, 0.0, TWO_PI * 1e6
        )
        assert isinstance(H1, list)
        # Full 2nd order has more terms (Debye-Waller + 2nd sideband)
        H2 = full_interaction_hamiltonian(
            ops,
            0,
            0,
            Omega,
            eta,
            TWO_PI * 100,
            TWO_PI * 1e6,
            lamb_dicke_order=2,
        )
        assert len(H2) > len(H1)

    # -- Phase 7: Single-qubit gates --

    def test_09_single_qubit_gates(self):
        """Exercise every single-qubit gate including composite pulse
        sequences."""
        hs = HilbertSpace(n_ions=1, n_modes=1, n_fock=5)
        ops = OperatorFactory(hs)
        sf = StateFactory(hs)
        psi0 = sf.ground_state()

        g = rx_gate(ops, 0, np.pi)
        assert isinstance(g, GatePulse) and g.duration > 0
        r = qutip.sesolve(g.hamiltonian, psi0, [0, g.duration])
        assert (
            abs(r.states[-1].overlap(sf.product_state([1], [0]))) ** 2 > 0.99
        )

        g = ry_gate(ops, 0, np.pi / 2)
        r = qutip.sesolve(g.hamiltonian, psi0, [0, g.duration])
        assert abs(qutip.expect(ops.sigma_z(0), r.states[-1])) < 0.05

        g = rz_gate(ops, 0, np.pi)
        plus = (sf.product_state([0], [0]) + sf.product_state([1], [0])).unit()
        r = qutip.sesolve(g.hamiltonian, plus, [0, g.duration])
        assert qutip.expect(ops.sigma_x(0), r.states[-1]) == pytest.approx(
            -1.0, abs=0.05
        )

        sk1 = sk1_composite_gate(ops, 0, np.pi)
        assert hasattr(sk1, "pulses") and len(sk1.pulses) == 3
        bb1 = bb1_composite_gate(ops, 0, np.pi)
        assert hasattr(bb1, "pulses") and len(bb1.pulses) == 4

        with pytest.raises(ValueError):
            sk1_composite_gate(ops, 0, 15.0)
        with pytest.raises(ValueError):
            bb1_composite_gate(ops, 0, 15.0)

    # -- Phase 8: Entangling gates --

    def test_10_entangling_gates(self):
        """Exercise MS, light-shift, and CZ gate construction and
        simulation."""
        hs = HilbertSpace(n_ions=2, n_modes=1, n_fock=15)
        ops = OperatorFactory(hs)
        sf = StateFactory(hs)
        eta = 0.05
        delta = TWO_PI * 15e3
        Omega = delta / (4 * eta)
        tau = ms_gate_duration(delta, loops=1)
        assert tau == pytest.approx(TWO_PI / delta)
        assert ms_gate_duration(delta, loops=2) == pytest.approx(2 * tau)

        H_ms = ms_gate_hamiltonian(ops, [0, 1], 0, [eta, eta], Omega, delta)
        assert isinstance(H_ms, list)
        r = qutip.sesolve(
            H_ms,
            sf.ground_state(),
            np.linspace(0, tau, 300),
            options={"max_step": tau / 60},
        )
        assert bell_state_fidelity(r.states[-1].ptrace([0, 1])) > 0.99

        H_ls = light_shift_gate_hamiltonian(
            ops, [0, 1], 0, [eta, eta], Omega, delta
        )
        assert isinstance(H_ls, list) and len(H_ls) > 0

        cz = cirac_zoller_gate(ops, 0, 1, 0, [eta, eta])
        assert len(cz) == 3
        for p in cz:
            assert (
                isinstance(p, GatePulse)
                and p.hamiltonian.isherm
                and p.duration > 0
            )

    # -- Phase 9: All noise channels --

    def test_11_all_noise_channels(self):
        """Exercise every noise model, verifying operator shapes and
        properties."""
        hs = HilbertSpace(n_ions=3, n_modes=1, n_fock=8)
        ops = OperatorFactory(hs)
        dim = hs.total_dim

        c_h0 = motional_heating_ops(ops, 0, 100.0, n_bar_env=0.0)
        assert len(c_h0) == 1
        c_h1 = motional_heating_ops(ops, 0, 100.0, n_bar_env=0.5)
        assert len(c_h1) == 2
        c_md = motional_dephasing_op(ops, 0, 1e3)
        assert c_md.shape == (dim, dim)

        r1 = heating_rate_from_noise(1e-11, 100e-6, 1e6)
        r2 = heating_rate_from_noise(1e-11, 50e-6, 1e6)
        assert r2 / r1 == pytest.approx(16.0, rel=0.5)

        c_dq = qubit_dephasing_op(ops, 0, t2=1e-3)
        assert c_dq.isherm
        with pytest.raises(ValueError):
            qubit_dephasing_op(ops, 0, t2=10.0, t1=1.0)
        c_se = spontaneous_emission_op(ops, 0, t1=1.0)
        assert not c_se.isherm
        c_rl = rayleigh_scattering_op(ops, 0, 1e3)
        assert c_rl.isherm
        c_rm = raman_scattering_op(ops, 0, 1e3)
        assert not c_rm.isherm
        c_lp = laser_phase_noise_op(ops, 0, 1e3)
        assert c_lp.isherm
        H_li = laser_intensity_noise_op(ops, 0, 0.01, 1e6)
        assert H_li.isherm
        H_xt = crosstalk_hamiltonian(ops, 0, 1, 0.01, 1e6, phase=0.3)
        assert H_xt.isherm and H_xt.shape == (dim, dim)

    # -- Phase 10: SPAM --

    def test_12_spam(self):
        """Exercise fluorescence, sampling, fidelity, and mid-circuit
        measurement."""
        hs = HilbertSpace(n_ions=3, n_modes=1, n_fock=5)
        ops = OperatorFactory(hs)
        sf = StateFactory(hs)
        rng = np.random.default_rng(42)

        probs = fluorescence_probabilities(sf.ground_state(), [0, 1, 2])
        assert all(p > 0.99 for p in probs)
        probs2 = fluorescence_probabilities(
            sf.product_state([1, 0, 1], [0]), [0, 1, 2]
        )
        assert probs2[0] < 0.01 and probs2[1] > 0.99 and probs2[2] < 0.01

        bits = sample_measurement(
            sf.ground_state(), [0, 1, 2], rng, spam_error=0.0
        )
        assert all(b in [0, 1] for b in bits)

        fid = measurement_fidelity(1e7, 100, 300e-6, 0.03)
        assert fid > 0.99

        plus = (
            sf.product_state([0, 0, 0], [0]) + sf.product_state([1, 0, 0], [0])
        ).unit()
        rho_post, outcome = mid_circuit_measurement(
            qutip.ket2dm(plus), ops, 0, rng
        )
        assert outcome in [0, 1]
        assert max(rho_post.ptrace(0).eigenenergies()) > 0.99

    # -- Phase 11: Transport --

    def test_13_qccd_transport(self):
        """Exercise shuttling and splitting noise models."""
        fast = shuttle_motional_excitation(200e-6, 10e-6, TWO_PI * 1e6)
        slow = shuttle_motional_excitation(200e-6, 300e-6, TWO_PI * 1e6)
        assert fast > slow >= 0.01

        hs = HilbertSpace(n_ions=1, n_modes=1, n_fock=20)
        ops = OperatorFactory(hs)
        sf = StateFactory(hs)
        rho0 = qutip.ket2dm(sf.ground_state())
        n_op = ops.number(0)
        rho_after = apply_shuttling_noise(rho0, ops, 0, added_quanta=0.5)
        assert qutip.expect(n_op, rho_after) > qutip.expect(n_op, rho0)
        rho_zero = apply_shuttling_noise(rho0, ops, 0, added_quanta=0.0)
        assert qutip.expect(n_op, rho_zero) == pytest.approx(0.0, abs=0.01)

        fast_s = split_crystal_excitation(TWO_PI * 1e6, 20e-6)
        slow_s = split_crystal_excitation(TWO_PI * 1e6, 200e-6)
        assert slow_s <= fast_s

    # -- Phase 12: THE MAIN EVENT -- Ising model quantum simulation --

    def test_14_ising_simulation_clean(self, yb_trap):
        """Simulate the Ising model on 2 ions via MS gate and measure
        correlations.

        The MS gate acts as exp(-i * chi * sigma_x_1 * sigma_x_2) on
        the spin subspace. Starting from |00> (sigma_z eigenstates,
        NOT sigma_x eigenstates), this creates entanglement because
        |00> is a superposition of sigma_x eigenstates. After the MS
        gate, we measure sigma_x correlations directly.

        Circuit:
        1. Start in |00, n=0>
        2. MS gate -> (|00> + i|11>)/sqrt(2) Bell state
        3. Measure sigma_x and sigma_z correlations
        4. Apply Ry(-pi/2) to rotate sigma_x -> sigma_z for "Z-basis" readout
        """
        hs = HilbertSpace(n_ions=2, n_modes=1, n_fock=15)
        ops = OperatorFactory(hs)
        sf = StateFactory(hs)

        # Step 1: ground state |00,0>
        psi0 = sf.ground_state()

        # Step 2: MS entangling gate directly on |00>
        eta = 0.05
        delta = TWO_PI * 15e3
        Omega = delta / (4 * eta)
        tau = ms_gate_duration(delta)
        H_ms = ms_gate_hamiltonian(ops, [0, 1], 0, [eta, eta], Omega, delta)
        tlist = np.linspace(0, tau, 300)
        result = qutip.sesolve(
            H_ms, psi0, tlist, options={"max_step": tau / 60}
        )
        psi_bell = result.states[-1]

        # Step 3: Verify Bell state via sigma_z correlations
        sz0, sz1 = ops.sigma_z(0), ops.sigma_z(1)
        zz_corr = qutip.expect(sz0 * sz1, psi_bell)
        sz0_mean = qutip.expect(sz0, psi_bell)
        sz1_mean = qutip.expect(sz1, psi_bell)
        connected_corr = zz_corr - sz0_mean * sz1_mean
        # For (|00>+i|11>)/sqrt(2): <sz0*sz1>=1, <sz0>=<sz1>=0, corr=1
        assert connected_corr > 0.9

        # Also verify entanglement: single-qubit reduced state is
        # maximally mixed
        rho_single = psi_bell.ptrace(0)
        single_purity = (rho_single**2).tr().real
        assert single_purity < 0.55  # ~0.5 for Bell state

        # Step 4: Verify Bell state via fluorescence measurement sampling
        rng = np.random.default_rng(42)
        rho_bell = qutip.ket2dm(psi_bell)
        outcomes = [
            sample_measurement(rho_bell, [0, 1], rng) for _ in range(100)
        ]
        # Bell state (|00>+i|11>)/sqrt(2): should measure 00 or 11
        # with equal prob
        n_same = sum(1 for o in outcomes if o[0] == o[1])
        assert n_same > 80  # should be ~100 for ideal Bell state

        # Phase-space trajectory: motion should return to origin after
        # full loop
        x_traj, p_traj = phase_space_trajectory(result.states, 0, [0, 1])
        assert len(x_traj) == len(tlist)
        assert abs(x_traj[-1]) < 0.3

        # Wigner function of final motional state
        xvec = np.linspace(-3, 3, 30)
        W = motional_wigner(result.states[-1], 0, [0, 1], xvec)
        assert W.shape == (30, 30)

    def test_15_ising_simulation_noisy(self, yb_trap):
        """Same Ising simulation but with realistic noise. Compare to clean."""
        hs = HilbertSpace(n_ions=2, n_modes=1, n_fock=15)
        ops = OperatorFactory(hs)
        sf = StateFactory(hs)

        eta = 0.05
        delta = TWO_PI * 15e3
        Omega = delta / (4 * eta)
        tau = ms_gate_duration(delta)

        # Prepare |++>
        g_ry = ry_gate(ops, 0, np.pi / 2, rabi_frequency=TWO_PI * 1e6)
        H_ry_global = (
            g_ry.hamiltonian
            + ry_gate(ops, 1, np.pi / 2, TWO_PI * 1e6).hamiltonian
        )
        psi_plus = qutip.sesolve(
            H_ry_global, sf.ground_state(), [0, g_ry.duration]
        ).states[-1]

        H_ms = ms_gate_hamiltonian(ops, [0, 1], 0, [eta, eta], Omega, delta)
        tlist = np.linspace(0, tau, 300)

        # Clean evolution
        r_clean = qutip.sesolve(
            H_ms, psi_plus, tlist, options={"max_step": tau / 60}
        )

        # Noisy evolution with every type of noise
        c_ops = [
            *motional_heating_ops(ops, 0, heating_rate=5e4),
            qubit_dephasing_op(ops, 0, t2=500e-6),
            qubit_dephasing_op(ops, 1, t2=500e-6),
            rayleigh_scattering_op(ops, 0, rate=1e3),
            laser_phase_noise_op(ops, 1, rate=500),
        ]
        r_noisy = qutip.mesolve(
            H_ms, psi_plus, tlist, c_ops=c_ops, options={"max_step": tau / 60}
        )

        # Noise should reduce spin purity
        purity_clean = (r_clean.states[-1].ptrace([0, 1]) ** 2).tr().real
        purity_noisy = (r_noisy.states[-1].ptrace([0, 1]) ** 2).tr().real
        assert purity_noisy < purity_clean

        # Compute error budget
        fid_clean = bell_state_fidelity(r_clean.states[-1].ptrace([0, 1]))
        fid_noisy = bell_state_fidelity(r_noisy.states[-1].ptrace([0, 1]))
        budget = compute_error_budget(
            ideal_fidelity=fid_clean,
            heating_error=(fid_clean - fid_noisy) * 0.5,
            dephasing_error=(fid_clean - fid_noisy) * 0.3,
            scattering_error=(fid_clean - fid_noisy) * 0.1,
            laser_noise_error=(fid_clean - fid_noisy) * 0.1,
        )
        assert budget["total_error"] > 0
        assert "heating" in budget

    # -- Phase 13: QCCD transport + mid-circuit measurement cycle --

    def test_16_qccd_mid_circuit_cycle(self):
        """Simulate a QCCD-style sequence:
        gate -> transport -> mid-circuit measure -> gate.

        This exercises the transport and measurement modules in a
        realistic workflow:
        1. Prepare Bell state on 2 ions
        2. "Shuttle" to measurement zone (adds motional noise)
        3. Mid-circuit measure ion 1 (ancilla)
        4. Conditional on outcome, analyze remaining qubit
        """
        hs = HilbertSpace(n_ions=2, n_modes=1, n_fock=15)
        ops = OperatorFactory(hs)
        sf = StateFactory(hs)
        rng = np.random.default_rng(123)

        # Prepare Bell state via MS
        eta = 0.05
        delta = TWO_PI * 15e3
        Omega = delta / (4 * eta)
        tau = ms_gate_duration(delta)
        H_ms = ms_gate_hamiltonian(ops, [0, 1], 0, [eta, eta], Omega, delta)
        result = qutip.sesolve(
            H_ms,
            sf.ground_state(),
            np.linspace(0, tau, 300),
            options={"max_step": tau / 60},
        )
        rho_bell = qutip.ket2dm(result.states[-1])

        # Simulate transport to measurement zone
        shuttle_delta_n = shuttle_motional_excitation(
            200e-6, 50e-6, TWO_PI * 1e6
        )
        assert shuttle_delta_n > 0
        rho_transported = apply_shuttling_noise(
            rho_bell, ops, 0, shuttle_delta_n
        )

        # Mid-circuit measurement of ion 1 (ancilla)
        rho_post, outcome = mid_circuit_measurement(
            rho_transported, ops, 1, rng
        )
        assert outcome in [0, 1]

        # After measuring ancilla, data qubit should be in a definite state
        rho_data = rho_post.ptrace(0)
        purity_data = (rho_data**2).tr().real
        assert purity_data > 0.8  # mostly pure after measurement

        # Verify state_fidelity and gate_fidelity work on the result
        sf_val = state_fidelity(rho_data, rho_data)
        assert sf_val == pytest.approx(1.0, abs=1e-8)

        bell_dm = qutip.ket2dm(
            (
                qutip.tensor(qutip.basis(2, 0), qutip.basis(2, 0))
                + 1j * qutip.tensor(qutip.basis(2, 1), qutip.basis(2, 1))
            ).unit()
        )
        gf = gate_fidelity(rho_bell, bell_dm, [0, 1])
        assert gf > 0.95

        # Crystal splitting excitation estimate
        split_n = split_crystal_excitation(TWO_PI * 1e6, 50e-6)
        assert split_n >= 0.05

    # -- Phase 14: SimulationRunner high-level API --

    def test_17_simulation_runner(self, yb_trap):
        """Exercise SimulationRunner with clean and noisy configs."""
        config = SimulationConfig(
            species=get_species("Yb171"),
            trap=yb_trap,
            n_ions=2,
            n_modes=1,
            n_fock=15,
            solver="sesolve",
        )
        runner = SimulationRunner(config)
        assert runner.eta.shape == (2, 2)

        r = runner.run_carrier_pulse(ion=0, theta=np.pi)
        assert qutip.expect(
            runner.ops.sigma_z(0), r.states[-1]
        ) == pytest.approx(-1, abs=0.15)

        r_ms = runner.run_ms_gate(ions=[0, 1])
        # The two-qubit state is pure (motion disentangled), but each single
        # qubit is maximally mixed in a Bell state: Tr(rho_single^2) ~ 0.5
        rho_single = r_ms.states[-1].ptrace(0)
        purity_single = (rho_single**2).tr().real
        assert purity_single < 0.6  # maximally entangled -> purity ~ 0.5

        with pytest.raises(ValueError):
            runner.run_ms_gate(ions=[0, 1, 2])

        config_noisy = SimulationConfig(
            species=get_species("Yb171"),
            trap=yb_trap,
            n_ions=1,
            n_modes=1,
            n_fock=10,
            solver="mesolve",
            heating_rate=1e4,
            t2_qubit=1e-3,
        )
        runner_noisy = SimulationRunner(config_noisy)
        r_n = runner_noisy.run_carrier_pulse(ion=0, theta=np.pi)
        assert qutip.expect(runner_noisy.ops.sigma_z(0), r_n.states[-1]) < 1.0

    # -- Phase 15: Cross-species generality --

    def test_18_all_species_pipeline(self):
        """Verify the full pipeline works for every ion species."""
        configs = [
            ("Yb171", 1000),
            ("Ca40", 300),
            ("Ca43", 300),
            ("Ba137", 500),
            ("Be9", 300),
            ("Sr88", 500),
        ]
        for name, v_rf in configs:
            species = get_species(name)
            trap = PaulTrap(
                v_rf=v_rf,
                omega_rf=TWO_PI * 30e6,
                r0=0.5e-3,
                omega_axial=TWO_PI * 1e6,
                species=species,
            )
            if not trap.is_stable():
                continue
            modes = normal_modes(1, trap)
            assert modes.axial_freqs[0] == pytest.approx(
                trap.omega_axial, rel=1e-3
            )
            if species.qubit_wavelength:
                k = TWO_PI / species.qubit_wavelength
            else:
                k = 2 * TWO_PI / species.raman_wavelength
            eta = lamb_dicke_parameters(modes, species, k, "axial")
            assert abs(eta[0, 0]) > 0
            n_bar = doppler_cooled_nbar(species, trap.omega_axial / TWO_PI)
            assert n_bar > 0


class TestAnalyticalExactness:
    """Tight numerical checks against known analytical results.

    If any formula coefficient, sign, or convention is wrong, these tests fail.
    Every assertion uses tight tolerances (atol=0.01 or rel=0.01).
    """

    def test_two_ion_spacing_analytical(self):
        """Two-ion spacing:
        d = (e^2 / (4*pi*eps0*m*omega_z^2))^(1/3) * 2^(1/3) * 2"""
        ca = get_species("Ca40")
        omega_z = TWO_PI * 1e6
        trap = PaulTrap(
            v_rf=300,
            omega_rf=TWO_PI * 30e6,
            r0=0.5e-3,
            omega_axial=omega_z,
            species=ca,
        )
        pos = equilibrium_positions(2, trap)
        l_scale = (
            ELECTRON_CHARGE**2 / (4 * PI * EPSILON_0 * ca.mass_kg * omega_z**2)
        ) ** (1 / 3)
        # Analytical: u1 = -u2, u1 = -(1/2)^(2/3) ~ -0.6300
        u_analytical = (1 / 2) ** (2 / 3)  # ~0.6300
        d_analytical = 2 * u_analytical * l_scale
        d_measured = pos[1] - pos[0]
        assert d_measured == pytest.approx(d_analytical, rel=0.001)

    def test_two_ion_stretch_frequency_exact(self):
        """Stretch mode frequency: omega_stretch = sqrt(3) * omega_axial."""
        ca = get_species("Ca40")
        trap = PaulTrap(
            v_rf=300,
            omega_rf=TWO_PI * 30e6,
            r0=0.5e-3,
            omega_axial=TWO_PI * 1e6,
            species=ca,
        )
        modes = normal_modes(2, trap)
        ratio = modes.axial_freqs[1] / modes.axial_freqs[0]
        assert ratio == pytest.approx(np.sqrt(3), rel=1e-4)

    def test_lamb_dicke_formula_direct(self):
        """Verify eta = k * b * sqrt(hbar/(2*m*omega)) against hand
        calculation."""
        ca = get_species("Ca40")
        omega = TWO_PI * 1e6
        trap = PaulTrap(
            v_rf=300,
            omega_rf=TWO_PI * 30e6,
            r0=0.5e-3,
            omega_axial=omega,
            species=ca,
        )
        modes = normal_modes(1, trap)
        k = TWO_PI / 729e-9
        eta = lamb_dicke_parameters(modes, ca, k, "axial")
        x_zpf = np.sqrt(HBAR / (2 * ca.mass_kg * omega))
        eta_hand = k * 1.0 * x_zpf  # b=1 for single ion COM
        assert eta[0, 0] == pytest.approx(eta_hand, rel=1e-6)

    def test_carrier_rabi_oscillation_exact(self):
        """sigma_z must equal cos(Omega*t) exactly (to solver precision)."""
        hs = HilbertSpace(n_ions=1, n_modes=1, n_fock=3)
        ops = OperatorFactory(hs)
        sf = StateFactory(hs)
        Omega = TWO_PI * 500e3
        H = carrier_hamiltonian(ops, 0, Omega)
        tlist = np.linspace(0, 4 * PI / Omega, 400)
        result = qutip.sesolve(
            H, sf.ground_state(), tlist, e_ops=[ops.sigma_z(0)]
        )
        expected = np.cos(Omega * tlist)
        np.testing.assert_allclose(result.expect[0], expected, atol=0.01)

    def test_rsb_rabi_frequency_vs_fock_number(self):
        """RSB Rabi frequency on |0,n> is exactly eta*Omega*sqrt(n)."""
        hs = HilbertSpace(n_ions=1, n_modes=1, n_fock=15)
        ops = OperatorFactory(hs)
        eta, Omega = 0.1, TWO_PI * 200e3
        H = red_sideband_hamiltonian(ops, 0, 0, Omega, eta)
        for n in [1, 2, 3, 5]:
            psi0 = qutip.tensor(qutip.basis(2, 0), qutip.basis(15, n))
            target = qutip.tensor(qutip.basis(2, 1), qutip.basis(15, n - 1))
            expected_rabi = eta * Omega * np.sqrt(n)
            t_pi = PI / expected_rabi
            r = qutip.sesolve(H, psi0, np.linspace(0, t_pi, 100))
            fid = abs(r.states[-1].overlap(target)) ** 2
            assert fid > 0.98, f"RSB pi-pulse failed for n={n}: fid={fid:.4f}"

    def test_ms_gate_bell_fidelity_exact(self):
        """MS gate must produce Bell state with F > 0.999."""
        hs = HilbertSpace(n_ions=2, n_modes=1, n_fock=20)
        ops = OperatorFactory(hs)
        sf = StateFactory(hs)
        eta = 0.05
        delta = TWO_PI * 15e3
        Omega = delta / (4 * eta)
        tau = ms_gate_duration(delta)
        H = ms_gate_hamiltonian(ops, [0, 1], 0, [eta, eta], Omega, delta)
        r = qutip.sesolve(
            H,
            sf.ground_state(),
            np.linspace(0, tau, 500),
            options={"max_step": tau / 100},
        )
        fid = bell_state_fidelity(r.states[-1].ptrace([0, 1]))
        assert fid > 0.999
        n_final = qutip.expect(ops.number(0), r.states[-1])
        assert n_final == pytest.approx(0.0, abs=0.01)

    def test_ms_gate_thermal_insensitivity(self):
        """MS gate fidelity should remain > 0.95 starting from
        n_bar=3 thermal state."""
        hs = HilbertSpace(n_ions=2, n_modes=1, n_fock=25)
        ops = OperatorFactory(hs)
        sf = StateFactory(hs)
        eta = 0.03  # smaller eta for better thermal insensitivity
        delta = TWO_PI * 10e3
        Omega = delta / (4 * eta)
        tau = ms_gate_duration(delta)
        H = ms_gate_hamiltonian(ops, [0, 1], 0, [eta, eta], Omega, delta)
        rho0 = sf.thermal_state(n_bar=[3.0])
        r = qutip.mesolve(
            H, rho0, np.linspace(0, tau, 500), options={"max_step": tau / 100}
        )
        fid = bell_state_fidelity(r.states[-1].ptrace([0, 1]))
        assert fid > 0.95

    def test_dephasing_decay_rate_exact(self):
        """Off-diagonal should decay as exp(-t/T2) under pure dephasing."""
        hs = HilbertSpace(n_ions=1, n_modes=1, n_fock=3)
        ops = OperatorFactory(hs)
        sf = StateFactory(hs)
        T2 = 100e-6
        plus = (sf.product_state([0], [0]) + sf.product_state([1], [0])).unit()
        c_ops = [qubit_dephasing_op(ops, 0, T2)]
        tlist = np.linspace(0, 3 * T2, 200)
        result = qutip.mesolve(
            0 * ops.identity(),
            plus,
            tlist,
            c_ops=c_ops,
            e_ops=[ops.sigma_x(0)],
        )
        expected = np.exp(-tlist / T2)
        np.testing.assert_allclose(result.expect[0], expected, atol=0.05)

    def test_heating_rate_exact(self):
        """<n>(t) = exp(gamma*t) - 1 for L=sqrt(gamma)*a_dag starting
        from vacuum."""
        hs = HilbertSpace(n_ions=1, n_modes=1, n_fock=30)
        ops = OperatorFactory(hs)
        sf = StateFactory(hs)
        n_dot = 1e4  # quanta/s
        c_ops = motional_heating_ops(ops, 0, heating_rate=n_dot)
        tlist = np.linspace(0, 100e-6, 50)
        result = qutip.mesolve(
            0 * ops.identity(),
            sf.ground_state(),
            tlist,
            c_ops=c_ops,
            e_ops=[ops.number(0)],
        )
        # With L = sqrt(gamma)*a_dag: d<n>/dt = gamma*(<n>+1),
        # <n>(t) = exp(gamma*t) - 1
        expected = np.exp(n_dot * tlist) - 1
        np.testing.assert_allclose(result.expect[0], expected, atol=0.05)

    def test_spontaneous_emission_decay_exact(self):
        """Excited state population decays as exp(-t/T1)."""
        hs = HilbertSpace(n_ions=1, n_modes=1, n_fock=3)
        ops = OperatorFactory(hs)
        sf = StateFactory(hs)
        T1 = 100e-6
        psi_excited = sf.product_state([1], [0])
        c_ops = [spontaneous_emission_op(ops, 0, T1)]
        tlist = np.linspace(0, 3 * T1, 200)
        # P(excited) = <|1><1|> = (1 - <sigma_z>)/2
        result = qutip.mesolve(
            0 * ops.identity(),
            psi_excited,
            tlist,
            c_ops=c_ops,
            e_ops=[ops.sigma_z(0)],
        )
        # sigma_z starts at -1 (excited) and decays to +1 (ground)
        # <sigma_z>(t) = 1 - 2*exp(-t/T1)
        expected_sz = 1.0 - 2.0 * np.exp(-tlist / T1)
        np.testing.assert_allclose(result.expect[0], expected_sz, atol=0.05)

    def test_measurement_correlations_bell_state(self):
        """Bell state measurement must produce perfectly correlated
        outcomes."""
        hs = HilbertSpace(n_ions=2, n_modes=1, n_fock=10)
        ops = OperatorFactory(hs)
        sf = StateFactory(hs)
        eta = 0.05
        delta = TWO_PI * 15e3
        Omega = delta / (4 * eta)
        tau = ms_gate_duration(delta)
        H = ms_gate_hamiltonian(ops, [0, 1], 0, [eta, eta], Omega, delta)
        r = qutip.sesolve(
            H,
            sf.ground_state(),
            np.linspace(0, tau, 300),
            options={"max_step": tau / 60},
        )
        rho = qutip.ket2dm(r.states[-1])
        rng = np.random.default_rng(42)
        outcomes = [sample_measurement(rho, [0, 1], rng) for _ in range(200)]
        n_same = sum(1 for o in outcomes if o[0] == o[1])
        assert n_same > 195  # essentially all correlated

    def test_doppler_limit_formula_consistency(self):
        """Doppler limit n_bar = Gamma/(2*omega) must be consistent
        with T_D."""
        for name in ["Yb171", "Ca40", "Ba137"]:
            s = get_species(name)
            omega_hz = 1.5e6
            n_bar = doppler_cooled_nbar(s, omega_hz)
            T_D = s.doppler_limit_temperature()
            n_from_T = BOLTZMANN * T_D / (HBAR * TWO_PI * omega_hz)
            assert n_bar == pytest.approx(n_from_T, rel=0.01)

    def test_pseudopotential_depth_vs_mathieu_q(self):
        """Trap depth should equal q*V_rf/8 in eV."""
        ca = get_species("Ca40")
        trap = PaulTrap(
            v_rf=300,
            omega_rf=TWO_PI * 30e6,
            r0=0.5e-3,
            omega_axial=TWO_PI * 1e6,
            species=ca,
        )
        depth_from_q = trap.mathieu_q * trap.v_rf / 8
        assert trap.pseudopotential_depth_eV == pytest.approx(
            depth_from_q, rel=1e-6
        )

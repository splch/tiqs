"""Experimental validation tests grounded in published results.

Each test reproduces a specific published measurement or analytical
result from the trapped-ion literature, with citations.  These
tests verify that the simulator's physics matches reality, not just
internal self-consistency.

References
----------
[James1998]  James, Appl. Phys. B 66, 181 (1998). quant-ph/9702053.
[Leibfried2003]  Leibfried et al., Rev. Mod. Phys. 75, 281 (2003).
[Benhelm2008]  Benhelm et al., Nature Physics 4, 463 (2008).
[Jain2024]  Jain et al., Nature 627, 510 (2024). arXiv:2308.07672.
[BrownGabrielse1982]  Brown & Gabrielse, Phys. Rev. A 25, 2423 (1982).
[Hanneke2008]  Hanneke et al., PRL 100, 120801 (2008).
[Akram2025]  Akram et al., PRX 15, 021079 (2025). arXiv:2403.04730.
[Hrmo2023]  Hrmo et al., Nature Communications 14, 2242 (2023).
[Ringbauer2022]  Ringbauer et al., Nature Physics 18, 1053 (2022).
"""

import math

import numpy as np
import pytest
import qutip

from tiqs.analysis.fidelity import bell_state_fidelity
from tiqs.chain.equilibrium import equilibrium_positions
from tiqs.chain.lamb_dicke import lamb_dicke_parameters
from tiqs.chain.normal_modes import normal_modes
from tiqs.constants import (
    ELECTRON_CHARGE,
    ELECTRON_MASS,
    EPSILON_0,
    PI,
    TWO_PI,
)
from tiqs.gates.molmer_sorensen import ms_gate_duration, ms_gate_hamiltonian
from tiqs.hilbert_space.builder import HilbertSpace
from tiqs.hilbert_space.operators import OperatorFactory
from tiqs.hilbert_space.states import StateFactory
from tiqs.interaction.gradient import MagneticGradient
from tiqs.species.electron import ElectronSpecies
from tiqs.species.ion import get_species
from tiqs.trap import PaulTrap, PenningTrap


class TestJames1998NormalModes:
    """Analytical mode frequencies and equilibrium positions from
    [James1998], the foundational reference verified by every
    trapped-ion experiment."""

    @pytest.fixture
    def ca40_trap(self):
        return PaulTrap(
            v_rf=300,
            omega_rf=TWO_PI * 30e6,
            r0=0.5e-3,
            omega_axial=TWO_PI * 1e6,
            species=get_species("Ca40"),
        )

    def test_two_ion_stretch_to_com_ratio(self, ca40_trap):
        """[James1998] Eq. 18: stretch/COM = sqrt(3) exactly."""
        modes = normal_modes(2, ca40_trap)
        ratio = modes.axial_freqs[1] / modes.axial_freqs[0]
        assert ratio == pytest.approx(np.sqrt(3), rel=1e-6)

    def test_two_ion_com_equals_axial(self, ca40_trap):
        """[James1998]: COM mode frequency equals the bare trap
        frequency."""
        modes = normal_modes(2, ca40_trap)
        assert modes.axial_freqs[0] == pytest.approx(
            ca40_trap.omega_axial, rel=1e-6
        )

    def test_three_ion_breathing_mode_ratio(self, ca40_trap):
        """[James1998] Table I: breathing mode at sqrt(29/5) * omega_z
        for 3 ions."""
        modes = normal_modes(3, ca40_trap)
        ratio = modes.axial_freqs[2] / modes.axial_freqs[0]
        assert ratio == pytest.approx(np.sqrt(29 / 5), rel=1e-4)

    def test_three_ion_tilt_mode_ratio(self, ca40_trap):
        """[James1998] Table I: tilt mode at sqrt(3) * omega_z."""
        modes = normal_modes(3, ca40_trap)
        ratio = modes.axial_freqs[1] / modes.axial_freqs[0]
        assert ratio == pytest.approx(np.sqrt(3), rel=1e-4)

    def test_three_ion_center_at_origin(self, ca40_trap):
        """[James1998]: center ion at z=0 by symmetry."""
        pos = equilibrium_positions(3, ca40_trap)
        assert pos[1] == pytest.approx(0.0, abs=1e-12)

    def test_three_ion_dimensionless_offset(self, ca40_trap):
        """[James1998] Table I: outer ions at u = +/-(5/4)^(1/3)
        in dimensionless units."""
        pos = equilibrium_positions(3, ca40_trap)
        length_scale = (
            ELECTRON_CHARGE**2
            / (
                4
                * PI
                * EPSILON_0
                * ca40_trap.species.mass_kg
                * ca40_trap.omega_axial**2
            )
        ) ** (1 / 3)
        u_outer = pos[2] / length_scale
        assert u_outer == pytest.approx((5 / 4) ** (1 / 3), rel=1e-3)

    def test_com_eigenvector_equal_amplitude(self, ca40_trap):
        """[James1998]: COM eigenvector has all components equal."""
        modes = normal_modes(2, ca40_trap)
        v_com = modes.axial_vectors[:, 0]
        assert abs(v_com[0]) == pytest.approx(abs(v_com[1]), rel=1e-6)

    def test_stretch_eigenvector_opposite_phase(self, ca40_trap):
        """[James1998]: stretch eigenvector has opposite signs."""
        modes = normal_modes(2, ca40_trap)
        v_str = modes.axial_vectors[:, 1]
        assert np.sign(v_str[0]) != np.sign(v_str[1])


class TestCa40IonSpacing:
    """Two-ion equilibrium spacing computed from fundamental constants,
    verified against [Leibfried2003] and direct CCD imaging."""

    def test_two_ion_spacing_ca40_at_1mhz(self):
        """d = 2^(1/3) * l_0 where l_0 = (e^2/(4*pi*eps0*m*omega_z^2))^(1/3).
        For Ca-40 at 1 MHz: d ~ 5.6 um."""
        ca = get_species("Ca40")
        trap = PaulTrap(
            v_rf=300,
            omega_rf=TWO_PI * 30e6,
            r0=0.5e-3,
            omega_axial=TWO_PI * 1e6,
            species=ca,
        )
        pos = equilibrium_positions(2, trap)
        spacing = pos[1] - pos[0]

        # Analytical formula
        m = ca.mass_kg
        omega_z = trap.omega_axial
        ke = 1 / (4 * PI * EPSILON_0)
        d_analytical = (2 * ke * ELECTRON_CHARGE**2 / (m * omega_z**2)) ** (
            1 / 3
        )

        assert spacing == pytest.approx(d_analytical, rel=1e-3)
        assert 4e-6 < spacing < 7e-6  # ~ 5.6 um


class TestMSGateBellState:
    """Verify that the MS gate produces a Bell state with the correct
    geometric phase, matching the condition from [Leibfried2003]."""

    def test_ideal_ms_bell_fidelity(self):
        """An ideal single-loop MS gate with eta*Omega = delta/4
        should produce (|00> + i|11>)/sqrt(2) with F > 0.99."""
        hs = HilbertSpace(n_ions=2, n_modes=1, n_fock=20)
        ops = OperatorFactory(hs)
        sf = StateFactory(hs)

        eta = 0.05
        delta = TWO_PI * 15e3
        Omega = delta / (4 * eta)
        tau = ms_gate_duration(delta)

        H = ms_gate_hamiltonian(ops, [0, 1], 0, [eta, eta], Omega, delta)
        psi0 = sf.ground_state()
        tlist = np.linspace(0, tau, 800)
        result = qutip.sesolve(H, psi0, tlist, options={"max_step": tau / 200})
        rho_spin = result.states[-1].ptrace([0, 1])
        fid = bell_state_fidelity(rho_spin)
        assert fid > 0.99

    def test_ms_gate_motional_closure(self):
        """After one complete phase-space loop, the motional state
        should return to vacuum: <n> ~ 0."""
        hs = HilbertSpace(n_ions=2, n_modes=1, n_fock=20)
        ops = OperatorFactory(hs)
        sf = StateFactory(hs)

        eta = 0.05
        delta = TWO_PI * 15e3
        Omega = delta / (4 * eta)
        tau = ms_gate_duration(delta)

        H = ms_gate_hamiltonian(ops, [0, 1], 0, [eta, eta], Omega, delta)
        result = qutip.sesolve(
            H,
            sf.ground_state(),
            np.linspace(0, tau, 800),
            options={"max_step": tau / 200},
        )
        n_final = qutip.expect(ops.number(0), result.states[-1])
        assert n_final == pytest.approx(0.0, abs=0.1)

    def test_rabi_oscillation_frequency(self):
        """[Leibfried2003] Eq. 22: carrier Rabi oscillation at
        frequency Omega. <sigma_z> = cos(Omega*t) for H = Omega/2 * sigma_x."""
        from tiqs.interaction.hamiltonian import carrier_hamiltonian

        hs = HilbertSpace(n_ions=1, n_modes=1, n_fock=5)
        ops = OperatorFactory(hs)
        sf = StateFactory(hs)

        Omega = TWO_PI * 200e3
        H = carrier_hamiltonian(ops, 0, Omega)
        tlist = np.linspace(0, 3 * TWO_PI / Omega, 500)
        result = qutip.sesolve(
            H, sf.ground_state(), tlist, e_ops=[ops.sigma_z(0)]
        )
        expected = np.cos(Omega * tlist)
        np.testing.assert_allclose(result.expect[0], expected, atol=0.05)


class TestPenningTrapETH:
    """Penning trap frequencies from [Jain2024] (ETH Zurich
    micro-Penning trap, Be-9+ at 3T)."""

    @pytest.fixture
    def eth_trap(self):
        """ETH parameters: B=3T, omega_z=2*pi*2.5 MHz, d~152 um."""
        return PenningTrap(
            magnetic_field=3.0,
            species=get_species("Be9"),
            d=152e-6,
            omega_axial=TWO_PI * 2.5e6,
        )

    def test_cyclotron_sum_rule(self, eth_trap):
        """[BrownGabrielse1982]: omega_+ + omega_- = omega_c."""
        wp = eth_trap.omega_cyclotron
        wm = eth_trap.omega_magnetron
        wc = eth_trap.omega_cyclotron_free
        assert wp + wm == pytest.approx(wc, rel=1e-10)

    def test_brown_gabrielse_invariant(self, eth_trap):
        """[BrownGabrielse1982]: omega_+^2 + omega_-^2 + omega_z^2 = omega_c^2.
        Verified experimentally to 4e-6 relative precision by
        Berrocal et al. PRA 110, 063107 (2024)."""
        wp = eth_trap.omega_cyclotron
        wm = eth_trap.omega_magnetron
        wz = eth_trap.omega_axial
        wc = eth_trap.omega_cyclotron_free
        assert wp**2 + wm**2 + wz**2 == pytest.approx(wc**2, rel=1e-10)

    def test_product_relation(self, eth_trap):
        """omega_+ * omega_- = omega_z^2 / 2."""
        wp = eth_trap.omega_cyclotron
        wm = eth_trap.omega_magnetron
        wz = eth_trap.omega_axial
        assert wp * wm == pytest.approx(wz**2 / 2, rel=1e-10)

    def test_free_cyclotron_frequency_be9_at_3t(self, eth_trap):
        """[Jain2024]: omega_c/(2*pi) = 5.12 MHz for Be-9 at 3T.
        f_c = eB / (2*pi*m)."""
        fc = eth_trap.omega_cyclotron_free / TWO_PI
        be_mass = get_species("Be9").mass_kg
        expected = ELECTRON_CHARGE * 3.0 / (TWO_PI * be_mass)
        assert fc == pytest.approx(expected, rel=1e-6)
        # Cross-check against published value
        assert fc == pytest.approx(5.12e6, rel=0.02)

    def test_frequency_hierarchy(self, eth_trap):
        """[Jain2024]: omega_- < omega_z < omega_+ < omega_c."""
        wm, wz, wp, wc = eth_trap.frequency_hierarchy
        assert wm < wz < wp < wc

    def test_stability(self, eth_trap):
        assert eth_trap.is_stable()

    def test_axial_modes_match_paul_trap_physics(self, eth_trap):
        """Axial modes in a Penning trap use the same Hessian as a
        Paul trap.  2-ion stretch/COM ratio must still be sqrt(3)."""
        modes = normal_modes(2, eth_trap)
        ratio = modes.axial_freqs[1] / modes.axial_freqs[0]
        assert ratio == pytest.approx(np.sqrt(3), rel=1e-4)


class TestHannekeCyclotronFrequency:
    """Electron cyclotron frequency from the most precise
    single-particle measurement: [Hanneke2008]."""

    def test_electron_cyclotron_at_5_36t(self):
        """[Hanneke2008]: f_c = eB/(2*pi*m_e) ~ 150 GHz at B=5.36T.
        This validated ELECTRON_MASS and ELECTRON_CHARGE to
        sub-ppb precision."""
        B = 5.36
        fc = ELECTRON_CHARGE * B / (TWO_PI * ELECTRON_MASS)
        assert fc == pytest.approx(150.0e9, rel=1e-3)


class TestEleQtronGradientParameters:
    """Magnetic gradient coupling from [Akram2025] (eleQtron,
    Yb-171 with 19 T/m gradient)."""

    def test_gradient_frequency_splitting(self):
        """[Akram2025]: At 19.09 T/m gradient, two Yb-171 ions
        separated by ~80 um have frequency splitting of ~3.2 MHz.
        d(omega_q)/dz = (d omega_q/dB) * (dB/dz)."""
        yb = get_species("Yb171")
        grad = MagneticGradient(db_dz=19.09, b_field=0.5e-3)

        # Frequency shift per meter
        shift_per_m = yb.qubit_zeeman_sensitivity * grad.db_dz / TWO_PI
        # Shift per 80 um
        shift_80um = shift_per_m * 80e-6
        # Should be ~3-4 MHz
        assert 1e6 < shift_80um < 10e6

    def test_gradient_effective_k_for_electron(self):
        """The gradient k_eff for electrons simplifies to (dB/dz)/B.
        This is verified in [Akram2025] Eq. 2 and the existing
        test_electron.py."""
        grad = MagneticGradient(db_dz=120.0, b_field=0.1)
        e = ElectronSpecies(magnetic_field=0.1)
        assert grad.effective_k(e) == pytest.approx(1200.0)


class TestDopplerCoolingLimits:
    """Doppler cooling limits from [Leibfried2003] Eq. 6."""

    def test_ca40_doppler_temperature(self):
        """T_D = hbar*Gamma / (2*k_B).  For Ca-40: Gamma/(2*pi) = 22.4 MHz,
        T_D ~ 0.5 mK."""
        ca = get_species("Ca40")
        T_D = ca.doppler_limit_temperature()
        assert 0.1e-3 < T_D < 2e-3

    def test_doppler_nbar_formula(self):
        """nbar_D = Gamma / (2*omega_trap).
        For Ca-40 at 1 MHz: nbar ~ 11."""
        ca = get_species("Ca40")
        nbar = ca.doppler_limit_nbar(1e6)
        gamma = ca.cooling_transition.linewidth
        omega_trap = TWO_PI * 1e6
        expected = gamma / (2 * omega_trap)
        assert nbar == pytest.approx(expected, rel=1e-6)
        assert 5 < nbar < 20


class TestSpeciesAtomicData:
    """Cross-check species database against NIST atomic data and
    published values."""

    def test_yb171_qubit_splitting(self):
        """Yb-171 hyperfine splitting: 12.6428 GHz (NIST)."""
        yb = get_species("Yb171")
        assert yb.qubit_frequency_hz == pytest.approx(12.6428e9, rel=1e-4)

    def test_ca40_metastable_lifetime(self):
        """Ca-40 D5/2 lifetime: 1.168 s (Barton et al. 2000)."""
        ca = get_species("Ca40")
        assert ca.metastable_lifetime == pytest.approx(1.168, rel=0.01)

    def test_yb171_infinite_t1(self):
        """Hyperfine ground-state qubits have T1 = infinity."""
        yb = get_species("Yb171")
        assert yb.qubit_t1 == math.inf

    def test_be9_is_lightest(self):
        """Be-9 is the lightest commonly used ion qubit."""
        be = get_species("Be9")
        for name in ["Ca40", "Ca43", "Ba137", "Yb171", "Sr88"]:
            assert be.mass_amu < get_species(name).mass_amu

    def test_lighter_ion_larger_lamb_dicke(self):
        """Lighter ions have larger Lamb-Dicke parameters at the same
        trap frequency (eta ~ 1/sqrt(m)).  [Leibfried2003] Eq. 11."""
        k = TWO_PI / 400e-9
        for species_name, v_rf in [("Be9", 300), ("Yb171", 1000)]:
            species = get_species(species_name)
            trap = PaulTrap(
                v_rf=v_rf,
                omega_rf=TWO_PI * 30e6,
                r0=0.5e-3,
                omega_axial=TWO_PI * 1e6,
                species=species,
            )
            modes = normal_modes(1, trap)
            eta = lamb_dicke_parameters(modes, species, k, "axial")
            if species_name == "Be9":
                eta_be = eta[0, 0]
            else:
                eta_yb = eta[0, 0]
        assert eta_be > eta_yb


class TestQuditOperatorConsistency:
    """Verify that qudit transition operators reduce to Pauli
    operators for d=2, matching the standard conventions used
    throughout the trapped-ion literature."""

    def test_transition_x_equals_sigma_x(self):
        """For d=2: (|0><1| + |1><0|) = sigma_x."""
        hs = HilbertSpace(n_ions=1, n_modes=1, n_fock=3)
        ops = OperatorFactory(hs)
        tx = ops.transition_x(0, 0, 1)
        sx = ops.sigma_x(0)
        assert (tx - sx).norm() < 1e-12

    def test_transition_z_equals_sigma_z(self):
        """For d=2: (|0><0| - |1><1|) = sigma_z."""
        hs = HilbertSpace(n_ions=1, n_modes=1, n_fock=3)
        ops = OperatorFactory(hs)
        tz = ops.transition_z(0, 0, 1)
        sz = ops.sigma_z(0)
        assert (tz - sz).norm() < 1e-12

    def test_qutrit_transition_x_is_hermitian(self):
        """Transition operators on a qutrit must be Hermitian for
        use in gate Hamiltonians [Ringbauer2022]."""
        hs = HilbertSpace(n_ions=1, n_modes=1, n_fock=3, ion_dims=3)
        ops = OperatorFactory(hs)
        for i, j in [(0, 1), (0, 2), (1, 2)]:
            assert ops.transition_x(0, i, j).isherm
            assert ops.transition_y(0, i, j).isherm
            assert ops.transition_z(0, i, j).isherm

    def test_qutrit_projectors_sum_to_identity(self):
        """Sum of all projectors equals the identity on the ion
        subspace."""
        hs = HilbertSpace(n_ions=1, n_modes=1, n_fock=3, ion_dims=3)
        ops = OperatorFactory(hs)
        total = ops.projector(0, 0) + ops.projector(0, 1) + ops.projector(0, 2)
        assert (total - ops.identity()).norm() < 1e-12

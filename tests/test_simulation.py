import numpy as np
import pytest
import qutip

from tiqs.constants import TWO_PI
from tiqs.simulation.config import SimulationConfig
from tiqs.simulation.runner import SimulationRunner
from tiqs.species.ion import get_species
from tiqs.trap import PaulTrap


@pytest.fixture
def ca40_config():
    trap = PaulTrap(
        v_rf=300.0,
        omega_rf=TWO_PI * 30e6,
        r0=0.5e-3,
        omega_axial=TWO_PI * 1.0e6,
        species=get_species("Ca40"),
    )
    return SimulationConfig(
        species=get_species("Ca40"),
        trap=trap,
        n_ions=2,
        n_modes=1,
        n_fock=15,
        solver="sesolve",
    )


class TestSimulationConfig:
    def test_create_config(self, ca40_config):
        assert ca40_config.n_ions == 2
        assert ca40_config.solver == "sesolve"

    def test_config_with_noise(self):
        trap = PaulTrap(
            v_rf=300.0,
            omega_rf=TWO_PI * 30e6,
            r0=0.5e-3,
            omega_axial=TWO_PI * 1e6,
            species=get_species("Ca40"),
        )
        config = SimulationConfig(
            species=get_species("Ca40"),
            trap=trap,
            n_ions=1,
            n_modes=1,
            n_fock=10,
            solver="mesolve",
            heating_rate=10.0,
            t2_qubit=1.0,
        )
        assert config.solver == "mesolve"
        assert config.heating_rate == 10.0


class TestSimulationRunner:
    def test_single_qubit_rabi(self, ca40_config):
        runner = SimulationRunner(ca40_config)
        result = runner.run_carrier_pulse(ion=0, theta=np.pi, duration=None)
        sz = runner.ops.sigma_z(0)
        final_sz = qutip.expect(sz, result.states[-1])
        # Pi-pulse flips |0> (sz=+1) to |1> (sz=-1)
        assert final_sz == pytest.approx(-1.0, abs=0.1)

    def test_ms_gate_entangles(self, ca40_config):
        runner = SimulationRunner(ca40_config)
        result = runner.run_ms_gate(ions=[0, 1], mode=0)
        rho_spin = qutip.ket2dm(result.states[-1]).ptrace([0, 1])
        rho_single = rho_spin.ptrace(0)
        purity = (rho_single * rho_single).tr().real
        assert purity < 0.9  # entangled: reduced state is mixed

    def test_runner_with_noise(self):
        trap = PaulTrap(
            v_rf=300.0,
            omega_rf=TWO_PI * 30e6,
            r0=0.5e-3,
            omega_axial=TWO_PI * 1e6,
            species=get_species("Ca40"),
        )
        config = SimulationConfig(
            species=get_species("Ca40"),
            trap=trap,
            n_ions=1,
            n_modes=1,
            n_fock=10,
            solver="mesolve",
            heating_rate=1e4,
            t2_qubit=1e-3,
        )
        runner = SimulationRunner(config)
        result = runner.run_carrier_pulse(ion=0, theta=np.pi)
        # With noise, fidelity should be less than perfect
        sz = runner.ops.sigma_z(0)
        final_sz = qutip.expect(sz, result.states[-1])
        assert final_sz < 1.0  # imperfect due to noise


class TestAnharmonicSimulation:
    def test_config_accepts_potentials(self):
        """SimulationConfig can be created with a potentials dict."""
        from tiqs.potential import DuffingPotential

        trap = PaulTrap(
            v_rf=300.0,
            omega_rf=TWO_PI * 30e6,
            r0=0.5e-3,
            omega_axial=TWO_PI * 1.0e6,
            species=get_species("Ca40"),
        )
        pot = DuffingPotential(
            omega=TWO_PI * 1e6,
            anharmonicity=-TWO_PI * 50e3,
        )
        config = SimulationConfig(
            species=get_species("Ca40"),
            trap=trap,
            n_ions=1,
            n_modes=1,
            n_fock=15,
            solver="sesolve",
            potentials={0: pot},
        )
        assert 0 in config.potentials

    def test_runner_with_duffing_potential(self):
        """Carrier pulse with anharmonic mode still produces Rabi
        oscillations (anharmonic correction is on the motion,
        not the spin)."""
        from tiqs.potential import DuffingPotential

        trap = PaulTrap(
            v_rf=300.0,
            omega_rf=TWO_PI * 30e6,
            r0=0.5e-3,
            omega_axial=TWO_PI * 1.0e6,
            species=get_species("Ca40"),
        )
        pot = DuffingPotential(
            omega=TWO_PI * 1e6,
            anharmonicity=-TWO_PI * 50e3,
        )
        config = SimulationConfig(
            species=get_species("Ca40"),
            trap=trap,
            n_ions=1,
            n_modes=1,
            n_fock=15,
            solver="sesolve",
            potentials={0: pot},
        )
        runner = SimulationRunner(config)
        result = runner.run_carrier_pulse(ion=0, theta=np.pi)
        sz = runner.ops.sigma_z(0)
        final_sz = qutip.expect(sz, result.states[-1])
        assert final_sz == pytest.approx(-1.0, abs=0.15)

    def test_no_potentials_backward_compatible(self):
        """Simulations without potentials produce identical results."""
        trap = PaulTrap(
            v_rf=300.0,
            omega_rf=TWO_PI * 30e6,
            r0=0.5e-3,
            omega_axial=TWO_PI * 1.0e6,
            species=get_species("Ca40"),
        )
        config = SimulationConfig(
            species=get_species("Ca40"),
            trap=trap,
            n_ions=1,
            n_modes=1,
            n_fock=15,
            solver="sesolve",
        )
        assert config.potentials == {}
        runner = SimulationRunner(config)
        result = runner.run_carrier_pulse(ion=0, theta=np.pi)
        sz = runner.ops.sigma_z(0)
        final_sz = qutip.expect(sz, result.states[-1])
        assert final_sz == pytest.approx(-1.0, abs=0.1)

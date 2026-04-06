# TIQS - Trapped Ion Quantum Simulator

Simulate trapped-ion quantum computers from first principles. TIQS builds time-dependent Hamiltonians and Lindblad noise models from physical parameters - trap voltages, laser frequencies, ion species data - and solves them with [QuTiP](https://qutip.org). No black-box gate models: everything derives from the underlying atomic physics.

## Quick start

Prepare a Bell state on two Ca-40 ions and measure it:

```python
import numpy as np
from tiqs import get_species, PaulTrap, normal_modes, lamb_dicke_parameters
from tiqs import HilbertSpace, OperatorFactory, StateFactory, SimulationRunner, SimulationConfig

TWO_PI = 2 * np.pi

# Physical setup: Ca-40 ions in a linear Paul trap
species = get_species("Ca40")
trap = PaulTrap(v_rf=300, omega_rf=TWO_PI * 30e6, r0=0.5e-3,
                omega_axial=TWO_PI * 1e6, species=species)

# Simulate an MS entangling gate through the full pipeline
config = SimulationConfig(species=species, trap=trap, n_ions=2, n_modes=1, n_fock=15)
runner = SimulationRunner(config)
result = runner.run_ms_gate(ions=[0, 1])

# Verify Bell state
from tiqs.analysis.fidelity import bell_state_fidelity
fid = bell_state_fidelity(result.states[-1].ptrace([0, 1]))
print(f"Bell state fidelity: {fid:.4f}")  # 1.0000
```

Or build the Hamiltonian yourself for full control:

```python
from tiqs.gates.molmer_sorensen import ms_gate_hamiltonian, ms_gate_duration
import qutip

hs = HilbertSpace(n_ions=2, n_modes=1, n_fock=20)
ops = OperatorFactory(hs)
sf = StateFactory(hs)

# MS gate parameters
eta = 0.05                          # Lamb-Dicke parameter
delta = TWO_PI * 15e3               # sideband detuning
Omega = delta / (4 * eta)           # maximally entangling condition
tau = ms_gate_duration(delta)       # gate time = 2*pi/delta

# Construct the time-dependent Hamiltonian and solve
H = ms_gate_hamiltonian(ops, ions=[0, 1], mode=0, eta=[eta, eta],
                        rabi_frequency=Omega, detuning=delta)
result = qutip.sesolve(H, sf.ground_state(), np.linspace(0, tau, 500))
```

## What you can simulate

TIQS models every layer of a trapped-ion quantum computer:

| Layer | What it computes |
|-------|-----------------|
| **Ion species** | Atomic data for Yb-171, Ca-40, Ca-43, Ba-137, Be-9, Sr-88: mass, qubit splitting, transition wavelengths, linewidths, branching ratios |
| **Trap** | Paul trap Mathieu stability, secular frequencies, pseudopotential depth, micromotion |
| **Coulomb crystal** | N-ion equilibrium positions, axial + radial normal mode frequencies and participation vectors via Hessian eigendecomposition |
| **Laser-ion coupling** | Carrier, red/blue sideband Hamiltonians with configurable Lamb-Dicke order (1st or 2nd, including Debye-Waller corrections) |
| **Entangling gates** | Molmer-Sorensen (bichromatic sigma_x), light-shift (sigma_z), Cirac-Zoller (sequential sideband) |
| **Single-qubit gates** | Rx, Ry, Rz rotations with SK1 and BB1 composite pulse sequences |
| **Cooling** | Doppler limit, resolved sideband cooling (analytical + simulated), EIT cooling |
| **Decoherence** | Motional heating (d^-4 scaling, 1/f noise), motional dephasing, qubit T1/T2, off-resonant photon scattering (Raman + Rayleigh), laser phase/intensity noise, addressing crosstalk |
| **SPAM** | Optical pumping initialization, fluorescence detection with Poisson photon counting, joint-distribution measurement sampling, mid-circuit measurement |
| **Transport** | QCCD shuttling and crystal splitting with motional excitation models |
| **Analysis** | State/gate/Bell fidelity, Wigner functions, phase-space trajectories, error budgets |

## Installation

Requires Python 3.14+.

```bash
pip install -e ".[dev]"
```

Dependencies: [QuTiP](https://qutip.org) >= 5.2.3, NumPy, SciPy.

## Adding noise

Every noise source is a Lindblad collapse operator or Hamiltonian perturbation that plugs directly into QuTiP's master equation solver:

```python
from tiqs.noise.motional import motional_heating_ops
from tiqs.noise.qubit import qubit_dephasing_op
from tiqs.noise.photon_scattering import rayleigh_scattering_op

c_ops = (
    motional_heating_ops(ops, mode=0, heating_rate=1e4)           # 10^4 quanta/s
    + [qubit_dephasing_op(ops, ion=0, t2=1e-3)]                  # T2 = 1 ms
    + [rayleigh_scattering_op(ops, ion=0, rate=500)]              # elastic scattering
)

result = qutip.mesolve(H, initial_state, tlist, c_ops=c_ops)
```

## Architecture

```
src/tiqs/
    species/       Ion species database (6 species with full atomic data)
    trap/          Paul trap physics (Mathieu equation, secular frequencies)
    chain/         Coulomb crystal equilibrium, normal modes, Lamb-Dicke parameters
    hilbert_space/ Composite tensor-product space, operator/state factories
    interaction/   Laser-ion Hamiltonians (carrier, sidebands, Raman transitions)
    gates/         Single-qubit (Rx/Ry/Rz, SK1, BB1) and entangling (MS, ZZ, CZ)
    cooling/       Doppler, resolved sideband, EIT cooling
    noise/         All decoherence channels as Lindblad operators
    spam/          State preparation (optical pumping) and measurement (fluorescence)
    transport/     QCCD shuttling and crystal splitting
    simulation/    SimulationRunner orchestrating the full pipeline
    analysis/      Fidelity metrics, Wigner functions, error budgets
```

## Testing

```bash
pytest tests/ -v
```

178 tests, 99% line coverage. Includes 13 analytical exactness checks that verify every formula against known results (Rabi frequencies, normal mode ratios, decoherence rates, Bell state fidelities) with tight tolerances.

## Supported ion species

| Species | Qubit type | Splitting | Cooling | Raman |
|---------|-----------|-----------|---------|-------|
| Yb-171 | Hyperfine | 12.643 GHz | 369.5 nm (19.6 MHz) | 355 nm |
| Ca-40 | Optical | 729 nm | 397 nm (22.4 MHz) | - |
| Ca-43 | Hyperfine | 3.226 GHz | 397 nm (22.4 MHz) | 397 nm |
| Ba-137 | Hyperfine | 8.038 GHz | 493 nm (20.3 MHz) | 515 nm |
| Be-9 | Hyperfine | 1.25 GHz | 313 nm (19.4 MHz) | 313 nm |
| Sr-88 | Optical | 674 nm | 422 nm (21.5 MHz) | - |

## References

The physics implemented follows:

- Leibfried, Blatt, Monroe, Wineland. "Quantum dynamics of single trapped ions." Rev. Mod. Phys. 75, 281 (2003)
- Molmer, Sorensen. "Multiparticle entanglement of hot trapped ions." Phys. Rev. Lett. 82, 1835 (1999)
- Wineland, Monroe, Itano, Leibfried, King, Meekhof. "Experimental issues in coherent quantum-state manipulation." J. Res. NIST 103, 259 (1998)
- Brownnutt, Kumph, Rabl, Blatt. "Ion-trap measurements of electric-field noise near surfaces." Rev. Mod. Phys. 87, 1419 (2015)

## License

BSD-3-Clause

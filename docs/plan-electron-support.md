## Plan: End-to-End Electron Trap Simulation

### What's done (this branch)

- `constants.py`: Added `ELECTRON_MASS`, `BOHR_MAGNETON`, `ELECTRON_G_FACTOR`.
- `species/electron.py`: `ElectronSpecies` frozen dataclass. One
  parameter (`magnetic_field`), two derived properties (`mass_kg`,
  `qubit_frequency_hz`). `data.py` is unchanged.
- Tests: 3 electron species tests, 3 constant value tests.
  Full suite passes (184/184).

### What remains

#### Step 1: Widen `PaulTrap.species` type

`PaulTrap` uses only `species.mass_kg`. Change the type annotation
from `IonSpecies` to `IonSpecies | ElectronSpecies`. Same for
`SimulationConfig.species`. No logic changes needed.

#### Step 2: End-to-end test

A test that simulates 2 trapped electrons using the existing
low-level APIs:

1. `PaulTrap` with GHz RF drive and `ElectronSpecies(0.1)`
2. `equilibrium_positions` and `normal_modes` (Coulomb physics is
   mass-agnostic -- just plug in electron mass)
3. Compute Lamb-Dicke eta from magnetic gradient coupling:
   `eta = g_e * mu_B * (dB/dz) * b * sqrt(hbar / (2*m*omega))
   / (hbar * omega_qubit)`
4. `HilbertSpace`, `OperatorFactory`, `StateFactory`
5. `ms_gate_hamiltonian` with the computed eta
6. `qutip.sesolve`, `bell_state_fidelity`

No new modules needed. The Hamiltonian layer is already
species-agnostic -- it takes operators and scalar floats.

### What does NOT need new code

The interaction, gates, noise, SPAM, analysis, and transport
modules are spin-operator algebra, not species-specific. They
work for electrons today with no changes. Resistive cooling
(nbar = k_B T / hbar omega) and gradient Lamb-Dicke parameters
are one-line formulas that belong in the test, not in dedicated
modules, until there is real usage to justify the abstraction.

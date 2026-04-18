# TIQS Examples

Example scripts demonstrating TIQS applied to trapped-electron and hybrid electron-ion quantum systems, based on Osada et al. *Phys. Rev. Research* **4**, 033245 (2022).

## Scripts

### `osada_reproduction.py` - Reproducing the Osada paper

Reproduces key results from Osada et al. (2022):

- **Table II** - Coulomb coupling parameters (g0, alpha_C) for the electron-ion hybrid system at various separations and frequencies
- **Eq. 9** - Optomechanical Hamiltonian dynamics of the electron-ion system
- **Sec. III.C** - Sympathetic cooling estimates showing ground-state cooling is feasible
- **Sec. II.C.2** - Dispersive readout coupling zeta for the electron-cavity-transmon system

### `penning_motional_qubit.py` - Motional-state qubit in a Penning trap

Simulates a motional-state qubit (Fock states |0> and |1> of the axial mode) in an anharmonic electron Penning trap:

- Anharmonic (Duffing/Kerr) energy spectrum vs. anharmonicity
- Single-qubit pi-pulse gate fidelity and leakage to |2>
- Gate fidelity vs. anharmonicity (the design tradeoff)
- Decoherence from motional heating and voltage-noise dephasing
- Dispersive readout via a coupled superconducting resonator

### `anharmonic_cooling.py` - Cooling bottleneck from anharmonicity

Simulates the full Lindblad master equation for an anharmonic electron mode sympathetically cooled via a Coulomb-coupled trapped ion. The Kerr anharmonicity detunes higher Fock-state transitions from the beam-splitter cooling drive, creating a population bottleneck that limits the achievable ground-state occupation.

- Coulomb coupling parameters at the Osada separation (10 um)
- Cooling dynamics: harmonic vs. anharmonic electron
- Cooling floor vs. anharmonicity (up to 128x worse than harmonic)
- Fock-state populations revealing the bottleneck mechanism

This is the most computationally non-trivial example: the two-mode open quantum system (Kerr oscillator + damped harmonic mode + beam-splitter coupling + heating) has no closed-form steady state.

## Running

```bash
uv run python examples/osada_reproduction.py
uv run python examples/penning_motional_qubit.py
uv run python examples/anharmonic_cooling.py
```

Each script runs in under 2 minutes and prints results to the terminal with `[check]` assertions verifying correctness.

# Sympathetic Cooling of an Anharmonic Electron Mode

Simulates the Osada et al. (2022) electron-ion hybrid cooling scheme with full quantum dynamics, going beyond the harmonic rate-equation approximation.

An electron with Kerr anharmonicity is coupled to a laser-cooled Be9+ ion via a beam-splitter interaction. The anharmonicity detunes higher Fock-state transitions from the cooling drive, creating a population bottleneck that limits the achievable ground-state occupation.

## What it computes

1. **Coulomb couplings** at the Osada separation (10 um): optomechanical g0, beam-splitter g_bs, and Coulomb self-Kerr alpha_C
2. **Cooling dynamics**: harmonic electron cools to nbar = 0.008 in 500 us; anharmonic (-30 kHz) only reaches 0.68
3. **Cooling floor vs. anharmonicity**: from 1.4x the harmonic prediction at alpha = 5 kHz to 128x at 50 kHz
4. **Fock-state populations**: |0> has 82% population but |2> through |9> each have 1-4%, revealing the bottleneck

## Why it matters

The two-mode open quantum system (Kerr oscillator + damped harmonic mode + beam-splitter + heating) has no closed-form steady state. The harmonic rate equation predicts nbar = 0.006 regardless of anharmonicity. The full Lindblad simulation shows the actual cooling floor is 100x worse at alpha = 50 kHz.

## Running

```bash
uv run python examples/anharmonic_cooling.py
```

Runs in about 60 seconds.

## Reference

Osada, A. et al. "Feasibility study on ground-state cooling and single-phonon readout of trapped electrons using hybrid quantum systems." *Phys. Rev. Research* **4**, 033245 (2022).

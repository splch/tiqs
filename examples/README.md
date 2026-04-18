# Sympathetic Cooling of an Anharmonic Electron Mode

Does the Kerr anharmonicity needed for a motional qubit prevent ground-state cooling? This script answers that question by simulating the full Lindblad master equation for the Osada et al. (2022) electron-ion hybrid cooling scheme.

## Running

```bash
uv run python examples/anharmonic_cooling.py
```

Takes about 3 seconds.

## Output

### 1. Coulomb couplings

```
g0/(2pi)      = 40.3 kHz (optomechanical)
g_bs/(2pi)    = 1251 kHz (beam-splitter)
alpha_C/(2pi) = 5.5 kHz (Coulomb Kerr)
g_eff/(2pi)   = 3 kHz (drive-activated)
gamma_ion/(2pi) = 10 kHz
g_eff/gamma   = 0.30 (weak coupling)
```

Computed from the Coulomb Taylor expansion at 10 um electron-ion separation. The weak-coupling condition (g_eff < gamma_ion) ensures the rate-equation comparison in section 3 is fair.

### 2. Cooling dynamics

```
                          harmonic    a=-30kHz
------------------------------------------------
  t=0:       0.990      0.990
  t=250us:   0.009      0.710
  t=500us:  0.008      0.683
```

The harmonic electron cools to near the ground state in 250 us. The anharmonic electron (-30 kHz Kerr) stalls at nbar = 0.68 -- 85x worse.

### 3. Cooling floor vs. anharmonicity

```
 alpha/(2pi)    nbar_sim     ratio
------------------------------------
        0 kHz      0.0085     1.4x
        5 kHz      0.0394     6.4x
       10 kHz      0.2186    35.3x
       20 kHz      0.5021    81.1x
       30 kHz      0.6585   106.4x
       50 kHz      0.7951   128.5x
```

The harmonic rate equation predicts nbar = 0.006 regardless of alpha. The full quantum simulation reveals a sharp cliff: at alpha = 10 kHz the cooling floor jumps to 0.22 (35x worse), and by 50 kHz it's 0.80 (128x worse). This cliff is the central design constraint -- the lab must keep alpha below ~5 kHz to reach the ground state with these coupling parameters.

### 4. Fock-state populations (alpha = -30 kHz)

```
  |0>: 0.8248  ################################
  |1>: 0.0102
  |2>: 0.0375  #
  |3>: 0.0447  #
  |4>: 0.0331  #
  |5>: 0.0210
```

This reveals the bottleneck mechanism. The cooling drive empties |1> efficiently (only 1% population) because it's resonant with the |1> -> |0> transition. But |2> through |5> each hold 2-4% of the population because their transitions are detuned by -30, -60, -90, -120 kHz from the drive. Population piles up in these detuned levels like cars behind a closed lane.

## Why this needs simulation

The two-mode system (Kerr oscillator + damped harmonic mode + beam-splitter coupling + heating) has no closed-form steady state. The harmonic rate equation misses the bottleneck entirely because it treats all Fock levels identically. The Fock-state-dependent detuning from the Kerr term creates coupled population dynamics across the entire ladder that only the full Lindblad master equation captures.

## Reference

Osada, A. et al. *Phys. Rev. Research* **4**, 033245 (2022). [doi:10.1103/PhysRevResearch.4.033245](https://doi.org/10.1103/PhysRevResearch.4.033245)

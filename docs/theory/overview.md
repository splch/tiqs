## Trapped-Ion and Trapped-Electron Quantum Computing

Trapped-particle quantum computers encode qubits in the internal electronic
states of individual atomic ions -- or the spin states of trapped electrons --
confined by electromagnetic traps under ultra-high vacuum, and manipulated
with precisely controlled laser, microwave, or magnetic-field-gradient drives.
The shared quantized vibrational motion of the particle chain serves as a quantum
bus that mediates entangling interactions between any pair of qubits.

Two trap architectures are supported. **Paul traps** confine charged particles
with oscillating RF electric fields and are the dominant platform for
trapped-ion quantum computing. **Penning traps** use a static magnetic field
for radial confinement and are particularly well-suited for trapping bare
electrons, where the absence of micromotion and the availability of strong
magnetic confinement for light particles enable fast gate operations. See
[trapping.md](trapping.md) for the detailed physics of both architectures.

As of early 2026, trapped-ion systems hold records for the highest gate
fidelities of any qubit platform: single-qubit gate errors as low as
$1.5 \times 10^{-7}$ and two-qubit gate errors of $8.4 \times 10^{-5}$.
The two records come from different groups and different qubit encodings;
see the record-fidelity table in [gates.md](gates.md) for the sources and
the caveats.

### The Physics Stack

TIQS models the full trapped-particle physics stack from the ground up:

| Layer | Physics | TIQS Package | Theory |
|-------|---------|--------------|--------|
| **Trapping** | Paul and Penning trap confinement, Mathieu equation, pseudopotential | `tiqs.trap` | [trapping.md](trapping.md) |
| **Ion chain** | Coulomb crystals, normal modes, Lamb-Dicke parameters | `tiqs.chain` | [normal_modes.md](normal_modes.md) |
| **Species** | Atomic structure, transitions, qubit encoding (ions and electrons) | `tiqs.species` | [species.md](species.md) |
| **Potentials** | Harmonic, Duffing (Kerr), and arbitrary anharmonic motional potentials | `tiqs.potential` | [potentials.md](potentials.md) |
| **Cooling** | Doppler, resolved sideband, and EIT cooling | `tiqs.cooling` | [cooling.md](cooling.md) |
| **Laser-ion** | Carrier and sideband Hamiltonians, Raman transitions | `tiqs.interaction` | [laser_ion_interaction.md](laser_ion_interaction.md) |
| **Gates** | Single-qubit rotations, MS, CZ, light-shift | `tiqs.gates` | [gates.md](gates.md) |
| **Noise** | Motional heating, qubit dephasing, photon scattering | `tiqs.noise` | [noise.md](noise.md) |
| **SPAM** | Optical pumping, fluorescence detection | `tiqs.spam` | [spam.md](spam.md) |
| **Transport** | QCCD shuttling, separation, merging | `tiqs.transport` | [transport.md](transport.md) |
| **Analysis** | Fidelity metrics, phase-space visualization, error budgets | `tiqs.analysis` | -- |

Each theory page explains the physics from first principles and shows the
corresponding API. You can read them in order (top to bottom) for a
textbook-style progression, or jump directly to a topic you need. The pages
mix two kinds of material: equations TIQS actually implements, and
background formulas provided for reference. Anything in the second category
is labelled *reference only* at the top of its section, and each page closes
with a **Model scope and approximations** list naming what its
implementations leave out. Read those lists before trusting a simulated
number: TIQS's gate Hamiltonians are idealised, so the fidelities it
reports are upper bounds.

### Quick Start

Two ions in a Paul trap -- compute normal modes and set up the Hilbert space
in under ten lines:

```python
import numpy as np
import tiqs

# Define species and trap
species = tiqs.get_species("Ca40")
trap = tiqs.PaulTrap(
    v_rf=200.0,
    omega_rf=2 * np.pi * 30e6,
    r0=400e-6,
    species=species,
    omega_axial=2 * np.pi * 1e6,
)
trap.mathieu_q  # 0.170 -- inside the 0.1-0.4 operating band

# Compute normal modes of the two-ion crystal
modes = tiqs.normal_modes(n_ions=2, trap=trap)

# Axial center-of-mass and stretch mode frequencies (rad/s)
modes.modes["axial"].freqs  # array([6.28e6, 1.09e7])

# Mode eigenvectors: columns are participation vectors
modes.modes["axial"].vectors  # [[-0.707, -0.707], [-0.707, 0.707]]

# Build the composite Hilbert space (2 qubits x 2 modes x 10 Fock states)
hs = tiqs.HilbertSpace(n_ions=2, n_modes=2, n_fock=10)
ops = tiqs.OperatorFactory(hs)
hs.dims  # [2, 2, 10, 10] -- total dimension 400
```

The Mathieu $q$ above matters: the pseudopotential approximation TIQS uses
for radial confinement loses accuracy above $q \approx 0.4$, and
`normal_modes` emits a `UserWarning` when any ion exceeds it. The
parameters above sit in the $q = 0.1$-$0.4$ band that real experiments use
(see [trapping.md](trapping.md)).

TIQS also supports Penning traps and trapped electrons. See
[species.md](species.md) for `ElectronSpecies` and
[trapping.md](trapping.md) for the `PenningTrap` class.

### How Simulation Works

The Quick Start above constructs the static building blocks: trap parameters,
normal modes, and a Hilbert space. To simulate dynamics, TIQS assembles the
full system Hamiltonian in the composite space
$\mathcal{H} = \mathcal{H}_\text{qubit}^{\otimes N} \otimes \mathcal{H}_\text{motion}^{\otimes M}$
and integrates it with QuTiP. `SimulationConfig.solver` selects
`"sesolve"` (unitary, the default and the only correct choice when there
are no collapse operators), `"mesolve"` (the Lindblad master equation), or
`"mcsolve"` (Monte Carlo trajectories of the same master equation). The
Lindblad form is:

$$
\frac{d\rho}{dt} = -i[H(t), \rho] + \sum_k \gamma_k \left( L_k \rho L_k^\dagger - \frac{1}{2}\lbrace L_k^\dagger L_k, \rho\rbrace \right)
$$

The Hamiltonian $H(t)$ includes qubit energies, motional mode energies, and
time-dependent laser or microwave drives (see
[laser_ion_interaction.md](laser_ion_interaction.md) and [gates.md](gates.md)).
The Lindblad operators $L_k$ model motional heating, qubit dephasing,
spontaneous emission, and photon scattering (see [noise.md](noise.md) for
each decoherence channel and its collapse operator).

### Model Scope and Approximations

Every theory page ends with its own scope list; this is the cross-cutting
summary. TIQS is a *lowest-level* simulator: it builds exact Hamiltonians
and Lindblad operators for a chosen idealisation and integrates them
faithfully. What it does **not** do is model the effects that separate a
clean idealisation from a real machine, so simulated gate fidelities are
upper bounds, not predictions.

Deliberately out of scope, repo-wide:

- **Pulse shaping and multi-mode gate solvers.** Every gate is a square
  pulse on a single motional mode. There is no amplitude, frequency, or
  phase modulation, and nothing solves the simultaneous multi-mode
  phase-space closure conditions, so the robust-gate techniques real
  hardware uses cannot be represented (see [gates.md](gates.md)).
- **Third levels.** Ions are strictly two-level (`dims = [2] * n_ions +
  fock_dims`). No shelving, no auxiliary level, no leakage out of the qubit
  manifold. This is why `cirac_zoller_gate` is not an entangling gate.
- **Micromotion in gates.** The RF drive sets trap frequencies through the
  pseudopotential and nothing else; no gate or interaction Hamiltonian
  depends on $\Omega_\text{RF}$.
- **Magnetic-field physics.** Zeeman structure, field-gradient qubit
  addressing, and field-noise-induced dephasing beyond the scalar $T_2$ are
  not modelled. No species carries a magnetic gradient, so for a particle
  with no optical or Raman transition (including `ElectronSpecies`)
  `SimulationRunner` raises rather than guessing a wavevector: supply
  `SimulationConfig.k_eff` computed from
  $k_\text{eff} = g\,\mu_B\,(\partial B/\partial z)/(\hbar\,\omega_m)$, or
  use `tiqs.chain.lamb_dicke.gradient_lamb_dicke_parameters` for the
  per-mode $\eta_{j,p}$ directly.
- **Mode-mode coupling.** Motional modes are independent harmonic (or
  single-mode anharmonic) oscillators; there is no cross-Kerr coupling
  between modes and no participation-weighted per-mode heating.
- **Trap electrode geometry.** Traps are parameterised
  ($V_\text{RF}$, $r_0$, $\kappa$, $z_0$), not solved from a surface-trap
  electrode layout.
- **Collective noise.** Dephasing is independent per ion; there is no
  correlated/collective dephasing channel.
- **Unwired channels.** `crosstalk_hamiltonian` and
  `laser_intensity_noise_op` are Hamiltonian terms rather than collapse
  operators, so `SimulationRunner` does not apply them; they must be used
  directly. `SimulationRunner` likewise has no configurable Lamb-Dicke
  order, and second-order physics is reachable only through
  `tiqs.interaction.hamiltonian.full_interaction_hamiltonian`.

### References

1. Leibfried, D. et al. "Quantum dynamics of single trapped ions."
   *Rev. Mod. Phys.* **75**, 281 (2003).
2. Bruzewicz, C.D. et al. "Trapped-ion quantum computing: Progress and
   challenges." *Appl. Phys. Rev.* **6**, 021314 (2019).
3. Cirac, J.I. & Zoller, P. "Quantum computations with cold trapped ions."
   *Phys. Rev. Lett.* **74**, 4091 (1995).
4. Jain, S. et al. "Penning micro-trap for quantum computing."
   *Nature* **627**, 510 (2024).
5. Ciaramicoli, G., Marzoli, I. & Tombesi, P. "Scalable quantum processor
   with trapped electrons." *Phys. Rev. Lett.* **91**, 017901 (2003).

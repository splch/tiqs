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

Each theory page explains the physics from first principles, defines the
equations TIQS implements, and shows the corresponding API. You can read
them in order (top to bottom) for a textbook-style progression, or jump
directly to a topic you need.

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
    r0=200e-6,
    species=species,
    omega_axial=2 * np.pi * 1e6,
)

# Compute normal modes of the two-ion crystal
modes = tiqs.normal_modes(n_ions=2, trap=trap)

# Axial center-of-mass and stretch mode frequencies (rad/s)
modes.modes["axial"].freqs   # array([6.28e6, 1.09e7])

# Mode eigenvectors: columns are participation vectors
modes.modes["axial"].vectors  # [[-0.707, -0.707], [-0.707, 0.707]]

# Build the composite Hilbert space (2 qubits x 2 modes x 10 Fock states)
hs = tiqs.HilbertSpace(n_ions=2, n_modes=2, n_fock=10)
ops = tiqs.OperatorFactory(hs)
hs.dims  # [2, 2, 10, 10] -- total dimension 400
```

TIQS also supports Penning traps and trapped electrons. See
[species.md](species.md) for `ElectronSpecies` and
[trapping.md](trapping.md) for the `PenningTrap` class.

### How Simulation Works

The Quick Start above constructs the static building blocks: trap parameters,
normal modes, and a Hilbert space. To simulate dynamics, TIQS assembles the
full system Hamiltonian in the composite space
$\mathcal{H} = \mathcal{H}_\text{qubit}^{\otimes N} \otimes \mathcal{H}_\text{motion}^{\otimes M}$
and integrates the Lindblad master equation using QuTiP:

$$
\frac{d\rho}{dt} = -i[H(t), \rho] + \sum_k \gamma_k \left( L_k \rho L_k^\dagger - \frac{1}{2}\lbrace L_k^\dagger L_k, \rho\rbrace \right)
$$

The Hamiltonian $H(t)$ includes qubit energies, motional mode energies, and
time-dependent laser or microwave drives (see
[laser_ion_interaction.md](laser_ion_interaction.md) and [gates.md](gates.md)).
The Lindblad operators $L_k$ model motional heating, qubit dephasing,
spontaneous emission, and photon scattering (see [noise.md](noise.md) for
each decoherence channel and its collapse operator).

### References

1. Leibfried, D. et al. "Quantum dynamics of single trapped ions."
   *Rev. Mod. Phys.* **75**, 281 (2003).
2. Bruzewicz, C.D. et al. "Trapped-ion quantum computing: Progress and
   challenges." *Appl. Phys. Rev.* **6**, 021314 (2019).
3. Cirac, J.I. & Zoller, P. "Quantum computations with cold trapped ions."
   *Phys. Rev. Lett.* **74**, 4091 (1995).
4. Rodriguez-Blanco, A. et al. "Penning micro-trap for quantum computing."
   *Nature* **627**, 510 (2024).
5. Ciaramicoli, G., Marzoli, I. & Tombesi, P. "Scalable quantum processor
   with trapped electrons." *Phys. Rev. Lett.* **91**, 017901 (2003).

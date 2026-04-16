## Trapped-Ion and Trapped-Electron Quantum Computing

Trapped-particle quantum computers encode qubits in the internal electronic
states of individual atomic ions -- or the spin states of trapped electrons --
confined by electromagnetic traps (Paul traps or Penning traps) under
ultra-high vacuum, and manipulated with precisely controlled laser, microwave,
or magnetic-gradient fields. The shared quantized vibrational motion of the
particle chain serves as a quantum bus that mediates entangling interactions
between any pair of qubits.

As of early 2026, trapped-ion systems hold records for the highest gate
fidelities of any qubit platform: single-qubit gate errors as low as
$1.5 \times 10^{-7}$ and two-qubit gate errors of $8.4 \times 10^{-5}$.

### The Physics Stack

TIQS models the full trapped-particle physics stack from the ground up:

| Layer | Physics | TIQS Package |
|-------|---------|--------------|
| **Trapping** | Paul and Penning trap confinement, Mathieu equation, pseudopotential | `tiqs.trap` |
| **Ion chain** | Coulomb crystals, normal modes, Lamb-Dicke parameters | `tiqs.chain` |
| **Species** | Atomic structure, transitions, qubit encoding | `tiqs.species` |
| **Potentials** | Harmonic, Duffing (Kerr), and arbitrary anharmonic motional potentials | `tiqs.potential` |
| **Cooling** | Doppler, resolved sideband, and EIT cooling | `tiqs.cooling` |
| **Laser-ion** | Carrier and sideband Hamiltonians, Raman transitions | `tiqs.interaction` |
| **Gates** | Single-qubit rotations, MS, CZ, light-shift | `tiqs.gates` |
| **Noise** | Motional heating, qubit dephasing, photon scattering | `tiqs.noise` |
| **SPAM** | Optical pumping, fluorescence detection | `tiqs.spam` |
| **Analysis** | Fidelity metrics, phase-space visualization, error budgets | `tiqs.analysis` |

### Quick Start

```python
import tiqs

# Define species and trap
species = tiqs.get_species("Ca40")
trap = tiqs.PaulTrap(
    v_rf=200.0,
    omega_rf=2 * 3.14159 * 30e6,
    r0=200e-6,
    species=species,
    omega_axial=2 * 3.14159 * 1e6,
)

# Compute normal modes
modes = tiqs.normal_modes(n_ions=2, trap=trap)

# Access axial mode frequencies and eigenvectors
modes.modes["axial"].freqs
modes.modes["axial"].vectors

# Build Hilbert space and operators
hs = tiqs.HilbertSpace(n_ions=2, n_modes=2, n_fock=10)
ops = tiqs.OperatorFactory(hs)
```

### How Simulation Works

TIQS constructs the full system Hamiltonian in the composite Hilbert space
$\mathcal{H} = \mathcal{H}_\text{qubit}^{\otimes N} \otimes \mathcal{H}_\text{motion}^{\otimes M}$
and integrates the Lindblad master equation using QuTiP:

$$
\frac{d\rho}{dt} = -i[H(t), \rho] + \sum_k \gamma_k \left( L_k \rho L_k^\dagger - \frac{1}{2}\lbrace L_k^\dagger L_k, \rho\rbrace \right)
$$

The Hamiltonian $H(t)$ includes qubit energies, motional mode energies, and
time-dependent laser drives. The Lindblad operators $L_k$ model motional
heating, qubit dephasing, spontaneous emission, and photon scattering.

### References

1. Leibfried, D. et al. "Quantum dynamics of single trapped ions."
   *Rev. Mod. Phys.* **75**, 281 (2003).
2. Bruzewicz, C.D. et al. "Trapped-ion quantum computing: Progress and
   challenges." *Appl. Phys. Rev.* **6**, 021314 (2019).
3. Cirac, J.I. & Zoller, P. "Quantum computations with cold trapped ions."
   *Phys. Rev. Lett.* **74**, 4091 (1995).

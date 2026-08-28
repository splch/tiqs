r"""Composite Hilbert space: tensor-product construction,
operators, and states.

## Hilbert Space Structure

TIQS simulates trapped-ion systems in the composite Hilbert space

$$
\mathcal{H} = \mathcal{H}_\text{qubit}^{\otimes N}
  \otimes \mathcal{H}_\text{motion}^{\otimes M}
$$

where $N$ is the number of ions (each a two-level qubit with $\dim = 2$) and
$M$ is the number of motional modes (each a truncated harmonic oscillator with
$\dim = n_\text{fock}$, containing Fock states $|0\rangle$ through
$|n_\text{fock}-1\rangle$). The total Hilbert space dimension is
$2^N \times n_\text{fock}^M$.

Operators acting on individual subsystems are embedded into the full space via
tensor products:

$$
\sigma_x^{(j)} = I \otimes \cdots
  \otimes \sigma_x \otimes \cdots \otimes I
$$

where $\sigma_x$ acts on the $j$-th qubit.

## Conventions

- `basis(2, 0)` $= |0\rangle$ is the qubit **ground** state, so QuTiP's
  matrix-raising $\sigma_+ = |0\rangle\langle 1|$ is *de-excitation*
  and $\sigma_- = |1\rangle\langle 0|$ is *excitation*. The excitation
  operator carries $e^{+i\phi}$ in the drive Hamiltonians.
- `OperatorFactory.position` / `momentum` are the unit-commutator
  quadratures $x = (a + a^\dagger)/\sqrt{2}$, $p = i(a^\dagger -
  a)/\sqrt{2}$ with $[x, p] = i$. `tiqs.potential` instead calls
  $q = a + a^\dagger = \sqrt{2}\,x$ the dimensionless position, so
  potentials written in $q$ must not be fed $x$ values.
- Fock cutoffs must be at least `builder.MIN_FOCK_DIM` $= 2$: a
  one-level mode has no ladder operator at all.

## Model scope and approximations

- Each ion is **strictly two-level**. There is no Zeeman substructure,
  no metastable ($D$-state) shelving level, and no leakage level, so
  species data such as `IonSpecies.metastable_lifetime` and
  `qubit_type` describe the intended physical qubit but do not add
  levels here. Shelving-based readout, leakage out of the qubit
  manifold, and repumping dynamics are therefore not simulable;
  `tiqs.spam` projects onto the two qubit states directly.
- Motional modes are truncated harmonic oscillators. Truncation always
  *reduces* the mean occupation of a thermal state, which
  `StateFactory.thermal_state` reports as a warning rather than
  correcting.
"""

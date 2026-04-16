## Motional Potentials

In standard trapped-ion physics, each normal mode is a quantum harmonic
oscillator with equally-spaced energy levels separated by $\hbar\omega$.
This simplification enables the elegant sideband framework that underlies
all cooling and gate protocols.

However, real trapping potentials can deviate from perfect harmonicity.
Higher-order terms in the electrode potential, intentional anharmonic
traps, or effective nonlinearities from strong drives produce
**anharmonic** mode Hamiltonians where the energy level spacing is no
longer uniform.

TIQS provides three potential models with a shared ``Potential`` protocol.

### The Potential Protocol

Any class exposing ``omega`` and ``single_mode_hamiltonian(n_fock)``
satisfies the protocol:

```python
class Potential(Protocol):
    @property
    def omega(self) -> float: ...

    def single_mode_hamiltonian(self, n_fock: int) -> qutip.Qobj: ...
```

This allows user-defined potentials to integrate seamlessly with TIQS
utility functions like ``energy_levels()``, ``transition_frequencies()``,
and ``mode_hamiltonian()``.

### Harmonic Potential

The default model. The single-mode Hamiltonian in natural units
($\hbar = 1$) is:

$$
H = \omega\,a^\dagger a
$$

Energy eigenvalues are $E_n = n\omega$ and all transition frequencies
equal $\omega$. This is what TIQS uses implicitly when no potential is
explicitly specified.

```python
import tiqs

pot = tiqs.HarmonicPotential(omega=2 * 3.14159 * 1e6)
H = pot.single_mode_hamiltonian(n_fock=20)
```

### Duffing (Kerr) Potential

A harmonic oscillator with a quartic nonlinearity, producing the
**Duffing** (also called **Kerr**) Hamiltonian:

$$
H = \omega\,\hat{n} + \frac{\alpha}{2}\,\hat{n}\,(\hat{n} - 1)
$$

where $\alpha$ is the **anharmonicity** parameter. The transition
frequency from $|n\rangle$ to $|n+1\rangle$ shifts linearly with $n$:

$$
\omega_{n \to n+1} = \omega + \alpha\,n
$$

- $\alpha < 0$: subharmonic (higher levels are closer together), as in
  transmon qubits or softening mechanical oscillators.
- $\alpha > 0$: superharmonic (higher levels are farther apart), as in
  stiffening potentials.

The $|0\rangle \to |1\rangle$ transition remains at $\omega$ regardless
of $\alpha$.

```python
import tiqs

pot = tiqs.DuffingPotential(
    omega=2 * 3.14159 * 1e6,
    anharmonicity=-2 * 3.14159 * 10e3,  # -10 kHz anharmonicity
)
freqs = tiqs.transition_frequencies(pot, n_fock=20)
# freqs[0] ~ omega, freqs[1] ~ omega + alpha, ...
```

### Arbitrary Potential

For potentials that cannot be expressed as a simple polynomial in
$\hat{n}$, ``ArbitraryPotential`` constructs the Hamiltonian from
a user-supplied potential energy function $V(x)$ expressed in terms
of the position operator:

$$
H = \frac{\hat{p}^2}{2m} + V(\hat{x})
$$

The position operator is built from the Fock-basis ladder operators:

$$
\hat{x} = x_\mathrm{zpf}\,(a + a^\dagger),
\qquad
x_\mathrm{zpf} = \sqrt{\frac{\hbar}{2m\omega}}
$$

The kinetic energy is computed indirectly as
$T = H_\mathrm{ref} - V_\mathrm{ref}$, where $H_\mathrm{ref}$ is the
reference harmonic oscillator and $V_\mathrm{ref}$ is its harmonic
potential.

**Important**: The user must provide the **full** potential $V(x)$,
including any harmonic part. For example, a quartic anharmonic
oscillator:

$$
V(x) = \frac{1}{2}m\omega^2 x^2 + \lambda x^4
$$

```python
import tiqs
import numpy as np

species = tiqs.get_species("Ca40")
omega = 2 * np.pi * 1e6
lam = 1e40

pot = tiqs.ArbitraryPotential(
    v_func=lambda x: 0.5 * species.mass_kg * omega**2 * x * x + lam * x**4,
    omega=omega,
    mass_kg=species.mass_kg,
)
E = tiqs.energy_levels(pot, n_fock=40)
```

#### Convergence

The Fock-basis representation converges best when the reference
frequency $\omega$ matches the curvature of $V(x)$ near its minimum.
Use ``check_convergence()`` to verify that the truncation dimension
``n_fock`` is sufficient:

```python
tiqs.check_convergence(pot, n_fock=40, n_levels=5)
```

This compares eigenvalues at ``n_fock`` vs ``n_fock + 5`` and warns
if any of the lowest 5 levels differ by more than $10^{-6}$ relative
tolerance.

#### Interaction Picture Caveat

Simulations with ``ArbitraryPotential`` should use the **Schrodinger
picture** rather than the interaction picture. The anharmonic correction
generally does not commute with the free harmonic Hamiltonian, so the
standard rotating-frame transformations used in sideband physics do not
apply directly.

### Utility Functions

``energy_levels(potential, n_fock)``
:   Diagonalizes the single-mode Hamiltonian and returns sorted
    eigenvalues in rad/s.

``transition_frequencies(potential, n_fock)``
:   Returns an array of $\omega_{n \to n+1}$ for $n = 0, \ldots,
    N_\mathrm{fock} - 2$.

``check_convergence(potential, n_fock, n_levels=5)``
:   Compares eigenvalues at two truncation sizes and warns if not
    converged.

``mode_hamiltonian(potential, ops, mode)``
:   Lifts a single-mode Hamiltonian into the full composite Hilbert
    space at the given mode index.

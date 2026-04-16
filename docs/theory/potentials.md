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
import numpy as np
import tiqs

pot = tiqs.HarmonicPotential(omega=2 * np.pi * 1e6)
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

- $\alpha < 0$: negative anharmonicity (higher levels are closer
  together), as in transmon qubits or softening nonlinearities.
- $\alpha > 0$: positive anharmonicity (higher levels are farther
  apart), as in stiffening nonlinearities.

The $|0\rangle \to |1\rangle$ transition remains at $\omega$ regardless
of $\alpha$.

```python
import numpy as np
import tiqs

pot = tiqs.DuffingPotential(
    omega=2 * np.pi * 1e6,
    anharmonicity=-2 * np.pi * 10e3,  # -10 kHz anharmonicity
)
freqs = tiqs.transition_frequencies(pot, n_fock=20)
# freqs[0] ~ omega, freqs[1] ~ omega + alpha, ...
```

### Arbitrary Potential

For potentials that cannot be expressed as a simple polynomial in
$\hat{n}$, ``ArbitraryPotential`` constructs the Hamiltonian from
a user-supplied potential $V(q)$ defined in **dimensionless
coordinates**, where $q = a + a^\dagger$ is the dimensionless
position operator. The potential must return values in **angular
frequency units** (rad/s, i.e. $\hbar = 1$).

The kinetic energy in the reference harmonic basis is:

$$
T = \omega\,(\hat{n} + \tfrac{1}{2}) - \frac{\omega}{4}\,q^2
$$

The full Hamiltonian is $H = T + V(q)$. For a harmonic potential,
$V(q) = \omega q^2/4$, and $H$ reduces to $\omega(\hat{n} + 1/2)$.

**Important**: The user must provide the **full** potential $V(q)$
including any harmonic part. Working in dimensionless units avoids
the catastrophic cancellation that occurs when SI Joule-scale
energies ($\sim 10^{-28}$) are added to rad/s-scale values
($\sim 10^{6}$) in QuTiP matrix arithmetic.

For example, a quartic anharmonic oscillator:

$$
V(q) = \frac{\omega}{4}\,q^2 + \lambda\,q^4
$$

where $\lambda$ is in rad/s per unit $q^4$.

```python
import tiqs
import numpy as np

omega = 2 * np.pi * 1e6
lam = omega * 0.01  # 1% anharmonicity

pot = tiqs.ArbitraryPotential(
    v_func=lambda q: omega / 4 * q * q + lam * q**4,
    omega=omega,
)
E = tiqs.energy_levels(pot, n_fock=40)
```

#### Convergence

The Fock-basis representation converges best when the reference
frequency $\omega$ matches the curvature of $V(q)$ near its minimum.
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

### References

1. Koch, J. et al. "Charge-insensitive qubit design derived from the
   Cooper pair box." *Phys. Rev. A* **76**, 042319 (2007).
2. Krantz, P. et al. "A quantum engineer's guide to superconducting
   qubits." *Appl. Phys. Rev.* **6**, 021318 (2019).

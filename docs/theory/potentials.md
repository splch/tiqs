## Anharmonic Potentials

Each normal mode of a trapped-ion chain is a quantum harmonic oscillator
with equally spaced energy levels separated by $\hbar\omega$. The uniform
spacing means a single laser frequency addresses the
$|n\rangle \to |n{+}1\rangle$ transition for all $n$, which is essential
for resolved-sideband cooling and for the Molmer-Sorensen and light-shift
entangling gates.

Real trapping potentials deviate from perfect harmonicity. Higher-order
terms in the electrode potential, intentional anharmonic traps (e.g. for
long ion chains or Penning-trap electrons), or effective nonlinearities
from strong drives produce **anharmonic** mode Hamiltonians where each
sideband transition $|n\rangle \to |n{+}1\rangle$ occurs at a slightly
different frequency. Gate and cooling protocols designed for a harmonic
spectrum then acquire $n$-dependent errors.

All potentials in TIQS work in angular-frequency units (rad/s) with
$\hbar = 1$, so energies and frequencies are numerically identical.
To recover SI energy, multiply by $\hbar$.

### Harmonic Potential

The single-mode Hamiltonian for a harmonic oscillator is:

$$
H = \omega\,a^\dagger a
$$

Energy eigenvalues are $E_n = n\omega$ and all transition frequencies
equal $\omega$. This is the implicit default when no potential is
specified.

### Duffing (Kerr) Potential

A harmonic oscillator with a quartic nonlinearity produces the
**Duffing** (also called **Kerr**) Hamiltonian:

$$
H = \omega\,\hat{n} + \frac{\alpha}{2}\,\hat{n}\,(\hat{n} - 1)
$$

where $\alpha$ is the **anharmonicity** parameter (equivalently,
$(\alpha/2)\,\hat{n}(\hat{n}-1) = (\alpha/2)\,a^{\dagger 2} a^2$).
The transition frequency from $|n\rangle$ to $|n{+}1\rangle$ shifts
linearly with $n$:

$$
\omega_{n \to n+1} = \omega + \alpha\,n
$$

For $\alpha < 0$ (negative anharmonicity), higher levels are closer
together, as in transmon superconducting qubits or softening trap
nonlinearities. For $\alpha > 0$ (positive anharmonicity), higher
levels are farther apart, as in stiffening nonlinearities. The
$|0\rangle \to |1\rangle$ transition remains at $\omega$ regardless
of $\alpha$.

**Effect on gate fidelity.** In a Molmer-Sorensen gate the bichromatic
drive is tuned to $\omega_0 \pm (\omega_p + \delta)$. When the motional
mode is anharmonic, the transition $|n\rangle \to |n{+}1\rangle$ occurs
at $\omega_p + \alpha n$ rather than $\omega_p$, so higher Fock states
are progressively off-resonant from the gate drive. This shifts the
phase-space closure condition and produces residual spin-motion
entanglement at the nominal gate time. For typical trapped-ion
parameters ($|\alpha|/\omega \sim 10^{-6}$-$10^{-4}$), the effect is
small but measurable.

### Arbitrary Potential

For potentials that cannot be expressed as a polynomial in $\hat{n}$,
TIQS constructs the Hamiltonian from a user-supplied function $V(q)$
of the **dimensionless position operator** $q = a + a^\dagger$, where
$V(q)$ returns values in angular-frequency units (rad/s).

The kinetic energy in the reference harmonic basis is:

$$
T = \omega\bigl(\hat{n} + \tfrac{1}{2}\bigr) - \frac{\omega}{4}\,q^2
$$

The full Hamiltonian is $H = T + V(q)$. For a harmonic potential
$V(q) = \omega\,q^2/4$, this reduces to $H = \omega(\hat{n} + 1/2)$.

The function $V(q)$ must include the **full** potential, including the
harmonic part. Working in dimensionless units avoids catastrophic
cancellation that occurs when SI Joule-scale energies ($\sim 10^{-28}$)
are added to rad/s-scale values ($\sim 10^{6}$) in matrix arithmetic.

For example, a quartic anharmonic oscillator:

$$
V(q) = \frac{\omega}{4}\,q^2 + \lambda\,q^4
$$

where $\lambda$ has units of rad/s per unit $q^4$.

**Convergence.** The Fock-basis representation converges best when the
reference frequency $\omega$ matches the curvature of $V(q)$ near its
minimum. The `check_convergence()` utility compares eigenvalues at
truncation dimension $N_\text{fock}$ versus $N_\text{fock} + 5$ and
warns if any of the lowest levels differ by more than $10^{-6}$ relative
tolerance.

**Interaction picture.** Simulations with arbitrary potentials should use
the Schrodinger picture rather than the interaction picture. The
anharmonic correction generally does not commute with the free harmonic
Hamiltonian, so the standard rotating-frame transformations used in
sideband physics do not apply directly.

### References

1. Koch, J. et al. "Charge-insensitive qubit design derived from the
   Cooper pair box." *Phys. Rev. A* **76**, 042319 (2007).
2. Krantz, P. et al. "A quantum engineer's guide to superconducting
   qubits." *Appl. Phys. Rev.* **6**, 021318 (2019).
3. Home, J.P. "Quantum science and metrology with mixed-species ion
   chains." *Adv. At. Mol. Opt. Phys.* **62**, 231 (2013).
4. Lin, G.-D. et al. "Large-scale quantum computation in an anharmonic
   linear ion trap." *Europhys. Lett.* **86**, 60004 (2009).

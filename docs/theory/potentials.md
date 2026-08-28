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

`omega` must be a **positive-energy** mode frequency: the level ladder built
here ascends. The one frequency in TIQS for which that is false is the
Penning magnetron $\omega_-$, whose true ladder descends (see
[trapping.md](trapping.md)); feeding it here, or to `mode_hamiltonian()`,
silently produces the wrong free evolution.

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

`anharmonicity` is a **phenomenological** parameter: TIQS does not derive it
from trap geometry. Nothing connects `tiqs.potential` to `tiqs.trap` or
`tiqs.chain`, so the value must come from the user's own model or
measurement.

**Softening turnover.** For $\alpha < 0$ the ladder is only monotonic up to
$n = \omega/|\alpha|$; above that, $\omega_{n \to n+1}$ goes negative and
energy-ascending eigenvalue order stops matching Fock order. Use
`transition_frequencies()`, which reads the Fock-ordered diagonal and
reports those negative gaps honestly, rather than differencing
`energy_levels()`, which sorts by energy.

**Effect on gate fidelity.** In a Molmer-Sorensen gate the bichromatic
drive is tuned to $\omega_0 \pm (\omega_p - \delta)$ (the tone placement
`molmer_sorensen.py` implements -- see the convention note in
[gates.md](gates.md)). When the motional
mode is anharmonic, the transition $|n\rangle \to |n{+}1\rangle$ occurs
at $\omega_p + \alpha n$ rather than $\omega_p$, so higher Fock states
are progressively off-resonant from the gate drive. This shifts the
phase-space closure condition and produces residual spin-motion
entanglement at the nominal gate time.

**Measured scales.** For a single ion the intrinsic self-Kerr is small: Home
et al. measured diagonal Kerr shifts of $-2.9$, $-0.9$ and $-0.1$ Hz per
quantum for the three modes (7, 5, 1.8 MHz) of a $^{25}$Mg$^+$ ion 30 $\mu$m
above a surface trap, i.e. $|\alpha|/\omega \sim 6\times10^{-8}$ to
$4\times10^{-7}$ -- "the largest shifts are a few parts in $10^7$ per
quantum". Their two-ion matrix reaches 6.5 Hz at $\sim$3.1 MHz,
$2\times10^{-6}$. Coulomb-induced **cross**-Kerr couplings between modes of
a multi-ion crystal are larger: Ding et al. measured about 20 Hz/phonon at
trap frequencies of $2\pi \times (1042, 979, 587)$ kHz
($\sim 2\times10^{-5}$ relative), rising to $\sim$300 Hz/phonon
($\sim 3\times 10^{-4}$) only when deliberately tuned near the
$2\omega_b - \omega_a$ parametric resonance. Note that a cross-Kerr
$\chi\, n_a n_b$ is a different object from the self-Kerr
$(\alpha/2) n(n-1)$ that `DuffingPotential` implements, and that smaller
traps give *larger* anharmonicity -- the Home et al. trap is already small.

### Arbitrary Potential

For potentials that cannot be expressed as a polynomial in $\hat{n}$,
TIQS constructs the Hamiltonian from a user-supplied function $V(q)$
of the **dimensionless position operator** $q = a + a^\dagger$, where
$V(q)$ returns values in angular-frequency units (rad/s).

> **Coordinate convention.** This $q$ is *not* the unit-commutator
> quadrature $x = (a + a^\dagger)/\sqrt{2}$ returned by
> `OperatorFactory.position` (the convention `tiqs.analysis.phase_space`
> also follows). The two differ by $q = \sqrt{2}\,x$, hence a factor 2 in
> any quadratic term. Expectation values taken with `ops.position` must be
> rescaled before being fed to a $V(q)$ written for this class.

The kinetic energy in the reference harmonic basis is:

$$
T = \omega\bigl(\hat{n} + \tfrac{1}{2}\bigr) - \frac{\omega}{4}\,q^2
$$

The full Hamiltonian is $H = T + V(q)$. For a harmonic potential
$V(q) = \omega\,q^2/4$, this reduces to $H = \omega(\hat{n} + 1/2)$.
`single_mode_hamiltonian` raises `ValueError` if $T + V(q)$ comes out
non-Hermitian (beyond $10^{-10}$ of $\max_{ij}|H_{ij}|$), since $V$ must be
a real function of the Hermitian operator $q$ and a complex $V$ would
otherwise yield eigenvalues whose imaginary parts are silently discarded.

The function $V(q)$ must include the **full** potential, including the
harmonic part. Two reasons for the dimensionless convention: the Fock basis
is itself dimensionless, and it keeps $\hbar$ bookkeeping out of the
user-supplied $V$. (It is not about floating-point conditioning. Double
precision carries $\sim$16 significant digits at any exponent, so a
Joule-scale Hamiltonian is perfectly well conditioned; the mild
cancellation that does occur in $T$ above is a factor of 2 -- its interior
diagonal is exactly $\omega(n + 1/2)/2$ -- nowhere near catastrophic.)

For example, a quartic anharmonic oscillator:

$$
V(q) = \frac{\omega}{4}\,q^2 + \lambda\,q^4
$$

where $\lambda$ has units of rad/s per unit $q^4$.

**Converting from SI.** The dimensionless coordinate maps to physical
displacement as $x = x_\text{zpf}\, q$ with
$x_\text{zpf} = \sqrt{\hbar/2m\omega}$, so a potential $V_\text{SI}(x)$ in
joules becomes $V(q) = V_\text{SI}(x_\text{zpf}\, q)/\hbar$. For a quartic
term $V_\text{SI} = c_4 x^4$ this gives
$\lambda = c_4\, x_\text{zpf}^4/\hbar$, and first-order perturbation theory
($\langle n|q^4|n\rangle = 6n(n+1) + 3$) maps it onto the Duffing
anharmonicity as

$$
\alpha = 12\,\lambda
$$

**Convergence.** The Fock-basis representation converges best when the
reference frequency $\omega$ matches the curvature of $V(q)$ near its
minimum. The `check_convergence()` utility compares the lowest levels at
truncation dimension $N_\text{fock}$ against $2 N_\text{fock}$ -- a
*geometric* step, because a fixed additive step cannot resolve tail
contributions that grow slowly with the truncation -- and returns `True`
when the largest level shift is at most $10^{-6}$ of the largest checked
level magnitude $\max_i |E_i|$, warning with that same ratio otherwise.

> Convergence here is necessary, **not sufficient**. A potential that is
> unbounded below has no ground state, so no truncation is converged and
> this check cannot certify it. With
> $V(q) = \omega q^2/4 - 10^{-3}\,\omega\, q^4$ the check reports
> converged for `n_fock` from 15 to 30 with $E_0/\omega$ frozen at 0.4970,
> and only later collapses -- $E_0/\omega = -6.62$ at `n_fock` $= 80$.
> Confirm independently that $V(q) \to +\infty$ as $|q| \to \infty$.

**Interaction picture.** Simulations with arbitrary potentials should use
the Schrodinger picture rather than the interaction picture. The
anharmonic correction generally does not commute with the free harmonic
Hamiltonian, so the standard rotating-frame transformations used in
sideband physics do not apply directly.

`transition_frequencies()` reports this honestly: an `ArbitraryPotential`
whose $V$ does not exactly cancel the reference curvature is not diagonal
in the Fock basis, so no $|n\rangle \to |n{+}1\rangle$ ladder exists at all.
In that case the function emits a `UserWarning` and returns gaps between
adjacent eigenvalues in ascending energy order instead.

### Model Scope and Approximations

- **Per-mode, diagonal-in-mode-index.** Every potential here is a
  single-mode object, and `mode_hamiltonian()` lifts it into the composite
  space one mode at a time. There is no coupling between the modes of a
  chain -- no Coulomb-derived cross-Kerr, no cubic axial-radial coupling
  (Marquet et al.). In-chain mode-mode coupling is an unimplemented scope
  gap, like full $N$-particle Penning transverse modes.
- **Phenomenological parameters.** `DuffingPotential.anharmonicity` and
  `ArbitraryPotential.v_func` are supplied by the user; nothing derives
  them from electrode geometry, ion positions, or mode vectors.
- **Positive-energy frequencies only.** `omega` sets an ascending ladder
  (above).
- **Truncation is unverified unless checked.** `single_mode_hamiltonian`
  builds a fixed `n_fock` matrix and does not test convergence itself;
  call `check_convergence()`, and note its limits above.
- **No time dependence.** Potentials are static; a drive-induced effective
  nonlinearity must be supplied as an already-averaged $\alpha$ or $V(q)$.

### References

1. Home, J.P., Hanneke, D., Jost, J.D., Leibfried, D. & Wineland, D.J.
   "Normal modes of trapped ions in the presence of anharmonic trap
   potentials." *New J. Phys.* **13**, 073026 (2011).
2. Ding, S., Maslennikov, G., Hablutzel, R. & Matsukevich, D.
   "Cross-Kerr nonlinearity for phonon counting."
   *Phys. Rev. Lett.* **119**, 193602 (2017).
3. Marquet, C., Schmidt-Kaler, F. & James, D.F.V. "Phonon-phonon
   interactions due to non-linear effects in a linear ion trap."
   *Appl. Phys. B* **76**, 199 (2003).
4. Home, J.P. "Quantum science and metrology with mixed-species ion
   chains." *Adv. At. Mol. Opt. Phys.* **62**, 231 (2013).
5. Lin, G.-D. et al. "Large-scale quantum computation in an anharmonic
   linear ion trap." *Europhys. Lett.* **86**, 60004 (2009).
6. Koch, J. et al. "Charge-insensitive qubit design derived from the
   Cooper pair box." *Phys. Rev. A* **76**, 042319 (2007).
7. Krantz, P. et al. "A quantum engineer's guide to superconducting
   qubits." *Appl. Phys. Rev.* **6**, 021318 (2019).

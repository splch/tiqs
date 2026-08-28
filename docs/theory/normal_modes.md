## Ion Chain Physics

Multiple trapped ions form **Coulomb crystals**: in a linear trap, ions line
up into a chain with collective small oscillations that decompose into quantized
**normal modes**. These modes serve as the quantum bus for entangling gates.

This page develops the single-species case first, because the closed forms are
simpler and the standard results (COM at $\omega_z$, uniform participation)
hold there. TIQS also supports **mixed-species** chains, for which several of
those results are false; the differences are collected in the
**Mixed-Species Chains** section below and flagged inline.

### Equilibrium Positions

For $N$ ions of mass $m$ and charge $e$ in a harmonic axial potential with
frequency $\omega_z$, the total potential energy is:

$$
V = \sum_{i=1}^{N} \frac{1}{2} m \omega_z^2 z_i^2 + \sum_{i \lt j} \frac{e^2}{4\pi\epsilon_0 |z_i - z_j|}
$$

Defining the characteristic length scale $l_0 = \left(\frac{e^2}{4\pi\epsilon_0\, m\, \omega_z^2}\right)^{1/3}$
and dimensionless positions $u_i = z_i / l_0$, the equilibrium condition
$\partial V / \partial u_i = 0$ gives $N$ coupled equations:

$$
u_i = \sum_{j \neq i} \frac{\text{sgn}(u_i - u_j)}{(u_i - u_j)^2}
$$

Closed-form solutions exist only for $N = 2$ and $N = 3$ (James Eqs. (2.6)
and (2.7)); the $N = 4$ and $N = 5$ entries below are the numerical values of
James Table 1.

| $N$ | Equilibrium positions $u_i$ | source |
|-----|---------------------------|--------|
| 2 | $\pm (1/2)^{2/3} = \pm 0.6300$ | analytic |
| 3 | $0, \pm (5/4)^{1/3} = \pm 1.0772$ | analytic |
| 4 | $\pm 0.4544, \pm 1.4368$ | numerical |
| 5 | $0, \pm 0.8221, \pm 1.7429$ | numerical |

TIQS solves the dimensionless equations numerically for **every** $N \ge 2$,
using MINPACK's hybrid Powell trust-region solver (`scipy.optimize.root`
with `method="hybr"`) supplied with the analytic Jacobian -- which is the
dimensionless axial Hessian below, so the step is a true Newton step. Only
$N = 0$ and $N = 1$ are special-cased. The initial guess is scaled as
$N^{0.56}$, because the chain half-width grows only that fast (James
Eq. (2.8) gives the minimum spacing $u_\text{min} = 2.018\,N^{-0.559}$); a
fixed spacing diverges above $N \approx 60$. Convergence is gated on the
force residual and on strict ordering, not on the solver's own success flag.

### Hessian Matrix and Normal Modes

Expanding $V$ to second order around equilibrium yields the dynamical matrix.
TIQS builds the **mass-weighted** dynamical matrix
$D = M^{-1/2} V'' M^{-1/2}$ (with $M = \operatorname{diag}(m_i)$), whose
eigenvalues are squared frequencies. For a single species this reduces to the
familiar mass-normalized Hessian
$A_{jk} = \frac{1}{m}\frac{\partial^2 V}{\partial z_j\, \partial z_k}$, and
in that case, for **axial modes**, the $N \times N$ matrix has elements:

$$
\frac{A_{jj}}{\omega_z^2} = 1 + 2\sum_{k \neq j} \frac{1}{|u_j - u_k|^3},
\qquad
\frac{A_{jk}}{\omega_z^2} = \frac{-2}{|u_j - u_k|^3}
$$

For **radial modes** (transverse to the chain axis):

$$
\frac{A^x_{jj}}{\omega_x^2} = 1 - \frac{\omega_z^2}{\omega_x^2}\sum_{k \neq j} \frac{1}{|u_j - u_k|^3},
\qquad
\frac{A^x_{jk}}{\omega_x^2} = \frac{\omega_z^2}{\omega_x^2} \frac{1}{|u_j - u_k|^3}
$$

Note the sign difference: Coulomb repulsion **stiffens** axial modes but
**softens** radial modes.

### Mode Frequencies and Participation Vectors

Diagonalizing $D$ yields $N$ normal modes:

$$
D\, \mathbf{b}^{(p)} = \omega_p^2\, \mathbf{b}^{(p)}
$$

where $\omega_p$ is the frequency of mode $p$ and
$\mathbf{b}^{(p)} = (b_{1,p}, \ldots, b_{N,p})$ is the **mass-weighted**
participation vector. $D$ has units of $\text{rad}^2/\text{s}^2$ (the masses
are divided out in its construction), so its eigenvalues are $\omega_p^2$
directly.

The normalization is a contract that later formulas depend on: because $D$ is
diagonalized by `numpy.linalg.eigh`, the columns are **Euclidean**
orthonormal,

$$
\sum_i b_{i,m}\, b_{i,n} = \delta_{mn}
$$

and the zero-point amplitudes below are written for exactly that convention.
Renormalizing to the generalized convention
$\mathbf{b}^T M \mathbf{b} = I$ (for example by switching to
`scipy.linalg.eigh(V, M)`) would silently invalidate them.

For a **single-species** chain the **center-of-mass (COM) mode** is the
lowest axial mode:

$$
\omega_\text{COM} = \omega_z, \qquad b_{i,\text{COM}} = \frac{1}{\sqrt{N}}
$$

For $N = 2$, the two axial modes are:

| Mode | Frequency | Participation vector |
|------|-----------|---------------------|
| COM | $\omega_z$ | $(1, 1)/\sqrt{2}$ |
| Stretch | $\sqrt{3}\,\omega_z$ | $(1, -1)/\sqrt{2}$ |

For $N = 3$:

| Mode | Frequency | Participation vector |
|------|-----------|---------------------|
| COM | $\omega_z$ | $(1, 1, 1)/\sqrt{3}$ |
| Tilt | $\sqrt{3}\,\omega_z$ | $(1, 0, -1)/\sqrt{2}$ |
| Breathe | $\sqrt{29/5}\,\omega_z$ | $(1, -2, 1)/\sqrt{6}$ |

None of the three statements in this section -- COM lowest, COM at
$\omega_z$, uniform participation -- survives unequal masses.

### Quantization

Each normal mode is an independent quantum harmonic oscillator:

$$
H_\text{motion} = \sum_{p=1}^{N} \hbar\omega_p \left(a_p^\dagger a_p + \frac{1}{2}\right)
$$

The position of ion $j$ in terms of mode operators:

$$
z_j = z_j^{(0)} + \sum_p \frac{b_{j,p}}{\sqrt{m_j}}\,
      \sqrt{\frac{\hbar}{2\,\omega_p}} \left(a_p + a_p^\dagger\right)
    = z_j^{(0)} + \sum_p b_{j,p}\, z_{0,p}^{(j)}
      \left(a_p + a_p^\dagger\right)
$$

where $z_{0,p}^{(j)} = \sqrt{\hbar / (2 m_j \omega_p)}$ is the zero-point
motion of ion $j$ in mode $p$. The mass carried here is the mass of **that
ion**, not a chain-wide constant -- for a single species the distinction
collapses, but it is what makes the mass-weighted eigenvector the right
object for a mixed chain.

One exception to the sum above: the Penning **magnetron** mode is an
$\mathbf{E} \times \mathbf{B}$ drift whose total energy *decreases* with
radius, so it enters as $-\hbar\omega_-(n_- + 1/2)$ and the positive-energy
quantization written here does not apply to it. See
[trapping.md](trapping.md) for the consequences (metastability, exchanged
sideband roles).

### Lamb-Dicke Parameters

The **Lamb-Dicke parameter** for ion $j$ and mode $p$ characterizes how
strongly a laser with wavevector $k_j$ couples the ion's internal state to
that motional mode:

$$
\eta_{j,p} = k_j\, b_{j,p}\, z_{0,p}^{(j)}
= k_j\, b_{j,p} \sqrt{\frac{\hbar}{2 m_j \omega_p}}
$$

Both the wavevector and the mass are per ion, which matters for mixed
species addressed by different lasers. Typical values are
$\eta \sim 0.05$-$0.2$. The **Lamb-Dicke regime**
$\eta\sqrt{2\bar{n}+1} \ll 1$ ensures that sideband transitions are
well-resolved and higher-order terms are suppressed.

`lamb_dicke_parameters()` accepts a scalar or a per-ion sequence for both
`species` and `k_eff`. Its `k_eff` is per *ion*, so it cannot express a
coupling whose strength depends on the mode frequency;
`gradient_lamb_dicke_parameters()` covers the magnetic-gradient (MAGIC) case,
where $k_{\text{eff},m} = g\mu_B(\partial B/\partial z)/\hbar\omega_m$ and
hence $\eta \propto \omega_m^{-3/2}$.

### Mixed-Species Chains

`normal_modes(n_ions, trap, masses=...)` takes per-ion masses, and the
mass-weighted matrix is then genuinely not proportional to any single-mass
Hessian. With $C = e^2/4\pi\epsilon_0$ and $d_{ij}$ the equilibrium
separations, the implemented elements are

$$
D^z_{ii} = \omega_{z,i}^2 + \sum_{j \neq i} \frac{2C}{m_i\, d_{ij}^3},
\qquad
D^z_{ij} = \frac{-2C}{\sqrt{m_i m_j}\; d_{ij}^3}
$$

for axial modes, with the sign flipped and the factor 2 replaced by 1 for
radial modes. The trap contributes a **mass-independent** axial spring
constant $K = m_\text{ref}\,\omega_\text{axial}^2$, so each ion has its own
$\omega_{z,i} = \sqrt{K/m_i}$, and each its own Mathieu $q_i \propto 1/m_i$
and radial frequency $\omega_{r,i}$.

The eigenvectors are orthonormal in the mass-weighted coordinates
$\xi_i = \sqrt{m_i}\,\delta z_i$, so:

> `vectors[i, m]` is **not** how far ion $i$ moves. The physical
> displacement amplitude is `vectors[i, m] / sqrt(m_i)`.

Worked example, $^9$Be$^+$ and $^{40}$Ca$^+$ with the axial spring constant
of a $2\pi \times 1$ MHz Ca-40 trap:

| quantity | value |
|---|---|
| axial mode frequencies | $2\pi \times (1.1857,\ 3.0762)$ MHz |
| per-ion $\omega_{z,i}/2\pi$ | 2.1058 MHz (Be), 1.0000 MHz (Ca) |
| lowest mode / $\omega_\text{axial}$ | 1.1857 (**not** 1) |
| $\mathbf{b}^{(0)}$ | $(-0.2716,\ -0.9624)$ (**not** $1/\sqrt{2}$ each) |
| physical amplitudes, mode 0 | $\propto (-0.594,\ -1.000)$ |
| physical amplitudes, mode 1 | $\propto (-1.000,\ +0.134)$ |

The out-of-phase amplitude ratio is $7.46$ physically but only $3.54$ read
straight off `vectors` -- a factor $\sqrt{m_\text{Ca}/m_\text{Be}} = 2.11$
for anyone who skips the mass weighting. The physical amplitudes above agree
exactly with the generalized eigenproblem $V''\mathbf{v} = \omega^2 M
\mathbf{v}$, which is the independent check.

For a mixed-species **Penning** chain the transverse frequencies are also
mass dependent, since $\omega_c = eB/m_i$ (below).

### The NormalModeResult Structure

``tiqs.normal_modes(n_ions, trap, masses=None)`` returns a
``NormalModeResult`` dataclass with two fields:

- ``positions``: equilibrium positions in meters, shape $(N,)$.
- ``modes``: a dictionary mapping physical names to ``ModeGroup`` objects.

``masses`` is an optional per-ion array in kg, shape $(N,)$, ordered to match
the sorted equilibrium positions (``masses[0]`` is the leftmost ion). When
omitted, every ion uses ``trap.species.mass_kg``.

Each ``ModeGroup`` contains:

- ``freqs``: angular frequencies in rad/s, shape $(N,)$, sorted ascending.
- ``vectors``: mass-weighted eigenvector matrix, shape $(N, N)$. Column $m$
  is the participation vector for mode $m$: ``vectors[i, m]`` $= b_{i,m}$,
  orthonormal over $i$. Physical displacement $=$
  ``vectors[i, m] / sqrt(m_i)``.

The dictionary keys depend on the trap type:

| Trap type | Mode keys |
|-----------|-----------|
| ``PaulTrap`` | ``"axial"``, ``"radial_x"``, ``"radial_y"`` |
| ``PenningTrap`` | ``"axial"``, ``"modified_cyclotron"``, ``"magnetron"`` |

For Paul traps, ``radial_y`` is **exactly degenerate with ``radial_x`` by
construction**, not merely in the absence of a symmetry-breaking field: the
model hard-codes the radially symmetric DC split
$a = -2\omega_z^2/\Omega_\text{RF}^2$ (Wübbena et al. Eq. (6) with
$\alpha = 1/2$) and ``PaulTrap`` exposes no asymmetry parameter, so the key
exists only for API symmetry. Real linear traps deliberately split the two
radial frequencies; modelling that would need a new trap parameter.

``normal_modes`` raises ``ValueError`` when any ion falls outside the first
Mathieu stability region ($q_i \ge 0.908$) or loses radial confinement
($\beta_i^2 \le 0$), when a Penning ion violates
$\omega_c > \sqrt{2}\,\omega_z$, and when the radial dynamical matrix is not
positive definite -- the zigzag case below.

```python
import numpy as np
import tiqs

species = tiqs.get_species("Ca40")
trap = tiqs.PaulTrap(
    v_rf=200.0,
    omega_rf=2 * np.pi * 30e6,
    r0=300e-6,
    species=species,
    omega_axial=2 * np.pi * 1e6,
)
result = tiqs.normal_modes(n_ions=2, trap=trap)

# Axial mode frequencies (rad/s, sorted ascending)
result.modes["axial"].freqs

# Participation vector for axial COM mode (mode index 0)
result.modes["axial"].vectors[:, 0]

# Lamb-Dicke parameters for the Ca+ 729 nm quadrupole transition
wavelength = 729e-9
k_eff = 2 * np.pi / wavelength
eta = tiqs.lamb_dicke_parameters(result, species, k_eff, direction="axial")
# eta[i, m] is the Lamb-Dicke parameter for ion i, mode m
```

That trap sits at Mathieu $q = 0.302$, inside the $q \approx 0.1$-$0.4$ band
where the pseudopotential approximation is accurate, and its anisotropy
$\omega_\text{rad}/\omega_z = 3.12$ leaves the chain linear up to $N = 5$
(see the zigzag thresholds below), so the example extends safely. Shrinking
`r0` to 200 um would push it to $q = 0.68$: still formally inside the first
stability region, but it emits the library's `Mathieu q > 0.4` warning and
underestimates the true secular frequency by roughly 10%.

### Linear-to-Zigzag Transition

Above a critical anisotropy the linear chain buckles into a zigzag, and the
1D equilibrium becomes a saddle point of the 3D potential rather than a
minimum. Written as a threshold on the frequency ratio, the linear chain is
stable when $\omega_\text{rad} > c_N\,\omega_z$.

The widely quoted estimate is Steane's Eq. (11),
$c_N \approx 0.73\, N^{0.86}$, derived from a uniform-spacing approximation
(and reproduced by Wineland et al. after their Eq. (7), together with the
exact small-$N$ values $c_2 = 1$ and $c_3 = 1.55$). It is a **conservative
overestimate**, by 15-33% for $N \le 10$. Steane's genuine $N \gg 1$ form is
his Eq. (12), $c_N > 0.77\,N/\sqrt{\log N}$, which is the *smaller* of the
two at large $N$ (35.9 versus 38.3 at $N = 100$).

The exact criterion for this model is available directly from the axial
spectrum. In units of $\omega_z^2$ the axial and radial dynamical matrices
are $I + 2A$ and $(\omega_\text{rad}/\omega_z)^2 I - A$ with the *same*
Coulomb matrix $A$, so positive definiteness of the radial matrix is exactly

$$
\frac{\omega_\text{rad}}{\omega_z} > c_N
= \sqrt{\frac{\mu_{N,\max} - 1}{2}}
$$

with $\mu_{N,\max}$ the largest axial eigenvalue in units of $\omega_z^2$
(James Table 2).

| $N$ | $c_N$ exact | $0.73\,N^{0.86}$ | overestimate |
|---|---|---|---|
| 2 | 1.0000 | 1.3250 | $+32.5\%$ |
| 3 | 1.5492 | 1.8778 | $+21.2\%$ |
| 4 | 2.0382 | 2.4049 | $+18.0\%$ |
| 5 | 2.4975 | 2.9136 | $+16.7\%$ |
| 10 | 4.5957 | 5.2884 | $+15.1\%$ |

$c_2 = 1$ is independently known: the two-ion radial rocking mode is
$\sqrt{\omega_\text{rad}^2 - \omega_z^2}$, which vanishes at
$\omega_\text{rad} = \omega_z$.

`normal_modes()` tests the exact condition -- it diagonalizes the radial
matrix and raises `ValueError` on any negative eigenvalue, rather than
applying the $0.73\,N^{0.86}$ fit -- and the error message quotes the
measured ratio against $c_N$. Note that the fit has no mixed-species
meaning at all, since each ion then has its own $\omega_{r,i}$ and
$\omega_{z,i}$; the eigenvalue test still applies unchanged.

### Penning Trap Modes

In a Penning trap, the three eigenmotions are axial oscillation, modified
cyclotron motion, and magnetron drift. The **axial modes** of an $N$-ion
crystal are computed identically to the Paul trap case: the Coulomb-coupled
dynamical matrix is diagonalized in the axial harmonic potential.

The **transverse modes** (modified cyclotron and magnetron) are qualitatively
different from Paul trap radial modes because the radial dynamics involve the
Coriolis-like coupling from the magnetic field. TIQS currently computes
Penning transverse modes in a **single-particle approximation**: each ion
oscillates independently at its own single-particle modified cyclotron
frequency $\omega_+(m_i)$ or magnetron frequency $\omega_-(m_i)$ with no
inter-ion coupling. A ``UserWarning`` is emitted when $N > 1$ to flag this.
Full $N$-particle transverse mode analysis with rotating-frame Coulomb
coupling is a planned extension.

In the returned ``ModeGroup``:

- **Single species**: ``freqs`` contains $N$ identical entries and
  ``vectors`` is the $N \times N$ identity matrix (each "mode" is localized
  on one ion).
- **Mixed species**: ``freqs`` holds the per-ion values $\omega_\pm(m_i)$
  sorted ascending, and ``vectors`` is the corresponding *permutation*
  matrix. For $^9$Be$^+$/$^{40}$Ca$^+$ at $B = 7$ T the modified cyclotron
  entries are $2\pi \times (2.64,\ 11.88)$ MHz -- a factor 4.50, set mainly
  by the mass ratio 4.43 through $\omega_c = eB/m_i$, with the remainder
  from the mass-dependent $\omega_{z,i}$ inside the square root.

The ``"magnetron"`` group is the negative-energy exception noted under
**Quantization** above: ``freqs`` stores the unsigned $\omega_-$, and
nothing downstream applies the sign. Passing ``direction="magnetron"`` to
``lamb_dicke_parameters`` returns the zero-point spread of a metastable mode
whose sideband roles are exchanged.

### Model Scope and Approximations

- **Harmonic (small-oscillation) modes only.** The chain is expanded to
  second order about equilibrium; there is no mode-mode (phonon-phonon)
  coupling between the modes of one chain, cross-Kerr or otherwise, so
  Marquet et al.'s cubic axial-radial coupling is not implemented even
  though the transverse-mode matrix from the same paper is. (TIQS does
  compute a geometry-derived Coulomb coupling between *separate* traps, in
  `tiqs.interaction.coulomb_coupling`.)
- **1D axial equilibrium.** `equilibrium_positions` solves only the axial
  force balance. Above the zigzag threshold that stationary point is a
  saddle of the 3D potential; `normal_modes` detects it from the radial
  matrix and rejects the whole result, including the axial part.
- **Pseudopotential radial frequencies.** Paul-trap radial modes are built
  from the secular $\omega_{r,i}$, so everything in
  [trapping.md](trapping.md)'s scope note applies here too: no micromotion,
  no exact Floquet treatment, exact radial degeneracy.
- **Single-particle Penning transverse modes** (above), and no encoding of
  the magnetron energy sign.
- **No trap anharmonicity in the mode structure.** Anharmonic *mode*
  Hamiltonians are available separately via
  [potentials.md](potentials.md), but they are applied per mode after
  diagonalization; the equilibrium positions and mode vectors themselves
  always come from a harmonic axial trap.

### References

1. James, D.F.V. "Quantum dynamics of cold trapped ions with application
   to quantum computation." *Appl. Phys. B* **66**, 181 (1998).
2. Marquet, C., Schmidt-Kaler, F. & James, D.F.V. "Phonon-phonon
   interactions due to non-linear effects in a linear ion trap."
   *Appl. Phys. B* **76**, 199 (2003).
3. Steane, A.M. "The ion trap quantum information processor."
   *Appl. Phys. B* **64**, 623 (1997).
4. Wineland, D.J. et al. "Experimental issues in coherent quantum-state
   manipulation of trapped atomic ions." *J. Res. NIST* **103**, 259 (1998).
5. Home, J.P. "Quantum science and metrology with mixed-species ion
   chains." *Adv. At. Mol. Opt. Phys.* **62**, 231 (2013).
6. Sosnova, K., Carter, A. & Monroe, C. "Character of motional modes for
   entanglement and sympathetic cooling of mixed-species trapped-ion
   chains." *Phys. Rev. A* **103**, 012610 (2021).
7. Wübbena, J.B. et al. "Sympathetic cooling of mixed-species two-ion
   crystals for precision spectroscopy."
   *Phys. Rev. A* **85**, 043412 (2012).
8. Mintert, F. & Wunderlich, C. "Ion-trap quantum logic using
   long-wavelength radiation." *Phys. Rev. Lett.* **87**, 257904 (2001).
9. Brown, L.S. & Gabrielse, G. "Geonium theory: Physics of a single
   electron or ion in a Penning trap." *Rev. Mod. Phys.* **58**, 233 (1986).

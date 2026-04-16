## Ion Chain Physics

Multiple trapped ions form **Coulomb crystals**: in a linear trap, ions line
up into a chain with collective small oscillations that decompose into quantized
**normal modes**. These modes serve as the quantum bus for entangling gates.

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

Analytical solutions exist for small $N$:

| $N$ | Equilibrium positions $u_i$ |
|-----|---------------------------|
| 2 | $\pm 0.6300$ |
| 3 | $0, \pm 1.0772$ |
| 4 | $\pm 0.4544, \pm 1.4368$ |
| 5 | $0, \pm 0.8221, \pm 1.7429$ |

For $N > 5$, numerical root-finding (Newton-Raphson) is required.

### Hessian Matrix and Normal Modes

Expanding $V$ to second order around equilibrium yields the dynamical matrix.
TIQS constructs the mass-normalized Hessian
$A_{jk} = \frac{1}{m}\frac{\partial^2 V}{\partial z_j\, \partial z_k}$
so that its eigenvalues are squared frequencies.
For **axial modes**, the $N \times N$ Hessian has elements:

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

Diagonalizing $A$ yields $N$ normal modes:

$$
A\, \mathbf{b}^{(p)} = \omega_p^2\, \mathbf{b}^{(p)}
$$

where $\omega_p$ is the frequency of mode $p$ and
$\mathbf{b}^{(p)} = (b_{1,p}, \ldots, b_{N,p})$ is the orthonormal
participation vector describing how much each ion moves in that mode.
The matrix $A$ has units of $\text{rad}^2/\text{s}^2$ (the mass is
already divided out in the Hessian construction), so its eigenvalues are
$\omega_p^2$ directly.

The **center-of-mass (COM) mode** is always the lowest axial mode:

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

### Quantization

Each normal mode is an independent quantum harmonic oscillator:

$$
H_\text{motion} = \sum_{p=1}^{N} \hbar\omega_p \left(a_p^\dagger a_p + \frac{1}{2}\right)
$$

The position of ion $j$ in terms of mode operators:

$$
z_j = z_j^{(0)} + \sum_p b_{j,p}\, z_{0,p} \left(a_p + a_p^\dagger\right)
$$

where $z_{0,p} = \sqrt{\hbar / (2m\omega_p)}$ is the zero-point motion of
mode $p$.

### Lamb-Dicke Parameters

The **Lamb-Dicke parameter** for ion $j$ and mode $p$ characterizes how
strongly a laser with wavevector $k$ couples the ion's internal state to that
motional mode:

$$
\eta_{j,p} = k\, b_{j,p}\, z_{0,p} = k\, b_{j,p} \sqrt{\frac{\hbar}{2m\omega_p}}
$$

Typical values are $\eta \sim 0.05$-$0.2$. The **Lamb-Dicke regime**
$\eta\sqrt{2\bar{n}+1} \ll 1$ ensures that sideband transitions are
well-resolved and higher-order terms are suppressed.

### The NormalModeResult Structure

``tiqs.normal_modes()`` returns a ``NormalModeResult`` dataclass with two fields:

- ``positions``: equilibrium positions in meters, shape $(N,)$.
- ``modes``: a dictionary mapping physical names to ``ModeGroup`` objects.

Each ``ModeGroup`` contains:

- ``freqs``: angular frequencies in rad/s, shape $(N,)$, sorted ascending.
- ``vectors``: eigenvector matrix, shape $(N, N)$. Column $m$ is the
  participation vector for mode $m$: ``vectors[i, m]`` $= b_{i,m}$.

The dictionary keys depend on the trap type:

| Trap type | Mode keys |
|-----------|-----------|
| ``PaulTrap`` | ``"axial"``, ``"radial_x"``, ``"radial_y"`` |
| ``PenningTrap`` | ``"axial"``, ``"modified_cyclotron"``, ``"magnetron"`` |

For Paul traps, ``radial_x`` and ``radial_y`` are identical (degenerate)
unless a symmetry-breaking field is applied.

```python
import numpy as np
import tiqs

species = tiqs.get_species("Ca40")
trap = tiqs.PaulTrap(
    v_rf=200.0, omega_rf=2*np.pi*30e6,
    r0=200e-6, species=species,
    omega_axial=2*np.pi*1e6,
)
result = tiqs.normal_modes(n_ions=2, trap=trap)

# Axial mode frequencies (rad/s, sorted ascending)
result.modes["axial"].freqs

# Participation vector for axial COM mode (mode index 0)
result.modes["axial"].vectors[:, 0]

# Lamb-Dicke parameters for counter-propagating Raman beams
wavelength = 729e-9  # Ca+ S-D transition
k_eff = 2 * 2 * np.pi / wavelength
eta = tiqs.lamb_dicke_parameters(result, species, k_eff, direction="axial")
# eta[i, m] is the Lamb-Dicke parameter for ion i, mode m
```

### Linear-to-Zigzag Transition

The linear chain is stable when $\omega_\text{rad} > \omega_z \cdot c_N$,
where $c_N$ is a critical ratio. For large $N$, $c_N \sim 0.73\, N^{0.86}$.
Beyond this threshold, the chain buckles into a zigzag configuration.

### Penning Trap Modes

In a Penning trap, the three eigenmotions are axial oscillation, modified
cyclotron motion, and magnetron drift. The **axial modes** of an $N$-ion
crystal are computed identically to the Paul trap case: the Coulomb-coupled
Hessian is diagonalized in the axial harmonic potential.

The **transverse modes** (modified cyclotron and magnetron) are qualitatively
different from Paul trap radial modes because the radial dynamics involve the
Coriolis-like coupling from the magnetic field. TIQS currently computes
Penning transverse modes in a **single-particle approximation**: each ion
oscillates independently at the single-particle modified cyclotron frequency
$\omega_+$ or magnetron frequency $\omega_-$ with no inter-ion coupling.
In the returned ``ModeGroup``, ``freqs`` contains $N$ identical entries and
``vectors`` is the $N \times N$ identity matrix (each "mode" is localized
on one ion). A ``UserWarning`` is emitted when $N > 1$ to flag this
approximation. Full $N$-particle transverse mode analysis with
rotating-frame Coulomb coupling is a planned extension.

### References

1. James, D.F.V. "Quantum dynamics of cold trapped ions with application
   to quantum computation." *Appl. Phys. B* **66**, 181 (1998).
2. Marquet, C., Schmidt-Kaler, F. & James, D.F.V. "Phonon-phonon
   interactions due to non-linear effects in a linear ion trap."
   *Appl. Phys. B* **76**, 199 (2003).

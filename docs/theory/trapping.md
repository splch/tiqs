## Paul Trap Physics

A Paul trap confines charged particles using oscillating electric fields that
create a time-averaged restoring force, the **pseudopotential**. Static electric
fields cannot create a three-dimensional potential minimum for a charged particle
(Earnshaw's theorem), so RF fields at radio frequencies are used instead.

### The RF Quadrupole Potential

In a linear Paul trap, four elongated electrodes carry RF voltage to provide
radial confinement, while segmented DC endcap electrodes provide axial
confinement. The ideal 2D quadrupole potential is:

$$
\Phi(x, y, t) = \frac{U_\text{DC} + V_\text{RF}\cos(\Omega_\text{RF} t)}{2 r_0^2}(x^2 - y^2)
$$

where $V_\text{RF}$ is the RF amplitude, $\Omega_\text{RF}$ is the RF drive
frequency (typically 10-100 MHz), and $r_0$ is the ion-to-electrode distance.

### The Mathieu Equation

The ion's equation of motion in each transverse direction reduces to the
**Mathieu equation**:

$$
\frac{d^2 u}{d\xi^2} + \bigl(a_u - 2q_u \cos 2\xi\bigr)\, u = 0
$$

where $\xi = \Omega_\text{RF} t / 2$ and the dimensionless stability parameters
are:

$$
a = \frac{4\, e\, U_\text{DC}}{m\, \Omega_\text{RF}^2\, r_0^2}, \qquad
q = \frac{2\, e\, V_\text{RF}}{m\, \Omega_\text{RF}^2\, r_0^2}
$$

Stable trapping occurs within bounded regions of the $(a, q)$ parameter space.
Most experiments operate in the first stability region with $a \approx 0$ and
$q \approx 0.1$-$0.4$, well below the stability boundary at $q = 0.908$.

### Secular Motion and Micromotion

In the pseudopotential approximation (valid for $q \ll 1$), the ion's motion
decomposes into two components:

**Secular motion**: slow harmonic oscillation at the secular frequency:

$$
\omega_\text{rad} = \frac{q\, \Omega_\text{RF}}{2\sqrt{2}}
$$

This is the "useful" oscillatory motion that serves as the quantum bus, with
typical values $\omega_\text{sec}/2\pi \sim 1$-$5$ MHz.

**Micromotion**: fast, driven oscillation at $\Omega_\text{RF}$, with
amplitude proportional to the ion's displacement from the RF null. At the exact
trap center, micromotion vanishes. Stray DC fields push ions off-center,
causing "excess micromotion" that broadens spectral lines and degrades gate
fidelities. Compensation is achieved by precisely nulling stray fields with
DC electrodes.

The axial secular frequency from static endcap confinement is:

$$
\omega_z = \sqrt{\frac{2\, e\, \kappa\, U_\text{end}}{m\, z_0^2}}
$$

### Pseudopotential and Trap Depth

The time-averaged pseudopotential is:

$$
\Psi_\text{pseudo}(\mathbf{r}) = \frac{e^2 |\nabla \Phi_\text{RF}|^2}{4\, m\, \Omega_\text{RF}^2}
= \frac{1}{2} m\, \omega_\text{rad}^2 (x^2 + y^2)
$$

Typical trap depths are $\sim 0.1$-$10$ eV, far exceeding the ions' thermal
energy after laser cooling ($\sim 10^{-4}$ eV), enabling confinement for hours
or days.

### Trap Geometries

**Linear Paul traps** use four rod or blade electrodes for radial RF
confinement, with DC endcap electrodes providing weaker axial confinement.
Ions line up along the RF null axis forming a 1D Coulomb crystal with typical
inter-ion spacings of 2-10 $\mu$m. Radial frequencies must exceed axial
frequencies to prevent the chain from buckling into a zigzag.

**Surface-electrode (planar) traps** place all electrodes in a single plane,
with ions trapped 30-150 $\mu$m above the surface. Compatible with
semiconductor lithography, they enable complex multi-zone QCCD architectures.
The main challenge is **anomalous motional heating**, which scales as
$\sim d^{-4}$ with ion-electrode distance $d$ and is suppressed $\sim 100\times$
by cryogenic operation at 4-15 K.

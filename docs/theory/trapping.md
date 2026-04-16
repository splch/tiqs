## Trapping Physics

TIQS supports two trap architectures: **Paul traps** using oscillating
electric fields for radial confinement, and **Penning traps** using a static
magnetic field for radial confinement with an electrostatic axial potential.
Both conform to the ``Trap`` protocol, which requires ``omega_axial``,
``species``, and ``is_stable()``.

### The Trap Protocol

TIQS defines a structural ``Trap`` protocol that any trap must satisfy:

```python
class Trap(Protocol):
    @property
    def omega_axial(self) -> float: ...

    @property
    def species(self) -> Species: ...

    def is_stable(self) -> bool: ...
```

Functions like ``equilibrium_positions()`` accept any ``Trap``-conforming
object. ``normal_modes()`` currently requires ``PaulTrap`` or ``PenningTrap``
because transverse mode computation depends on trap-specific physics.

Both ``PaulTrap`` and ``PenningTrap`` can be constructed either from a known
axial frequency or from a DC voltage via the ``from_dc_voltage()`` class
method:

```python
import numpy as np
import tiqs

species = tiqs.get_species("Ca40")

# Construct with known axial frequency
trap = tiqs.PaulTrap(
    v_rf=200.0, omega_rf=2*np.pi*30e6,
    r0=200e-6, species=species,
    omega_axial=2*np.pi*1e6,
)
trap.is_stable()  # True

# Construct from DC endcap voltage
trap = tiqs.PaulTrap.from_dc_voltage(
    v_rf=200.0, omega_rf=2*np.pi*30e6,
    r0=200e-6, species=species,
    u_dc_axial=10.0,
)
```

### Paul Traps

A Paul trap confines charged particles using oscillating electric fields that
create a time-averaged restoring force, the **pseudopotential**. Static electric
fields cannot create a three-dimensional potential minimum for a charged particle
(Earnshaw's theorem), so RF fields at radio frequencies are used instead.

#### The RF Quadrupole Potential

In a linear Paul trap, four elongated electrodes carry RF voltage to provide
radial confinement, while segmented DC endcap electrodes provide axial
confinement. The ideal 2D quadrupole potential is:

$$
\Phi(x, y, t) = \frac{U_\text{DC} + V_\text{RF}\cos(\Omega_\text{RF} t)}{2 r_0^2}(x^2 - y^2)
$$

where $V_\text{RF}$ is the RF amplitude, $\Omega_\text{RF}$ is the RF drive
frequency (typically 10-100 MHz), and $r_0$ is the ion-to-electrode distance.

#### The Mathieu Equation

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

TIQS computes $q$ from the RF parameters directly. For $a$, it uses a
simplified model that accounts for radial defocusing from the axial DC
potential:

$$
a \approx \frac{-2\,\omega_z^2}{\Omega_\text{RF}^2}
$$

This is negative because the axial DC field weakens radial confinement.

Stable trapping occurs within bounded regions of the $(a, q)$ parameter space.
Most experiments operate in the first stability region with $a \approx 0$ and
$q \approx 0.1$-$0.4$, well below the stability boundary at $q = 0.908$.

#### Secular Motion and Micromotion

In the pseudopotential approximation (valid for $q \ll 1$), the ion's motion
decomposes into two components:

**Secular motion**: slow harmonic oscillation at the secular frequency. The
full expression, which TIQS implements, includes the effect of the DC axial
potential on the radial motion:

$$
\omega_\text{rad} = \frac{\Omega_\text{RF}}{2}\sqrt{a + \frac{q^2}{2}}
$$

For typical operating parameters where $|a| \ll q^2/2$, this simplifies to:

$$
\omega_\text{rad} \approx \frac{q\, \Omega_\text{RF}}{2\sqrt{2}}
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
\omega_z = \sqrt{\frac{\kappa\, e\, U_\text{end}}{m\, z_0^2}}
$$

#### Pseudopotential and Trap Depth

The time-averaged pseudopotential is:

$$
\Psi_\text{pseudo}(\mathbf{r}) = \frac{e^2 |\nabla \Phi_\text{RF}|^2}{4\, m\, \Omega_\text{RF}^2}
= \frac{1}{2} m\, \omega_\text{rad}^2 (x^2 + y^2)
$$

Typical trap depths are $\sim 0.1$-$10$ eV, far exceeding the ions' thermal
energy after laser cooling ($\sim 10^{-4}$ eV), enabling confinement for hours
or days.

#### Trap Geometries

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

### Penning Traps

A Penning trap confines charged particles using a **static, uniform magnetic
field** for radial confinement and a **static electric quadrupole** for axial
confinement. Unlike Paul traps, there is no time-varying RF field and hence
no micromotion.

#### Axial Confinement

The electrostatic potential between hyperbolic or cylindrical electrodes
creates a harmonic axial well:

$$
\omega_z = \sqrt{\frac{e\, V_\mathrm{dc}}{m\, d^2}}
$$

where $d$ is the characteristic trap dimension
($d^2 = (z_0^2 + r_0^2/2)/2$ for a hyperbolic trap) and $V_\mathrm{dc}$ is
the DC trapping voltage. TIQS provides ``PenningTrap.from_dc_voltage()`` to
construct from voltage, or you can pass ``omega_axial`` directly:

```python
import numpy as np
import tiqs

species = tiqs.ElectronSpecies(magnetic_field=1.0)
trap = tiqs.PenningTrap(
    magnetic_field=1.0, species=species,
    d=1e-3, omega_axial=2*np.pi*100e6,
)
trap.is_stable()  # True
```

#### Radial Confinement

The axial magnetic field $B$ causes the particle to undergo cyclotron motion.
The **free cyclotron frequency** is:

$$
\omega_c = \frac{eB}{m}
$$

The combination of cyclotron motion and electrostatic defocusing in the radial
plane produces two eigenmotions with distinct frequencies:

**Modified cyclotron** (fast circular orbit):

$$
\omega_+ = \frac{\omega_c}{2}
+ \sqrt{\left(\frac{\omega_c}{2}\right)^2 - \frac{\omega_z^2}{2}}
$$

**Magnetron** (slow $\mathbf{E} \times \mathbf{B}$ drift orbit):

$$
\omega_- = \frac{\omega_c}{2}
- \sqrt{\left(\frac{\omega_c}{2}\right)^2 - \frac{\omega_z^2}{2}}
$$

These satisfy the useful identities:

$$
\omega_+ + \omega_- = \omega_c, \qquad
\omega_+\,\omega_- = \frac{\omega_z^2}{2}, \qquad
\omega_+^2 + \omega_-^2 + \omega_z^2 = \omega_c^2
$$

The last relation is the **Brown-Gabrielse invariance theorem**, which
allows precision measurements of the cyclotron frequency without measuring
all three eigenfrequencies individually. TIQS exposes all three as
properties: ``omega_cyclotron``, ``omega_modified_cyclotron``, and
``omega_magnetron``.

#### Stability Condition

Confinement requires the discriminant to be positive:

$$
\left(\frac{\omega_c}{2}\right)^2 > \frac{\omega_z^2}{2}
\qquad\Longleftrightarrow\qquad
\omega_c > \sqrt{2}\,\omega_z
$$

If this is violated, the magnetron frequency becomes imaginary and radial
confinement is lost.

#### Advantages for Electrons

Penning traps are particularly well-suited for trapping **bare electrons**:

- No micromotion eliminates a major source of decoherence.
- Magnetic field strengths of 1-5 T provide strong radial confinement
  for light particles.
- Axial frequencies of 50-200 MHz are achievable, enabling fast gate
  operations.
- The spin qubit frequency ($\omega_s = g_e \mu_B B / \hbar$) is set by
  the same magnetic field that provides confinement.

### References

1. Leibfried, D. et al. "Quantum dynamics of single trapped ions."
   *Rev. Mod. Phys.* **75**, 281 (2003).
2. Brown, L.S. & Gabrielse, G. "Geonium theory: Physics of a single
   electron or ion in a Penning trap." *Rev. Mod. Phys.* **58**, 233 (1986).
3. Jain, S. et al. "Penning micro-trap for quantum computing."
   *Nature* **627**, 510 (2024).
4. Wineland, D.J. et al. "Experimental issues in coherent quantum-state
   manipulation of trapped atomic ions." *J. Res. NIST* **103**, 259 (1998).
5. Mintert, F. & Wunderlich, C. "Ion-trap quantum logic using
   long-wavelength radiation." *Phys. Rev. Lett.* **87**, 257904 (2001).
6. Ball, H. et al. "Site-resolved imaging of beryllium ion crystals in a
   high-optical-access Penning trap with inbore optomechanics."
   *Rev. Sci. Instrum.* **90**, 053103 (2019).

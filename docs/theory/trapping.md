## Trapping Physics

Two trap architectures dominate charged-particle quantum computing:
**Paul traps** using oscillating electric fields for radial confinement, and
**Penning traps** using a static magnetic field for radial confinement with
an electrostatic axial potential.

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
\Phi(x, y, t) = \frac{U_\text{rod} + V_\text{RF}\cos(\Omega_\text{RF} t)}{2 r_0^2}(x^2 - y^2)
$$

where $V_\text{RF}$ is the RF amplitude, $\Omega_\text{RF}$ is the RF drive
frequency (typically 10-100 MHz), $U_\text{rod}$ is any static offset on the
RF rods, and $r_0$ is the ion-to-electrode distance.

$r_0$ is an **effective** distance, not a purely geometric one. Every
expression below assumes the electrodes lie on equipotentials of
$\Phi \propto (x^2 - y^2)/r_0^2$; real rods and blades do not, so the field
carries a geometric efficiency factor of order unity that must be absorbed
into $r_0$. Wineland et al. state the caveat after their Eq. (1) ("unless the
rods conform to equipotentials ... this equation must be multiplied by a
constant factor on the order of 1"), Leibfried et al. keep it as explicit
parameters $\alpha, \alpha'$ in RMP Eq. (7), and Berkeland et al. distinguish
an effective $R'$ from the geometric $R$. `PaulTrap.r0` documents the same
point; for the PCB slot trap in the TIQS electron tests, the ideal-geometry
value underestimates the measured radial frequency by 18%.

#### The Mathieu Equation

The ion's equation of motion in each transverse direction reduces to the
**Mathieu equation**:

$$
\frac{d^2 u}{d\xi^2} + \bigl(a_u + 2q_u \cos 2\xi\bigr)\, u = 0
$$

where $\xi = \Omega_\text{RF} t / 2$. The $+2q$ sign convention is the one
used by Wineland et al. Eq. (3), Berkeland et al. Eq. (4) and Leibfried et
al. RMP Eq. (17), and it is the convention under which the following $a$ and
$q$ are both positive-signed as written:

$$
a_\text{rod} = \frac{4\, e\, U_\text{rod}}{m\, \Omega_\text{RF}^2\, r_0^2},
\qquad
q = \frac{2\, e\, V_\text{RF}}{m\, \Omega_\text{RF}^2\, r_0^2}
$$

Two physically distinct DC contributions add into $a$, and they have
different symmetry:

- **Rod DC** ($U_\text{rod}$, above) is *antisymmetric*:
  $a_x = -a_y = a_\text{rod}$. TIQS sets it to zero -- `PaulTrap` has no
  rod-DC parameter.
- **Endcap DC** (the axial confinement, below) is *symmetric* and radially
  defocusing: $a_x = a_y = -a_z/2 < 0$.

Wineland et al. Eq. (3) keeps both in one expression,
$a_x = (4e/m\Omega_\text{RF}^2)\bigl(U_\text{rod}/r_0^2 -
\kappa U_\text{end}/z_0^2\bigr)$. With TIQS' rod DC at zero, and using the
axial relation of the next section, the endcap term is *exactly*

$$
a = \frac{-4\, e\, \kappa\, U_\text{end}}{m\, \Omega_\text{RF}^2\, z_0^2}
  = \frac{-2\,\omega_z^2}{\Omega_\text{RF}^2}
$$

This is an identity for the ideal Laplace geometry, not an approximation --
`PaulTrap.mathieu_a` implements the right-hand form.

Stable trapping occurs within bounded regions of the $(a, q)$ parameter space.
Most experiments operate in the first stability region with $a \approx 0$ and
$q \approx 0.1$-$0.4$, well below the stability boundary at $q = 0.908$.

`PaulTrap.is_stable()` tests $0 < q < 0.908$ together with
$\beta^2 = a + q^2/2 > 0$. Both are **lowest-order** conditions, not
refinements of the exact criterion. The exact condition on the Mathieu
characteristic exponent is $0 \le \beta \le 1$, with $\beta$ from the
continued fraction of Leibfried et al. RMP Eqs. (11)-(13);
$\beta^2 = a + q^2/2$ is only its leading term, and $q < 0.908$ is the exact
boundary only at $a = 0$. Near either boundary the classification can be
wrong in both directions: at $(q, a) = (0.4, -0.0793)$ the exact motion is
unstable although $\beta^2 > 0$, and at $(q, a) = (0.95, -0.10)$ it is stable
although the $q$ cut rejects it.

#### Secular Motion and Micromotion

In the pseudopotential approximation (valid for $q \ll 1$), the ion's motion
decomposes into two components:

**Secular motion**: slow harmonic oscillation at the secular frequency. The
full expression includes the effect of the DC axial potential on the radial
motion:

$$
\omega_\text{rad} = \frac{\Omega_\text{RF}}{2}\sqrt{a + \frac{q^2}{2}}
$$

For typical operating parameters where $|a| \ll q^2/2$, this simplifies to
the RF-only secular frequency

$$
\omega_{r,\text{RF}} = \frac{q\, \Omega_\text{RF}}{2\sqrt{2}}
$$

Note that $|a| \ll q$ is *not* sufficient: the standard TIQS Yb-171 fixture
has $|a|/q = 0.017$ but $|a|/(q^2/2) = 0.28$, and dropping $a$ there
overestimates $\omega_\text{rad}$ by 17%.

Secular motion is the "useful" oscillation that serves as the quantum bus,
with typical values $\omega_\text{rad}/2\pi \sim 1$-$5$ MHz.

**Micromotion**: fast, driven oscillation at $\Omega_\text{RF}$, with
amplitude proportional to the ion's displacement from the RF null. At the exact
trap center, micromotion vanishes. Stray DC fields push ions off-center,
causing "excess micromotion" that broadens spectral lines and degrades gate
fidelities. Compensation is achieved by precisely nulling stray fields with
DC electrodes.

TIQS' `micromotion_amplitude()` and `stray_field_displacement()` implement
Berkeland et al. Eqs. (15) and (16), but they are **diagnostics only**. The
pseudopotential approximation time-averages micromotion away, so no
Hamiltonian, Rabi frequency, or Lamb-Dicke parameter in this package carries
a micromotion correction, and no micromotion channel appears in
`compute_error_budget`. To include it by hand, scale a carrier Rabi
frequency by $J_0(k_\text{eff} x_\text{mm})$ and the $n$-th RF sideband by
$J_n(k_\text{eff} x_\text{mm})$.

The axial secular frequency from static endcap confinement is:

$$
\omega_z = \sqrt{\frac{2\,\kappa\, e\, U_\text{end}}{m\, z_0^2}}
$$

The factor 2 follows from Laplace's equation. The only harmonic quadrupole
with axial curvature $\kappa U_\text{end}/z_0^2$ is
$\Phi_\text{dc} = (\kappa U_\text{end}/z_0^2)\bigl[z^2 - (x^2+y^2)/2\bigr]$,
so $m\ddot{z} = -e\,\partial_z \Phi_\text{dc} =
-2 e \kappa U_\text{end} z / z_0^2$. See Wineland et al. Eq. (2) and
Berkeland et al. Eqs. (5) and (9). `PaulTrap.from_dc_voltage()` and
`PaulTrap.u_dc_axial` implement this relation and its exact inverse,
$U_\text{end} = m\omega_z^2 z_0^2/(2\kappa e)$.

#### Pseudopotential and Trap Depth

The time-averaged RF pseudopotential is:

$$
\Psi_\text{pseudo}(\mathbf{r}) = \frac{e^2 |\nabla \Phi_\text{RF}|^2}{4\, m\, \Omega_\text{RF}^2}
= \frac{1}{2} m\, \omega_{r,\text{RF}}^2 (x^2 + y^2)
$$

The second equality holds with the **RF-only** secular frequency
$\omega_{r,\text{RF}} = q\Omega_\text{RF}/2\sqrt{2}$, not with
$\omega_\text{rad}$: the pseudopotential is generated by the RF field alone,
and the DC term contributes separately. `pseudopotential_depth_eV`
accordingly returns this expression evaluated **at the electrode radius**
$r = r_0$, which reduces to $e\,q\,V_\text{RF}/8$ (Wineland et al. Eq. (6)).
Substituting the $a$-inclusive $\omega_\text{rad}$ instead is a different
quantity -- on the standard Yb-171 fixture ($q = 0.127$, $a = -0.00222$) it
gives 11.52 eV against the RF-only 15.89 eV, 27% lower -- and because the
DC axial confinement is radially defocusing ($a < 0$), the true radial well
depth is indeed the lower of the two.

Typical trap depths are $\sim 0.1$-$10$ eV, far exceeding the ions' thermal
energy after laser cooling. The Doppler limit for the species TIQS ships is
$T_D \approx 0.43$-$0.54$ mK (`doppler_limit_temperature()`), i.e.
$k_B T_D \sim 4 \times 10^{-8}$ eV -- around $10^{-7}$ eV, six to eight
orders of magnitude below the well depth, which is why ions stay confined for
hours or days. (For scale, $10^{-4}$ eV would be $T = 1.2$ K, and room
temperature is $0.026$ eV.)

#### Trap Geometries

**Linear Paul traps** use four rod or blade electrodes for radial RF
confinement, with DC endcap electrodes providing weaker axial confinement.
Ions line up along the RF null axis forming a 1D Coulomb crystal with typical
inter-ion spacings of 2-10 $\mu$m.

Radial confinement must exceed axial confinement to prevent the chain from
buckling into a zigzag, and the required margin **grows with the number of
ions**. Wineland et al. give $\omega_\text{rad}/\omega_z > 1$ for two ions
and $> 1.55$ for three, with the estimate
$(\omega_\text{rad}/\omega_z)_c \approx 0.73\, N^{0.86}$ for larger chains
(Steane's Eq. (11)). Evaluated: 1.32 at $N = 2$, 1.88 at $N = 3$, 2.91 at
$N = 5$, 5.29 at $N = 10$. That fit is a conservative overestimate of the
exact threshold; `tiqs.normal_modes()` tests the exact criterion directly
from the radial dynamical matrix and raises when the chain has buckled. See
[normal_modes.md](normal_modes.md) for both.

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

The electrostatic potential between the electrodes creates a harmonic axial
well:

$$
\omega_z = \sqrt{\frac{e\, C_2\, V_\mathrm{dc}}{m\, d^2}}
$$

where $d$ is the characteristic trap dimension and $C_2$ is the quadrupole
coefficient of the expansion
$V = (V_\text{dc}/2)\sum_k C_k (r/d)^k P_k(\cos\theta)$.

For **ideal hyperbolic** electrodes $C_2 = 1$ and
$d^2 = (z_0^2 + r_0^2/2)/2$. `PenningTrap.from_dc_voltage()` and
`PenningTrap.v_dc` implement exactly that case, with `v_dc` the magnitude
$|V_0|$ -- the polarity that produces axial confinement is set by the sign of
the charge, opposite for the electron.

For **cylindrical** and multi-ring geometries $C_2$ is design dependent and
generally below 1, and must be obtained numerically (Gabrielse & Mackintosh).
Ignoring it is not a small error: for the published cylindrical trap with
$\rho_0 = 4.5$ mm, $2z_0 = 7.7$ mm and $\nu_z = 200$ MHz, the ideal formula
demands $V_0 = 112.0$ V where the measured value is 101.4 V -- 10.4% high in
voltage, 4.8% low in frequency, i.e. $C_2 \approx 0.91$. Ball et al. obtained
$C_2$ for their multi-ring trap by solving Laplace's equation on a numerical
grid.

#### Radial Confinement

The axial magnetic field $B$ causes the particle to undergo cyclotron motion.
The **free cyclotron frequency** is:

$$
\omega_c = \frac{eB}{m}
$$

TIQS fixes the charge magnitude at $e$: only singly charged ions and the
electron are representable. Note that this is the free-space value, not the
trap-shifted $\bar\omega_c \equiv \omega_+$ that precision experiments
report.

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

These satisfy the identities:

$$
\omega_+ + \omega_- = \omega_c, \qquad
\omega_+\,\omega_- = \frac{\omega_z^2}{2}, \qquad
\omega_+^2 + \omega_-^2 + \omega_z^2 = \omega_c^2
$$

The last relation is the **Brown-Gabrielse invariance theorem**, and the
first two are *not* on the same footing. The sum and product rules hold only
for the ideal trap; the invariance theorem stays exact when the magnetic
field is tilted with respect to the electrode axis and when the quadrupole
has an ellipticity, because its proof needs only Laplace's equation and a
uniform $B$. With a 10 degree tilt and 0.1 ellipticity the quadrature sum
reproduces $\omega_c^2$ to $5\times 10^{-15}$ while the sum rule is off by
$1\times10^{-5}$ and the product rule by $6\times10^{-2}$.

That robustness is the point of the theorem, and it means precision
experiments measure **all three** shifted eigenfrequencies
$\bar\omega_+, \bar\omega_z, \bar\omega_-$ and combine them in quadrature to
recover the free-space cyclotron frequency, rather than avoiding one of them.
(Electron $g$-2 work can skip the magnetron by using the theorem's expansion
$\omega_c \approx \bar\omega_+ + \bar\omega_z^2/2\bar\omega_+$, but that is a
corollary of measuring all three, not an alternative to it.) TIQS implements
only the ideal trap, so all three relations are trivially exact here.

#### The Magnetron Mode Carries Negative Energy

The radial electrostatic force is *defocusing*, so the magnetron term enters
the Hamiltonian with the opposite sign to the other two:

$$
H = \hbar\omega_+\left(n_+ + \tfrac{1}{2}\right)
  + \hbar\omega_z\left(n_z + \tfrac{1}{2}\right)
  - \hbar\omega_-\left(n_- + \tfrac{1}{2}\right)
$$

Three consequences follow, and none of them is optional bookkeeping:

- The magnetron motion is only **metastable**: its total energy *decreases*
  as the orbit radius grows, so the drift orbit expands toward the electrodes
  if left alone.
- The roles of the upper and lower motional sidebands are **exchanged**.
  Dehmelt put it plainly: "the roles of upper and lower side-bands are
  reversed here from the case of a particle in a well where the energy
  increases with amplitude because the magnetron motion is metastable and the
  total energy of this motion decreases with radius." Removing a magnetron
  quantum requires the **blue**-detuned tone.
- Cooling it therefore means *raising* $n_-$, and in practice requires
  coupling it to the modified-cyclotron mode. Jain et al. use "a weak
  axialization rf quadrupolar electric field ... at the bare cyclotron
  frequency, which resonantly couples the magnetron and modified-cyclotron
  motions."

`PenningTrap.omega_magnetron` returns the unsigned $\omega_-$ and documents
this, but **nothing downstream applies the sign for you**. Do not hand it to
code that assumes an ascending ladder -- `HarmonicPotential`,
`mode_hamiltonian`, `full_interaction_hamiltonian`'s `mode_frequency`, or
`sideband_cooling_nbar` -- without flipping the sign yourself. The same
warning applies to `normal_modes()`'s `"magnetron"` `ModeGroup`; see
[normal_modes.md](normal_modes.md).

#### Stability Condition

Confinement requires the discriminant to be positive:

$$
\left(\frac{\omega_c}{2}\right)^2 > \frac{\omega_z^2}{2}
\qquad\Longleftrightarrow\qquad
\omega_c > \sqrt{2}\,\omega_z
$$

If this is violated, **both** radial eigenfrequencies acquire imaginary
parts,

$$
\omega_\pm = \frac{\omega_c}{2}
\pm i\sqrt{\frac{\omega_z^2}{2} - \left(\frac{\omega_c}{2}\right)^2}
$$

(a complex-conjugate pair with real part $\omega_c/2 \neq 0$, not a purely
imaginary root), the radial motion spirals outward, and confinement is lost.
`PenningTrap.is_stable()` and the transverse-frequency properties share one
predicate: the discriminant must be strictly positive, so the exactly
critical trap $\omega_c = \sqrt{2}\,\omega_z$ is rejected too, where
$\omega_+$ and $\omega_-$ are degenerate.

#### Advantages for Electrons

Penning traps are particularly well-suited for trapping **bare electrons**:

- No micromotion eliminates a major source of decoherence.
- Magnetic field strengths of 1-5 T provide strong radial confinement
  for light particles.
- Axial frequencies of 50-200 MHz are achievable, enabling fast gate
  operations.
- The spin qubit frequency ($\omega_s = g_e \mu_B B / \hbar$) is set by
  the same magnetic field that provides confinement.

### Model Scope and Approximations

- **Pseudopotential only.** TIQS never integrates the Mathieu equation. All
  Paul-trap dynamics use the time-averaged secular frequency, so micromotion
  is a diagnostic (above) rather than a coupling: no Rabi frequency,
  Lamb-Dicke parameter, or error-budget channel carries a micromotion
  correction.
- **Ideal quadrupole geometry.** $q$, $\omega_\text{rad}$ and the trap depth
  hold for electrodes on equipotentials; real geometries need the efficiency
  factor absorbed into $r_0$ (above). No surface-trap or segmented-trap
  geometry classes exist, so trap-design quantities are not computed from
  electrode layouts.
- **Lowest-order stability.** `is_stable()` is the leading-order Floquet
  criterion, and can misclassify near either boundary (above).
- **Symmetric radial DC.** `PaulTrap` hard-codes $a_x = a_y$, so the two
  radial directions are exactly degenerate; there is no rod-DC or
  radial-asymmetry parameter with which to split them.
- **Ideal Penning trap.** $C_2 = 1$, no tilt, no ellipticity, no higher
  multipoles $C_4, C_6, \ldots$, so all three invariance identities are
  exact here and the anharmonicity compensation of real traps has no
  analogue. The magnetron energy sign is documented but not encoded (above).
- **Singly charged particles.** The charge magnitude is fixed at $e$
  throughout; there is no `charge_state` on the species.

### References

1. Leibfried, D. et al. "Quantum dynamics of single trapped ions."
   *Rev. Mod. Phys.* **75**, 281 (2003).
2. Brown, L.S. & Gabrielse, G. "Geonium theory: Physics of a single
   electron or ion in a Penning trap." *Rev. Mod. Phys.* **58**, 233 (1986).
3. Jain, S. et al. "Penning micro-trap for quantum computing."
   *Nature* **627**, 510 (2024).
4. Wineland, D.J. et al. "Experimental issues in coherent quantum-state
   manipulation of trapped atomic ions." *J. Res. NIST* **103**, 259 (1998).
5. Berkeland, D.J. et al. "Minimization of ion micromotion in a Paul trap."
   *J. Appl. Phys.* **83**, 5025 (1998).
6. Steane, A.M. "The ion trap quantum information processor."
   *Appl. Phys. B* **64**, 623 (1997).
7. Mintert, F. & Wunderlich, C. "Ion-trap quantum logic using
   long-wavelength radiation." *Phys. Rev. Lett.* **87**, 257904 (2001).
8. Dehmelt, H. "Experiments with an isolated subatomic particle at rest."
   Nobel lecture (1989).
9. Gabrielse, G. & Mackintosh, F.C. "Cylindrical Penning traps with
   orthogonalized anharmonicity compensation."
   *Int. J. Mass Spectrom. Ion Processes* **57**, 1 (1984).
10. Ball, H. et al. "Site-resolved imaging of beryllium ion crystals in a
    high-optical-access Penning trap with inbore optomechanics."
    *Rev. Sci. Instrum.* **90**, 053103 (2019).

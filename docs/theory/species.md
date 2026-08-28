## Ion Species and Qubit Encoding

The choice of ion species determines nearly every engineering decision --
laser wavelengths, qubit coherence, gate mechanisms, and scalability. All
species used are singly-charged, alkaline-earth-like atoms with a single
valence electron.

### Qubit Encoding Types

**Hyperfine qubits** use two hyperfine levels of the electronic ground state,
typically $m_F = 0$ "clock" states that are first-order insensitive to
magnetic field fluctuations. Both qubit states live in the ground state, so
$T_1$ is effectively infinite and memory coherence times reach thousands
of seconds (Wang et al., 2021 report an *estimated*
$T_2 \approx 5500$ s for ${}^{171}\text{Yb}^+$ -- the time constant of a
fit extrapolated beyond the measured range, not a directly observed
decay).

**Optical qubits** encode between the ground state and a metastable excited
state connected by a narrow electric-quadrupole transition. A single
narrow-linewidth laser directly drives rotations. The trade-off is finite
$T_1$ limited by the metastable state lifetime ($\sim 1$ s).

**Zeeman qubits** use magnetic sublevels within the same manifold. Their
linear sensitivity to magnetic field noise limits coherence to milliseconds
without dynamical decoupling.

### Species Comparison

| Species | Qubit type | Splitting | Cooling $\lambda$ | Key advantage |
|---------|-----------|-----------|-------------------|---------------|
| ${}^{171}\text{Yb}^+$ | Hyperfine | 12.6428 GHz | 369.526 nm | Long coherence (estimated $T_2 \approx 5500$ s); mature ecosystem |
| ${}^{40}\text{Ca}^+$ | Optical (also run as Zeeman) | 729.347 nm transition | 396.959 nm | Simple level structure; all diode lasers |
| ${}^{43}\text{Ca}^+$ | Hyperfine | 3.2256 GHz | 396.959 nm | Record single-qubit fidelity ($1.5 \times 10^{-7}$); $T_2^* \approx 50$ s |
| ${}^{137}\text{Ba}^+$ | Hyperfine | 8.0377 GHz | 493.545 nm | All visible wavelengths; scalable photonics |
| ${}^{9}\text{Be}^+$ | Hyperfine | 1.2500 GHz | 313.133 nm | Lightest ion; fastest gates |
| ${}^{88}\text{Sr}^+$ | Optical | 674.026 nm transition | 421.671 nm | Quantum networking (fiber-friendly photons) |

All wavelengths in this chapter and in `tiqs.species` are **vacuum**
values from the NIST Atomic Spectra Database, to 6-7 significant
figures. Masses follow the NIST/AME2020 evaluation: `IonSpecies.mass_amu`
is the **neutral-atom** relative atomic mass, while the mass used by the
dynamics, `IonSpecies.mass_kg`, subtracts one electron
($m = m_\text{amu}u - m_e$). Ignoring that electron would overestimate
the mass by $3.2\times10^{-6}$ (Yb-171) to $6.1\times10^{-5}$ (Be-9)
relative, and any $\omega \propto m^{-1/2}$ by half as much.

### Key Atomic Properties

**${}^{171}\text{Yb}^+$** (nuclear spin $I = 1/2$): Qubit states
$|F{=}0, m_F{=}0\rangle$ and $|F{=}1, m_F{=}0\rangle$ in the ${}^2S_{1/2}$
manifold, split by 12.642812118 GHz. Cooling via
${}^2S_{1/2} \to {}^2P_{1/2}$ at 369.5262 nm (total upper-state
linewidth $\Gamma/2\pi = 19.6$ MHz). Repumper at 935.187 nm clears the
metastable ${}^2D_{3/2}$ state. Gates driven by stimulated Raman
transitions via 355 nm pulsed lasers or direct microwave drive at
12.6 GHz. Used by IonQ (Forte) and Quantinuum (H1, H2).

**${}^{40}\text{Ca}^+$** (zero nuclear spin): Optical qubit between $4S_{1/2}$
and metastable $3D_{5/2}$ (lifetime 1.168 s) at 729.347 nm. Doppler
cooling at 396.959 nm, repumping at 866.452 nm ($3D_{3/2}$) and
854.444 nm ($3D_{5/2}$ clear-out). All wavelengths accessible with diode
lasers. Used by the Innsbruck group and AQT. The same isotope is also
run as a *Zeeman* qubit on the $4S_{1/2}$ sublevels
($\omega_0 \approx 2\pi \times 240$ MHz at $B_0 \approx 8.5$ mT) by
Oxford Ionics / IonQ, which is the encoding behind their published
$8.4\times10^{-6}$ single-qubit and $8.4(7)\times10^{-5}$ two-qubit
errors.

**${}^{43}\text{Ca}^+$** ($I = 7/2$): Clock qubit at 3.2256 GHz with
long memory coherence, $T_2^* \approx 50$ s (Harty et al., 2014; the
same platform now reports $T_2 \approx 70$ s), and the record
single-qubit gate error $1.5(4) \times 10^{-7}$ per Clifford with
microwave-driven gates (Smith et al., 2024). Note that $T_2^*$
(free-induction) and the dynamically-decoupled $T_2$ quoted for
${}^{171}\text{Yb}^+$ above are different quantities and are not
directly comparable. Only 0.135% natural abundance requires
isotope-selective loading. Used by the University of Oxford ion-trap
group (Clarendon Laboratory).

**${}^{137}\text{Ba}^+$** ($I = 3/2$): All primary wavelengths in the visible
spectrum: cooling at 493.5454 nm, repumping at 649.869 nm ($5D_{3/2}$)
and 614.341 nm ($5D_{5/2}$ clear-out). The $5D_{5/2}$ metastable state
has a $\sim 30$ s lifetime for high-fidelity electron shelving; TIQS
uses 31.2(9) s (Auchter et al., 2014), but published values span
roughly 25.6-31.2 s -- Mohanty et al. (2015) report 26.4(1.7) s and
theory gives 29.8(3) s -- so treat it as good to $\sim 20\%$.
Quantinuum's Helios processor (2025) was the first commercial system
using ${}^{137}\text{Ba}^+$.

**${}^{9}\text{Be}^+$** ($I = 3/2$): Clock qubit at 1.250017674 GHz;
cooling on ${}^2S_{1/2} \to {}^2P_{3/2}$ at 313.133 nm. TIQS uses
$\Gamma/2\pi = 17.97$ MHz for that line, from the NIST
$A_{ki} = 1.1292\times10^8$ s$^{-1}$ (the ${}^2P_{3/2}$ state has no
other decay channel); the widely propagated 19.4 MHz is $\sim 8\%$
higher and is not supported by the transition probability. The
difference matters: it moves the Doppler limit from 466 $\mu$K to
431 $\mu$K and rescales every sympathetic-cooling rate that consumes
$\Gamma$.

### The Species Protocol

TIQS defines a structural `Species` protocol:

```python
class Species(Protocol):
    @property
    def mass_kg(self) -> float: ...

    @property
    def qubit_frequency_hz(self) -> float: ...
```

Both `IonSpecies` and `ElectronSpecies` satisfy this protocol. Any custom
class exposing these two properties will be accepted by TIQS functions that
take a `Species` argument (e.g. `lamb_dicke_parameters()`).

### Trade-offs

The fundamental trade-off: **heavier ions** (Ba$^+$, Yb$^+$) offer convenient
wavelengths and long-lived states but slower gate speeds, while **lighter
ions** (Be$^+$, Mg$^+$) enable faster dynamics but demand challenging UV
optics. The mass scaling of the secular frequencies is not uniform:
at fixed DC voltage the axial mode scales as
$\omega_z \propto 1/\sqrt{m}$, while at fixed RF amplitude the radial
modes scale as $\omega_r \propto 1/m$, since the Mathieu parameter
$q \propto V_\text{rf}/(m\Omega_\text{rf}^2 r_0^2)$ (see
[trapping](trapping.md)). The 2025-2026 industry trend strongly favors
barium for its visible-wavelength scalability.

### Trapped Electrons

Bare electrons confined in Paul traps or Penning traps are a candidate
platform for quantum computing. The qubit is the electron spin-1/2 in an
applied magnetic field, with Zeeman splitting
$f = g_e \mu_B B / h = 28.02$ GHz/T. Unlike atomic
ions, electrons have no internal level structure, so cooling is resistive
(via an RLC tank circuit) and spin-motion coupling is mediated by a magnetic
field gradient rather than a laser wavevector. TIQS models trapped electrons
via `tiqs.ElectronSpecies`.

| Property | Electron | Typical ion (${}^{40}\text{Ca}^+$) |
|----------|----------|---------------------|
| Mass | $9.1 \times 10^{-31}$ kg | $6.6 \times 10^{-26}$ kg |
| Qubit encoding | Spin Zeeman | Optical or hyperfine |
| Qubit frequency | Tunable via $B$ at 28.02 GHz/T ($\sim$1-30 GHz below 1.07 T; $\sim$28-140 GHz over the Penning range) | Fixed: $\sim$1-13 GHz (hyperfine) or $\sim$411 THz (optical) |
| $T_1$ | $\infty$ (no decay channel) | $\infty$ (hyperfine) or $\sim$1 s (optical) |
| Cooling | Resistive (RLC circuit) | Laser Doppler + sideband |
| Spin-motion coupling | Magnetic field gradient | Laser wavevector |
| Trap type | Paul (GHz RF) or Penning (1-5 T) | Paul (10-100 MHz RF) |
| Secular frequencies | 30 MHz - 2 GHz | 1-5 MHz |

The magnetic field gradient $dB/dz$ couples the electron spin to its motional
mode. Writing the quantized position as
$\hat{z} = z_0(\hat{a} + \hat{a}^\dagger)$ where
$z_0 = \sqrt{\hbar / 2m\omega_z}$, the gradient interaction is:

$$
H_\text{grad} = \frac{g_e \mu_B}{2} \frac{dB}{dz}\, z_0\bigl(\hat{a} + \hat{a}^\dagger\bigr)\, \sigma_z
$$

This is a $\sigma_z$-dependent force with the same mathematical structure as
the light-shift gate Hamiltonian (see [gates](gates.md)), making it the
**native entangling operation** for gradient-coupled particles (rather than
the MS gate, which requires additional microwave dressing to rotate the spin
basis). Because an electron has no laser transition, TIQS cannot guess an
effective wavevector for it: `SimulationConfig.k_eff` must be supplied
explicitly (e.g. $k_\text{eff} = g_e\mu_B (dB/dz)/(\hbar\omega_m)$ per
mode, following Mintert and Wunderlich) or the runner raises.

### Model scope and approximations

- **Singly charged ions only.** The charge is fixed at $+e$ by every
  consumer and `mass_kg` subtracts exactly one electron; there is no
  `charge_state` field.
- **No sublevel structure.** No Zeeman or hyperfine sublevels, no dipole
  matrix elements or saturation intensities, and no magnetic response
  ($df/dB$, $d^2f/dB^2$, Lande $g_F$). Every ion in a chain therefore
  shares one qubit frequency, magnetic dephasing can only be entered as
  a phenomenological $T_2$ (see [noise](noise.md)), and photon
  scattering rates are user inputs rather than derived quantities.
- **Linewidth semantics differ by entry.** `cooling_transition.linewidth`
  is the *total* natural linewidth of the upper state;
  `repump_transitions[i].linewidth` is the *partial* Einstein
  coefficient $A_{ki}/2\pi$ of that one line. `branching_ratio` is
  populated for the cooling transitions and left at its 1.0 default
  (i.e. not characterised) for repumpers.
- **Species coverage.** Six isotopes: Yb-171, Ca-40, Ca-43, Ba-137,
  Be-9, Sr-88. Notably absent are Mg-24/Mg-25 and Ba-138, so the
  canonical sympathetic coolant for a Be-9 chain cannot be built, and
  no `"zeeman"` entry exists even though the `qubit_type` field admits
  the string.
- The ${}^{171}\text{Yb}^+$ 760 nm ${}^2F_{7/2}$ clear-out linewidth is
  an order-of-magnitude placeholder (no published rate was found); it is
  unused by any code path.

### References

1. Harty, T.P. et al. "High-fidelity preparation, gates, memory, and readout
   of a trapped-ion quantum bit." *Phys. Rev. Lett.* **113**, 220501 (2014).
   ${}^{43}\text{Ca}^+$ clock qubit, $T_2^* = 50$ s.
2. Harty, T.P., Sepiol, M.A., Allcock, D.T.C., Ballance, C.J., Tarlton,
   J.E. & Lucas, D.M. "High-fidelity trapped-ion quantum logic using
   near-field microwaves." *Phys. Rev. Lett.* **117**, 140501 (2016).
3. Smith, M.C., Leu, A.D., Miyanishi, K., Gely, M.F. & Lucas, D.M.
   "Single-qubit gates with errors at the $10^{-7}$ level."
   arXiv:2412.04421 (2024). Source of the $1.5(4)\times10^{-7}$
   error per Clifford in a ${}^{43}\text{Ca}^+$ hyperfine clock qubit.
4. Löschnauer, C.M. et al. "Scalable, high-fidelity all-electronic
   control of trapped-ion qubits." arXiv:2407.07694 (2024);
   *PRX Quantum* **6**, 040313 (2025). Oxford Ionics
   ${}^{40}\text{Ca}^+$ Zeeman qubit, $8.4\times10^{-6}$ single-qubit
   error.
5. Hughes, A.C. et al. "Trapped-ion two-qubit gates with $>99.99\%$
   fidelity without ground-state cooling." arXiv:2510.17286 (2025).
   ${}^{40}\text{Ca}^+$ Zeeman qubit, $8.4(7)\times10^{-5}$ two-qubit
   error.
6. Wang, P. et al. "Single ion qubit with estimated coherence time
   exceeding one hour." *Nat. Commun.* **12**, 233 (2021).
7. Auchter, C., Noel, T.W., Hoffman, M.R., Williams, S.R. & Blinov,
   B.B. "Measurement of the branching fractions and lifetime of the
   $5D_{5/2}$ level of Ba$^+$." *Phys. Rev. A* **90**, 060501(R)
   (2014).
8. Mohanty, A. et al. "Lifetime measurement of the $5d\,{}^2D_{5/2}$
   state in Ba$^+$." *Hyperfine Interact.* (2015),
   doi:10.1007/s10751-015-1161-9 (arXiv:1504.03023): 26.4(1.7) s.
9. Yu, Q. et al. "Feasibility study of quantum computing using trapped
   electrons." *Phys. Rev. A* **105**, 022420 (2022).
10. Huang, A. et al. "Numerical investigations of electron dynamics in a
    linear Paul trap." arXiv:2503.12379 (2025).

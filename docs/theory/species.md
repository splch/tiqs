## Ion Species and Qubit Encoding

The choice of ion species determines nearly every engineering decision --
laser wavelengths, qubit coherence, gate mechanisms, and scalability. All
species used are singly-charged, alkaline-earth-like atoms with a single
valence electron.

### Qubit Encoding Types

**Hyperfine qubits** use two hyperfine levels of the electronic ground state,
typically $m_F = 0$ "clock" states that are first-order insensitive to
magnetic field fluctuations. Both qubit states live in the ground state, so
$T_1$ is effectively infinite and coherence times reach thousands of seconds.

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
| ${}^{171}\text{Yb}^+$ | Hyperfine | 12.6 GHz | 369.5 nm | Long coherence ($T_2 > 5000$ s); mature ecosystem |
| ${}^{40}\text{Ca}^+$ | Optical | 729 nm transition | 397 nm | Simple level structure; all diode lasers |
| ${}^{43}\text{Ca}^+$ | Hyperfine | 3.226 GHz | 397 nm | Record single-qubit fidelity ($1.5 \times 10^{-7}$); $T_2^* \approx 50$ s |
| ${}^{137}\text{Ba}^+$ | Hyperfine | $\sim 8$ GHz | 493 nm | All visible wavelengths; scalable photonics |
| ${}^{9}\text{Be}^+$ | Hyperfine | 1.25 GHz | 313 nm | Lightest ion; fastest gates |
| ${}^{88}\text{Sr}^+$ | Optical | 674 nm transition | 422 nm | Quantum networking (fiber-friendly photons) |

### Key Atomic Properties

**${}^{171}\text{Yb}^+$** (nuclear spin $I = 1/2$): Qubit states
$|F{=}0, m_F{=}0\rangle$ and $|F{=}1, m_F{=}0\rangle$ in the ${}^2S_{1/2}$
manifold. Cooling via ${}^2S_{1/2} \to {}^2P_{1/2}$ at 369.5 nm (linewidth
$\Gamma/2\pi \approx 19.6$ MHz). Repumper at 935 nm clears the metastable
${}^2D_{3/2}$ state. Gates driven by stimulated Raman transitions via 355 nm
pulsed lasers or direct microwave drive at 12.6 GHz. Used by IonQ (Forte)
and Quantinuum (H1, H2).

**${}^{40}\text{Ca}^+$** (zero nuclear spin): Optical qubit between $4S_{1/2}$
and metastable $3D_{5/2}$ (lifetime $\sim 1.17$ s) at 729 nm. Doppler cooling
at 397 nm, repumping at 866 nm. All wavelengths accessible with diode lasers.
Used by Innsbruck group and AQT.

**${}^{43}\text{Ca}^+$** ($I = 7/2$): Clock qubit at 3.226 GHz. Holds record
coherence time $T_2^* \approx 50$ s and record single-qubit gate error
$1.5 \times 10^{-7}$ with microwave-driven gates. Only 0.135% natural
abundance requires isotope-selective loading. Used by Oxford Ionics / IonQ.

**${}^{137}\text{Ba}^+$** ($I = 3/2$): All primary wavelengths in the visible
spectrum: cooling at 493 nm, repumping at 650 nm. The $5D_{5/2}$ metastable
state has $\sim 30$ s lifetime for high-fidelity electron shelving. Quantinuum's
Helios processor (2025) was the first commercial system using ${}^{137}\text{Ba}^+$.

### The Species Protocol

TIQS defines a structural ``Species`` protocol:

```python
class Species(Protocol):
    @property
    def mass_kg(self) -> float: ...

    @property
    def qubit_frequency_hz(self) -> float: ...
```

Both ``IonSpecies`` and ``ElectronSpecies`` satisfy this protocol. Any custom
class exposing these two properties will be accepted by TIQS functions that
take a ``Species`` argument (e.g. ``lamb_dicke_parameters()``).

### Trade-offs

The fundamental trade-off: **heavier ions** (Ba$^+$, Yb$^+$) offer convenient
wavelengths and long-lived states but slower gate speeds (motional frequencies
scale as $1/\sqrt{m}$), while **lighter ions** (Be$^+$, Mg$^+$) enable faster
dynamics but demand challenging UV optics. The 2025-2026 industry trend strongly
favors barium for its visible-wavelength scalability.

### Trapped Electrons

Bare electrons confined in Paul traps or Penning traps are a candidate
platform for quantum computing. The qubit is the electron spin-1/2 in an applied magnetic field,
with Zeeman splitting $f = g_e \mu_B B / h \approx 28$ GHz/T. Unlike atomic
ions, electrons have no internal level structure, so cooling is resistive
(via an RLC tank circuit) and spin-motion coupling is mediated by a magnetic
field gradient rather than a laser wavevector. TIQS models trapped electrons
via `tiqs.ElectronSpecies`.

| Property | Electron | Typical ion (${}^{40}\text{Ca}^+$) |
|----------|----------|---------------------|
| Mass | $9.1 \times 10^{-31}$ kg | $6.6 \times 10^{-26}$ kg |
| Qubit encoding | Spin Zeeman | Optical or hyperfine |
| Qubit frequency | Tunable via $B$ ($\sim$1-30 GHz) | Fixed ($\sim$1-13 GHz) |
| $T_1$ | $\infty$ (no decay channel) | $\infty$ (hyperfine) or $\sim$1 s (optical) |
| Cooling | Resistive (RLC circuit) | Laser Doppler + sideband |
| Spin-motion coupling | Magnetic field gradient | Laser wavevector |
| Trap drive | 1-10 GHz (Paul) or 1-5 T (Penning) | 10-100 MHz |
| Secular frequencies | 30 MHz - 2 GHz | 1-5 MHz |

The magnetic field gradient $dB/dz$ couples the electron spin to its motional
mode:

$$
H_\text{grad} = \frac{g_e \mu_B}{2} \frac{dB}{dz}\, \hat{z}\, \sigma_z
$$

This is a $\sigma_z$-dependent force, making the **light-shift gate** the
native entangling operation (rather than the MS gate, which requires additional
microwave dressing to rotate the spin basis).

#### References

1. Yu, Q. et al. "Feasibility study of quantum computing using trapped
   electrons." *Phys. Rev. A* **105**, 022420 (2022).
2. Huang, A. et al. "Numerical investigations of electron dynamics in a linear
   Paul trap." arXiv:2503.12379 (2025).

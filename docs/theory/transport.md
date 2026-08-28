## QCCD Transport Theory

A quantum charge-coupled device (QCCD) processor stores ions in a
segmented linear trap with many zones: dedicated regions for gates,
cooling, measurement, and storage. Because laser-driven entangling
gates require a shared motional mode between nearby ions, each gate
step demands that the right pair of ions be physically co-located in
the same trapping zone. After the gate, the ions must be returned to
storage or moved to another zone for the next operation.

A single layer of a quantum algorithm may therefore involve tens or
hundreds of ion transport operations. Every transport step adds
motional excitation (phonons) to the ion's motion, degrading the
quality of subsequent gates whose fidelity depends on the motional
state being near the ground state. Understanding and minimizing
this motional excitation is one of the central engineering
challenges in scaling trapped-ion processors.

### Ion Shuttling

Shuttling translates an ion (or a small crystal) from one trapping
zone to another by smoothly sweeping the voltage waveform on the
segmented DC electrodes so that the electrostatic potential minimum
glides along the trap axis. The ion follows the moving well, but
any part of the motion that is too fast for the ion to follow
smoothly leaves it displaced from the instantaneous potential
minimum and deposits energy into its secular motion.

#### The Excitation Is a Fourier Component of the Waveform

Model the shuttle as a harmonic well of *fixed* frequency $\omega$
whose centre follows a prescribed trajectory $z_0(t)$, so
$H(t) = p^2/2m + \tfrac{1}{2} m\omega^2 (z - z_0(t))^2$. In the
co-moving frame this is a driven oscillator,
$H(t) = \hbar\omega\, a^\dagger a + f(t)(a + a^\dagger)$, whose exact
propagator is a **coherent displacement**. The final state is a
coherent state $|\alpha\rangle$ of the destination well with

$$
\alpha = \frac{1}{2\,x_\text{zpf}}
    \int_0^T \dot{z}_0(t)\, e^{i\omega t}\, dt ,
\qquad
\Delta\bar{n} = |\alpha|^2 ,
\qquad
x_\text{zpf} = \sqrt{\frac{\hbar}{2 m \omega}}
$$

This is Bowler et al. Eq. (1) (credited there to Lau & James), and
the same Fourier-component result appears in Reichle et al.: the
excitation is the Fourier component of the well's *velocity* profile
at the secular frequency, expressed in units of $x_\text{zpf}$. It
depends only on the waveform, the trap frequency, and the mass --
**not** on the initial motional state.

Two consequences are worth stating explicitly, because both are easy
to get backwards:

- **The generic decay is a power law, not an exponential.** For a
  hard-edged constant-velocity ramp the integral gives Bowler's
  Eq. (2), $\Delta\bar{n} = (d/x_\text{zpf})^2
  \sin^2(\omega T/2)/(\omega T)^2$: an oscillation under a
  $(\omega T)^{-2}$ envelope. The scaling argument behind that
  envelope is a **velocity** ratio -- the trap-centre speed $d/T$
  measured against the zero-point velocity $\omega\, x_\text{zpf}$,
  so $\Delta\bar{n} \sim \bigl(d/(T\,\omega\, x_\text{zpf})\bigr)^2$
  -- not an acceleration ratio. Exponential suppression in $\omega T$
  requires a specially engineered analytic ($C^\infty$) ramp; a
  merely smooth "minimum-jerk" profile does not produce it.
- **That power law is the slow limit, not the fast one.** The
  $1/(\omega T)^2$ envelope is the *large*-$\omega T$ asymptotic. As
  $\omega T \to 0$ the excitation instead **saturates**: the ion
  never has time to move, so it is left displaced by the full
  distance $d$ and $\Delta\bar{n} \to (d/2x_\text{zpf})^2$.

#### The Implemented Reference Waveform

`shuttle_motional_excitation()` evaluates the integral above exactly
for one named profile, the smooth $\sin^2$ velocity ramp

$$
\dot{z}_0(t) = \frac{2d}{T}\,\sin^2\!\left(\frac{\pi t}{T}\right)
$$

which starts and stops with zero velocity *and* zero acceleration.
Writing $N = \omega T/2\pi$ for the number of secular periods during
the shuttle, the integral evaluates in closed form to

$$
\Delta\bar{n} = \left(\frac{d}{2\,x_\text{zpf}}\right)^2
    \frac{\operatorname{sinc}^2(N - 1)}{N^2\,(N + 1)^2} ,
\qquad
\operatorname{sinc}(x) = \frac{\sin \pi x}{\pi x}
$$

(equivalently $\sin^2(\pi N) / \bigl[\pi^2 N^2 (N^2-1)^2\bigr]$
times the same prefactor; the $\operatorname{sinc}$ form is used in
the code because it stays finite at $N = 1$). Its behaviour:

| regime | value |
|--------|-------|
| Sudden, $N \to 0$ | $\to (d/2x_\text{zpf})^2$ (saturation) |
| $N = 1$ | $(d/2x_\text{zpf})^2/4$ -- finite, **not** a null |
| Envelope, large $N$ | $\propto N^{-6}$ |
| $N = 2, 3, 4, \ldots$ | exactly zero (**catch condition**) |

The nulls at integer $N \ge 2$ are the *catch condition*: "an ion
starting in its ground state of motion is caught back in the ground
state" (Bowler et al.). They are exact only for perfect timing and
this idealized waveform, so a returned value far below the trap's
anomalous-heating contribution is a property of the model, not a
prediction. Pass `heating_rate` to add that floor (see the
**Anomalous Heating** section below).

Because the prefactor is $(d/2x_\text{zpf})^2$, the result is exactly
proportional to $d^2$ and to the ion mass $m$ -- both `distance` and
`mass_kg` are load-bearing arguments.

#### Practical Numbers

| case | $d$, $T$, $\omega/2\pi$, species | model | measured |
|---|---|---|---|
| $x_\text{zpf}$ reference | Ca-40 at 1 MHz | 11.25 nm | -- |
| QCCD hop | 200 um, 10.5 us, 1 MHz, Ca-40 | 6.09 quanta | -- |
| QCCD hop | 200 um, 50.5 us, 1 MHz, Ca-40 | $4.8\times10^{-4}$ | -- |
| Walther et al. | 280 um, 3.6 us, 1.41 MHz, Ca-40 | 78 quanta | 0.10(1) |
| Bowler et al. | 370 um, 8 us, 1.972 MHz, Be-9 | 0.33 quanta | 0.1 gain |
| Sterk et al. | 210 um, 6 us, 2.5 MHz, Yb-171 | 0 ($N=15$) | 0.36(8) |

Read those last three rows as the honest caveat: the $\sin^2$ ramp is
**unoptimized**, and in the diabatic regime engineered waveforms beat
it by orders of magnitude (Walther et al.) while an exact catch lands
below what real timing jitter allows -- the Sterk et al. figure is a
round-trip excitation against a model prediction of exactly zero,
since 6 us at 2.5 MHz is $N = 15$ periods on the nose. Treat the
return value as the excitation of one specific reference profile, not
as a prediction for tuned hardware.

In the adiabatic regime the model and the literature agree on the
practical rule: surface-electrode traps with tailored waveforms
routinely achieve sub-quantum excitation for durations beyond roughly
50 $\mu$s at 1-3 MHz, i.e. $\sim$50-150 secular periods per shuttle
(the model gives $4.8\times10^{-4}$ quanta for a 200 um hop at
$N = 50.5$). Fast diabatic protocols with optimized bang-bang
waveforms reach $\sim$5-10 $\mu$s at the cost of more complex
electrode control and tighter calibration.

### Transport Noise Channel

The post-transport state is a coherent displacement with
$|\alpha|^2 = \Delta\bar{n}$ and a **deterministic** phase: it is set
by the waveform, which is why experiments cancel it rather than
average over it. Bowler et al. drove an ion to $\bar{n} \approx 1.6$
during transport and returned it to $\bar{n} = 0.19 \pm 0.02$ with a
waveform-locked electric-field impulse chosen so
$\alpha_E = -\alpha(t_T)$, or alternatively by waiting an integer
number of secular periods before transporting back. Sterk et al.
resolved the same coherence as a periodic dependence of the final
excitation on the dwell time at the destination.

TIQS does not track that phase, so `apply_shuttling_noise()` applies
the **phase-averaged displacement** channel

$$
\rho \mapsto \frac{1}{P}\sum_{p=0}^{P-1}
    D(\alpha_p)\,\rho\,D^\dagger(\alpha_p) ,
\qquad
\alpha_p = \sqrt{\Delta\bar{n}}\; e^{2\pi i p / P}
$$

with $P = 24$ phases by default. Since

$$
\langle D^\dagger(\alpha)\, \hat n\, D(\alpha)\rangle
= \langle \hat n\rangle
+ \alpha^*\langle a\rangle + \alpha\langle a^\dagger\rangle
+ |\alpha|^2
$$

and a uniform phase grid cancels the linear terms exactly, the
channel adds exactly $\Delta\bar{n}$ for **any** input state and
leaves $\langle a \rangle$ unchanged. It is a mixture of unitaries,
hence trace preserving and positive, and it acts only on the
addressed mode, so a motional mode entangled with qubits or other
modes keeps the correlations that should survive -- the reason a
channel is used rather than a state replacement.

From vacuum the output is a phase-averaged coherent state, whose Fock
populations are **Poissonian** with mean $\Delta\bar{n}$ and variance
$\Delta\bar{n}$. That matches Bowler's measured "Fock state
populations consistent with coherent states". It is *not* a thermal
state, which at the same mean would have variance
$\bar{n}^2 + \bar{n}$ -- three times larger at $\bar{n} = 2$, with
2.5$\times$ the ground-state population.

Two model notes:

- A gain channel $L = \sqrt{\gamma}\,a^\dagger$ would be the wrong
  choice here even though it produces the right state from vacuum: it
  adds $\Delta\bar{n}\,(1 + \bar{n}_0)$ rather than
  $\Delta\bar{n}$, and it amplifies a pre-existing coherent
  amplitude by $\sqrt{1 + \Delta\bar{n}}$ (for $\Delta\bar{n} = 1$
  and an input $|\alpha_0| = 1$: $\langle a\rangle \to 1.414$ and
  $\langle n\rangle: 1 \to 3$ instead of 2). There is no bosonic
  stimulated-emission enhancement in a waveform kick -- the
  displacement is state independent.
- Coherences $\rho_{n n'}$ with $|n - n'|$ a nonzero multiple of $P$
  survive the finite phase average. Raise `n_phases` to at least the
  mode's Fock dimension if an exactly diagonal output is wanted.
- For **long** crystal separations the thermal description is
  actually the correct one (Ruster et al.: "for fast separation times,
  oscillatory motion is excited, while a predominantly thermal state
  is obtained for long times"), but the broadening mechanism there is
  shot-to-shot scatter in $|\alpha|$ from mV-level electrode noise,
  not phase averaging.

### Crystal Splitting

Many QCCD operations require separating a two-ion crystal into
individual ions in distinct wells -- for example after a two-qubit
gate, when the ions must go to different zones. Splitting is
physically more demanding than linear shuttling because the axial
potential must be continuously reshaped from a single harmonic well
into a double well with a barrier between the two ions.

#### The Critical Point

Expanding the axial potential about the crystal centre as
$U(z) = a(t) z^2 + b(t) z^4$, splitting requires $a(0) > 0$,
$b(0) = 0$ and $a(t_s) < 0$, $b(t_s) > 0$. The **critical point** is
where $a$ passes through zero: the external harmonic confinement
vanishes and the potential is locally quartic.

What does *not* vanish is the ion motion's frequency. Coulomb
repulsion plus the quartic term keep the crystal's normal modes
harmonic and finite throughout. At $a = 0$ the two-ion equilibrium
separation is $d = (2 k_e e^2 / b)^{1/5}$, giving
$\omega_\text{COM} = \sqrt{3 b d^2/m}$ and
$\omega_\text{str} = \sqrt{5 b d^2/m}$ -- both strictly nonzero for
any $b > 0$, with a ratio $\sqrt{5/3} = 1.29$. Bowler et al.
simulated minima of 700 kHz (COM) and 880 kHz (stretch), a ratio of
1.26; Kaufmann et al. Table 1 lists $\omega_\text{crit}/2\pi =
0.11$-$0.29$ MHz for six segmented traps.

So adiabatic splitting is **achievable, not impossible**: Kaufmann et
al. reach $\bar{n} < 0.1$ with optimized ramps at $T = 60$-$70$ us.
The real limits are control precision and anomalous heating, not a
frequency zero-crossing -- Bowler et al. found that a $+3$ mV offset
on a single electrode moved the result from $\bar{n} = 1.14$ to
$> 15$.

The frequency that controls adiabaticity is therefore
$\omega_\text{crit}$, the minimum crystal mode frequency at the
critical point, typically a factor 3-20 *below* the initial
single-well frequency. That, not the initial frequency, is the
argument `split_crystal_excitation()` takes.

#### Excitation Model

In the impulsive (diabatic) regime the energy gain scales as the
square of the rate at which the control parameter is swept through
the critical point, $\delta E \propto T^{-2}$ (Kaufmann et al.).
TIQS uses that scaling anchored to one measurement:

$$
\Delta\bar{n}_\text{split} \approx
  \Delta\bar{n}_\text{ref}
  \left(\frac{\omega_\text{ref}\, T_\text{ref}}
             {\omega_\text{crit}\, T}\right)^{2}
  + \dot{\bar{n}}\, T ,
\qquad
\begin{aligned}
\Delta\bar{n}_\text{ref} &= 2 \\
\omega_\text{ref} T_\text{ref} &= 2\pi \cdot 0.7\,\text{MHz}
  \cdot 55\,\mu\text{s} \approx 242
\end{aligned}
$$

The anchor is Bowler et al., who separated a two-ion crystal in 55 us
and measured Fock populations consistent with coherent states of
$\bar{n} = 2.1 \pm 0.1$ and $\bar{n} = 1.9 \pm 0.1$ in the two
destination zones, with a simulated critical-point COM frequency of
$2\pi \times 0.7$ MHz.

Splitting is genuinely worse than shuttling at equal duration, and
now for the right reason: the $T^{-2}$ envelope decays far more
slowly than the shuttle's $N^{-6}$, and there are no catch nulls.

Representative values:

| $\omega_\text{crit}/2\pi$ | $T$ | model | measured |
|---|---|---|---|
| 0.7 MHz | 55 us | 2.00 quanta | 2.1(1) / 1.9(1) (Bowler) |
| 0.7 MHz | 200 us | 0.151 quanta | -- |
| 0.175 MHz | 55 us | 32 quanta | -- |
| 0.18 MHz | 80 us | 14.3 quanta | 4.16(16) (Ruster) |

The last two rows show both the sensitivity to $\omega_\text{crit}$
(a factor 4 in frequency is a factor 16 in excitation) and the
accuracy ceiling of a one-anchor power law. Expect
factor-of-a-few over-estimates for spectroscopically calibrated
ramps, which can also cross into Kaufmann's adiabatic regime
($\chi < 1$), where an extra exponential suppression appears that
this form does not contain.

### Anomalous Heating

Both estimators take an optional `heating_rate` $\dot{\bar{n}}$ in
quanta/s -- the same units as `tiqs.noise.motional.heating_rate` --
and add $\dot{\bar{n}}\,T$. This is the irreducible contribution at
long durations, and unlike a constant floor it **grows** with
duration:

$$
\Delta\bar{n}_\text{th} = \int_0^T \Gamma_h(\omega(t))\, dt
\approx \dot{\bar{n}}\, T ,
\qquad
\Gamma_h(\omega) = \frac{S_E(\omega)\, e^2}{4 m \hbar \omega}
$$

Heating is worst exactly where the confinement is weakest, so for a
split $\dot{\bar{n}}$ should be evaluated near $\omega_\text{crit}$,
not at the single-well frequency: Kaufmann et al. measured
$\Gamma_h \approx 6.3\,(\omega/2\pi\,\text{MHz})^{-1.81}$ quanta/ms
in their trap. Sterk et al. report a 295(24) quanta/s background,
which reaches 0.01 quanta in 34 us.

With $\dot{\bar{n}} > 0$ the split total has a minimum at

$$
T^* = \left(\frac{2\,\Delta\bar{n}_\text{ref}\,
  \omega_\text{ref}^2 T_\text{ref}^2}
  {\dot{\bar{n}}\, \omega_\text{crit}^2}\right)^{1/3}
$$

and splitting more slowly than $T^*$ is counterproductive:
"anomalous heating will strongly contribute to the energy gain at
large splitting times" (Kaufmann et al.). For
$\omega_\text{crit} = 2\pi \times 0.7$ MHz and
$\dot{\bar{n}} = 10^3$/s this gives $T^* = 230$ us with
$\Delta\bar{n}(T^*) = 0.34$ quanta.

### Error Budget Context

In a full QCCD circuit, transport errors accumulate across the
many shuttle and split operations required to bring ion pairs
together for entangling gates and return them to their home zones.
Consider a simple two-qubit gate layer in a processor with $N$
ions: at minimum, two ions must be shuttled into a gate zone (2
shuttles), the gate is performed, and the ions are shuttled back
(2 more shuttles). If the ions were part of a crystal, a split
and merge operation are also needed. A single algorithmic layer
can easily involve $O(N)$ transport steps.

Because phase-averaged transport kicks add **linearly**, the
accumulation is simple arithmetic: even at $\Delta\bar{n} = 0.1$ per
step, 20 transport operations before the next recooling raise the
motional occupation to $\bar{n} = 2$ exactly. For a typical
Molmer-Sorensen gate with Lamb-Dicke parameter $\eta \sim 0.1$, the
Lamb-Dicke condition $\eta\sqrt{2\bar{n}+1} \ll 1$ starts to break
down and gate errors grow quadratically with $\bar{n}$.

Sympathetic sideband cooling after transport restores the motional
ground state, but at a cost: each cooling cycle takes 1-10 ms per
mode (see [cooling.md](cooling.md)). Moses et al. tabulate the
per-circuit budget of the Quantinuum H2 race-track processor; for
2Q randomized benchmarking at $\ell = 128$ the split is 2% quantum
operations, 30% transport, 68% cooling, the largest cooling
fraction among the circuits they list (as also tabulated by Fallek
et al.). Reducing $\Delta\bar{n}$ per transport step directly
reduces the recooling frequency needed, and thus improves overall
algorithm throughput.

### Model Scope and Approximations

- The shuttle estimate is the **exact** excitation of **one**
  reference waveform (the $\sin^2$ velocity ramp). TIQS does not
  optimize waveforms; optimized and bang-bang profiles do far better
  in the diabatic regime, hard-edged ramps far worse.
- Constant trap frequency during transport. No anharmonicity, no
  frequency modulation along the path, no trap tilt, one mode, one
  ion -- crystal-internal modes during a hop are not modeled.
- The catch nulls assume perfect timing and the idealized profile.
  Real residual excitation is set by timing jitter and waveform
  error, so values below the anomalous-heating term are unphysical.
- The split model is a one-anchor power law, not a trap model: no
  critical-point geometry, no adiabatic-regime exponential, no
  separation into COM and stretch modes, no tilt or offset control
  parameters. The returned value is per ion.
- The noise channel deposits energy into a single mode. It does not
  model the correlated kick two ions receive during a split, nor
  recoil heating during the ramp beyond the $\dot{\bar{n}}T$ term,
  nor the deterministic phase that real experiments exploit to
  cancel the displacement.

### References

1. Bowler, R. et al. "Coherent diabatic ion transport and separation
   in a multizone trap array." *Phys. Rev. Lett.* **109**, 080502 (2012).
2. Walther, A. et al. "Controlling fast transport of cold trapped ions."
   *Phys. Rev. Lett.* **109**, 080501 (2012).
3. Reichle, R. et al. "Transport dynamics of single ions in segmented
   microstructured Paul trap arrays." *Fortschr. Phys.* **54**,
   666 (2006).
4. Lau, H.-K. & James, D.F.V. "Decoherence and dephasing errors caused
   by the dc Stark effect in rapid ion transport."
   *Phys. Rev. A* **83**, 062330 (2011).
5. Kaufmann, H. et al. "Dynamics and control of fast ion crystal
   splitting in segmented Paul traps."
   *New J. Phys.* **16**, 073012 (2014).
6. Ruster, T. et al. "Experimental realization of fast ion separation
   in segmented Paul traps." *Phys. Rev. A* **90**, 033410 (2014).
7. Rowe, M.A. et al. "Transport of quantum states and separation of
   ions in a dual RF ion trap."
   *Quantum Inf. Comput.* **2**, 257 (2002).
8. Sterk, J.D. et al. "Closed-loop optimization of fast trapped-ion
   shuttling with sub-quanta excitation."
   *npj Quantum Inf.* **8**, 68 (2022).
9. Pino, J.M. et al. "Demonstration of the trapped-ion quantum CCD
   computer architecture." *Nature* **592**, 209 (2021).
10. Moses, S.A. et al. "A race-track trapped-ion quantum processor."
    *Phys. Rev. X* **13**, 041052 (2023).
11. Fallek, S.D. et al. "Rapid exchange cooling with trapped ions."
    *Nat. Commun.* **15**, 1089 (2024).

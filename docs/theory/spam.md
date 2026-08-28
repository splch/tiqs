## State Preparation and Measurement

State preparation and measurement (SPAM) bookend every quantum circuit.
Their combined error floor bounds the fidelity of any gate
characterization.

**Bright/dark convention (TIQS-internal).** $|0\rangle$
(`basis(2, 0)`) is the ground state, the state optical pumping prepares,
and the state this library labels *bright*; $|1\rangle$ is *dark*, and
`fluorescence_probabilities` returns the population of $|0\rangle$. That
matches optical/shelving qubits (${}^{40}\text{Ca}^+$ $S_{1/2}$ bright
versus $D_{5/2}$ dark) but is **inverted** relative to the usual
labeling of direct-fluorescence hyperfine qubits, where
$|F{=}0, m_F{=}0\rangle$ is dark and $|F{=}1, m_F{=}0\rangle$ is bright.
So for ${}^{171}\text{Yb}^+$ the physical *dark* $|F{=}0\rangle$ level
maps onto TIQS $|0\rangle$; relabel on output if the hyperfine
convention is wanted. It is not a universal convention either way.

### State Preparation via Optical Pumping

Qubits are initialized into a known state via **optical pumping**, a
dissipative process where polarized laser light drives the ion into a target
"dark state" that is decoupled from the pump field.

For ${}^{171}\text{Yb}^+$, frequency-selective pumping at 369.5 nm drives
$|F{=}1\rangle \to {}^2P_{1/2}\,|F'{=}1\rangle$ (in practice via a
2.1 GHz EOM sideband on the detection light), from which the ion has a
$1/3$ chance of decaying to $|F{=}0\rangle$. Population accumulates in
$|F{=}0, m_F{=}0\rangle$ because its only dipole-allowed transition,
$|F{=}0\rangle \to {}^2P_{1/2}\,|F'{=}1\rangle$, is detuned from the
pump by the full 12.64 GHz ground-state hyperfine splitting
($|F{=}0\rangle \to |F'{=}0\rangle$ is forbidden by selection rules).
This takes 5-20 $\mu$s at fidelities exceeding 99.9%.

Repumper lasers prevent population trapping in metastable $D$ states: 935 nm
for Yb$^+$, 866 nm for Ca$^+$, 650 nm for Ba$^+$.

TIQS models this as a single qubit-local Lindblad channel driving
$|1\rangle \to |0\rangle$ at rate $\Gamma_p$, so
$p_0(t) = 1 - e^{-\Gamma_p t}$ with no error floor, no branching into
the metastable $D$ states the repumper text describes, and no photon
recoil into the motional modes (the motional state is left
bit-identical). Measured preparation errors saturate near $10^{-4}$.

### Fluorescence Detection

Readout exploits **state-dependent fluorescence**: one qubit state ("bright")
strongly scatters photons on a cycling transition, while the other ("dark")
is decoupled.

For ${}^{171}\text{Yb}^+$: the $|F{=}1\rangle$ state fluoresces at 369.5 nm
on ${}^2S_{1/2} \to {}^2P_{1/2}\,|F'{=}0\rangle$, at an effective
$\sim 10^7$ photons/s (below the two-level saturated value
$\Gamma/2 \approx 6\times 10^7$/s because of $D_{3/2}$ branching and
repumping), while $|F{=}0\rangle$ is 14.7 GHz off-resonance -- the
ground-state 12.64 GHz plus the ${}^2P_{1/2}$ 2.1 GHz hyperfine
splitting -- and effectively invisible.

**Poisson threshold model**: photon counts from the bright and dark
states follow Poisson distributions with means

$$
\mu_b = R_b\, t_\text{det}\, \eta_c + R_\text{bg}\, t_\text{det},
\qquad
\mu_d = R_d\, t_\text{det}\, \eta_c + R_\text{bg}\, t_\text{det}
$$

where $R_b$ and $R_d$ are the *ion's* bright and off-resonant dark
scattering rates (both attenuated by the collection efficiency
$\eta_c$), and $R_\text{bg}$ is the detector-side background plus dark
count rate, which is already a detected rate and so is *not* attenuated
by $\eta_c$. (Noek et al. use $R_d$ for the bright$\to$dark *pumping*
rate; here it is the dark state's own scattering rate.) The optimal
threshold for equal priors is where the two Poisson likelihoods cross,
$n^* = (\mu_b - \mu_d)/\ln(\mu_b/\mu_d)$, and the readout fidelity is

$$
F_\text{readout} = \frac{1}{2}\Bigl[P(n \geq n^* \mid \mu_b) + P(n < n^* \mid \mu_d)\Bigr]
$$

maximized over integer thresholds bracketing $n^*$.

Worked example, so the three inputs stay consistent: at
$R_b = 10^7$/s the mean count is $\mu_b = 20$ for
($t_\text{det} = 100\;\mu$s, $\eta_c = 2\%$) and $\mu_b = 250$ for
($500\;\mu$s, $5\%$) -- the whole $100$-$500\;\mu$s, $2$-$5\%$ box spans
20-250 collected photons, not tens. Real operating points are lower on
both axes: Gaebler et al. scatter $\approx 900$ photons in a
$120\;\mu$s window and collect $1.7\%$ of them, $\approx 15$ detected.
Demonstrated total collection efficiencies are $0.1\%$ (NA $\approx
0.3$ objective, Olmschenk et al.), $0.19\%$ (Myerson et al.), $1.7\%$
(Gaebler et al.) and $2.2\%$ with a dedicated NA = 0.6 objective (Noek
et al.), so $2$-$5\%$ is at or above the best ever demonstrated. The
dark state's own scattering contributes far less than one count -- for
Yb$^+$ the Lorentzian suppression at 14.7 GHz is
$(\Gamma/2)^2/[(\Gamma/2)^2 + \Delta^2] = 4\times 10^{-7}$, i.e.
$\mu_d \sim 4\times10^{-5}$ over $300\;\mu$s at $\eta_c = 3\%$ -- so the
"0-1 counts" seen on dark shots are detector background $R_\text{bg}$,
not ion fluorescence.

### Electron Shelving

For species with metastable states (e.g., Ba$^+$ with a $D_{5/2}$
lifetime of $\sim 30$ s; published values span roughly 25.6-31.2 s, see
[species](species.md)), **electron shelving** improves discrimination:
one qubit state is transferred to the metastable level before
fluorescence detection, making it dark for the detection window
(up to its finite lifetime). Measured performance, distinguishing the
two figures of merit: average **SPAM** fidelity 99.971(6)% in
${}^{133}\text{Ba}^+$ (Christensen et al., 2020), and single-shot
**readout** fidelity 99.991(1)% for the ${}^{40}\text{Ca}^+$ optical
qubit using time-resolved maximum-likelihood analysis (Myerson et al.,
2008) -- readout fidelity excludes state-preparation error, so the two
are not comparable.

TIQS qubits are strictly two-level (`HilbertSpace.dims` is
`[2]*n_ions + fock`), so no metastable shelf exists in the simulator and
this section describes physics that cannot be represented; the shelving
lifetime enters only as the `metastable_lifetime` species datum.

### Detection Error Sources

The dominant error sources are:

- **Off-resonant pumping** (dominant for hyperfine readout): the
  *bright* state is off-resonantly excited to
  ${}^2P_{1/2}\,|F'{=}1\rangle$ -- detuned by only the
  $\Delta_\text{HFP} = 2.1$ GHz excited-state splitting -- and decays
  into the dark state, truncating the bright photon distribution
  (making it non-Poissonian) and producing false *dark* counts. The
  reverse process, dark pumped bright, is detuned by
  $\Delta_\text{HFP} + \Delta_\text{HFS} = 14.7$ GHz: suppressed by
  $[(\Delta_\text{HFP}+\Delta_\text{HFS})/\Delta_\text{HFP}]^2 \approx
  49$ before branching factors, giving a net rate ratio of $\approx 16$
  (Noek et al. Eqs. 2-3). Because the bright distribution is truncated,
  this bounds the Poisson model above and is why time-resolved analysis
  helps.
- **Finite photon count**: statistical overlap between the bright and dark
  photon-count distributions, reduced by longer detection windows or higher
  collection efficiency.
- **State decay during measurement**: the shelved state can decay during
  detection (probability $\sim t_\text{det} / \tau_D$).

More sophisticated approaches use **time-resolved maximum-likelihood analysis**
to account for state decay during measurement, achieving the single-shot
readout fidelity of 99.991% quoted above.

### Mid-Circuit Measurement

Measuring ancilla qubits while preserving data qubits is critical for quantum
error correction. In QCCD architectures ions are separated and
spatially isolated before detection. The published numbers, by device
generation:

- Pino et al. (2021): two adjacent gate zones -- which are also the
  detection zones -- separated by $750\;\mu$m; SPAM error $3(1)\times
  10^{-3}$ with measurement error above $10^{-3}$, limited by detector
  solid angle.
- Gaebler et al. (2021): unmeasured ions held $110\;\mu$m to either side
  of the detection beam, with measured crosstalk from $\sim 10^{-4}$ to
  a few $\times 10^{-3}$.
- Moses et al. (2023): measurement crosstalk $4.5(6)\times 10^{-6}$ on
  H2. The suppression comes from the **micromotion hiding** technique of
  Gaebler et al. (over an order of magnitude), not from larger zone
  separation -- H2's gate-zone spacing is the same $750\;\mu$m as H1.

`mid_circuit_measurement` is an ideal instantaneous projection: no
photon recoil into the motional modes, no scattered-light AC Stark shift
or depumping of neighbors (the crosstalk quoted above is not modeled),
and no state decay during the detection window.

### Model scope and approximations

- **No decay or off-resonant repumping during detection.**
  `measurement_fidelity` is photon statistics only, hence an *upper
  bound*. At Myerson's published conditions ($R_b = 55800$/s,
  $R_d = 442$/s, $t_b = 420\;\mu$s, both already detected) it returns an
  error of $1.3\times10^{-6}$ against the measured threshold-method
  $1.8(1)\times10^{-4}$, which is dominated by $D_{5/2}$ decay at
  $t_b/\tau_D = 3.6\times10^{-4}$. It can and does return exactly 1.0
  for well-separated means.
- **Rates are user inputs.** No species carries dipole matrix elements
  or a saturation intensity, so $R_b$ and $R_d$ are not derived from
  atomic data.
- **Two-level ions only**: no shelving level, no leakage level, no
  Zeeman or hyperfine substructure (see Electron Shelving).
- **Optical pumping** has no error floor, no $D$-state branching and no
  recoil heating (see State Preparation).
- `sample_measurement`'s `spam_error` is a symmetric, independent
  per-bit flip probability supplied by the caller; it is not derived
  from any of the models above.

### References

1. Olmschenk, S. et al. "Manipulation and detection of a trapped
   Yb$^+$ ion hyperfine qubit." *Phys. Rev. A* **76**, 052314 (2007).
2. Myerson, A.H. et al. "High-fidelity readout of trapped-ion qubits."
   *Phys. Rev. Lett.* **100**, 200502 (2008).
3. Noek, R. et al. "High speed, high fidelity detection of an atomic
   hyperfine qubit." *Opt. Lett.* **38**, 4735 (2013).
4. Christensen, J.E., Hucul, D., Campbell, W.C. & Hudson, E.R. "High
   fidelity manipulation of a qubit built from a synthetic nucleus."
   *npj Quantum Inf.* **6**, 35 (2020). SPAM fidelity 99.971(6)% for
   ${}^{133}\text{Ba}^+$ with $D_{5/2}$ shelving.
5. Pino, J.M. et al. "Demonstration of the trapped-ion quantum CCD
   computer architecture." *Nature* **592**, 209 (2021).
6. Gaebler, J.P. et al. "Suppression of mid-circuit measurement
   crosstalk errors with micromotion." *Phys. Rev. A* **104**, 062440
   (2021).
7. Moses, S.A. et al. "A race track trapped-ion quantum processor."
   *Phys. Rev. X* **13**, 041052 (2023). Table II: measurement
   crosstalk $4.5(6)\times10^{-6}$.

## State Preparation and Measurement

Initializing qubits to a known state and reading them out at the end of a
circuit are the first and last operations in any quantum algorithm. Together
they set the SPAM error floor that gates must stay below.

### State Preparation via Optical Pumping

Qubits are initialized into a known state via **optical pumping**, a
dissipative process where polarized laser light drives the ion into a target
"dark state" that is decoupled from the pump field.

For ${}^{171}\text{Yb}^+$, frequency-selective pumping at 369.5 nm drives
the $|F{=}1\rangle$ manifold of ${}^2S_{1/2} \to {}^2P_{1/2}$ while
$|F{=}0\rangle$ remains dark (separated by 14.7 GHz from the nearest
excited-state transition). Population accumulates in $|F{=}0, m_F{=}0\rangle$
within 5-20 $\mu$s at fidelities exceeding 99.9%.

Repumper lasers prevent population trapping in metastable $D$ states: 935 nm
for Yb$^+$, 866 nm for Ca$^+$, 650 nm for Ba$^+$.

### Fluorescence Detection

Readout exploits **state-dependent fluorescence**: one qubit state ("bright")
strongly scatters photons on a cycling transition, while the other ("dark")
is decoupled.

For ${}^{171}\text{Yb}^+$: the $|F{=}1\rangle$ state fluoresces at 369.5 nm
($\sim 10^7$ photons/s at saturation), while $|F{=}0\rangle$ is 14.7 GHz
off-resonance and effectively invisible.

During a detection window of 100-500 $\mu$s, the bright state yields
$\sim 10$-$30$ collected photons (limited by $\sim 2$-$5\%$ total collection
efficiency), while the dark state yields 0-1 background counts. A threshold
discriminator separates the states.

**Poisson threshold model**: Photon counts from the bright and dark states
follow Poisson distributions with means
$\mu_b = R_b\, t_\text{det}\, \eta_c$ and
$\mu_d = R_d\, t_\text{det}\, \eta_c$, where $R_b$ and $R_d$ are the
scattering rates and $\eta_c$ is the collection efficiency. The optimal
threshold $n^*$ minimizes the total error:

$$
F_\text{readout} = \frac{1}{2}\Bigl[P(n \geq n^* \mid \mu_b) + P(n < n^* \mid \mu_d)\Bigr]
$$

### Electron Shelving

For species with metastable states (e.g., Ba$^+$ with $D_{5/2}$ lifetime
$\sim 30$ s), **electron shelving** dramatically improves discrimination: one
qubit state is transferred to the metastable level before fluorescence
detection, making it completely dark during the detection window. This enables
SPAM fidelities exceeding 99.99%.

### Detection Error Sources

The dominant error sources are:

- **Off-resonant excitation**: The dark state has a small but nonzero scattering
  rate, producing false bright counts.
- **Finite photon count**: Statistical overlap between the bright and dark
  photon-count distributions, reduced by longer detection windows or higher
  collection efficiency.
- **State decay during measurement**: The shelved state can decay during
  detection (probability $\sim t_\text{det} / \tau_D$).

More sophisticated approaches use **time-resolved maximum-likelihood analysis**
to account for state decay during measurement, achieving single-shot readout
fidelity of 99.991%.

### Mid-Circuit Measurement

Measuring ancilla qubits while preserving data qubits is critical for quantum
error correction. In QCCD architectures, ions are physically shuttled to
separated detection zones ($\sim 370\;\mu$m apart), reducing measurement
crosstalk to $\sim 2 \times 10^{-5}$.

### References

1. Myerson, A.H. et al. "High-fidelity readout of trapped-ion qubits."
   *Phys. Rev. Lett.* **100**, 200502 (2008).
2. Noek, R. et al. "High speed, high fidelity detection of an atomic
   hyperfine qubit." *Opt. Lett.* **38**, 4735 (2013).
3. Pino, J.M. et al. "Demonstration of the trapped-ion quantum CCD
   computer architecture." *Nature* **592**, 209 (2021).

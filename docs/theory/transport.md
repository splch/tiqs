## QCCD Transport Theory

In a quantum charge-coupled device (QCCD) architecture, ions are
shuttled between spatially separated zones for gates, cooling, and
readout. Transport operations add motional excitation that must be
accounted for in error budgets.

### Ion Shuttling

An ion confined at angular frequency $\omega$ is translated a
distance $d$ in time $T$ by sweeping the trapping potential
minimum. The residual motional excitation depends on how
adiabatically the transport is performed relative to the secular
frequency.

For an optimized waveform the residual excitation decays
exponentially with the number of trap oscillation periods during
transport:

$$
\Delta\bar{n} \sim \left(\frac{d}{x_\text{zpf}}\right)^2
    \exp\!\bigl(-\pi\,\omega\,T\bigr)
$$

where $x_\text{zpf} = \sqrt{\hbar / 2m\omega}$ is the zero-point
fluctuation amplitude. In the simplified model used here the
excitation scales as

$$
\Delta\bar{n} \sim A\,\frac{d^2}{T^2\,\omega^2}
$$

with $A$ a geometry-dependent constant of order unity for optimized
waveforms. Modern electrode designs with tailored voltage waveforms
achieve sub-quantum excitation ($\Delta\bar{n} < 0.1$) for
durations exceeding roughly 50 $\mu$s at typical trap frequencies
of 1-3 MHz.

### Thermal Noise Channel

The effect of shuttling on the quantum state is modeled as a
thermal excitation channel. A Lindblad collapse operator

$$
L = \sqrt{\gamma}\; a^\dagger
$$

applied to the relevant motional mode drives the master equation

$$
\frac{d\langle n\rangle}{dt} = \gamma\,(\langle n\rangle + 1)
$$

The solution $\langle n\rangle(t) = (\langle n\rangle_0 + 1)\,
e^{\gamma t} - 1$ is inverted to find the rate $\gamma$ that
deposits the desired number of quanta $\Delta\bar{n}$ in a short
fictitious evolution time $\tau$:

$$
\gamma = \frac{\ln(\Delta\bar{n} + 1)}{\tau}
$$

QuTiP's `mesolve` integrates the resulting Lindblad equation,
mapping an input density matrix $\rho$ to the post-transport state
with the correct added phonon population and associated coherence
loss.

### Crystal Splitting

Separating a two-ion crystal into individual wells requires
reshaping the axial potential from a single harmonic well to a
double well. During the transition the axial frequency passes
through a near-zero minimum, making the process intrinsically less
adiabatic than simple linear shuttling.

The adiabaticity parameter is $\omega\,T_\text{split}$, and the
excitation is modeled as

$$
\Delta\bar{n}_\text{split} \approx
  2\,\exp\!\bigl(-\omega\,T_\text{split} / 5\bigr)
$$

with a floor of 0.05 quanta for highly adiabatic splits
($\omega\,T_\text{split} > 50$). Experimentally, optimized
splitting protocols on surface traps achieve excitations of
0.02-0.1 quanta at durations of 50-200 $\mu$s.

### Error Budget Context

In a full QCCD circuit, transport errors accumulate across the
many shuttle and split operations required to bring ion pairs
together for entangling gates and return them to their home zones.
A single algorithmic layer may involve tens of transport steps,
so even sub-quantum excitation per step can build up significantly.
Recooling (sympathetic sideband cooling) after transport is
therefore essential and can dominate total algorithm runtime.

### References

1. Bowler, R. et al. "Coherent diabatic ion transport and separation
   in a multizone trap array." *Phys. Rev. Lett.* **109**, 080502 (2012).
2. Walther, A. et al. "Controlling fast transport of cold trapped ions."
   *Phys. Rev. Lett.* **109**, 080501 (2012).
3. Pino, J.M. et al. "Demonstration of the trapped-ion quantum CCD
   computer architecture." *Nature* **592**, 209 (2021).

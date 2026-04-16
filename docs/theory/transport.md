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
any imperfection in the waveform, or any part of the motion that
is too fast for the ion to follow smoothly, kicks the ion away
from the instantaneous potential minimum and deposits energy into
its secular motion.

#### Adiabaticity

The key parameter governing shuttling quality is **adiabaticity**:
the ratio of the transport duration $T$ to the ion's oscillation
period $\tau_\text{sec} = 2\pi/\omega$.

When $T \gg \tau_\text{sec}$ (many oscillation periods during
transport), the ion continuously tracks the instantaneous ground
state of the moving well. Its wavepacket follows the potential
minimum without acquiring excess kinetic energy, and the residual
excitation is exponentially small (the **adiabatic limit**).

When $T \sim \tau_\text{sec}$ or shorter, the potential minimum
moves significantly between successive oscillation cycles. The ion
cannot keep up and ends the transport in a coherent state displaced
from the new potential minimum. This displacement maps to a nonzero
mean phonon number.

For an optimized transport waveform (smooth start and stop, minimum
jerk), the residual excitation decays exponentially with the number
of trap oscillation periods during the shuttle:

$$
\Delta\bar{n} \sim \left(\frac{d}{x_\text{zpf}}\right)^2
    \exp\!\bigl(-\pi\,\omega\,T\bigr)
$$

where $x_\text{zpf} = \sqrt{\hbar / 2m\omega}$ is the zero-point
fluctuation amplitude and $d$ is the shuttling distance. The
prefactor $(d / x_\text{zpf})^2$ reflects the fact that longer
shuttles displace the ion further from its final equilibrium in
phase space, so more adiabatic margin is needed. The exponential
term is the payoff of adiabaticity: each additional oscillation
period during transport suppresses the excitation by a constant
factor.

In the opposite, fast (diabatic) limit where $\omega T$ is of
order unity, the exponential saturates and the excitation instead
scales as a power law:

$$
\Delta\bar{n} \sim A\,\frac{d^2}{T^2\,\omega^2}
$$

with $A$ a geometry-dependent constant of order unity for optimized
waveforms. The displacement error is proportional to the acceleration
of the potential minimum ($\sim d/T^2$), measured in units of the
trap's restoring acceleration ($\sim \omega^2 x_\text{zpf}$).

#### Practical Numbers

Modern surface-electrode traps with tailored voltage waveforms
routinely achieve sub-quantum excitation ($\Delta\bar{n} < 0.1$)
for durations exceeding roughly 50 $\mu$s at trap frequencies of
1-3 MHz. This corresponds to $\sim$50-150 oscillation periods per
shuttle, safely in the exponentially suppressed regime. Fast
"diabatic" protocols with optimized bang-bang waveforms can push to
$\sim$5-10 $\mu$s at the cost of more complex electrode control
and tighter calibration requirements.

### Thermal Noise Channel Model

After the shuttle completes, the ion is in a coherent state
displaced from the ground state of the final well. On the
timescale of subsequent operations, the phase of this coherent
oscillation is effectively random (it depends on sub-nanosecond
timing details of the waveform), so the motional state is well
approximated as a **thermal mixture** with mean phonon number
$\Delta\bar{n}$. This is the same kind of state produced by
coupling the oscillator to a hot reservoir.

TIQS therefore models shuttling as a thermal excitation channel
applied to the affected motional mode: a Lindblad master equation
with a single collapse operator that creates phonons.

A collapse operator $L = \sqrt{\gamma}\; a^\dagger$ acting on a
harmonic oscillator mode produces transitions
$|n\rangle \to |n+1\rangle$ at rate $\gamma(n+1)$. The factor
$(n+1)$ is the bosonic stimulated emission enhancement: the more
phonons already present, the faster new ones are added. The
resulting equation of motion for the mean occupation is:

$$
\frac{d\langle n\rangle}{dt} = \gamma\,(\langle n\rangle + 1)
$$

This is a first-order linear ODE with the exact solution:

$$
\langle n\rangle(t) = (\langle n\rangle_0 + 1)\,e^{\gamma t} - 1
$$

Starting from vacuum ($\langle n\rangle_0 = 0$), after a time
$\tau$ the occupation is $\langle n\rangle(\tau) = e^{\gamma\tau} - 1$.
Setting the deposited quanta to exactly $\Delta\bar{n}$ and inverting:

$$
\gamma = \frac{\ln(\Delta\bar{n} + 1)}{\tau}
$$

The implementation uses a short fictitious evolution time
$\tau = 1\;\mu$s (much shorter than any real dynamics) with the
corresponding $\gamma$, so that the Lindblad master equation maps
the input density matrix $\rho$ to the post-transport state. This
captures two effects simultaneously: the increase in mean phonon
number and the loss of motional coherence (off-diagonal elements of
$\rho$ in the Fock basis decay), both of which degrade subsequent
gate fidelities.

The Lindblad channel is preferred over a simple state replacement
because it correctly handles the case where the motional mode is
entangled with qubits or other modes: it acts locally on the
affected subsystem within the full density matrix, preserving any
correlations that should survive while adding the appropriate
decoherence.

### Crystal Splitting

Many QCCD operations require separating a two-ion crystal into
individual ions in distinct wells. For example, after a two-qubit
gate, the ions must be sent to different zones for subsequent
single-qubit operations or further transport. This splitting is
physically more demanding than linear shuttling because the axial
potential must be continuously reshaped from a single harmonic well
into a double well with a barrier between the two ions.

During the transition from single well to double well, the axial
confinement frequency passes through a near-zero minimum. At the
critical point where the barrier first appears, the potential is
nearly flat at its center, roughly quartic rather than quadratic.
The instantaneous trap frequency drops to nearly zero, and the
adiabatic condition $T \gg 2\pi/\omega$ becomes impossible to
satisfy at that instant no matter how slowly the split is performed.

This means there is always a "dangerous moment" during the split
where the ions are weakly confined and susceptible to excitation.
The practical consequence is that crystal splitting always produces
more heating than a linear shuttle of comparable duration, and the
excitation decays more slowly with increasing duration.

#### Excitation Model

The adiabaticity parameter for splitting is
$\omega\,T_\text{split}$, where $\omega$ is the initial (single-well)
trap frequency and $T_\text{split}$ is the total splitting
duration. The excitation is modeled as:

$$
\Delta\bar{n}_\text{split} \approx
  2\,\exp\!\bigl(-\omega\,T_\text{split} / 5\bigr)
$$

The prefactor of 2 (compared to $\sim$5 for shuttling) and the
slower decay constant (dividing by 5 rather than 3) reflect the
intrinsically non-adiabatic nature of the frequency zero-crossing.
A floor of 0.05 quanta is imposed for highly adiabatic splits
($\omega\,T_\text{split} > 50$), representing residual excitation
from waveform imperfections and finite electrode resolution that
cannot be eliminated even with arbitrarily slow splitting.

Experimentally, optimized splitting protocols on surface traps
achieve excitations of 0.02-0.1 quanta at durations of
50-200 $\mu$s. The same Lindblad thermal noise channel described
above is used to apply the splitting excitation to the quantum
state.

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

Even at $\Delta\bar{n} = 0.1$ per step, 20 transport operations
before the next recooling would raise the motional occupation to
$\bar{n} \sim 2$. For a typical Molmer-Sorensen gate with
Lamb-Dicke parameter $\eta \sim 0.1$, the Lamb-Dicke condition
$\eta\sqrt{2\bar{n}+1} \ll 1$ starts to break down and gate
errors grow quadratically with $\bar{n}$.

Sympathetic sideband cooling after transport restores the motional
ground state, but at a cost: each cooling cycle takes 1-10 ms per
mode (see [cooling.md](cooling.md)), and this recooling overhead can
dominate total algorithm runtime. The Quantinuum H1 system, for
example, spends roughly 68% of its execution time on recooling
between transport steps. This makes transport excitation one of the
key bottlenecks for scaling QCCD processors: reducing
$\Delta\bar{n}$ per transport step directly reduces the recooling
frequency needed and thus improves overall algorithm throughput.

### References

1. Bowler, R. et al. "Coherent diabatic ion transport and separation
   in a multizone trap array." *Phys. Rev. Lett.* **109**, 080502 (2012).
2. Walther, A. et al. "Controlling fast transport of cold trapped ions."
   *Phys. Rev. Lett.* **109**, 080501 (2012).
3. Pino, J.M. et al. "Demonstration of the trapped-ion quantum CCD
   computer architecture." *Nature* **592**, 209 (2021).
4. Kaufmann, H. et al. "Fast ion swapping for quantum-information
   processing." *Phys. Rev. A* **95**, 052319 (2017).

## Noise and Decoherence

All decoherence channels in TIQS are modeled via the Lindblad master equation:

$$
\frac{d\rho}{dt} = -i[H, \rho] + \sum_k \gamma_k \left(L_k \rho L_k^\dagger - \frac{1}{2}\lbrace L_k^\dagger L_k, \rho\rbrace\right)
$$

where $L_k$ are collapse (jump) operators and $\gamma_k$ are the corresponding
rates.

### Motional Heating

Electric field noise from electrode surfaces drives the trapped ion's motion,
causing transitions between motional Fock states. This is the dominant
"physics-to-hardware" limitation.

**Lindblad operators** for each mode $p$:

$$
L_\text{heat} = \sqrt{\dot{\bar{n}}_p\,(\bar{n}_\text{th}+1)}\; a_p^\dagger, \qquad
L_\text{cool} = \sqrt{\dot{\bar{n}}_p\,\bar{n}_\text{th}}\; a_p
$$

where $\dot{\bar{n}}_p$ is the heating rate (phonons/second), related to the
electric field noise spectral density:

$$
\dot{\bar{n}} = \frac{e^2 S_E(\omega_p)}{4m\hbar\omega_p}
$$

**Scaling laws**: $S_E(\omega) \propto d^{-\beta}\, \omega^{-\alpha}$ with
$\beta \approx 4$ (ion-electrode distance) and $\alpha \approx 1$ (1/f noise).
Cryogenic cooling to 4 K reduces rates by $\sim 100\times$.

### Motional Dephasing

Trap frequency fluctuations (from voltage noise on DC electrodes) cause
dephasing of motional superpositions:

$$
L_\text{deph} = \sqrt{\gamma_\text{deph}}\; a_p^\dagger a_p
$$

Off-diagonal density matrix elements decay as
$\langle n|\rho|n'\rangle \to \langle n|\rho|n'\rangle\, e^{-\gamma_\text{deph}(n-n')^2 t/2}$.

### Qubit Dephasing

Fluctuations in the qubit frequency (from magnetic field noise, AC Stark
shifts) cause pure dephasing:

$$
L_\phi = \sqrt{\gamma_\phi / 2}\; \sigma_z
$$

The $T_2$ dephasing time is $T_2 = 1/\gamma_\phi$. For clock-state qubits
with second-order magnetic sensitivity:

$$
\gamma_\phi = 2\pi \left|\frac{d^2f}{dB^2}\right| \cdot 2B_0 \cdot \delta B_\text{rms}
$$

### Spontaneous Emission

For optical qubits, the excited state decays at rate $\Gamma_D = 1/\tau_D$:

$$
L_\text{decay} = \sqrt{\Gamma_D}\; \sigma_-
$$

Population in $|e\rangle$ decays to $|g\rangle$ at rate $\Gamma_D$, and
coherence decays at $\Gamma_D / 2$. For ${}^{40}\text{Ca}^+$, $\tau_D = 1.17$ s.

### Off-Resonant Photon Scattering

During Raman-driven gates, off-resonant coupling to the excited $P$ state
causes spontaneous scattering with two components:

**Raman scattering** (inelastic) changes the internal state, causing bit-flip
errors:

$$
L_\text{Raman} = \sqrt{\Gamma_\text{Raman}(t)}\; \sigma_-
$$

**Rayleigh scattering** (elastic) causes dephasing without population transfer.
The dephasing rate involves interference between scattering amplitudes from
the two qubit states.

Total gate error from scattering: $\epsilon_\text{sc} \sim C / |\Delta|$, where
$\Delta$ is the detuning from the excited state. This sets a **fundamental error
floor** for laser-driven gates ($\sim 10^{-4}$ for typical detunings) that can
only be reduced by increasing $\Delta$ (requiring more laser power) or
eliminated entirely by using microwave-driven gates.

### Laser Phase Noise

Phase noise on the laser maps to qubit dephasing. For Raman transitions, the
relevant noise is on the **difference frequency** $\delta\phi_\text{eff} = \delta\phi_1 - \delta\phi_2$,
so common-mode noise from a single laser source is rejected. For white phase
noise with spectral density $S_\phi$:

$$
L_\text{phase} = \sqrt{\pi S_\phi \Omega^2 / 2}\; \sigma_z
$$

### Laser Intensity Noise

Intensity fluctuations cause Rabi frequency errors
$\Omega(t) = \Omega_0(1 + \epsilon(t))$, producing **coherent rotation errors**
rather than decoherence. For a $\pi$-pulse:

$$
1 - F = \frac{\pi^2 \langle\epsilon^2\rangle}{4}
$$

This is best modeled by classical noise averaging rather than a Lindblad
operator.

### Crosstalk

Residual laser intensity on neighboring ions causes spurious rotations. With
beam waists of $\sim 1$-$2\;\mu$m and ion spacings of $\sim 5\;\mu$m,
nearest-neighbor intensity is $\sim 10^{-3}$-$10^{-4}$. The crosstalk
Hamiltonian adds off-target single-qubit rotations proportional to the
spillover intensity.

### Complete Dissipator

The total Lindblad superoperator sums all channels:

$$
\mathcal{D}[\rho] = \sum_p \mathcal{D}_{\text{heat},p}[\rho] + \sum_p \mathcal{D}_{\text{deph},p}[\rho] + \sum_j \mathcal{D}_{\phi,j}[\rho] + \sum_j \mathcal{D}_{\text{decay},j}[\rho] + \sum_j \mathcal{D}_{\text{scatter},j}[\rho]
$$

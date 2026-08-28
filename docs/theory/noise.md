## Noise and Decoherence

All decoherence channels in TIQS are modeled via the Lindblad master equation:

$$
\frac{d\rho}{dt} = -i[H, \rho] + \sum_k \left(L_k \rho L_k^\dagger - \frac{1}{2}\lbrace L_k^\dagger L_k, \rho\rbrace\right)
$$

where the $L_k$ are collapse (jump) operators. **Rate convention**: each
rate is absorbed into its own operator, $L_k = \sqrt{\gamma_k}\,A_k$, so
no separate $\gamma_k$ prefactor appears in the dissipator. This is the
convention of `qutip.mesolve`, to which TIQS passes bare `c_ops`; every
operator written below already carries its $\sqrt{\text{rate}}$.

**Symbol convention**: this page uses the textbook labels
$\sigma_+ = |e\rangle\langle g|$ (excitation) and
$\sigma_- = |g\rangle\langle e|$ (de-excitation), as does
[laser_ion_interaction](laser_ion_interaction.md). The *code* labels are
inverted relative to those symbols because $|0\rangle$ is the ground
state, so `OperatorFactory.sigma_plus` $= |0\rangle\langle 1|$ is
de-excitation and `sigma_minus` $= |1\rangle\langle 0|$ is excitation.
An operator written $\propto \sigma_-$ here is therefore built with
`ops.sigma_plus` in the source.

### Motional Heating

Electric field noise from electrode surfaces drives the trapped ion's motion,
causing transitions between motional Fock states. This is the dominant
"physics-to-hardware" limitation.

The measured quantity is the **ground-state heating rate**
$\dot{\bar{n}}_p$ in quanta/second, related to the electric field noise
spectral density by

$$
\dot{\bar{n}} = \frac{e^2 S_E(\omega_p)}{4m\hbar\omega_p}
$$

(Brownnutt et al., *Rev. Mod. Phys.* **87**, 1419 (2015) Eq. 12).

**Lindblad operators** for a damped thermal bath at mean occupation
$\bar{n}_\text{th}$ (Turchette et al., *Phys. Rev. A* **62**, 053807
(2000) Eq. 4; Brownnutt Eq. 14) -- note that $\bar{n}_\text{th}+1$
multiplies the *annihilation* operator:

$$
L_\uparrow = \sqrt{\Gamma\,\bar{n}_\text{th}}\; a_p^\dagger, \qquad
L_\downarrow = \sqrt{\Gamma\,(\bar{n}_\text{th}+1)}\; a_p, \qquad
\Gamma = \dot{\bar{n}}_p / \bar{n}_\text{th}
$$

giving $d\langle n\rangle/dt = \dot{\bar{n}}_p - \Gamma\langle n\rangle$
(Brownnutt Eq. 18): initial slope $\dot{\bar{n}}_p$ from the ground
state, and equilibration at $\langle n\rangle = \bar{n}_\text{th}$, i.e.
$\langle n\rangle(t) = \bar{n}_\text{th}(1 - e^{-\Gamma t})$ from
vacuum. Swapping the two factors would instead give
$d\langle n\rangle/dt =
+\dot{\bar{n}}_p(\langle n\rangle + \bar{n}_\text{th} + 1)$, which has
no fixed point and grows exponentially.

Anomalous heating actually sits in the **infinite-temperature limit**:
$\bar{n}_\text{th} \sim 10^5$-$10^7$ for $\omega_p/2\pi = 1$ MHz at
4-300 K (Brownnutt Sec. III). Taking $\Gamma \to 0$ at fixed
$\Gamma\bar{n}_\text{th} = \dot{\bar{n}}_p$ leaves

$$
L_\uparrow = \sqrt{\dot{\bar{n}}_p}\; a_p^\dagger, \qquad
L_\downarrow = \sqrt{\dot{\bar{n}}_p}\; a_p
$$

so $d\langle n\rangle/dt = \dot{\bar{n}}_p$ exactly and
$\langle n\rangle(t) = \langle n\rangle_0 + \dot{\bar{n}}_p t$ -- the
linear growth from the ground state measured by Turchette et al.
Sec. III.A.3. This is what `motional_heating_ops` builds by default
(`n_bar_env=None`); a finite `n_bar_env` selects the damped bath above,
and `n_bar_env=0` is rejected because a $T = 0$ bath only cools.

**Scaling laws**: $S_E(\omega) \propto d^{-\beta}\, \omega^{-\alpha}$ with
$\alpha \approx 1$ (1/f noise) and $\beta \approx 4$ for planar patch
potentials -- though measured exponents span roughly 2 to 4 across trap
geometries (Brownnutt Sec. IV), which is why `beta` is a parameter of
`heating_rate_from_noise` rather than a constant. Cryogenic cooling to
4 K reduces rates by $\sim 100\times$.

### Motional Dephasing

Trap frequency fluctuations (from voltage noise on DC electrodes) cause
dephasing of motional superpositions:

$$
L_\text{deph} = \sqrt{\gamma_\text{deph}}\; a_p^\dagger a_p
$$

Off-diagonal density matrix elements decay as
$\langle n|\rho|n'\rangle \to \langle n|\rho|n'\rangle\, e^{-\gamma_\text{deph}(n-n')^2 t/2}$.
Since $\gamma_\text{deph}$ multiplies a dimensionless exponent it is a
decay rate in s$^{-1}$, not an angular frequency.

### Qubit Dephasing

Fluctuations in the qubit frequency (from magnetic field noise, AC Stark
shifts) cause pure dephasing:

$$
L_\phi = \sqrt{\gamma_\phi / 2}\; \sigma_z
$$

The pure dephasing rate relates to the coherence time as
$\gamma_\phi = 1/T_2 - 1/(2T_1)$; for hyperfine qubits where
$T_1 \to \infty$, this simplifies to $\gamma_\phi = 1/T_2$.

For a clock qubit with only second-order magnetic sensitivity, write
$f(B) = f_0 + \tfrac12 (d^2f/dB^2)\,B^2$, so
$df/dB = (d^2f/dB^2)\,B_0$ and the frequency spread is

$$
\gamma_\phi \sim 2\pi \left|\frac{d^2f}{dB^2}\right| B_0\,
  \delta B_\text{rms}
$$

(one factor of $B_0$, not two). This is an order-of-magnitude estimate
of $1/T_2$ valid for *fast* (Markovian) field noise only: quasi-static
Gaussian field noise gives Gaussian coherence decay
$e^{-\delta\omega_\text{rms}^2 t^2/2}$, not the exponential a $\sigma_z$
Lindblad term produces. **Not implemented**: no species carries
$df/dB$, $d^2f/dB^2$ or Lande $g_F$, and there is no magnetic field or
gradient object for ions, so magnetic dephasing can only be entered as
the phenomenological `t2` of `SimulationConfig`. Consequently every ion
in a chain necessarily shares one qubit frequency.

### Spontaneous Emission

For optical qubits, the excited state decays at rate $\Gamma_D = 1/\tau_D$:

$$
L_\text{decay} = \sqrt{\Gamma_D}\; \sigma_-
$$

Population in $|e\rangle$ decays to $|g\rangle$ at rate $\Gamma_D$, and
coherence decays at $\Gamma_D / 2$. For ${}^{40}\text{Ca}^+$, $\tau_D = 1.17$ s.
Decay out of the qubit manifold (leakage) is not modeled.

### Off-Resonant Photon Scattering

During Raman-driven gates, off-resonant coupling to the excited $P$ state
causes spontaneous scattering with two components.

**Raman scattering** (inelastic) projects the ion into one of the ground
sublevels at a rate that does not depend on which qubit state it started
from (Ozeri et al., *Phys. Rev. A* **75**, 042329 (2007) Eqs. 9-11). With
equal branching to the two sublevels, half of the $\Gamma_\text{Raman}$
events change the qubit state, so the channel is a **bidirectional pair**:

$$
L_\downarrow = \sqrt{\Gamma_\text{Raman}/2}\; \sigma_-, \qquad
L_\uparrow = \sqrt{\Gamma_\text{Raman}/2}\; \sigma_+
$$

This is population transfer (a depolarizing channel), not a $\sigma_x$
bit flip; a single-direction operator would be pure amplitude damping
and would produce *zero* error from the prepared ground state. Note that
$L_\downarrow$ is the same operator as $L_\text{decay}$ above, so adding
both channels for one physical process double-counts decay. Raman events
that leave the qubit manifold entirely (Ozeri's $\epsilon_D$ leakage into
metastable $D$ levels) are not modeled -- there is no third level.

**Rayleigh scattering** (elastic) causes dephasing without population
transfer:

$$
L_\text{Rayleigh} = \sqrt{\Gamma_\text{el}/4}\; \sigma_z
$$

decaying coherences at $\Gamma_\text{el}/2$. Here $\Gamma_\text{el}$ is
the elastic-scattering **decoherence** rate (Uys et al., *Phys. Rev.
Lett.* **105**, 200401 (2010) Eq. 7), *not* the Rayleigh scattering
rate: elastic events exchange no energy or angular momentum with the
internal state, so they dephase only through the *difference* between
the two qubit states' scattering amplitudes, and
$\Gamma_\text{el} \le \Gamma_\text{Rayleigh}$ always. For clock qubits
the two amplitudes nearly coincide and the suppression is severe,
$\Gamma_\text{el}/\Gamma_\text{Rayleigh} \approx (\omega_0/\Delta)^2$
for $\Delta \gg \omega_0$ (Ozeri Eq. 66) -- e.g. $4\times 10^{-5}$ for a
${}^9$Be$^+$ clock qubit at $\Delta = 2\pi \times 197$ GHz. The
photon-recoil kick accompanying each elastic event (Ozeri's
$\epsilon_R$, a motional error) is not modeled.

For a single-qubit Raman rotation the scattering error is (Ozeri
Eq. 13)

$$
\epsilon_\text{sc} \sim \frac{2\pi\gamma}{3}\,
  \frac{\omega_f}{\lvert\Delta\,(\Delta - \omega_f)\rvert}
$$

with $\gamma$ the excited-state linewidth, $\Delta$ the Raman detuning
and $\omega_f$ the excited-state fine-structure splitting -- so for
$|\Delta| \gg \omega_f$ the error falls as $\Delta^{-2}$,
*quadratically* in the detuning rather than linearly. This sets a
**fundamental error floor** for laser-driven gates ($\sim 10^{-4}$ at
the detunings and powers of Ozeri Tables II-III) that can only be
reduced by increasing $\Delta$ (requiring more laser power) or
eliminated entirely by using microwave-driven gates. For two-qubit
gates at fixed gate time the error is set by available laser power
rather than by $\Delta$ (Ozeri Eq. 28).

### Laser Phase Noise

Phase noise on the laser maps to qubit dephasing. For Raman transitions, the
relevant noise is on the **difference frequency** $\delta\phi_\text{eff} = \delta\phi_1 - \delta\phi_2$,
so common-mode noise from a single laser source is rejected.

TIQS parameterizes this by the **FWHM linewidth** $W$ (rad/s) of the
laser or Raman beat note:

$$
L_\text{phase} = \sqrt{W / 4}\; \sigma_z
$$

A phase-diffusing (white-*frequency*-noise) laser has
$\langle[\varphi(t)-\varphi(0)]^2\rangle = 2Dt$, first-order coherence
$e^{-D|\tau|}$ and hence a Lorentzian spectrum of half width $D$, i.e.
$W = 2D$; the qubit coherence therefore decays at
$1/T_2 = W/2 = \pi W_\text{Hz}$, which is what the operator above
produces. White *phase* noise does not random-walk the phase and gives
no exponential decay, so it is not the right input here; servo bumps and
$1/f$ laser noise give non-exponential decay and need explicit
stochastic averaging. Phase noise *on a resonant drive* is a separate,
transverse channel ($\propto \Omega\varphi(t)\,\sigma_y/2$) and is not
modeled.

### Laser Intensity Noise

Intensity fluctuations cause Rabi frequency errors
$\Omega(t) = \Omega_0(1 + \epsilon(t))$, producing **coherent rotation errors**
rather than decoherence. For a $\pi$-pulse:

$$
1 - F = \frac{\pi^2 \langle\epsilon^2\rangle}{4}
$$

For a resonant single-photon drive $\Omega \propto \sqrt{I}$, so
$\delta\Omega/\Omega = \tfrac12\,\delta I/I$; for a two-photon Raman
drive from one laser $\Omega \propto I$ and
$\delta\Omega/\Omega = \delta I/I$. Model this by classical noise
averaging -- sample $\epsilon$, solve with $H = (1+\epsilon)H_0$,
average the results -- rather than as a Lindblad operator, which would
be dimensionally wrong and wildly overestimate the error.

### Crosstalk

Residual laser intensity on neighboring ions causes spurious rotations,
adding off-target single-qubit rotations on the same axis as the target
drive. Be explicit about which ratio is meant: the *intensity* ratio of
a Gaussian beam of waist $w_0$ at ion spacing $s$ is
$e^{-2s^2/w_0^2}$, so the **Rabi-frequency** fraction (the quantity
`crosstalk_hamiltonian` takes) is $\epsilon = e^{-s^2/w_0^2}$. At
$w_0 = 2\;\mu$m and $s = 5\;\mu$m that is $\epsilon = 1.9\times10^{-3}$
(intensity ratio $3.7\times10^{-6}$), and at $w_0 = 1\;\mu$m it is
utterly negligible, $1.4\times10^{-11}$. Measured nearest-neighbor
crosstalk in addressed-beam systems is typically
$10^{-3}$-$10^{-2}$ in Rabi fraction, i.e. *above* the Gaussian
estimate, because it is limited by aberrations and scattered light
rather than by the ideal beam profile.

### Complete Dissipator

The total Lindblad superoperator sums the channels the solver is given:

$$
\mathcal{D}[\rho] = \sum_p \mathcal{D}_{\text{heat},p}[\rho] + \sum_p \mathcal{D}_{\text{deph},p}[\rho] + \sum_j \mathcal{D}_{\phi,j}[\rho] + \sum_j \mathcal{D}_{\text{decay},j}[\rho] + \sum_j \mathcal{D}_{\text{Raman},j}[\rho]
$$

### Model scope and approximations

- **Uniform per-mode heating.** `motional_heating_ops` builds one bath
  per mode from one rate. Spatially uniform (long-wavelength) field
  noise actually heats only the center-of-mass mode, at
  $N_\text{ion}\dot{\bar{n}}$ (Brownnutt Eqs. 22-23); per-mode rates
  should be derived from mode participation rather than shared.
- **Independent per-ion dephasing.** Magnetic-field noise on a chain is
  largely common-mode, giving collective dephasing
  ($L \propto \sum_i \sigma_z^{(i)}$, which leaves the
  $|01\rangle$-$|10\rangle$ decoherence-free subspace intact) and, for
  slow $1/f$ noise, Gaussian rather than exponential decay. Neither is
  modeled.
- **Scattering.** Only the inelastic Raman branch is wired into the
  simulation runner (from `photon_scattering_rate`); the elastic
  Rayleigh branch has a species- and detuning-dependent branching ratio
  and must be added explicitly as $\Gamma_\text{el}$. No photon recoil
  reaches the motion from any scattering channel.
- **Hamiltonian-valued channels are not collapse operators.**
  `crosstalk_hamiltonian` and `laser_intensity_noise_op` return
  Hamiltonian terms, so the runner does not apply them; add them to a
  gate Hamiltonian yourself.
- **No magnetic-field physics for ions** (see Qubit Dephasing), no
  micromotion coupling, and no leakage level anywhere: qubits are
  strictly two-level.

### References

1. Turchette, Q.A. et al. "Decoherence and decay of motional quantum states
   of a trapped atom coupled to engineered reservoirs." *Phys. Rev. A*
   **62**, 053807 (2000). Eq. (4) thermal-bath dissipator; Sec. III.A.3
   linear anomalous heating.
2. Ozeri, R. et al. "Errors in trapped-ion quantum gates due to spontaneous
   photon scattering." *Phys. Rev. A* **75**, 042329 (2007). Eqs. (9)-(11)
   Raman error structure, Eq. (13) single-qubit scaling, Eq. (28)
   two-qubit error, Eq. (66) Rayleigh clock-qubit suppression.
3. Uys, H. et al. "Decoherence due to elastic Rayleigh scattering."
   *Phys. Rev. Lett.* **105**, 200401 (2010). Eq. (7) defines the elastic
   decoherence rate $\Gamma_\text{el}$.
4. Brownnutt, M. et al. "Ion-trap measurements of electric-field noise near
   surfaces." *Rev. Mod. Phys.* **87**, 1419 (2015). Eq. (12) heating rate,
   Eq. (14) bath operators, Eq. (18) rate equation, Sec. IV distance
   scaling.

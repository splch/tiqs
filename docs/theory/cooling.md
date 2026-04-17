## Cooling Theory

Laser cooling brings trapped ions from thermal energies ($\sim$room temperature
after loading) to the quantum ground state of motion, enabling high-fidelity
quantum gates.

### Doppler Cooling

The first cooling stage uses a laser red-detuned from a strong dipole
transition ($\Gamma/2\pi \sim 20$ MHz). An ion moving toward the beam sees
the light Doppler-shifted closer to resonance, preferentially absorbing
momentum kicks opposing its motion.

The **Doppler temperature limit** (optimized at detuning $\Delta = -\Gamma/2$):

$$
T_D = \frac{\hbar\Gamma}{2k_B}
$$

giving $\sim 0.5$ mK for Ca$^+$. The corresponding mean phonon occupation is:

$$
\bar{n}_D = \frac{\Gamma}{2\omega_z}
$$

For typical trap frequencies $\omega_z / 2\pi \sim 1$-$3$ MHz and
$\Gamma/2\pi \sim 20$ MHz, this gives $\bar{n} \approx 5$-$20$, cooled
significantly but far from the motional ground state.

### Resolved Sideband Cooling

To approach $\bar{n} \to 0$ for high-fidelity gates, **resolved sideband
cooling (RSC)** exploits a narrow transition whose linewidth is much smaller
than the trap frequency ($\Gamma_\text{eff} \ll \omega_z$).

The cooling cycle repeats two steps:

1. A **red sideband** $\pi$-pulse drives
   $|g, n\rangle \to |e, n{-}1\rangle$, removing one phonon.

2. **Optical pumping** returns the ion to $|g\rangle$ via a rapid transition,
   predominantly at the carrier frequency (preserving $n$) in the Lamb-Dicke
   regime.

Each cycle removes one quantum of motion. The steady-state phonon number is:

$$
\bar{n}_\text{SBC} \approx \left(\frac{\Gamma_\text{eff}}{2\omega_z}\right)^2
$$

Since $\omega_z \gg \Gamma_\text{eff}$ in the resolved-sideband regime,
$\bar{n} \ll 1$. Experimentally, $\bar{n} < 0.01$ ($>99\%$ ground-state
probability) is routinely achieved after 20-50 cycles.

**Cooling rate per cycle**: The phonon removal rate scales as
$W_- \propto (\eta\Omega)^2 n / \Gamma_\text{eff}$, while the off-resonant
heating rate scales as $W_+ \propto \eta^2 \Gamma_\text{eff}(n+1)$.

**Total cooling time**: $t_\text{cool} \sim \bar{n}_0 (t_\pi + t_\text{repump})$
where $\bar{n}_0$ is the initial phonon number after Doppler cooling. Typical
total: 1-10 ms per mode. In QCCD architectures, sideband cooling can consume
$\sim 68\%$ of total algorithm runtime.

### EIT Cooling

**Electromagnetically induced transparency (EIT) cooling** uses two laser
beams to create a coherent dark state in a three-level $\Lambda$ system,
producing a narrow Fano-like absorption profile:

- **Carrier absorption is suppressed** (the EIT transparency window)
- **Red sideband absorption is enhanced** (the Fano peak)
- **Blue sideband absorption is suppressed**

The $\Lambda$ system Hamiltonian with pump ($\Omega_p$) and probe ($\Omega_{pr}$)
fields creates a dark state:

$$
|D\rangle = \frac{\Omega_{pr}|1\rangle - \Omega_p|2\rangle}{\sqrt{\Omega_p^2 + \Omega_{pr}^2}}
$$

This state is decoupled from the excited state and does not scatter photons.

The effective EIT linewidth is set by the Rabi frequencies and the natural
linewidth:

$$
\gamma_\text{EIT} \approx \frac{\Omega_{pr}^2\,\Gamma}{\Omega_p^2}
$$

The steady-state phonon number in the ideal case is:

$$
\bar{n}_\text{EIT} \approx \left(\frac{\gamma_\text{EIT}}{2\omega_z}\right)^2
$$

analogous to the resolved-sideband result with $\gamma_\text{EIT}$ replacing
$\Gamma_\text{eff}$. In practice, residual carrier absorption through an
imperfect dark state raises this floor to
$\bar{n} \approx \epsilon\,\gamma_\text{EIT}/(2\omega_z)$, where
$\epsilon \ll 1$ is the suppression factor.

**Bandwidth advantage**: Unlike RSC (which cools one mode at a time), EIT
cooling provides broad cooling bandwidth ($\sim \Omega_p$). By choosing
$\Omega_p$ appropriately, all transverse modes of an $N$-ion chain can be
cooled simultaneously, demonstrated for 40+ ions in $< 300\;\mu$s.

### Sympathetic Cooling

A co-trapped ion of a **different species** (the *coolant*) is laser-cooled
while the computational qubits are cooled **indirectly** through the Coulomb
interaction that couples all ions via shared normal modes. The cooling laser
addresses only the coolant species (far off-resonance from qubit transitions),
so qubit quantum states are preserved.

#### Coolant participation

The cooling rate of mode $m$ depends on how much the coolant ions participate
in that mode. The **coolant participation** is:

$$
P_m = \sum_{k \in \text{coolant}} |b_{k,m}|^2
$$

where $b_{k,m}$ is the mass-weighted eigenvector component of coolant ion $k$
in mode $m$. Since the eigenvectors are orthonormal,
$\sum_i |b_{i,m}|^2 = 1$ for every mode, so $0 \le P_m \le 1$. When all
ions are coolants (single-species chain), $P_m = 1$.

Modes where the coolant has near-zero participation are called **spectator
modes** and cannot be efficiently cooled sympathetically.

#### Sympathetic Doppler limit

The steady-state phonon number per mode under sympathetic Doppler cooling is:

$$
\bar{n}_m = \frac{\Gamma}{2\,\omega_m\,P_m}
$$

where $\Gamma$ is the coolant cooling-transition linewidth and $\omega_m$ is
the mode frequency. The factor $1/P_m$ arises because the effective cooling
rate on mode $m$ is reduced by $P_m$ compared to direct cooling. When
$P_m = 1$, this reduces to the standard Doppler limit $\Gamma/(2\omega)$.

#### Cooling rate

At optimum detuning $\Delta = -\Gamma/2$ and saturation parameter
$s = I/I_\text{sat}$:

$$
\Gamma_m^\text{cool} = \frac{\Gamma}{2}\,\frac{s}{1+s}\,P_m
$$

Each mode's phonon number relaxes exponentially toward the Doppler limit:

$$
\bar{n}(t) = \bar{n}_\text{ss}
  + (\bar{n}_0 - \bar{n}_\text{ss})\,e^{-\Gamma_m^\text{cool}\,t}
$$

#### Sympathetic sideband cooling limit

After Doppler pre-cooling, resolved sideband cooling on the coolant ion
further reduces the phonon number:

$$
\bar{n}_m^\text{SBC} = \frac{1}{P_m}
  \left(\frac{\gamma_\text{eff}}{2\,\omega_m}\right)^2
$$

#### Species pairing considerations

The mass ratio between coolant and qubit species controls the normal mode
structure. Similar masses ($m_\text{cool}/m_\text{qubit} \approx 1$)
maximize mode hybridization, giving all modes significant coolant
participation. Disparate masses cause modes to localize on one species,
creating spectator modes. Common pairings include ${}^9\text{Be}^+$ /
${}^{40}\text{Ca}^+$, ${}^{138}\text{Ba}^+$ / ${}^{171}\text{Yb}^+$,
and same-element isotope pairs like ${}^{40}\text{Ca}^+$ /
${}^{43}\text{Ca}^+$.

### References

1. Wineland, D.J. & Itano, W.M. "Laser cooling of atoms." *Phys. Rev. A*
   **20**, 1521 (1979).
2. Monroe, C. et al. "Resolved-sideband Raman cooling of a bound atom to
   the 3D zero-point energy." *Phys. Rev. Lett.* **75**, 4011 (1995).
3. Lechner, R. et al. "Electromagnetically-induced-transparency ground-state
   cooling of long ion strings." *Phys. Rev. A* **93**, 053401 (2016).
4. Sosnova, K. et al. "Character of motional modes for entanglement and
   sympathetic cooling of mixed-species trapped-ion chains." *Phys. Rev. A*
   **103**, 012610 (2021).
5. Bruzewicz, C.D. et al. "Trapped-ion quantum computing: Progress and
   challenges." *Appl. Phys. Rev.* **6**, 021314 (2019).

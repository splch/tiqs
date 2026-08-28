## Cooling Theory

Laser cooling brings trapped ions from the energies at which they are
captured -- up to $\sim 1$ eV ($\sim 10^4$ K, roughly one tenth of the
trap depth) immediately after loading -- down to the quantum ground
state of motion, enabling high-fidelity quantum gates.

### Doppler Cooling

The first cooling stage uses a laser red-detuned from a strong dipole
transition ($\Gamma/2\pi \sim 20$ MHz). An ion moving toward the beam sees
the light Doppler-shifted closer to resonance, preferentially absorbing
momentum kicks opposing its motion.

The **Doppler temperature limit** (low saturation, optimized at detuning
$\Delta = -\Gamma/2$):

$$
T_D = \frac{\hbar\Gamma}{2k_B}
$$

giving $0.54$ mK for Ca$^+$ ($\Gamma/2\pi = 22.4$ MHz). The
corresponding mean phonon occupation is:

$$
\bar{n}_D = \frac{\Gamma}{2\omega_z}
$$

For typical trap frequencies $\omega_z / 2\pi \sim 1$-$3$ MHz and
$\Gamma/2\pi \sim 20$ MHz, this gives $\bar{n} \approx 3$-$10$ (10.0 at
1 MHz, 3.3 at 3 MHz), cooled significantly but far from the motional
ground state.

**Regime caveats.** Both expressions are the textbook $s \to 0$,
$\bar{n} \gg 1$ limit, and drop three $O(1)$ effects of Leibfried
et al., *Rev. Mod. Phys.* **75**, 281 (2003) Eqs. (105)-(106):

- the optimal detuning is $\Delta = -\Gamma\sqrt{1+s}/2$, so the form
  above assumes low saturation $s = I/I_\text{sat} \ll 1$;
- the emission-recoil geometry factor $\xi$ ($2/5$ for dipole
  radiation) gives $k_B T_\text{min} =
  \hbar\Gamma\,(1+\xi)\sqrt{1+s}/4$, i.e. $0.7\times$ the value above
  as $s \to 0$;
- $\bar{n} = k_B T/(\hbar\omega_z)$ is a classical result, so it
  overestimates when $\Gamma \lesssim 2\omega_z$: Leibfried's
  ${}^9$Be$^+$ example ($\omega_z/2\pi = 11.2$ MHz,
  $\Gamma/2\pi = 19.4$ MHz) gives $0.87$ against a measured $0.47(5)$.
  (TIQS uses the NIST-derived $\Gamma/2\pi = 17.97$ MHz for that Be$^+$
  line, so the same formula returns $0.80$; see
  [species](species.md).)

Unit warning: `tiqs.cooling.doppler_cooled_nbar` takes the trap
frequency in **Hz** (`trap_frequency_hz`), unlike
`sideband_cooling_nbar` and `eit_cooling_nbar`, which take angular
frequencies in rad/s like the rest of the library.

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

Each cycle removes at most one quantum of motion. The steady-state
phonon number is:

$$
\bar{n}_\text{SBC} \approx \left(\frac{\Gamma_\text{eff}}{2\omega_z}\right)^2
$$

Since $\omega_z \gg \Gamma_\text{eff}$ in the resolved-sideband regime,
$\bar{n} \ll 1$. Experimentally, $\bar{n} < 0.01$ ($>99\%$ ground-state
probability) is routinely achieved after 20-50 cycles.

This is the bracket-free form quoted by Wineland et al., *J. Res.
NIST* **103**, 259 (1998) Sec. 3.1. Leibfried RMP Eq. (112) carries an
additional $O(1)$ factor,

$$
\bar{n} \approx
  \left(\frac{\Gamma_\text{eff}}{2\omega_z}\right)^2
  \left[\frac{\tilde{\eta}^2}{\eta^2} + \frac{1}{4}\right],
$$

where $\tilde{\eta}$ is the Lamb-Dicke parameter of the
*spontaneously emitted* photon and $\eta$ that of the cooling drive.
The bracket is $1/4$ for $\tilde{\eta} \ll \eta$ and $\approx 5/4$ for
$\tilde{\eta} = \eta$, so `sideband_cooling_nbar` -- which returns the
bracket-free value -- is accurate only to a factor of a few.

**Cooling and heating rates per cycle**: on the red sideband the phonon
removal rate is $W_- \approx (\eta\Omega)^2 n / \Gamma_\text{eff}$.
Heating is *off-resonant*, so each channel carries its own excitation
probability: off-resonant carrier excitation (probability
$(\Omega/2\omega_z)^2$) followed by a recoil-carrying spontaneous
emission, plus off-resonant blue-sideband excitation (probability
$(\eta\Omega/4\omega_z)^2$) followed by carrier decay,

$$
W_+ \approx \left[\tilde{\eta}^2
    \left(\frac{\Omega}{2\omega_z}\right)^2
  + \eta^2\left(\frac{\Omega}{4\omega_z}\right)^2\right]
  \Gamma_\text{eff}\,(n+1)
$$

whose ratio to $W_-$ reproduces the steady state above,
$W_+/W_- = (\Gamma_\text{eff}/2\omega_z)^2
[\tilde{\eta}^2/\eta^2 + 1/4]$ (Leibfried RMP Eqs. (111)-(112)).
Dropping the excitation probabilities would leave
$W_+/W_- = \Gamma_\text{eff}^2/\Omega^2$, which carries no $\omega_z$
dependence and so cannot produce the $(\Gamma_\text{eff}/2\omega_z)^2$
scaling at all.

**Total cooling time**: $t_\text{cool} \sim \bar{n}_0 (t_\pi + t_\text{repump})$
where $\bar{n}_0$ is the initial phonon number after Doppler cooling. Typical
total: 1-10 ms per mode. In the Quantinuum QCCD demonstrations of Pino
et al. (2021) and Moses et al. (2023), cooling consumes as much as
$\sim 68\%$ of the total algorithm duration, as tabulated by Fallek
et al., *Nat. Commun.* **15**, 1089 (2024).

**What `sideband_cooling_simulate` models**: sequential fixed-duration
RSB $\pi$-pulses ($t_\pi = \pi/|\eta\Omega|$) alternating with
optical-pumping segments. It has no off-resonant carrier or blue
sideband and no repump recoil, so $|g, 0\rangle$ is an exact dark
state and the $(\Gamma_\text{eff}/2\omega_z)^2$ floor above is *not*
reproduced -- the mode frequency does not even enter. What limits it
instead is the fixed pulse duration: the RSB coupling grows as
$\eta\Omega\sqrt{n}$, so a $t_\pi$ calibrated on
$|g,1\rangle \to |e,0\rangle$ transfers only $\sin^2(\pi\sqrt{n}/2)$
out of $|g,n\rangle$ and leaves the Fock states with $\sqrt{n}$ even
($n = 4, 16, 36, \ldots$) completely dark. Starting from
$\bar{n} = 3$ the sequence therefore stalls near $\bar{n} = 1.4$ no
matter how many cycles are run. Real sequences vary the pulse duration
between cycles.

### EIT Cooling

**Electromagnetically induced transparency (EIT) cooling** uses two laser
beams to create a coherent dark state in a three-level $\Lambda$ system,
producing a narrow Fano-like absorption profile:

- **Carrier absorption is suppressed** (the EIT transparency window)
- **Red sideband absorption is enhanced** (the Fano peak)
- **Blue sideband absorption is suppressed**

Take the strong pump $\Omega_p$ on $|1\rangle \to |e\rangle$ and the
weak probe $\Omega_{pr}$ on $|2\rangle \to |e\rangle$, both detuned by
$\Delta$ from the excited state. The dark state is the combination
whose two excitation amplitudes cancel, so each ground state carries
the *other* leg's Rabi frequency:

$$
|D\rangle = \frac{\Omega_{pr}|1\rangle - \Omega_p|2\rangle}{\sqrt{\Omega_p^2 + \Omega_{pr}^2}}
$$

(Morigi, Eschner & Keitel, *Phys. Rev. Lett.* **85**, 4458 (2000)
Eq. (1)). For $\Omega_p \gg \Omega_{pr}$ the ion is pumped almost
entirely into the probe-coupled state $|2\rangle$. This state is
decoupled from the excited state and does not scatter photons.

The narrow bright resonance beside the transparency window is the
Autler-Townes dressed state of $\{|1\rangle, |e\rangle\}$. It sits at
the AC Stark shift

$$
\delta_\text{AC} = \frac{\sqrt{\Delta^2 + \Omega_p^2} - |\Delta|}{2}
  \approx \frac{\Omega_p^2}{4|\Delta|}
$$

and its width is that dressed state's excited-state admixture times
the natural linewidth:

$$
\gamma_\text{EIT} = \Gamma \sin^2\theta
  \approx \frac{\Gamma\,\Omega_p^2}{4\Delta^2}
  = \frac{\Gamma\,\delta_\text{AC}}{|\Delta|},
\qquad
\tan\theta = \frac{\sqrt{\Delta^2 + \Omega_p^2} - \Delta}{\Omega_p}
$$

as the full width at half maximum, valid for
$\Delta \gg \Omega_p, \Gamma$ and $\Omega_{pr} \ll \Omega_p$. The
*probe* Rabi frequency does not set this width: it enters only as an
overall scattering-rate prefactor, which cancels in the steady state
(raising it broadens the feature by saturation, never narrows it).
Cooling is engineered by tuning the bright resonance onto the mode,
$\delta_\text{AC} = \omega_z$, which fixes $\Omega_p$ once $\Delta$ is
chosen.

The steady-state phonon number for a perfect dark state is then

$$
\bar{n}_\text{EIT} \approx \left(\frac{\gamma_\text{EIT}}{4\omega_z}\right)^2
  = \left(\frac{\Gamma}{4|\Delta|}\right)^2
$$

(Leibfried RMP Eq. (128); Morigi, *Phys. Rev. A* **67**, 033402 (2003)
Eq. (32)), so $\Delta$ trades the achievable occupation against the
bandwidth below. Residual carrier absorption through an *imperfect*
dark state adds to this floor rather than scaling it:

$$
\bar{n} \approx \epsilon
  + \left(\frac{\gamma_\text{EIT}}{4\omega_z}\right)^2,
\qquad
\epsilon = \frac{W(\text{carrier})}{W(\text{red sideband})} \ll 1
$$

Carrier scattering is recoil diffusion: it enters the Lamb-Dicke rate
coefficients $A_+$ and $A_-$ equally, so it cancels from the cooling
rate $A_- - A_+$ and survives only in the numerator of
$\bar{n} = A_+/(A_- - A_+)$. The carrier-limited floor is therefore
$\bar{n} \to \epsilon$ (up to an $O(1)$ recoil/geometry projection),
with no $\gamma_\text{EIT}/\omega_z$ suppression -- which is what
`eit_cooling_nbar` returns. It takes $\gamma_\text{EIT}$ (as the FWHM)
and $\epsilon$ as inputs; neither is derived from a level structure.

**Bandwidth advantage**: unlike RSC, which cools one mode at a time,
EIT cooling cools every mode that falls inside the bright resonance.
That window is centred on $\delta_\text{AC}$ and has width
$\sim\gamma_\text{EIT} = \Gamma\Omega_p^2/(4\Delta^2)$, so it is set
by $\Gamma$ and $\Delta$; $\Omega_p$ alone only positions the window.
Lechner et al. ground-state cooled all radial modes of strings of up to
18 ions in under 1 ms with a three-level $\Lambda$ scheme (a single
short pulse requires the modes bunched within a few hundred kHz),
while Feng et al. cooled the complete transverse spectrum of up to 40
ions -- over 3 MHz of bandwidth -- in less than 300 $\mu$s, using a
four-level *tripod* scheme rather than the $\Lambda$ system derived
above.

### Sympathetic Cooling

A co-trapped ion of a **different species** (the *coolant*) is laser-cooled
while the computational qubits are cooled **indirectly** through the Coulomb
interaction that couples all ions via shared normal modes. The cooling laser
addresses only the coolant species (far off-resonance from qubit transitions),
so qubit quantum states are preserved.

#### Coolant participation

The cooling *rate* of mode $m$ depends on how much the coolant ions
participate in that mode. The **coolant participation** is:

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
\bar{n}_m = \frac{\Gamma}{2\,\omega_m}
  + \frac{\dot{n}_{\text{ext},m}}{\Gamma_m^\text{cool}}
$$

where $\Gamma$ is the coolant cooling-transition linewidth and $\omega_m$ is
the mode frequency. The first term is the ordinary Doppler limit and is
**independent of the coolant participation**: a friction force on the
coolant damps mode $m$ at $\alpha|b_{c,m}|^2/m_c$, while photon-recoil
momentum diffusion feeds the same mass-weighted normal coordinate at
$D_p|b_{c,m}|^2/m_c$, so the geometric factor cancels in the balance
$E_\text{ss} = D_p/\alpha$. Wübbena et al., *Phys. Rev. A* **85**,
043412 (2012) show the cancellation explicitly in their Eqs. (21)-(23),
state it in the text at Eq. (26), and give $E_D = \hbar\Gamma/2$ in
Eq. (27). $P_m$ sets how *fast* a mode approaches the limit, not where
the limit lies -- a slower approach does not move a fixed point.

Participation returns only through *external* heating (the second
term, structurally Wübbena Eq. (32)): electric-field-noise heating does
not carry the same $|b_{c,m}|^2/m_c$ factor, so it does not cancel, and
a weakly participating mode equilibrates higher. In TIQS this term is
the optional `ndot_ext` argument, in the same quanta/s units as the
heating rate that `tiqs.noise.motional_heating_ops` injects as a
Lindblad channel.

#### Cooling rate

The rate is set by the coolant's **recoil frequency**, not by its
linewidth. Linearizing the velocity-dependent radiation-pressure force

$$
F(v) = \frac{\hbar k \Gamma}{2}\,
  \frac{s}{1 + s + \bigl(2(\Delta - kv)/\Gamma\bigr)^2},
\qquad s = I/I_\text{sat}
$$

about $v = 0$ gives the damping coefficient
$\alpha = -\partial F/\partial v|_{v=0}$; the mode energy (hence
$\langle n\rangle$) relaxes at $P_m\,\alpha/m_c$:

$$
\Gamma_m^\text{cool}
  = -8\,\omega_R\,
    \frac{s\,(\Delta/\Gamma)}
         {\bigl[1 + s + (2\Delta/\Gamma)^2\bigr]^2}\,P_m,
\qquad
\omega_R = \frac{\hbar k_c^2}{2\,m_c}
$$

At $\Delta = -\Gamma/2$ this reduces to
$\Gamma_m^\text{cool} = 4\,\omega_R\,s/(2+s)^2\,P_m$, whose global
maximum over both $s$ and $\Delta$ is $(\omega_R/2)\,P_m$, attained at
$s = 2$. The linewidth sets no scale here -- it enters only through the
dimensionless ratio $2\Delta/\Gamma$. Note also that
$\Delta = -\Gamma/2$ is the *temperature* optimum; the rate alone peaks
at $2\Delta/\Gamma = -\sqrt{(1+s)/3}$.

For scale, the $s = 2$, $P_m = 1$ ceiling $\omega_R/2$ is
$2\pi \times 113$ kHz for a ${}^9$Be$^+$ coolant and
$2\pi \times 4.3$ kHz for ${}^{171}$Yb$^+$. The resonant photon
*scattering* rate $(\Gamma/2)\,s/(1+s)$ is a different quantity
altogether: at $s = 1$ it exceeds that ceiling by $40\times$ for
Be$^+$ and $1150\times$ for Yb$^+$, and would damp a 1 MHz mode in
$\sim 3\%$ of one trap period, violating the secular requirement
$\Gamma_m^\text{cool} \ll \omega_m \ll \Gamma$.

Each mode's phonon number relaxes exponentially toward the limit:

$$
\bar{n}(t) = \bar{n}_\text{ss}
  + (\bar{n}_0 - \bar{n}_\text{ss})\,e^{-\Gamma_m^\text{cool}\,t}
$$

#### Sympathetic sideband cooling limit

After Doppler pre-cooling, resolved sideband cooling on the coolant ion
further reduces the phonon number:

$$
\bar{n}_m^\text{SBC}
  = \left(\frac{\gamma_\text{eff}}{2\,\omega_m}\right)^2
  + \frac{\dot{n}_{\text{ext},m}}{\Gamma_m^\text{cool}}
$$

The same cancellation applies: $A_+$ and $A_-$ both carry
$\eta_{c,m}^2 \propto P_m$, so $\bar{n} = A_+/(A_- - A_+)$ is
participation-independent, and only external heating reintroduces a
$P_m$ dependence through $\Gamma_m^\text{cool} \propto P_m$. The first
term drops the same $O(1)$ Leibfried bracket as
$\bar{n}_\text{SBC}$ above.

#### Species pairing considerations

The mass ratio between coolant and qubit species controls the normal mode
structure. Similar masses ($m_\text{cool}/m_\text{qubit} \approx 1$)
maximize mode hybridization, giving all modes significant coolant
participation. Disparate masses cause modes to localize on one species,
creating spectator modes. Pairings in experimental use include
${}^9\text{Be}^+$ / ${}^{24}\text{Mg}^+$, ${}^{40}\text{Ca}^+$ /
${}^{88}\text{Sr}^+$, ${}^{138}\text{Ba}^+$ / ${}^{171}\text{Yb}^+$,
and same-element isotope pairs such as ${}^{40}\text{Ca}^+$ /
${}^{43}\text{Ca}^+$. A deliberately disparate pair
(${}^9\text{Be}^+$ / ${}^{40}\text{Ca}^+$, mass ratio 4.4) is the worst
case by this criterion and produces strongly localized spectator modes;
see Sosnova et al. (2021) on mode character.

### Model scope and approximations

The cooling module returns *analytic estimates*, not solutions of the
optical Bloch equations:

- `doppler_cooled_nbar`, `sideband_cooling_nbar`, `eit_cooling_nbar`
  and the two sympathetic limits are closed-form Lamb-Dicke
  rate-equation results. Their $O(1)$ prefactors are dropped (the
  Doppler geometry factor $\xi$ and $\sqrt{1+s}$, the Leibfried
  Eq. (112) bracket, the EIT recoil/laser-mode projection
  $\alpha/\cos^2\theta$ of Schmidt-Kaler et al. (2001)), and none of
  $\Gamma_\text{eff}$, $\gamma_\text{EIT}$ or $\epsilon$ is derived from
  a level structure -- they are inputs.
- `sideband_cooling_simulate` solves an idealized RSB-only master
  equation: no off-resonant carrier or blue sideband, no repump recoil,
  and no check of the resolved-sideband condition, so it has no cooling
  floor of its own (see above).
- Sympathetic cooling is applied as a per-mode thermal Lindblad channel
  with a single coolant species and the laser-geometry projection
  $l_x^2 = |\hat{k}\cdot\hat{e}_m|^2$ taken as 1; a mixed coolant set
  is not representable. Target occupations comparable to the Fock
  cutoff are rejected rather than silently truncated, so sympathetic
  Doppler cooling needs a generous `n_fock` (a Be$^+$ coolant on a
  1.2 MHz mode targets $\bar{n} \approx 7.6$).
- Not modeled anywhere: micromotion, mode-mode (Kerr) coupling,
  spectator-mode heating during a cooling pulse, motional recoil during
  optical pumping, and polarization-gradient or exchange-cooling
  protocols.

### References

1. Wineland, D.J. & Itano, W.M. "Laser cooling of atoms." *Phys. Rev. A*
   **20**, 1521 (1979). Velocity-dependent radiation-pressure force and
   the damping coefficient behind the sympathetic cooling rate.
2. Stenholm, S. "The semiclassical theory of laser cooling."
   *Rev. Mod. Phys.* **58**, 699 (1986).
3. Monroe, C. et al. "Resolved-sideband Raman cooling of a bound atom to
   the 3D zero-point energy." *Phys. Rev. Lett.* **75**, 4011 (1995).
4. Wineland, D.J. et al. "Experimental issues in coherent
   quantum-state manipulation of trapped atomic ions."
   *J. Res. NIST* **103**, 259 (1998).
5. Morigi, G., Eschner, J. & Keitel, C.H. "Ground state laser cooling
   using electromagnetically induced transparency."
   *Phys. Rev. Lett.* **85**, 4458 (2000).
6. Schmidt-Kaler, F., Eschner, J., Morigi, G., Roos, C.F., Leibfried,
   D., Mundt, A. & Blatt, R. "Laser cooling with electromagnetically
   induced transparency: application to trapped samples of ions or
   neutral atoms." *Appl. Phys. B* **73**, 807 (2001)
   (arXiv:quant-ph/0107087). Eqs. (1)-(3) for the Lamb-Dicke rate
   coefficients and the $O(1)$ recoil/laser-mode projections.
7. Morigi, G. "Cooling atomic motion with quantum interference."
   *Phys. Rev. A* **67**, 033402 (2003).
8. Leibfried, D., Blatt, R., Monroe, C. & Wineland, D. "Quantum
   dynamics of single trapped ions." *Rev. Mod. Phys.* **75**, 281
   (2003). Eqs. (105)-(106) Doppler limit, (111)-(112) sideband
   cooling, (119)-(128) EIT cooling.
9. Wübbena, J.B., Amairi, S., Mandel, O. & Schmidt, P.O. "Sympathetic
   cooling of mixed-species two-ion crystals for precision
   spectroscopy." *Phys. Rev. A* **85**, 043412 (2012). Eqs. (21)-(23)
   participation scaling, (26)-(27) participation-independent limit,
   (32) externally heated limit.
10. Home, J.P. "Quantum science and metrology with mixed-species ion
    chains." *Adv. At. Mol. Opt. Phys.* **62**, 231 (2013).
11. Lechner, R. et al. "Electromagnetically-induced-transparency ground-state
    cooling of long ion strings." *Phys. Rev. A* **93**, 053401 (2016).
    Up to 18 ions, all radial modes, under 1 ms ($\Lambda$ scheme).
12. Bruzewicz, C.D. et al. "Trapped-ion quantum computing: Progress and
    challenges." *Appl. Phys. Rev.* **6**, 021314 (2019).
13. Feng, L. et al. "Efficient ground-state cooling of large trapped-ion
    chains with an electromagnetically-induced-transparency tripod
    scheme." *Phys. Rev. Lett.* **125**, 053001 (2020). Up to 40 ions,
    $>3$ MHz bandwidth, $<300\;\mu$s, four-level tripod.
14. Sosnova, K. et al. "Character of motional modes for entanglement and
    sympathetic cooling of mixed-species trapped-ion chains." *Phys. Rev. A*
    **103**, 012610 (2021). Mode character and the participation
    definition (Eqs. 4 and 12); it contains no cooling-rate formula.
15. Fallek, S.D. et al. "Rapid exchange cooling with trapped ions."
    *Nat. Commun.* **15**, 1089 (2024). Source of the $\sim 68\%$
    runtime figure, tabulated from Pino et al. (2021) and Moses
    et al. (2023).

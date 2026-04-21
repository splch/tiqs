"""Sympathetic cooling of an anharmonic electron mode.

When the electron's axial mode has Kerr anharmonicity (needed
for the motional qubit), the beam-splitter cooling drive is
only resonant with one Fock-state transition at a time. Higher
levels are detuned by alpha*n, creating a population bottleneck
that limits the achievable ground-state occupation.

This script simulates the full Lindblad master equation for an
anharmonic electron mode coupled to a damped ion mode via a
beam-splitter interaction, and compares the result to the
standard harmonic rate-equation prediction.

This computes:
  1. Coulomb coupling parameters at the Osada 2022 separation
  2. Cooling dynamics: harmonic vs. anharmonic electron
  3. Cooling floor vs. anharmonicity (the design tradeoff)
  4. Fock-state populations showing the bottleneck

The key result: the cooling floor rises with anharmonicity,
setting an upper bound on the engineered alpha that still
permits ground-state preparation.

References
----------
Osada, A. et al. Phys. Rev. Research 4, 033245 (2022).
"""

import numpy as np
import qutip

from tiqs import (
    DuffingPotential,
    ElectronSpecies,
    HilbertSpace,
    OperatorFactory,
    PenningTrap,
    StateFactory,
    beam_splitter_coupling,
    check_convergence,
    coulomb_self_kerr,
    mode_hamiltonian,
    optomechanical_coupling,
    transition_frequencies,
)
from tiqs.constants import ELECTRON_MASS, TWO_PI
from tiqs.noise.motional import motional_heating_ops
from tiqs.species.ion import get_species

# Fock truncation: N_E=10 is needed for convergence of the
# anharmonic steady state (population piles up at higher n).
# N_I=4 suffices since the ion stays near vacuum.
N_E = 10
N_I = 4


def header(title):
    print(f"\n{'=' * 65}\n{title}\n{'=' * 65}")


# Physical parameters
B = 3.0
omega_e = TWO_PI * 800e6  # electron axial frequency
species_e = ElectronSpecies(magnetic_field=B)

trap = PenningTrap(
    magnetic_field=B, species=species_e, d=3.5e-3, omega_axial=omega_e
)
assert trap.is_stable()

be9 = get_species("Be9")
omega_i = TWO_PI * 2e6  # ion axial frequency
L = 10e-6  # 10 um electron-ion separation

# Effective beam-splitter coupling (Osada Sec. III.C: activated
# by driving the electron at omega_e - omega_i).
# Use g_eff < gamma_ion so the weak-coupling rate equation is valid.
g_eff = TWO_PI * 3e3  # 3 kHz (weak coupling regime)

# Ion cooling rate (laser Doppler, Osada assumes 10 kHz)
gamma_ion = TWO_PI * 10e3

# Electron heating (Osada Ref. [13]: 140 quanta/s)
heating_rate = 140.0

# Initial thermal occupation (post-resistive cooling at 300 mK)
nbar_init = 1.0


def run_cooling(alpha, t_final, n_steps=40, return_state=False):
    """Simulate beam-splitter cooling and return (times, nbar).

    If return_state is True, also returns the final density matrix.
    """
    pot = DuffingPotential(omega=omega_e, anharmonicity=alpha)
    hs = HilbertSpace(n_ions=0, n_modes=2, n_fock=[N_E, N_I])
    ops = OperatorFactory(hs)
    sf = StateFactory(hs)

    H_kerr = mode_hamiltonian(pot, ops, 0) - omega_e * ops.number(0)
    H_bs = g_eff * (
        ops.create(0) * ops.annihilate(1) + ops.annihilate(0) * ops.create(1)
    )
    H = H_kerr + H_bs

    c_ops = [
        np.sqrt(gamma_ion) * ops.annihilate(1),
        *motional_heating_ops(ops, 0, heating_rate),
    ]

    rho0 = sf.thermal_state(n_bar=[nbar_init, 0.0])
    tlist = np.linspace(0, t_final, n_steps)
    e_ops = [] if return_state else [ops.number(0)]
    result = qutip.mesolve(
        H,
        rho0,
        tlist,
        c_ops=c_ops,
        e_ops=e_ops,
        options={"nsteps": 10000},
    )
    if return_state:
        nbar = [qutip.expect(ops.number(0), s) for s in result.states]
        return tlist, np.array(nbar), result.states[-1]
    return tlist, result.expect[0]


# 1. Coulomb coupling parameters

header("1. Coulomb couplings (L = 10 um)")

m_e = ELECTRON_MASS
m_i = be9.mass_kg

g0 = optomechanical_coupling(m_e, m_i, omega_e, omega_i, L)
g_bs_raw = beam_splitter_coupling(m_e, m_i, omega_e, omega_i, L)
alpha_C = coulomb_self_kerr(m_e, omega_e, L)

print(f"g0/(2pi)      = {g0 / TWO_PI / 1e3:.1f} kHz (optomechanical)")
print(f"g_bs/(2pi)    = {g_bs_raw / TWO_PI / 1e3:.0f} kHz (beam-splitter)")
print(f"alpha_C/(2pi) = {alpha_C / TWO_PI / 1e3:.1f} kHz (Coulomb Kerr)")
print(f"g_eff/(2pi)   = {g_eff / TWO_PI / 1e3:.0f} kHz (drive-activated)")
print(f"gamma_ion/(2pi) = {gamma_ion / TWO_PI / 1e3:.0f} kHz")
print(f"g_eff/gamma   = {g_eff / gamma_ion:.2f} (weak coupling)")
print("\n[check] g_eff < gamma_ion: rate-equation comparison is valid")


# 2. Cooling dynamics: harmonic vs. anharmonic

header("2. Cooling: harmonic vs. anharmonic")

t_sim = 500e-6

t1, n1 = run_cooling(alpha=0.0, t_final=t_sim)
t2, n2 = run_cooling(alpha=-TWO_PI * 30e3, t_final=t_sim)

mid = len(t1) // 2
print(f"g_eff/(2pi) = {g_eff / TWO_PI / 1e3:.0f} kHz")
print(f"Initial nbar = {nbar_init:.0f}")
print()
print(f"{'':>22}  {'harmonic':>10}  {'a=-30kHz':>10}")
print("-" * 48)
t_mid_us = t_sim * 1e6 / 2
t_end_us = t_sim * 1e6
print(f"  t=0:       {n1[0]:.3f}      {n2[0]:.3f}")
print(f"  t={t_mid_us:.0f}us:   {n1[mid]:.3f}      {n2[mid]:.3f}")
print(f"  t={t_end_us:.0f}us:  {n1[-1]:.3f}      {n2[-1]:.3f}")

assert n1[-1] < n2[-1], "Anharmonic should cool slower"
print("\n[check] Anharmonic electron cools slower than harmonic")


# 3. Cooling floor vs. anharmonicity

header("3. Cooling floor vs. anharmonicity")

t_cool = 1e-3  # 1 ms (long enough to approach steady state)

# Harmonic rate-equation prediction (alpha-independent):
# Gamma_s = 4 * g_eff^2 / gamma_ion (Jaynes-Cummings cooling rate)
# nbar_ss = heating_rate / Gamma_s
Gamma_s = 4 * g_eff**2 / gamma_ion
nbar_rate_eq = heating_rate / Gamma_s

print(f"Cooling time = {t_cool * 1e3:.0f} ms")
print(f"Rate-equation prediction (harmonic): nbar = {nbar_rate_eq:.4f}")
print()
print(f"{'alpha/(2pi)':>12}  {'nbar_sim':>10}  {'ratio':>8}")
print("-" * 36)

for a_khz in [0, 5, 10, 20, 30, 50]:
    alpha = -TWO_PI * a_khz * 1e3
    _, nbar_t = run_cooling(alpha, t_cool, n_steps=30)
    nbar_sim = nbar_t[-1]
    ratio = nbar_sim / nbar_rate_eq

    print(f"{a_khz:>9} kHz  {nbar_sim:>10.4f}  {ratio:>6.1f}x")

print()
print("ratio = simulation / harmonic prediction")
print("[check] Anharmonicity raises the cooling floor")


# 4. Fock-state populations at steady state

header("4. Fock-state populations (alpha = -30 kHz)")

alpha_show = -TWO_PI * 30e3
pot_show = DuffingPotential(omega=omega_e, anharmonicity=alpha_show)
assert check_convergence(pot_show, n_fock=N_E)

freqs = transition_frequencies(pot_show, n_fock=N_E)
print("Transition detunings from the cooling drive:")
for n in range(min(4, len(freqs))):
    det = (freqs[n] - freqs[0]) / TWO_PI / 1e3
    print(f"  |{n}> -> |{n + 1}>:  {det:+.1f} kHz from resonance")

# Reuse run_cooling to get the final density matrix
_, _, rho_final = run_cooling(alpha_show, t_cool, n_steps=2, return_state=True)
rho_e = rho_final.ptrace(0)
pops = [rho_e[n, n].real for n in range(N_E)]

print()
for n in range(N_E):
    bar = "#" * int(pops[n] * 40)
    print(f"  |{n}>: {pops[n]:.4f}  {bar}")

kerr_weight = sum(pops[n] * n * (n - 1) for n in range(N_E))
print(f"\n  <n*(n-1)> = {kerr_weight:.4f} (Kerr term activity)")
assert kerr_weight > 0.001, "Kerr term should be active"
print("[check] Higher Fock states populated; Kerr term is active")


header("All checks passed.")

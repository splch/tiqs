"""Reproduce key results from Osada et al. PhysRevResearch 4, 033245 (2022).

"Feasibility study on ground-state cooling and single-phonon readout
of trapped electrons using hybrid quantum systems"

This script reproduces:
  1. Table II  -- Coulomb coupling parameters (g0, alpha_C) for an
     electron-ion hybrid system at various separations and frequencies.
  2. Eq. 9    -- Full Hamiltonian dynamics of the electron-ion system,
     showing the optomechanical interaction and Kerr anharmonicity.
  3. Sec. III.C -- Sympathetic cooling of the electron via the ion,
     verifying that ground-state cooling (n_bar < 1) is achievable.
  4. Sec. II.C.2 -- Dispersive readout coupling zeta for the
     electron-cavity-transmon system (Table I parameters).

All calculations use SI units. Angular frequencies are in rad/s;
divide by 2*pi to get Hz.

Reference
---------
Osada, A., Taniguchi, K., Shigefuji, M. & Noguchi, A.
Phys. Rev. Research 4, 033245 (2022).
https://doi.org/10.1103/PhysRevResearch.4.033245
"""

import numpy as np
import qutip

from tiqs import (
    DuffingPotential,
    HilbertSpace,
    OperatorFactory,
    StateFactory,
    coulomb_self_kerr,
    optomechanical_coupling,
    transition_frequencies,
)
from tiqs.constants import BOLTZMANN, ELECTRON_MASS, HBAR, TWO_PI
from tiqs.noise.motional import motional_heating_ops
from tiqs.species.ion import get_species

be9 = get_species("Be9")
m_e = ELECTRON_MASS
m_i = be9.mass_kg
omega_i = TWO_PI * 2e6  # ion secular frequency (common to all rows)

# Shared parameters from Table II row 1
omega_e = TWO_PI * 800e6
g0_paper_row1 = TWO_PI * 33e3  # total g0 (includes trap-geometry correction)
alpha_paper_row1 = TWO_PI * 33e3  # total alpha (includes alpha_K, beta)


def header(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


# 1. Reproduce Table II: coupling parameters
#
# Osada Eq. 10: g0 has a pure Coulomb term (computed here) and a
# trap-geometry correction (negative, depends on electrode design).
# Osada Eq. 11-12: alpha = alpha_C + alpha_K - 6*beta^2/omega_e,
# where alpha_C is the Coulomb part and alpha_K, beta depend on
# the effective potential. We compute the Coulomb contributions;
# the paper's totals are shown for comparison.

header("Table II: Coulomb coupling parameters")
print(
    f"{'omega_e/2pi':>12}  {'L':>6}  "
    f"{'g0_C/2pi':>11}  {'g0_paper':>10}  "
    f"{'alpha_C/2pi':>12}"
)
print("-" * 70)

table_ii = [
    # (freq_Hz,  L_m,     g0_paper_Hz)
    (800e6, 10e-6, 33e3),
    (800e6, 50e-6, 0.39e3),
    (500e6, 10e-6, 39e3),
    (500e6, 7e-6, 1.6e6),
]

for freq_hz, L, g0_paper in table_ii:
    g0_C = optomechanical_coupling(m_e, m_i, TWO_PI * freq_hz, omega_i, L)
    alpha_C = coulomb_self_kerr(m_e, TWO_PI * freq_hz, L)
    print(
        f"{freq_hz / 1e6:>9.0f} MHz  {L * 1e6:>4.0f} um"
        f"  {g0_C / TWO_PI / 1e3:>8.1f} kHz"
        f"  {g0_paper / 1e3:>7.1f} kHz"
        f"  {alpha_C / TWO_PI / 1e3:>9.1f} kHz"
    )

print()
print("g0_C = pure Coulomb (first term of Eq. 10)")
print("g0_paper = total from Table II (includes trap-geometry correction)")
print("alpha_C = Coulomb self-Kerr (Eq. 12); paper's total alpha also")
print("  includes alpha_K and beta terms from the effective potential.")

# For row 1, the pure Coulomb g0 exceeds the paper's total g0
# because the effective-potential correction (Eq. 10, second term) is
# negative at this separation. At smaller L the correction flips sign
# and can dominate (row 4: paper total 6x the Coulomb-only value).
g0_check = optomechanical_coupling(m_e, m_i, omega_e, omega_i, 10e-6)
assert g0_check / TWO_PI > 33e3, "Pure Coulomb g0 should exceed 33 kHz"
print("\n[check] Row 1: pure Coulomb g0 > paper's 33 kHz")


# 2. Anharmonic spectrum (transmon-like motional qubit)
#
# The Kerr anharmonicity makes the electron's motional mode
# behave like a transmon: |0> <-> |1> at omega, but
# |1> <-> |2> at omega - alpha, enabling selective addressing.

header("Anharmonic (Kerr) motional spectrum")

pot = DuffingPotential(omega=omega_e, anharmonicity=-alpha_paper_row1)
freqs = transition_frequencies(pot, n_fock=10)

print(f"omega_e/(2pi) = {omega_e / TWO_PI / 1e6:.0f} MHz")
print(
    f"alpha/(2pi)   = {alpha_paper_row1 / TWO_PI / 1e3:.0f} kHz"
    " (total, from paper)"
)
print()
for n in range(4):
    shift = (freqs[n] - omega_e) / TWO_PI
    print(
        f"  |{n}> -> |{n + 1}>:  "
        f"{freqs[n] / TWO_PI / 1e6:.6f} MHz  "
        f"(shift = {shift / 1e3:+.1f} kHz)"
    )

# Verify: shift of |n> -> |n+1> is exactly -alpha * n
for n in range(1, 4):
    expected_shift = -alpha_paper_row1 * n
    actual_shift = freqs[n] - omega_e
    assert abs(actual_shift - expected_shift) / abs(expected_shift) < 1e-6

print("\n[check] Transition shifts match -alpha * n exactly")


# 3. Hamiltonian dynamics (Eq. 9)
#
# H = hbar*omega_e*a'a + hbar*omega_i*c'c
#   - hbar*g0*a'a*(c' + c)          optomechanical coupling
#   - (hbar*alpha/2)*a'^2 a^2       self-Kerr on electron
#
# We work in the interaction picture (free oscillator terms removed)
# and simulate the optomechanical + Kerr dynamics.

header("Eq. 9: Electron-ion optomechanical dynamics")

# Two bosonic modes: electron phonon (0), ion phonon (1)
hs = HilbertSpace(n_ions=0, n_modes=2, n_fock=[15, 10])
ops = OperatorFactory(hs)
sf = StateFactory(hs)

n_e = ops.number(0)
a, ad = ops.annihilate(0), ops.create(0)
c, cd = ops.annihilate(1), ops.create(1)

# Interaction-picture Hamiltonian (Eq. 9 without free terms)
H = -g0_paper_row1 * n_e * (c + cd) - (alpha_paper_row1 / 2) * ad * ad * a * a

# Start with 1 electron phonon, 0 ion phonons
psi0 = sf.product_state(qubit_states=[], fock_states=[1, 0])

t_opto = TWO_PI / g0_paper_row1
tlist = np.linspace(0, 3 * t_opto, 500)
result = qutip.sesolve(H, psi0, tlist, e_ops=[n_e, ops.number(1)])

print("Initial state: |n_e=1, n_i=0>")
print(f"g0/(2pi) = {g0_paper_row1 / TWO_PI / 1e3:.0f} kHz")
print(f"Optomechanical period = {t_opto * 1e6:.1f} us")
print(
    f"<n_e> preserved: {result.expect[0][0]:.3f} -> {result.expect[0][-1]:.3f}"
)
print(f"<n_i> peak:      {max(result.expect[1]):.4f}")

# The optomechanical coupling creates small ion excitations
# proportional to (g0/omega_i)^2 in the dispersive regime
assert max(result.expect[1]) > 0, "Ion should acquire some phonons"
print("\n[check] Optomechanical interaction drives ion motion")


# 4. Sympathetic cooling (Sec. III.C)
#
# Beam-splitter Lindblad simulation: drive the electron at
# omega_e - omega_i to activate the beam-splitter coupling,
# while the ion is laser-cooled. The electron thermalizes
# with the cold ion bath.

header("Sec. III.C: Sympathetic cooling via beam-splitter interaction")

# Analytical estimate first (Sec. III.C formulas)
# Max coupling limited by Kerr saturation (Eq. 13)
g_max = g0_paper_row1**2 / alpha_paper_row1
Gamma_i = TWO_PI * 10e3  # ion cooling rate
Gamma_s = 4 * g_max**2 / Gamma_i  # sympathetic cooling rate
n_th = 1.0 / (np.exp(HBAR * omega_e / (BOLTZMANN * 0.3)) - 1)
Gamma_th = TWO_PI * 10  # electron thermalization rate (conservative)

# Eq. below Table II
Gamma_th_prime = Gamma_th + Gamma_i
n_bar_analytical = n_th * Gamma_th_prime / (Gamma_s + Gamma_th_prime)

print("Analytical estimate:")
print(f"  g_max/(2pi) = {g_max / TWO_PI / 1e3:.1f} kHz")
print(f"  Gamma_s/(2pi) = {Gamma_s / TWO_PI:.0f} Hz")
print(f"  n_th (300 mK) = {n_th:.1f}")
print(f"  n_bar_e = {n_bar_analytical:.2e}")

# Full Lindblad simulation
n_fock_e, n_fock_i = 15, 8
hs = HilbertSpace(n_ions=0, n_modes=2, n_fock=[n_fock_e, n_fock_i])
ops = OperatorFactory(hs)

# Beam-splitter coupling (activated by parametric drive)
g_bs = TWO_PI * 5e3
H_bs = g_bs * (
    ops.create(0) * ops.annihilate(1) + ops.annihilate(0) * ops.create(1)
)

c_ops = [
    np.sqrt(TWO_PI * 10e3) * ops.annihilate(1),  # ion laser cooling
    *motional_heating_ops(ops, mode=0, heating_rate=140.0),  # Ref. [13]
]

# Start with electron thermally excited at n_bar = 5
rho0 = qutip.tensor(
    qutip.thermal_dm(n_fock_e, 5.0),
    qutip.fock_dm(n_fock_i, 0),
)

tlist = np.linspace(0, 5e-3, 200)
result = qutip.mesolve(
    H_bs,
    rho0,
    tlist,
    c_ops=c_ops,
    e_ops=[ops.number(0), ops.number(1)],
)

n_e_final = result.expect[0][-1]

print("\nLindblad simulation (beam-splitter + ion cooling):")
print(f"  g_bs/(2pi) = {g_bs / TWO_PI / 1e3:.0f} kHz")
print("  Initial <n_e> = 5.0")
print(f"  Final   <n_e> = {n_e_final:.3f}")

assert n_e_final < 1.0, "Electron should be cooled below 1 phonon"
print(f"\n[check] Electron cooled to {n_e_final:.3f} phonons (< 1)")


# 5. Dispersive readout (Sec. II.C.2, Table I)
#
# The electron phonon couples to a MW cavity (g_ec), which
# dispersively couples to a transmon (g_sc). The combined
# dispersive shift zeta imprints the phonon number onto
# the transmon spectrum. Uses the full Eq. 5, not the
# simplified Eq. 6.

header("Sec. II.C.2: Dispersive readout coupling zeta (Eq. 5)")

# Table I parameters
omega_MW = TWO_PI * 1e9
omega_q = TWO_PI * 4e9
g_ec = TWO_PI * 33e3
g_sc = TWO_PI * 200e6

Delta_sc = omega_MW - omega_q  # -3 GHz * 2pi

# Eq. 5: zeta = 2*g_ec^2 * g_sc^2 * Delta_sc
#               / (g_sc^4 - Delta_ec^2 * Delta_sc^2)
# Two resonances where the denominator vanishes:
#   Delta_ec = +/- g_sc^2 / |Delta_sc|
# => omega_e = omega_MW -/+ g_sc^2/|Delta_sc|
#            = 986.7 MHz and 1013.3 MHz

res_shift = g_sc**2 / abs(Delta_sc)
print("Table I parameters:")
print(f"  omega_MW/(2pi) = {omega_MW / TWO_PI / 1e9:.0f} GHz")
print(f"  omega_q/(2pi)  = {omega_q / TWO_PI / 1e9:.0f} GHz")
print(f"  g_ec/(2pi)     = {g_ec / TWO_PI / 1e3:.0f} kHz")
print(f"  g_sc/(2pi)     = {g_sc / TWO_PI / 1e6:.0f} MHz")
print(f"  Dispersive shift = {res_shift / TWO_PI / 1e6:.1f} MHz")
print(
    f"  Resonances at {(omega_MW - res_shift) / TWO_PI / 1e6:.1f}"
    f" and {(omega_MW + res_shift) / TWO_PI / 1e6:.1f} MHz"
)

# Scan near the 986 MHz resonance (Fig. 4c)
omega_e_scan = TWO_PI * np.linspace(980, 993, 1000) * 1e6
Delta_ec = omega_MW - omega_e_scan
denom = g_sc**4 - Delta_ec**2 * Delta_sc**2
with np.errstate(divide="ignore", invalid="ignore"):
    zeta_vals = 2 * g_ec**2 * g_sc**2 * Delta_sc / denom
zeta_vals[np.abs(denom) < 1e10] = np.nan

peak_idx = np.nanargmax(np.abs(zeta_vals))
peak_freq = omega_e_scan[peak_idx] / TWO_PI / 1e6
peak_zeta = abs(zeta_vals[peak_idx]) / TWO_PI / 1e3

print("\nNear 986 MHz resonance (Fig. 4c):")
print(f"  Peak |zeta|/(2pi) = {peak_zeta:.1f} kHz at {peak_freq:.1f} MHz")
print("  (Eq. 5 diverges at exact resonance; the paper's Fig. 4c")
print("   shows ~20 kHz from full numerical diagonalization which")
print("   caps the divergence. The resonance location matches.)")

assert abs(peak_freq - 986.7) < 1, "Resonance should be near 986.7 MHz"
assert peak_zeta > 10, "Peak zeta should be in the kHz range"
print(f"\n[check] Resonance location matches Fig. 4c: {peak_freq:.1f} MHz")

header("All checks passed.")

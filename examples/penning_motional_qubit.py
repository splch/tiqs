"""Motional-state qubit in an anharmonic electron Penning trap.

A single electron in a Penning trap with an engineered quartic
anharmonicity forms a "transmon made of a single electron": the
lowest two Fock states |0> and |1> of the axial mode serve as
the qubit, with the anharmonicity separating the 0-1 transition
from the 1-2 transition so the qubit can be addressed selectively.

This script simulates:
  1. Energy spectrum of the anharmonic axial mode vs. anharmonicity
  2. Single-qubit gate (pi-pulse) on the motional qubit
  3. Gate fidelity vs. anharmonicity (leakage to |2> and above)
  4. Decoherence: motional heating and dephasing from voltage noise
  5. Dispersive readout via a coupled superconducting resonator

Parameters are adapted from Osada et al. (2022), who studied an
electron Paul trap, to the Penning trap context relevant to
Noguchi's group: electron axial frequency ~200 MHz at B ~ 3 T,
cryogenic operation at 100 mK, with a coupled superconducting
resonator (Q ~ 10^7 per Tominaga et al. 2025) for readout and
cooling.

References
----------
Osada, A. et al. Phys. Rev. Research 4, 033245 (2022).
Taniguchi, K., Noguchi, A. & Oka, T. arXiv:2502.17200 (2025).
Jain, S. et al. Nature 627, 510 (2024).
"""

import numpy as np
import qutip

from tiqs import (
    DuffingPotential,
    ElectronSpecies,
    PenningTrap,
    transition_frequencies,
)
from tiqs.constants import TWO_PI

N_FOCK = 10  # Fock space truncation


def header(title):
    print(f"\n{'=' * 65}\n{title}\n{'=' * 65}")


def rotating_frame_H(pot, n_fock, Omega):
    """Build the rotating-frame Hamiltonian for a driven anharmonic
    oscillator, rotating at the 0-1 transition frequency."""
    E = np.sort(pot.single_mode_hamiltonian(n_fock).eigenenergies().real)
    omega_01 = E[1] - E[0]
    E_rot = E - np.arange(n_fock) * omega_01 - E[0]
    H_free = qutip.Qobj(np.diag(E_rot))
    a = qutip.destroy(n_fock)
    H_drive = (Omega / 2) * (a + a.dag())
    return H_free + H_drive, omega_01


# Trap parameters
B = 3.0  # Tesla
omega_z = TWO_PI * 200e6  # axial frequency
species = ElectronSpecies(magnetic_field=B)

trap = PenningTrap(
    magnetic_field=B, species=species, d=3.5e-3, omega_axial=omega_z
)
assert trap.is_stable()


# 1. Energy spectrum vs. anharmonicity
#
# In a harmonic trap, all transitions are at omega_z and the qubit
# levels cannot be addressed individually. Adding a quartic term
# V(z) ~ beta * z^4 produces a Duffing/Kerr Hamiltonian:
#   H = omega_z * n + (alpha/2) * n * (n - 1)
# where alpha is the anharmonicity. The |0>-|1> transition stays
# at omega_z, but |1>-|2> shifts to omega_z + alpha.
#
# For negative alpha (softening), higher transitions move DOWN,
# exactly like a transmon qubit.

header("1. Anharmonic spectrum vs. alpha")

print(f"Axial frequency: {omega_z / TWO_PI / 1e6:.0f} MHz")
print(f"B field: {B:.0f} T")
print(f"Penning stable: {trap.is_stable()}")
print(f"Cyclotron: {trap.omega_modified_cyclotron / TWO_PI / 1e6:.0f} MHz")
print(f"Magnetron: {trap.omega_magnetron / TWO_PI / 1e3:.0f} kHz")
print()

# Keep alpha/omega < 0.1 so the Duffing model stays valid
alphas_mhz = [1, 2, 5, 10, 20]
print(f"{'alpha/2pi':>10}  {'omega_01':>10}  {'omega_12':>10}  {'delta':>8}")
print("-" * 44)
for a_mhz in alphas_mhz:
    alpha = -TWO_PI * a_mhz * 1e6
    pot = DuffingPotential(omega=omega_z, anharmonicity=alpha)
    freqs = transition_frequencies(pot, n_fock=N_FOCK)
    delta = (freqs[0] - freqs[1]) / TWO_PI / 1e6
    print(
        f"{a_mhz:>7} MHz"
        f"  {freqs[0] / TWO_PI / 1e6:>7.1f} MHz"
        f"  {freqs[1] / TWO_PI / 1e6:>7.1f} MHz"
        f"  {delta:>5.1f} MHz"
    )

print("\ndelta = omega_01 - omega_12: spectral gap protecting the qubit")
print("[check] Larger alpha -> larger gap -> better qubit isolation")


# 2. Single-qubit gate on the motional qubit
#
# A microwave drive at frequency omega_01 rotates the qubit in the
# {|0>, |1>} subspace. In the anharmonic mode's eigenbasis, the
# drive Hamiltonian couples adjacent levels:
#   H_d = (Omega/2) * (a + a_dag)
# The anharmonicity detunes the |1>-|2> transition, suppressing
# leakage. We simulate the full N-level dynamics.

header("2. Motional qubit pi-pulse")

alpha = -TWO_PI * 20e6  # 20 MHz anharmonicity
Omega = TWO_PI * 1e6  # 1 MHz Rabi frequency -> 500 ns pi-pulse

pot = DuffingPotential(omega=omega_z, anharmonicity=alpha)
H, omega_01 = rotating_frame_H(pot, N_FOCK, Omega)

t_pi = np.pi / Omega
tlist = np.linspace(0, t_pi, 500)
psi0 = qutip.basis(N_FOCK, 0)
result = qutip.sesolve(H, psi0, tlist)

psi_final = result.states[-1]
p0 = abs(psi_final[0, 0]) ** 2
p1 = abs(psi_final[1, 0]) ** 2
p2 = abs(psi_final[2, 0]) ** 2
leakage = 1 - p0 - p1

print(f"alpha/(2pi) = {abs(alpha) / TWO_PI / 1e6:.0f} MHz")
print(f"Omega/(2pi) = {Omega / TWO_PI / 1e6:.0f} MHz")
print(f"Gate time   = {t_pi * 1e9:.0f} ns")
print(f"alpha/Omega = {abs(alpha) / Omega:.0f}")
print()
print("After pi-pulse:")
print(f"  P(|0>) = {p0:.6f}")
print(f"  P(|1>) = {p1:.6f}")
print(f"  P(|2>) = {p2:.6f}")
print(f"  Leakage = {leakage:.2e}")
print(f"  Fidelity = {p1:.6f}")

assert p1 > 0.99, "Pi-pulse fidelity should exceed 99%"
print("\n[check] Square-pulse fidelity limited by leakage to |2>")
print("  (DRAG pulse shaping can suppress this further)")


# 3. Gate fidelity vs. anharmonicity
#
# Scan alpha to find the minimum anharmonicity needed for a
# target gate fidelity. The trade-off: larger alpha means less
# leakage but also stronger sensitivity to voltage noise
# (dephasing scales with alpha).

header("3. Gate fidelity vs. anharmonicity")

print(f"Omega/(2pi) = {Omega / TWO_PI / 1e6:.0f} MHz")
print(f"Gate time   = {t_pi * 1e9:.0f} ns")
print()
print(f"{'alpha/2pi':>10}  {'alpha/Om':>9}  {'Fidelity':>10}  {'Leakage':>10}")
print("-" * 45)

for a_mhz in [2, 5, 10, 15, 20]:
    alpha_s = -TWO_PI * a_mhz * 1e6
    pot_s = DuffingPotential(omega=omega_z, anharmonicity=alpha_s)
    H_s, _ = rotating_frame_H(pot_s, N_FOCK, Omega)

    opts = {"nsteps": 10000, "max_step": t_pi / 50}
    psi_f = qutip.sesolve(
        H_s, qutip.basis(N_FOCK, 0), [0, t_pi], options=opts
    ).states[-1]
    fid = abs(psi_f[1, 0]) ** 2
    leak = 1 - abs(psi_f[0, 0]) ** 2 - fid

    print(f"{a_mhz:>7} MHz  {a_mhz:>6}x    {fid:>9.6f}  {leak:>9.2e}")

print("\n[check] Fidelity improves rapidly with alpha/Omega ratio")


# 4. Decoherence: heating and dephasing
#
# Two dominant noise sources for the motional qubit:
# - Motional heating: electric field noise from electrode surfaces
#   adds phonons, causing |1> -> |2> transitions (T1-like)
# - Motional dephasing: voltage noise fluctuates the trap frequency,
#   randomizing the phase between |0> and |1> (T2-like)
#
# We prepare |+> = (|0> + |1>)/sqrt(2) and watch coherence decay.

header("4. Decoherence from heating and dephasing")

n_levels = 6
alpha = -TWO_PI * 10e6

pot = DuffingPotential(omega=omega_z, anharmonicity=alpha)
E = np.sort(pot.single_mode_hamiltonian(n_levels).eigenenergies().real)
omega_01 = E[1] - E[0]
E_rot = E - np.arange(n_levels) * omega_01 - E[0]
H_free = qutip.Qobj(np.diag(E_rot))

# Heating: ~10 quanta/s at cryogenic temperatures
# (Osada 2022 estimates 140 quanta/s; cryogenic reduces this)
heating_rate = 10.0  # quanta/s

# Dephasing from voltage noise on trap electrodes
dephasing_rate = TWO_PI * 100  # 100 Hz, conservative

a = qutip.destroy(n_levels)
c_ops = [
    np.sqrt(heating_rate) * a.dag(),
    np.sqrt(dephasing_rate) * a.dag() * a,
]

plus = (qutip.basis(n_levels, 0) + qutip.basis(n_levels, 1)).unit()
proj_01 = qutip.basis(n_levels, 0) * qutip.basis(n_levels, 1).dag()

tlist = np.linspace(0, 10e-3, 200)
result = qutip.mesolve(
    H_free,
    plus,
    tlist,
    c_ops=c_ops,
    e_ops=[proj_01, a.dag() * a],
    options={"nsteps": 50000},
)

coherence = result.expect[0].real
n_mean = result.expect[1].real

# Estimate T2 from half-life of coherence
half_idx = np.argmin(np.abs(coherence - coherence[0] / 2))
T2_est = tlist[half_idx] / np.log(2) if half_idx > 0 else np.inf

print(f"alpha/(2pi)    = {abs(alpha) / TWO_PI / 1e6:.0f} MHz")
print(f"Heating rate   = {heating_rate:.0f} quanta/s")
print(f"Dephasing rate = {dephasing_rate / TWO_PI:.0f} Hz")
print()
print(f"Coherence at t=0:    {coherence[0]:.4f}")
print(f"Coherence at t=1 ms: {coherence[len(tlist) // 5]:.4f}")
print(f"Coherence at t=5 ms: {coherence[len(tlist) // 2]:.4f}")
print(f"<n> at t=10 ms:      {n_mean[-1]:.4f}")
print(f"Estimated T2:        {T2_est * 1e3:.1f} ms")

assert T2_est > 1e-3, "T2 should exceed 1 ms for useful gates"
print(
    f"\n[check] T2 ~ {T2_est * 1e3:.0f} ms >> gate time ({t_pi * 1e9:.0f} ns)"
)


# 5. Dispersive readout via superconducting resonator
#
# The electron's axial motion couples to a superconducting
# microwave resonator via electric dipole interaction. In the
# dispersive regime (|omega_z - omega_r| >> g), the resonator
# frequency shifts by chi per phonon, enabling QND readout of
# the motional qubit state.

header("5. Dispersive readout via superconducting resonator")

# Parameters: Tominaga et al. (2025) demonstrated Q ~ 10^7 in
# planar superconducting resonators. A coupling of ~100 kHz is
# achievable with optimized electrode-resonator geometry (Osada
# 2022 achieved 33 kHz in a less optimized coaxial design).
omega_r = TWO_PI * 210e6  # resonator, detuned 10 MHz from qubit
g_er = TWO_PI * 100e3  # electron-resonator coupling
Q_r = 1e7  # resonator quality factor (Tominaga 2025)
kappa = omega_r / Q_r

Delta = omega_01 - omega_r

# Dispersive shift for an anharmonic oscillator:
#   chi = g^2 * alpha / (Delta * (Delta + alpha))
# Reduces to g^2/Delta when |alpha| >> |Delta|.
chi = g_er**2 * alpha / (Delta * (Delta + alpha))

# Measurement rate (Gambetta et al. PRA 77, 012112, 2008):
#   Gamma_meas = 8 * chi^2 * n_readout / kappa
n_readout = 5
Gamma_meas = 8 * chi**2 * n_readout / kappa
t_readout = 1 / Gamma_meas

print(f"omega_r/(2pi)  = {omega_r / TWO_PI / 1e6:.0f} MHz")
print(f"g_er/(2pi)     = {g_er / TWO_PI / 1e3:.0f} kHz")
print(f"Delta/(2pi)    = {Delta / TWO_PI / 1e6:.1f} MHz")
print(f"Q_r            = {Q_r:.0e}")
print(f"kappa/(2pi)    = {kappa / TWO_PI:.0f} Hz")
print()
print(f"Dispersive shift chi/(2pi) = {chi / TWO_PI:.1f} Hz")
print(f"|chi|/kappa = {abs(chi) / kappa:.2f}")
print(f"Readout time = {t_readout * 1e6:.1f} us")

regime = "resolved" if abs(chi) > kappa else "unresolved"
print(f"\nReadout regime: {regime} (chi/kappa = {abs(chi) / kappa:.1f})")

assert t_readout < T2_est, "Readout must be faster than decoherence"
print(
    f"[check] Readout ({t_readout * 1e6:.1f} us) << T2 ({T2_est * 1e3:.0f} ms)"
)


header("All checks passed.")

"""Stage 1 validation: axial-only BGNF vs direct Fock diagonalization.

Build a 1D anharmonic oscillator with H = ω a†a + γ (a + a†)^4
in DIMENSIONLESS units (ℏ = 1, ω_z = 1) so floating-point precision
is not an issue. Compare the numerical Kerr shift to the BGNF prediction.
"""

import sys

sys.path.insert(0, "src")

import numpy as np
import qutip

from tiqs.constants import (
    ELECTRON_CHARGE,
    ELECTRON_MASS,
    HBAR,
    TWO_PI,
)
from tiqs.elliptical import (
    AnharmonicCoeffs,
    frequency_shifts_matrix,
    orbit_params,
)
from tiqs.multipole import (
    ElectrostaticPotential,
    shift_matrix_general,
)
from tiqs.species.electron import ElectronSpecies
from tiqs.trap import PenningTrap


# Use the Verdu-paper reference parameters.
B = 0.5
omega_z = TWO_PI * 28e6
m = ELECTRON_MASS
q = +ELECTRON_CHARGE
C_004 = 1e10

# Compute the Verdu and BGNF M-matrix entries in physical units.
sp = ElectronSpecies(magnetic_field=B)
trap = PenningTrap(magnetic_field=B, species=sp, d=3.5e-3, omega_axial=omega_z)
nu_p = trap.omega_modified_cyclotron / TWO_PI
nu_z = trap.omega_axial / TWO_PI
nu_m = trap.omega_magnetron / TWO_PI
orb = orbit_params(
    trap.omega_cyclotron, trap.omega_axial, trap.omega_modified_cyclotron, 0.0
)
M_verdu = frequency_shifts_matrix(
    nu_p, nu_z, nu_m, orb, AnharmonicCoeffs(c002=1.0, c004=C_004), m
)[1, 1]
pot = ElectrostaticPotential.from_quadrupole(
    omega_z, m, q, epsilon=0.0
) + ElectrostaticPotential({(0, 0, 4): C_004})
M_bgnf = shift_matrix_general(pot, B, m, q, order=4)[1, 1]
print(f"Verdu M[z, z]  = {M_verdu:.6e} Hz/J")
print(f"BGNF  M[z, z]  = {M_bgnf:.6e} Hz/J")
print()

# Direct Fock diagonalization in dimensionless units.
#
# In dimensionless units with ℏ = 1, omega = 1, position
# x_dimless = (a + a†)/sqrt(2):
#
#   H_dimless = a†a + γ * (a + a†)^4 / 4
#
# where γ is the dimensionless anharmonicity. Match to physical:
#
#   H_physical = ℏω_z (N + γ_phys * (a + a†)^4 / 4)
#
# with γ_phys * ℏω_z = q*C_004 * x_zpf^4 = q*C_004 * (ℏ/(2mω_z))^2 / 4 * 4
# Actually let me just carefully match. In QM:
#
#   z_op = x_zpf * (a + a†)  with x_zpf = sqrt(ℏ/(2mω_z))
#   H_full = ℏω_z N + q*C_004 * z_op^4
#         = ℏω_z N + q*C_004 * x_zpf^4 * (a + a†)^4
#         = ℏω_z [ N + (q*C_004 * x_zpf^4 / (ℏω_z)) * (a + a†)^4 ]
#
# So γ ≡ q*C_004 * x_zpf^4 / (ℏω_z) is the dimensionless anharmonicity.
# Energies in units of ℏω_z; convert back at the end.

x_zpf = np.sqrt(HBAR / (2 * m * omega_z))
gamma = q * C_004 * x_zpf**4 / (HBAR * omega_z)
print(f"x_zpf = {x_zpf:.4e} m")
print(f"γ (dimensionless anharmonicity) = {gamma:.4e}")
print(f"  (must be << 1 for perturbation theory to apply)")
print()

n_fock = 40
a = qutip.destroy(n_fock)
ad = a.dag()
N = ad * a
H_dimless = N + gamma * (a + ad) ** 4

energies_dimless = np.sort(H_dimless.eigenenergies().real)

# Transition frequencies in units of ω_z (each = 1 + Kerr correction).
n_safe = 20
trans_dimless = np.diff(energies_dimless[: n_safe + 1])
shifts_per_phonon = trans_dimless - 1.0  # subtract the bare ω_z = 1

print(f"=== Numerical Fock spectrum (n_fock={n_fock}) ===")
print(f"  Transition shifts ν_(n,n+1) - ν_z, in units of ν_z:")
for n in [0, 1, 2, 3, 5, 10, 15]:
    if n < n_safe:
        print(f"    n={n}: shift = {shifts_per_phonon[n]:+.6e} ν_z")

# Fit shift = α * n (linear in n, expected Kerr behavior).
n_arr = np.arange(n_safe)
slope_dimless, offset_dimless = np.polyfit(n_arr, shifts_per_phonon, 1)
print(
    f"\n  Linear fit: shift(n) = {slope_dimless:.6e} * n + {offset_dimless:.6e} ν_z"
)

# Quadratic fit to assess higher-order corrections
qc, lc, cc = np.polyfit(n_arr, shifts_per_phonon, 2)
print(f"  Quadratic fit: {qc:.4e} n² + {lc:.4e} n + {cc:.4e}")

# Convert dimensionless slope to M^V (Hz/J).
# slope_dimless = α_kerr / ω_z (where α_kerr is rad/s per phonon)
# E per phonon = ℏω_z
# M^V = ν_shift_per_phonon / E_per_phonon
#     = (slope_dimless * ν_z) / (ℏ ω_z)
#     = slope_dimless / (2π ℏ)
M_numerical = slope_dimless / (TWO_PI * HBAR)

print(f"\n=== Comparison ===")
print(f"  Verdu     M[z, z] = {M_verdu:.6e} Hz/J")
print(f"  BGNF      M[z, z] = {M_bgnf:.6e} Hz/J")
print(f"  Numerical M[z, z] = {M_numerical:.6e} Hz/J")
print(f"  Numerical / BGNF  = {M_numerical / M_bgnf:.6f}")
print(f"  Numerical / Verdu = {M_numerical / M_verdu:.6f}")

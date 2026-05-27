"""Stage 2 validation: radial cross-mode BGNF vs 2D Fock diagonalization.

DIMENSIONLESS units to avoid a QuTiP arithmetic bug at the
~1e-24 J scale. Energy unit: ℏω_+. Action unit: ℏ. Then:

  H̃ = H/(ℏω_+) = N_+ - (ω_-/ω_+) N_- + γ̃ · X̃^4

where X̃ = x_op / x_zpf,+, γ̃ = q*C_400*x_zpf,+^4/(ℏω_+).

Energy shifts ΔẼ from diagonalization, fit to polynomial in (n_+, n_-),
gives the dimensionless M^I:

  M^Ĩ[α,β] = ∂²Ẽ/∂n_α∂n_β    (dimensionless: rad/s per ℏ²)

Convert back: M^I_phys[α,β] = M^Ĩ × ω_+ / ℏ  (rad/s per (J·s)²)
"""

import sys

sys.path.insert(0, "src")

import numpy as np
import qutip
import scipy.optimize

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
    linear_modes,
    shift_matrix_general,
)
from tiqs.species.electron import ElectronSpecies
from tiqs.trap import PenningTrap


B = 0.140
omega_z = TWO_PI * 2.20e9  # gives ω_+/ω_- ≈ 4
m = ELECTRON_MASS
q = +ELECTRON_CHARGE

print(f"=== Setup ===")
print(f"  B = {B} T, ω_z/(2π) = {omega_z / TWO_PI / 1e9:.4f} GHz")

pot_quad = ElectrostaticPotential.from_quadrupole(omega_z, m, q, epsilon=0.0)
modes = linear_modes(pot_quad, B, m, q)
omega_p = modes.omega_plus
omega_m = modes.omega_minus
print(f"  ω_+/(2π) = {omega_p / TWO_PI / 1e9:.4f} GHz")
print(f"  ω_-/(2π) = {omega_m / TWO_PI / 1e9:.4f} GHz")
print(f"  ratio ω_+/ω_- = {omega_p / omega_m:.3f}")
print()

# Verdú prediction
sp = ElectronSpecies(magnetic_field=B)
trap = PenningTrap(magnetic_field=B, species=sp, d=3.5e-3, omega_axial=omega_z)
nu_p = trap.omega_modified_cyclotron / TWO_PI
nu_z = trap.omega_axial / TWO_PI
nu_m = trap.omega_magnetron / TWO_PI
orb = orbit_params(
    trap.omega_cyclotron, trap.omega_axial, trap.omega_modified_cyclotron, 0.0
)

# We'll set C_400 to give a specific dimensionless γ_tilde.
# x_zpf,+ = sqrt(ℏ/(2 m ω_+)) = the cyclotron zero-point amplitude.
x_zpf_plus = np.sqrt(HBAR / (2 * m * omega_p))
gamma_tilde = 1e-6  # very small for clean perturbative limit
C_400 = gamma_tilde * HBAR * omega_p / (q * x_zpf_plus**4)

print(f"  x_zpf,+ = {x_zpf_plus:.3e} m")
print(f"  Chosen γ̃ = {gamma_tilde:.0e} (dimensionless perturbation strength)")
print(f"  C_400 = {C_400:.3e} V/m^4")
print()

M_verdu = frequency_shifts_matrix(
    nu_p, nu_z, nu_m, orb, AnharmonicCoeffs(c002=1.0, c400=C_400), m
)
pot = pot_quad + ElectrostaticPotential({(4, 0, 0): C_400})
M_bgnf = shift_matrix_general(pot, B, m, q, order=4)

# Dimensionless Fock setup: energy unit ℏω_+, action unit ℏ,
# basis |N_+, N_-⟩, H̃ = H/(ℏω_+).
n_fock = 12
a_p = qutip.tensor(qutip.destroy(n_fock), qutip.qeye(n_fock))
a_m = qutip.tensor(qutip.qeye(n_fock), qutip.destroy(n_fock))
N_p = a_p.dag() * a_p
N_m = a_m.dag() * a_m

# Build x̃ = x/x_zpf,+ as a Hermitian operator.
# In the symplectic transform from my code:
#   x = S[0,0] q_+ + S[0,2] q_- + S[0,3] p_+ + S[0,5] p_-
# with q_α, p_α being canonical mode coordinates. In QM, the bridge is:
#   q_α = sqrt(ℏ/2) (a_α + a_α†),   p_α = -i sqrt(ℏ/2) (a_α - a_α†)
# So:
#   x = sqrt(ℏ/2) [
#         S[0,0]·(a_+ + a_+†) + S[0,2]·(a_- + a_-†)
#       - i S[0,3]·(a_+ - a_+†) - i S[0,5]·(a_- - a_-†) ]
# Dividing by x_zpf,+:
#   x̃ = (sqrt(ℏ/2)/x_zpf,+) [...]
#      = sqrt(ω_+ m) [...]    (using x_zpf,+² = ℏ/(2 m ω_+))
S = modes.transform
prefac = np.sqrt(
    omega_p * m
)  # converts q_α (units sqrt(action)) → dimensionless when divided by x_zpf,+
x_dimless = (
    prefac * S[0, 0] * (a_p + a_p.dag())
    + prefac * S[0, 2] * (a_m + a_m.dag())
    + prefac * S[0, 3] * (-1j) * (a_p - a_p.dag())
    + prefac * S[0, 5] * (-1j) * (a_m - a_m.dag())
)
# Hermitize numerically.
x_dimless = (x_dimless + x_dimless.dag()) / 2

# Build perturbation V_op_dimless = γ̃ · X̃^4 in dimensionless form.
# Originally V = q*C_400*x^4 = γ̃·ℏω_+ · (x/x_zpf,+)^4 → V/(ℏω_+) = γ̃ X̃^4.
# But our x̃ above includes an extra sqrt factor; need to re-normalize.
# Test: ⟨0,0|x̃²|0,0⟩ should equal x_zpf,+²/x_zpf,+² = 1 (in BG cylindrical limit).
ground = qutip.tensor(qutip.fock(n_fock, 0), qutip.fock(n_fock, 0))
xx_expect = qutip.expect(x_dimless * x_dimless, ground)
print(f"=== Sanity ===")
print(f"  ⟨0,0|X̃²|0,0⟩ = {xx_expect:.4f} (should be ≈ 1)")
# In a perfect cylindrical case this is 1; with elliptical or off-axis
# couplings it can differ. For our BG cyl test it should be exactly 1.

# Adjust γ̃ definition: the "perturbation" in dimensionless form is
# V_dimless = q*C_400*x_op^4 / (ℏω_+).
# Since x_dimless = x_op/x_zpf,+, we have x_op^4 = x_zpf,+^4 · x_dimless^4.
# So V_dimless = q*C_400*x_zpf,+^4/(ℏω_+) · x_dimless^4 = γ̃ x_dimless^4.
V_dimless = gamma_tilde * x_dimless**4

# Quadratic Hamiltonian in dimensionless units, working with .full() arrays
# to avoid the QuTiP Dia subtraction bug.
H_2_arr = N_p.full() - (omega_m / omega_p) * N_m.full()
H_arr = H_2_arr + V_dimless.full()

# Diagonalize.
energies, vectors = np.linalg.eigh(H_arr)

# Identify perturbed states by max overlap with |n_+, n_-⟩.
n_plus_max = 4
n_minus_max = 4
shift_data = []
print(f"\n=== State identification ===")
for np_val in range(n_plus_max):
    for nm_val in range(n_minus_max):
        unpert_state = qutip.tensor(
            qutip.fock(n_fock, np_val), qutip.fock(n_fock, nm_val)
        )
        unpert_arr = unpert_state.full().flatten()
        unpert_E = np_val - (omega_m / omega_p) * nm_val  # dimensionless
        # Max-overlap eigenvector.
        overlaps = np.abs(vectors.conj().T @ unpert_arr) ** 2
        best_idx = int(np.argmax(overlaps))
        ov = overlaps[best_idx]
        if ov < 0.99:
            continue
        pert_E = energies[best_idx].real  # dimensionless
        delta_E = pert_E - unpert_E  # dimensionless ΔẼ
        shift_data.append((np_val, nm_val, delta_E, ov))

print(f"  {len(shift_data)} states identified with overlap > 0.99")
for np_val, nm_val, dE, ov in shift_data[:8]:
    print(f"    |{np_val}, {nm_val}⟩: ΔẼ = {dE:+.4e}, overlap = {ov:.6f}")
print()

# Fit ΔẼ(n_+, n_-) = a + b n_+ + c n_- + d n_+² + e n_+ n_- + f n_-²
shifts = np.array([s[2] for s in shift_data])
coords = np.array([(s[0], s[1]) for s in shift_data], dtype=float)


def fit_func(X, a, b, c, d, e, f):
    np_arr, nm_arr = X[:, 0], X[:, 1]
    return (
        a
        + b * np_arr
        + c * nm_arr
        + d * np_arr**2
        + e * np_arr * nm_arr
        + f * nm_arr**2
    )


popt, _ = scipy.optimize.curve_fit(fit_func, coords, shifts)
a, b, c, d, e, f = popt
print(f"=== Polynomial fit (dimensionless) ===")
print(f"  Constant:   {a:+.4e}")
print(f"  n_+:        {b:+.4e}")
print(f"  n_-:        {c:+.4e}")
print(f"  n_+²:       {d:+.4e}")
print(f"  n_+n_-:     {e:+.4e}")
print(f"  n_-²:       {f:+.4e}")
print()

# Convert dimensionless quadratic coefficients to physical M^I.
# E = ℏω_+ · Ẽ.
# K_4 has form: K_4 = (1/2) M^I_pp (ℏN_+)² + M^I_pm (ℏN_+)(ℏN_-) + (1/2) M^I_mm (ℏN_-)²
# Energies: <K_4>(n_+, n_-) ≈ (ℏ²/2) M^I_pp n_+² + ℏ² M^I_pm n_+ n_- + (ℏ²/2) M^I_mm n_-² + ...
# In dimensionless: ΔẼ = (ℏ/(2ω_+)) M^I_pp n_+² + (ℏ/ω_+) M^I_pm n_+ n_- + (ℏ/(2ω_+)) M^I_mm n_-²
# So d = (ℏ/(2ω_+)) M^I_pp, e = (ℏ/ω_+) M^I_pm, f = (ℏ/(2ω_+)) M^I_mm.

M_I_pp_num = 2 * d * omega_p / HBAR
M_I_pm_num = e * omega_p / HBAR
M_I_mm_num = 2 * f * omega_p / HBAR
print(f"=== Numerical M^I (action-derivative) ===")
print(f"  M^I[+,+] = {M_I_pp_num:.4e}")
print(f"  M^I[+,-] = {M_I_pm_num:.4e}")
print(f"  M^I[-,-] = {M_I_mm_num:.4e}")
print()

# BGNF M^I (already-extracted from K coefficients; my code's M^V × 2π × signed_β × ω_β)
sign = np.array([1, 1, -1])
M_I_bgnf = M_bgnf * (
    TWO_PI
    * (sign * np.array([omega_p, modes.omega_z, omega_m]))[np.newaxis, :]
)
M_I_verdu = M_verdu * (
    TWO_PI * np.array([omega_p, modes.omega_z, omega_m])[np.newaxis, :]
)

print(f"=== BGNF M^I ===")
print(f"  M^I[+,+] = {M_I_bgnf[0, 0]:.4e}")
print(f"  M^I[+,-] = {M_I_bgnf[0, 2]:.4e}")
print(f"  M^I[-,-] = {M_I_bgnf[2, 2]:.4e}")
print()

print(f"=== Verdú M^I (asymmetric matrix; show one side) ===")
print(f"  M^I[+,+] = {M_I_verdu[0, 0]:.4e}")
print(f"  M^I[+,-] = {M_I_verdu[0, 2]:.4e}")
print(f"  M^I[-,+] = {M_I_verdu[2, 0]:.4e}")
print(f"  M^I[-,-] = {M_I_verdu[2, 2]:.4e}")
print()

print(f"=== Ratios numerical/X ===")
print(f"             BGNF        Verdú")
for label, ai, bi in [
    ("M^I[+,+]", 0, 0),
    ("M^I[+,-]", 0, 2),
    ("M^I[-,-]", 2, 2),
]:
    n_val = {(0, 0): M_I_pp_num, (0, 2): M_I_pm_num, (2, 2): M_I_mm_num}[
        (ai, bi)
    ]
    b_val = M_I_bgnf[ai, bi]
    v_val = M_I_verdu[ai, bi]
    r_b = n_val / b_val if abs(b_val) > 1e-30 else float("nan")
    r_v = n_val / v_val if abs(v_val) > 1e-30 else float("nan")
    print(f"  {label}: {r_b:>10.4f}  {r_v:>10.4f}")

"""Validate the BGNF against cylindrical symmetry of the perturbation.

For a Penning trap with a cylindrically-symmetric perturbation V(ρ, z)
(where ρ² = x² + y²), the radial dynamics commutes with L_z rotations
in the (x, y) plane. This means the canonical angular momentum
L_z = N_+ - N_- is conserved, so the normal-form Hamiltonian K can
depend only on:

    K = K(N_+ + N_-,  (N_+ - N_-)², ...)

More precisely, the SECOND-ORDER expansion in actions is

    K = a (N_+ + N_-) + b (N_+ + N_-)² + c (N_+ - N_-)² + ...

giving the structural constraint

    M^I[+, +] = M^I[-, -]    (cylindrical symmetry)

This is a rigorous, convention-independent test of any code that
processes cylindrically-symmetric perturbations.

Independently, the cross-mode coefficient is

    M^I[+, -] = 2(b - c)

while the diagonal coefficients are M^I[+, +] = M^I[-, -] = 2(b + c).
So the ratio (M^I[+, -]) / (M^I[+, +]) carries physical information
about the b/c balance of the perturbation.
"""

import sys
sys.path.insert(0, "src")

import numpy as np

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


def cylindrical_C4_potential(omega_z, m, q, C4=1e10):
    """Cylindrically-symmetric Laplacian quartic V_4 = z^4 -3z²ρ² + (3/8)ρ^4."""
    return (
        ElectrostaticPotential.from_quadrupole(omega_z, m, q, epsilon=0.0)
        + ElectrostaticPotential({
            (0, 0, 4): C4 * 1.0,
            (2, 0, 2): C4 * -3.0,
            (0, 2, 2): C4 * -3.0,
            (4, 0, 0): C4 * 3.0 / 8.0,
            (0, 4, 0): C4 * 3.0 / 8.0,
            (2, 2, 0): C4 * 3.0 / 4.0,
        })
    )


def cylindrical_x2y2_pure(omega_z, m, q):
    """Just the angular partner C_220 x²y² (decidedly NOT cylindrical)."""
    return (
        ElectrostaticPotential.from_quadrupole(omega_z, m, q, epsilon=0.0)
        + ElectrostaticPotential({(2, 2, 0): 1e10})
    )


B = 0.140
m = ELECTRON_MASS
q = +ELECTRON_CHARGE

# Test multiple ω_z values
print("=" * 70)
print("CYLINDRICAL SYMMETRY TEST: M^I[+, +] == M^I[-, -]")
print("=" * 70)
print("For any cylindrically-symmetric V(ρ, z) perturbation, L_z = N_+ - N_-")
print("is conserved by the perturbation theory, so the diagonal radial")
print("entries M^I[+, +] and M^I[-, -] must be EQUAL.")
print()
print(f"{'ω_z [GHz]':<10} {'M^I[+,+]':>14} {'M^I[-,-]':>14} {'M^I[+,-]':>14} "
      f"{'(+/+) - (-/-)':>16} {'Rel':>10}")
print("-" * 80)

for omega_z_GHz in [0.5, 1.0, 1.5, 2.0, 2.5]:
    omega_z = TWO_PI * omega_z_GHz * 1e9
    pot = cylindrical_C4_potential(omega_z, m, q)
    try:
        M_V = shift_matrix_general(pot, B, m, q, order=4)
    except ValueError as e:
        print(f"{omega_z_GHz:<10.2f}   (unstable: {str(e)[:50]})")
        continue
    modes = linear_modes(
        pot.restrict_to_orders(min_order=2, max_order=2), B, m, q
    )
    sign = np.array([1, 1, -1])
    omegas = np.array([modes.omega_plus, modes.omega_z, modes.omega_minus])
    M_I = M_V * (TWO_PI * sign * omegas)[np.newaxis, :]
    diff = M_I[0, 0] - M_I[2, 2]
    scale = max(abs(M_I[0, 0]), abs(M_I[2, 2]), 1e-30)
    rel = abs(diff) / scale
    print(f"{omega_z_GHz:<10.2f} {M_I[0, 0]:>14.4e} {M_I[2, 2]:>14.4e} "
          f"{M_I[0, 2]:>14.4e} {diff:>16.4e} {rel:>10.2e}  "
          f"{'✓' if rel < 1e-10 else '✗'}")

# Test Verdú on the same potential
print()
print(f"Verdú elliptical.py at ω_z = 2.0 GHz, cylindrical C_4 combination:")
sp = ElectronSpecies(magnetic_field=B)
omega_z = TWO_PI * 2.0e9
trap = PenningTrap(magnetic_field=B, species=sp, d=3.5e-3, omega_axial=omega_z)
nu_p = trap.omega_modified_cyclotron / TWO_PI
nu_z = trap.omega_axial / TWO_PI
nu_m = trap.omega_magnetron / TWO_PI
orb = orbit_params(
    trap.omega_cyclotron, trap.omega_axial,
    trap.omega_modified_cyclotron, 0.0
)
coeffs = AnharmonicCoeffs(
    c002=1.0,
    c004=1.0,
    c202=-3.0,
    c022=-3.0,
    c400=3.0/8.0,
    c040=3.0/8.0,
    c220=3.0/4.0,
)
M_V_verdu = frequency_shifts_matrix(nu_p, nu_z, nu_m, orb, coeffs, m)
omegas_v = np.array([trap.omega_modified_cyclotron, trap.omega_axial, trap.omega_magnetron])
M_I_verdu = M_V_verdu * (TWO_PI * omegas_v)[np.newaxis, :]
diff = M_I_verdu[0, 0] - M_I_verdu[2, 2]
scale = max(abs(M_I_verdu[0, 0]), abs(M_I_verdu[2, 2]), 1e-30)
rel = abs(diff) / scale
print(f"  M^I[+,+] = {M_I_verdu[0, 0]:.4e}")
print(f"  M^I[-,-] = {M_I_verdu[2, 2]:.4e}")
print(f"  Diff:     {diff:.4e}  (relative {rel:.2e})")
print(f"  {'✓ Verdú passes cylindrical-symmetry test' if rel < 1e-10 else '✗ Verdú FAILS cylindrical-symmetry test'}")

# Now check pure C_220 (which is NOT cylindrically symmetric)
print()
print(f"For pure C_220 x²y² (NOT cylindrically symmetric), the diagonals")
print(f"should DIFFER:")
omega_z = TWO_PI * 2.0e9
pot_pure_C220 = cylindrical_x2y2_pure(omega_z, m, q)
M_V_220 = shift_matrix_general(pot_pure_C220, B, m, q, order=4)
modes = linear_modes(
    pot_pure_C220.restrict_to_orders(min_order=2, max_order=2), B, m, q
)
omegas = np.array([modes.omega_plus, modes.omega_z, modes.omega_minus])
sign = np.array([1, 1, -1])
M_I_220 = M_V_220 * (TWO_PI * sign * omegas)[np.newaxis, :]
print(f"  M^I[+,+] = {M_I_220[0, 0]:.4e}")
print(f"  M^I[-,-] = {M_I_220[2, 2]:.4e}")
print(f"  Diff:     {M_I_220[0, 0] - M_I_220[2, 2]:.4e}")
print(f"  (For BG cyl, M^I[+,+] = M^I[-,-] anyway because x²y² happens to")
print(f"   have the (x, y) swap symmetry of cylindrical V at BG cyl ε=0!)")

# Real test of non-cylindrical: ε ≠ 0
print()
print(f"With Kretzschmar ε = 0.3 (broken cylindrical symmetry):")
omega_z = TWO_PI * 2.0e9
pot_eps = (
    ElectrostaticPotential.from_quadrupole(omega_z, m, q, epsilon=0.3)
    + ElectrostaticPotential({
        (0, 0, 4): 1.0e10,
        (2, 0, 2): -3.0e10,
        (0, 2, 2): -3.0e10,
        (4, 0, 0): 3.0/8.0 * 1.0e10,
        (0, 4, 0): 3.0/8.0 * 1.0e10,
        (2, 2, 0): 3.0/4.0 * 1.0e10,
    })
)
M_V_eps = shift_matrix_general(pot_eps, B, m, q, order=4)
modes = linear_modes(
    pot_eps.restrict_to_orders(min_order=2, max_order=2), B, m, q
)
omegas = np.array([modes.omega_plus, modes.omega_z, modes.omega_minus])
sign = np.array([1, 1, -1])
M_I_eps = M_V_eps * (TWO_PI * sign * omegas)[np.newaxis, :]
print(f"  M^I[+,+] = {M_I_eps[0, 0]:.4e}")
print(f"  M^I[-,-] = {M_I_eps[2, 2]:.4e}")
print(f"  Diff: {(M_I_eps[0, 0] - M_I_eps[2, 2]):.4e} "
      f"(rel = {abs(M_I_eps[0, 0] - M_I_eps[2, 2]) / max(abs(M_I_eps[0, 0]), 1e-30):.3e})")
print(f"  (Diagonals SHOULD differ here because ε≠0 breaks cylindrical symmetry.)")


print()
print("=" * 70)
print("NUMERICAL FOCK-BASIS GROUND TRUTH for cylindrical C_4")
print("=" * 70)

import qutip
import scipy.optimize

omega_z = TWO_PI * 2.0e9
# Calibrate γ_tilde = 1e-5 for clean perturbative regime + numerical
# resolution.
quad = ElectrostaticPotential.from_quadrupole(omega_z, m, q, epsilon=0.0)
modes = linear_modes(quad, B, m, q)
omega_p = modes.omega_plus
omega_m = modes.omega_minus
x_zpf = np.sqrt(HBAR / (2 * m * omega_p))
# Choose C_4 so the dominant Cartesian piece (C_400 = 3/8 C_4) gives
# γ_tilde = 1e-5 in dimensionless x_zpf,+ units.
gamma_tilde_400 = 1e-5
C_400 = gamma_tilde_400 * HBAR * omega_p / (q * x_zpf**4)
C_4 = C_400 * 8.0 / 3.0  # so C_400 = (3/8) C_4
print(f"C_4 = {C_4:.3e}, C_400 = {C_400:.3e}")

pot_cyl = cylindrical_C4_potential(omega_z, m, q, C4=C_4)
M_V_bgnf = shift_matrix_general(pot_cyl, B, m, q, order=4)
sign = np.array([1, 1, -1])
omegas_p = np.array([omega_p, modes.omega_z, omega_m])
M_I_bgnf = M_V_bgnf * (TWO_PI * sign * omegas_p)[np.newaxis, :]
print(f"BGNF M^I:")
print(f"  M^I[+,+] = {M_I_bgnf[0, 0]:.4e}")
print(f"  M^I[+,-] = {M_I_bgnf[0, 2]:.4e}")
print(f"  M^I[-,-] = {M_I_bgnf[2, 2]:.4e}")
print()

# Numerical Fock-basis: only includes radial part of V (since 2D Fock
# basis), so the axial-mode contribution to cylindrically-symmetric V_4
# is dropped. The contribution from purely-radial part of V_4 is:
# V_radial = (3/8) (x² + y²)² + cross terms with z = 0.
# = (3/8) [(x²+y²)²]  -- the (3/8) C_4 piece for radial dynamics.

# Build only the radial part of the perturbation (set z = 0, i.e. skip
# any term with k > 0):
print("Numerical Fock (radial only): includes (3/8) ρ^4 piece of V_4")
print()

n_fock = 12
a_p = qutip.tensor(qutip.destroy(n_fock), qutip.qeye(n_fock))
a_m = qutip.tensor(qutip.qeye(n_fock), qutip.destroy(n_fock))
sqrt_h = np.sqrt(HBAR)
S = modes.transform
def cart_op(row):
    op = (
        S[row, 0] * sqrt_h / np.sqrt(2) * (a_p + a_p.dag())
        + S[row, 2] * sqrt_h / np.sqrt(2) * (a_m + a_m.dag())
        + S[row, 3] * (-1j * sqrt_h / np.sqrt(2)) * (a_p - a_p.dag())
        + S[row, 5] * (-1j * sqrt_h / np.sqrt(2)) * (a_m - a_m.dag())
    )
    return (op + op.dag()) / 2
x_op = cart_op(0)
y_op = cart_op(1)

Id = qutip.tensor(qutip.qeye(n_fock), qutip.qeye(n_fock)).full()
x_arr = x_op.full()
y_arr = y_op.full()
V_arr = np.zeros_like(Id)
# Include all radial (k = 0) terms.
pert_pot = pot_cyl.restrict_to_orders(min_order=3)
for (i, j, k), c in pert_pot.coeffs.items():
    if k != 0 or c == 0:
        continue
    op = Id.copy()
    for _ in range(i):
        op = op @ x_arr
    for _ in range(j):
        op = op @ y_arr
    V_arr = V_arr + (q * c) * op

N_p = (a_p.dag() * a_p).full()
N_m = (a_m.dag() * a_m).full()
H_arr = N_p - (omega_m / omega_p) * N_m + V_arr / (HBAR * omega_p)
eig, vecs = np.linalg.eigh(H_arr)

shifts = []
for np_v in range(4):
    for nm_v in range(4):
        un = qutip.tensor(
            qutip.fock(n_fock, np_v), qutip.fock(n_fock, nm_v)
        ).full().flatten()
        un_E = np_v - (omega_m / omega_p) * nm_v
        ovs = np.abs(vecs.conj().T @ un) ** 2
        bi = int(np.argmax(ovs))
        if ovs[bi] < 0.99:
            continue
        shifts.append((np_v, nm_v, eig[bi].real - un_E))

print(f"  States identified: {len(shifts)}")
sh = np.array([s[2] for s in shifts])
co = np.array([(s[0], s[1]) for s in shifts], dtype=float)

def fit(X, a, b, c, d, e, f):
    np_a, nm_a = X[:, 0], X[:, 1]
    return (
        a + b * np_a + c * nm_a
        + d * np_a**2 + e * np_a * nm_a + f * nm_a**2
    )

popt, _ = scipy.optimize.curve_fit(fit, co, sh)
_, _, _, d_fit, e_fit, f_fit = popt
M_I_pp_num = 2 * d_fit * omega_p / HBAR
M_I_pm_num = e_fit * omega_p / HBAR
M_I_mm_num = 2 * f_fit * omega_p / HBAR
print(f"Numerical M^I (from 2D Fock):")
print(f"  M^I[+,+] = {M_I_pp_num:.4e}")
print(f"  M^I[+,-] = {M_I_pm_num:.4e}")
print(f"  M^I[-,-] = {M_I_mm_num:.4e}")
print()
print(f"Note: numerical is the RADIAL-ONLY result (z = 0 in Fock basis).")
print(f"BGNF includes both radial and axial-radial cross terms (C_202, C_022).")
print(f"For the strict M^I[+,+] / M^I[-,-] / M^I[+,-] comparison restricted")
print(f"to radial actions, the radial-only BGNF result is:")
pot_radial_only = (
    quad
    + ElectrostaticPotential({
        (4, 0, 0): C_4 * 3.0 / 8.0,
        (0, 4, 0): C_4 * 3.0 / 8.0,
        (2, 2, 0): C_4 * 3.0 / 4.0,
    })
)
M_V_radial = shift_matrix_general(pot_radial_only, B, m, q, order=4)
M_I_radial = M_V_radial * (TWO_PI * sign * omegas_p)[np.newaxis, :]
print(f"  BGNF M^I (radial-only V): [+,+]={M_I_radial[0,0]:.3e}, "
      f"[+,-]={M_I_radial[0,2]:.3e}, [-,-]={M_I_radial[2,2]:.3e}")
print()
print(f"Ratios numerical / BGNF (radial-only V):")
for label, num, bgnf in [
    ("[+,+]", M_I_pp_num, M_I_radial[0, 0]),
    ("[+,-]", M_I_pm_num, M_I_radial[0, 2]),
    ("[-,-]", M_I_mm_num, M_I_radial[2, 2]),
]:
    r = num / bgnf
    tag = "✓" if 0.99 < r < 1.01 else ("≈" if 0.95 < r < 1.05 else "✗")
    print(f"  {label}: numerical={num:.3e}, BGNF_radial={bgnf:.3e}, ratio={r:.4f} {tag}")

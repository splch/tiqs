"""Validate the remaining BGNF gaps:

1. Cubic perturbation (tests the (1/2){W_3, H_3} Lie-Deprit factor)
2. v3p4 1:-2 resonance regime (should fail; expose resonance detector)
3. Higher-order numerical convergence
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
from tiqs.multipole import (
    ElectrostaticPotential,
    detect_resonances,
    linear_modes,
    shift_matrix_general,
)


m = ELECTRON_MASS
q = +ELECTRON_CHARGE


def numerical_M_I(
    potential,
    B,
    n_fock=14,
    n_plus_max=4,
    n_minus_max=4,
    overlap_threshold=0.99,
):
    """Direct Fock diagonalization: returns numerical M^I[+,+], M^I[+,-], M^I[-,-]."""
    quad_pot = potential.restrict_to_orders(min_order=2, max_order=2)
    pert_pot = potential.restrict_to_orders(min_order=3)
    modes = linear_modes(quad_pot, B, m, q)
    omega_p = modes.omega_plus
    omega_m = modes.omega_minus
    S = modes.transform

    a_p = qutip.tensor(qutip.destroy(n_fock), qutip.qeye(n_fock))
    a_m = qutip.tensor(qutip.qeye(n_fock), qutip.destroy(n_fock))

    sqrt_hbar = np.sqrt(HBAR)

    def cart_op(row):
        op = (
            S[row, 0] * sqrt_hbar / np.sqrt(2) * (a_p + a_p.dag())
            + S[row, 2] * sqrt_hbar / np.sqrt(2) * (a_m + a_m.dag())
            + S[row, 3] * (-1j * sqrt_hbar / np.sqrt(2)) * (a_p - a_p.dag())
            + S[row, 5] * (-1j * sqrt_hbar / np.sqrt(2)) * (a_m - a_m.dag())
        )
        return (op + op.dag()) / 2

    x_op = cart_op(0)
    y_op = cart_op(1)

    Id_arr = qutip.tensor(qutip.qeye(n_fock), qutip.qeye(n_fock)).full()
    x_arr = x_op.full()
    y_arr = y_op.full()
    V_arr_phys = np.zeros_like(Id_arr)
    for (i, j, k), c in pert_pot.coeffs.items():
        if k != 0 or c == 0:
            continue
        op = Id_arr.copy()
        for _ in range(i):
            op = op @ x_arr
        for _ in range(j):
            op = op @ y_arr
        V_arr_phys = V_arr_phys + (q * c) * op

    N_p = a_p.dag() * a_p
    N_m = a_m.dag() * a_m
    H_arr = (
        N_p.full()
        - (omega_m / omega_p) * N_m.full()
        + V_arr_phys / (HBAR * omega_p)
    )
    energies, vectors = np.linalg.eigh(H_arr)

    shift_data = []
    for np_val in range(n_plus_max):
        for nm_val in range(n_minus_max):
            unpert = (
                qutip
                .tensor(qutip.fock(n_fock, np_val), qutip.fock(n_fock, nm_val))
                .full()
                .flatten()
            )
            unpert_E = np_val - (omega_m / omega_p) * nm_val
            overlaps = np.abs(vectors.conj().T @ unpert) ** 2
            best = int(np.argmax(overlaps))
            if overlaps[best] < overlap_threshold:
                continue
            shift_data.append((
                np_val,
                nm_val,
                energies[best].real - unpert_E,
                overlaps[best],
            ))

    if len(shift_data) < 6:
        return None, len(shift_data), 0.0

    shifts = np.array([s[2] for s in shift_data])
    coords = np.array([(s[0], s[1]) for s in shift_data], dtype=float)
    min_overlap = min(s[3] for s in shift_data)

    def fit(X, a, b, c, d, e, f):
        np_arr, nm_arr = X[:, 0], X[:, 1]
        return (
            a
            + b * np_arr
            + c * nm_arr
            + d * np_arr**2
            + e * np_arr * nm_arr
            + f * nm_arr**2
        )

    popt, _ = scipy.optimize.curve_fit(fit, coords, shifts)
    _, _, _, d, e, f = popt
    M = np.array([
        [2 * d * omega_p / HBAR, 0, e * omega_p / HBAR],
        [0, 0, 0],
        [e * omega_p / HBAR, 0, 2 * f * omega_p / HBAR],
    ])
    return M, len(shift_data), min_overlap


def calibrate_C(omega_z, eps, kind, gamma_tilde):
    quad = ElectrostaticPotential.from_quadrupole(omega_z, m, q, epsilon=eps)
    modes = linear_modes(quad, B, m, q)
    x_zpf_p = np.sqrt(HBAR / (2 * m * modes.omega_plus))
    n_total = sum(int(c) for c in kind[1:])
    return gamma_tilde * HBAR * modes.omega_plus / (q * x_zpf_p**n_total)


B = 0.140

print("=" * 70)
print("TEST 1: Cubic perturbation -- exercises the (1/2){W_3, H_3} factor")
print("=" * 70)
print()
print("A pure C_300 (cubic in x) perturbation is non-resonant in BG cyl")
print("(no monomial a^k ā^l with sum |k - l|=3 satisfies k·Ω = 0 for")
print("generic ratios). The cubic terms are eliminated by W_3, but their")
print("contribution propagates to H_4 via (1/2){W_3, H_3}.")
print()

# Use a moderate ω_+/ω_- ratio
omega_z = TWO_PI * 2.20e9
gamma_tilde = 1e-4  # cubic shifts are O(γ̃²), need slightly larger γ̃

# C_300 perturbation
C_300 = calibrate_C(omega_z, 0.0, "c300", gamma_tilde)
pot = ElectrostaticPotential.from_quadrupole(
    omega_z, m, q, epsilon=0.0
) + ElectrostaticPotential({(3, 0, 0): C_300})
print(f"C_300 = {C_300:.3e} V/m^3 (gamma_tilde = {gamma_tilde})")

M_bgnf = shift_matrix_general(pot, B, m, q, order=4)
sign = np.array([1, 1, -1])
modes = linear_modes(pot.restrict_to_orders(min_order=2, max_order=2), B, m, q)
omegas = np.array([modes.omega_plus, modes.omega_z, modes.omega_minus])
M_I_bgnf = M_bgnf * (TWO_PI * sign * omegas)[np.newaxis, :]

print(f"BGNF M^I (cubic perturbation, after Lie-Deprit triangle):")
print(f"  [+,+] = {M_I_bgnf[0, 0]:.3e}")
print(f"  [+,-] = {M_I_bgnf[0, 2]:.3e}")
print(f"  [-,-] = {M_I_bgnf[2, 2]:.3e}")

M_num, n_states, min_ov = numerical_M_I(pot, B, n_fock=14)
print(
    f"\nNumerical M^I from Fock diagonalization (n_states={n_states}, min_overlap={min_ov:.4f}):"
)
if M_num is None:
    print("  Insufficient data!")
else:
    print(f"  [+,+] = {M_num[0, 0]:.3e}")
    print(f"  [+,-] = {M_num[0, 2]:.3e}")
    print(f"  [-,-] = {M_num[2, 2]:.3e}")
    print(f"\nRatios (numerical / BGNF):")
    for ai, bi, name in [(0, 0, "[+,+]"), (0, 2, "[+,-]"), (2, 2, "[-,-]")]:
        n = M_num[ai, bi]
        b_val = M_I_bgnf[ai, bi]
        if abs(b_val) > 1e-30 * (abs(n) + 1):
            r = n / b_val
            tag = "✓" if 0.99 < r < 1.01 else ("≈" if 0.9 < r < 1.1 else "✗")
            print(f"  {name}: {r:.4f} {tag}")
        else:
            print(f"  {name}: BGNF ≈ 0, numerical = {n:.3e}")
print()

print("=" * 70)
print("TEST 2: v3p4 1:-2 resonance regime")
print("=" * 70)
print()
print("Tune ω_z so ω_+ : ω_- ≈ 2:1 (chip-trap regime). Non-resonant")
print("Birkhoff is invalid; my code's detect_resonances should flag it,")
print("and the M-matrix may not match numerical.")
print()

# Find ω_z that gives ω_+/ω_- = 2 exactly (the v3p4 case)
# In BG cyl: ω_+/ω_- = (ω_c + ω_1)/(ω_c - ω_1) where ω_1 = sqrt(ω_c²-2ω_z²)
# Setting this = 2: ω_c + ω_1 = 2(ω_c - ω_1) → 3 ω_1 = ω_c → ω_1 = ω_c/3
# So ω_c² - 2ω_z² = ω_c²/9 → ω_z² = 4 ω_c²/9 → ω_z = (2/3) ω_c
omega_c = abs(q * B / m)
omega_z_resonant = (2.0 / 3.0) * omega_c
print(f"ω_c/(2π) = {omega_c / TWO_PI / 1e9:.4f} GHz")
print(
    f"ω_z/(2π) = {omega_z_resonant / TWO_PI / 1e9:.4f} GHz (chosen for 1:2 resonance)"
)
modes_r = linear_modes(
    ElectrostaticPotential.from_quadrupole(
        omega_z_resonant, m, q, epsilon=0.0
    ),
    B,
    m,
    q,
)
print(f"  ω_+/(2π) = {modes_r.omega_plus / TWO_PI / 1e9:.4f} GHz")
print(f"  ω_-/(2π) = {modes_r.omega_minus / TWO_PI / 1e9:.4f} GHz")
print(f"  ω_+/ω_-  = {modes_r.omega_plus / modes_r.omega_minus:.6f}")

resonances = detect_resonances(
    modes_r.omega_plus,
    modes_r.omega_z,
    modes_r.omega_minus,
    max_total_degree=4,
    relative_tol=1e-3,
)
print(f"\nDetected resonances ({len(resonances)} total):")
for k, residual in resonances[:5]:
    print(
        f"  k = {k}, |residual|/ω_max = {residual / max(modes_r.omega_plus, modes_r.omega_minus):.3e}"
    )

# Try BGNF -- it should either succeed (if the resonance doesn't enter
# at order 4) or raise.
gamma_tilde_v3p4 = 1e-6
C_400 = calibrate_C(omega_z_resonant, 0.0, "c400", gamma_tilde_v3p4)
pot_v3p4 = ElectrostaticPotential.from_quadrupole(
    omega_z_resonant, m, q, epsilon=0.0
) + ElectrostaticPotential({(4, 0, 0): C_400})

# The 1:-2 resonance enters at cubic order via monomials like a_+ a_-^2,
# but if there are no cubic terms in our potential (only C_400), and the
# resonance is exactly 1:2 (not coupled to ω_z), then quartic non-resonant
# should still work. Let's see.
print()
M_bgnf_v3p4 = shift_matrix_general(pot_v3p4, B, m, q, order=4)
M_I_bgnf_v3p4 = (
    M_bgnf_v3p4
    * (
        TWO_PI
        * sign
        * np.array([modes_r.omega_plus, modes_r.omega_z, modes_r.omega_minus])
    )[np.newaxis, :]
)

print(f"BGNF M^I (v3p4 quartic-only):")
print(f"  [+,+] = {M_I_bgnf_v3p4[0, 0]:.3e}")
print(f"  [+,-] = {M_I_bgnf_v3p4[0, 2]:.3e}")
print(f"  [-,-] = {M_I_bgnf_v3p4[2, 2]:.3e}")

M_num_v3p4, n_states_v3p4, ov_v3p4 = numerical_M_I(pot_v3p4, B, n_fock=14)
if M_num_v3p4 is None:
    print(
        f"\nInsufficient overlap data ({n_states_v3p4}/{4 * 4}); resonance likely"
    )
    print("mixes states heavily.")
else:
    print(
        f"\nNumerical M^I (n_states={n_states_v3p4}, min_overlap={ov_v3p4:.4f}):"
    )
    print(f"  [+,+] = {M_num_v3p4[0, 0]:.3e}")
    print(f"  [+,-] = {M_num_v3p4[0, 2]:.3e}")
    print(f"  [-,-] = {M_num_v3p4[2, 2]:.3e}")
    print(f"\nRatios numerical/BGNF:")
    for ai, bi, name in [(0, 0, "[+,+]"), (0, 2, "[+,-]"), (2, 2, "[-,-]")]:
        n = M_num_v3p4[ai, bi]
        b_val = M_I_bgnf_v3p4[ai, bi]
        if abs(b_val) > 1e-30 * (abs(n) + 1):
            r = n / b_val
            tag = "✓" if 0.99 < r < 1.01 else ("≈" if 0.9 < r < 1.1 else "✗")
            print(f"  {name}: {r:.4f} {tag}")
print()

print("=" * 70)
print("TEST 3: Cubic + quartic combined -- exercises full Lie-Deprit")
print("=" * 70)
print()

omega_z = TWO_PI * 2.20e9
gamma3 = 1e-4
gamma4 = 1e-6
C_300 = calibrate_C(omega_z, 0.0, "c300", gamma3)
C_400 = calibrate_C(omega_z, 0.0, "c400", gamma4)
pot = ElectrostaticPotential.from_quadrupole(
    omega_z, m, q, epsilon=0.0
) + ElectrostaticPotential({(3, 0, 0): C_300, (4, 0, 0): C_400})
print(f"C_300 = {C_300:.3e}, C_400 = {C_400:.3e}")

M_bgnf = shift_matrix_general(pot, B, m, q, order=4)
modes = linear_modes(pot.restrict_to_orders(min_order=2, max_order=2), B, m, q)
omegas = np.array([modes.omega_plus, modes.omega_z, modes.omega_minus])
M_I_bgnf = M_bgnf * (TWO_PI * sign * omegas)[np.newaxis, :]

print(f"BGNF M^I:")
print(f"  [+,+] = {M_I_bgnf[0, 0]:.3e}")
print(f"  [+,-] = {M_I_bgnf[0, 2]:.3e}")
print(f"  [-,-] = {M_I_bgnf[2, 2]:.3e}")

M_num, n_st, ov = numerical_M_I(pot, B, n_fock=14)
print(f"\nNumerical M^I (n_states={n_st}, min_overlap={ov:.4f}):")
if M_num is not None:
    print(f"  [+,+] = {M_num[0, 0]:.3e}")
    print(f"  [+,-] = {M_num[0, 2]:.3e}")
    print(f"  [-,-] = {M_num[2, 2]:.3e}")
    print(f"\nRatios numerical/BGNF:")
    for ai, bi, name in [(0, 0, "[+,+]"), (0, 2, "[+,-]"), (2, 2, "[-,-]")]:
        n = M_num[ai, bi]
        b_val = M_I_bgnf[ai, bi]
        if abs(b_val) > 1e-30 * (abs(n) + 1):
            r = n / b_val
            tag = "✓" if 0.99 < r < 1.01 else ("≈" if 0.9 < r < 1.1 else "✗")
            print(f"  {name}: {r:.4f} {tag}")

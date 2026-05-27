"""Validate BGNF cross-mode entries across multiple parameter regimes.

Sweep:
1. Different ω_+/ω_- ratios (3 to 10)
2. Different epsilon (Kretzschmar elliptical)
3. Different perturbation orders (C_400, C_220, C_040)
4. v3p4-like resonance regime (should disagree -- non-resonant Birkhoff fails)

For each: compare numerical Fock diagonalization to BGNF.
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


def numerical_M_I_radial(
    potential: ElectrostaticPotential,
    B: float,
    m: float,
    q: float,
    *,
    n_fock: int = 12,
    gamma_tilde_target: float = 1e-6,
    n_plus_max: int = 4,
    n_minus_max: int = 4,
    overlap_threshold: float = 0.99,
):
    """Numerically diagonalize H_2 + V in a 2D Fock basis and extract
    the action-derivative shift matrix M^I.

    Returns
    -------
    M_I : (3, 3) ndarray with M^I[+,+], M^I[+,-]=M^I[-,+], M^I[-,-] populated.
    n_states : number of confidently labeled states used in the fit.
    """
    quad_pot = potential.restrict_to_orders(min_order=2, max_order=2)
    pert_pot = potential.restrict_to_orders(min_order=3)
    modes = linear_modes(quad_pot, B, m, q)
    omega_p = modes.omega_plus
    omega_m = modes.omega_minus
    S = modes.transform

    # Build operators in dimensionless units (energy unit = ℏω_+).
    a_p = qutip.tensor(qutip.destroy(n_fock), qutip.qeye(n_fock))
    a_m = qutip.tensor(qutip.qeye(n_fock), qutip.destroy(n_fock))
    N_p = a_p.dag() * a_p
    N_m = a_m.dag() * a_m

    sqrt_hbar = np.sqrt(HBAR)

    # Build position operators x, y in physical units; convert to
    # dimensionless x̃ = x/x_zpf,+.
    def cart_op(row):
        op = (
            S[row, 0] * sqrt_hbar / np.sqrt(2) * (a_p + a_p.dag())
            + S[row, 2] * sqrt_hbar / np.sqrt(2) * (a_m + a_m.dag())
            + S[row, 3] * (-1j * sqrt_hbar / np.sqrt(2)) * (a_p - a_p.dag())
            + S[row, 5] * (-1j * sqrt_hbar / np.sqrt(2)) * (a_m - a_m.dag())
        )
        op = (op + op.dag()) / 2  # numerical Hermitization
        return op

    x_op = cart_op(0)
    y_op = cart_op(1)

    # Build perturbation V_op from the Cartesian Taylor expansion.
    # Work directly with .full() arrays to avoid any QuTiP arithmetic
    # issues at small scales (Dia format precision bug).
    Id_arr = qutip.tensor(qutip.qeye(n_fock), qutip.qeye(n_fock)).full()
    x_arr = x_op.full()
    y_arr = y_op.full()
    V_arr_phys = np.zeros_like(Id_arr)
    for (i, j, k), c in pert_pot.coeffs.items():
        if k != 0:
            continue  # axial component skipped in 2D radial Fock
        if c == 0:
            continue
        op = Id_arr.copy()
        for _ in range(i):
            op = op @ x_arr
        for _ in range(j):
            op = op @ y_arr
        V_arr_phys = V_arr_phys + (q * c) * op

    # Convert to dimensionless H̃ = H/(ℏω_+).
    H_2_arr = N_p.full() - (omega_m / omega_p) * N_m.full()
    V_arr = V_arr_phys / (HBAR * omega_p)
    H_arr = H_2_arr + V_arr

    # Diagonalize.
    energies, vectors = np.linalg.eigh(H_arr)

    # Identify perturbed states and extract energy shifts.
    shift_data = []
    for np_val in range(n_plus_max):
        for nm_val in range(n_minus_max):
            unpert_state = qutip.tensor(
                qutip.fock(n_fock, np_val), qutip.fock(n_fock, nm_val)
            )
            unpert_arr = unpert_state.full().flatten()
            unpert_E = np_val - (omega_m / omega_p) * nm_val
            overlaps = np.abs(vectors.conj().T @ unpert_arr) ** 2
            best_idx = int(np.argmax(overlaps))
            ov = overlaps[best_idx]
            if ov < overlap_threshold:
                continue
            pert_E = energies[best_idx].real
            shift_data.append((np_val, nm_val, pert_E - unpert_E))

    if len(shift_data) < 6:
        return None, len(shift_data)

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
    _, _, _, d, e, f = popt
    M_I = np.zeros((3, 3))
    M_I[0, 0] = 2 * d * omega_p / HBAR
    M_I[0, 2] = e * omega_p / HBAR
    M_I[2, 0] = M_I[0, 2]
    M_I[2, 2] = 2 * f * omega_p / HBAR
    return M_I, len(shift_data)


def get_predictions(potential, B, m, q):
    """Return BGNF M^I and Verdú M^I for the given potential."""
    M_bgnf_V = shift_matrix_general(potential, B, m, q, order=4)
    quad_pot = potential.restrict_to_orders(min_order=2, max_order=2)
    pert_pot = potential.restrict_to_orders(min_order=3)
    modes = linear_modes(quad_pot, B, m, q)
    sign = np.array([1, 1, -1])
    omegas = np.array([modes.omega_plus, modes.omega_z, modes.omega_minus])
    M_I_bgnf = M_bgnf_V * (TWO_PI * sign * omegas)[np.newaxis, :]

    sp = ElectronSpecies(magnetic_field=B)
    trap = PenningTrap(
        magnetic_field=B,
        species=sp,
        d=3.5e-3,
        omega_axial=quad_pot.coeffs.get((0, 0, 2), 0) * 2 * q / m,
    )
    # Use the modes' actual ω_z (which is what `from_quadrupole` produces)
    # rather than reverse-engineering it from C_002:
    sp = ElectronSpecies(magnetic_field=B)
    omega_z = np.sqrt(2 * abs(q) * quad_pot.coeffs.get((0, 0, 2), 0) / m)
    trap = PenningTrap(
        magnetic_field=B, species=sp, d=3.5e-3, omega_axial=omega_z
    )
    eps = -(
        quad_pot.coeffs.get((2, 0, 0), 0) - quad_pot.coeffs.get((0, 2, 0), 0)
    ) / quad_pot.coeffs.get((0, 0, 2), 0)
    trap = PenningTrap(
        magnetic_field=B,
        species=sp,
        d=3.5e-3,
        omega_axial=omega_z,
        epsilon=eps,
    )
    nu_p = trap.omega_modified_cyclotron / TWO_PI
    nu_z = trap.omega_axial / TWO_PI
    nu_m = trap.omega_magnetron / TWO_PI
    orb = orbit_params(
        trap.omega_cyclotron,
        trap.omega_axial,
        trap.omega_modified_cyclotron,
        eps,
    )
    coeff_kw = {f"c{i}{j}{k}": v for (i, j, k), v in pert_pot.coeffs.items()}
    coeffs = AnharmonicCoeffs(c002=1.0, **coeff_kw)
    M_verdu_V = frequency_shifts_matrix(nu_p, nu_z, nu_m, orb, coeffs, m)
    M_I_verdu = M_verdu_V * (TWO_PI * omegas)[np.newaxis, :]
    return M_I_bgnf, M_I_verdu


def run_test(label, potential, B, m, q):
    print(f"\n========== {label} ==========")
    M_I_bgnf, M_I_verdu = get_predictions(potential, B, m, q)
    M_I_num, n_states = numerical_M_I_radial(potential, B, m, q)
    print(f"  States identified: {n_states}")
    if M_I_num is None:
        print("  Insufficient data; skipping.")
        return

    print(
        f"  {'':<10} {'Num':>13} {'BGNF':>13} {'Verdu':>13} "
        f"{'Num/BGNF':>10} {'Num/Verdu':>10}"
    )
    for elem_label, ai, bi in [
        ("[+,+]", 0, 0),
        ("[+,-]", 0, 2),
        ("[-,-]", 2, 2),
    ]:
        n_val = M_I_num[ai, bi]
        b_val = M_I_bgnf[ai, bi]
        v_val = M_I_verdu[ai, bi]
        r_b = (
            n_val / b_val
            if abs(b_val) > 1e-30 * (abs(n_val) + 1)
            else float("nan")
        )
        r_v = (
            n_val / v_val
            if abs(v_val) > 1e-30 * (abs(n_val) + 1)
            else float("nan")
        )
        match_b = "✓" if 0.99 < r_b < 1.01 else "✗"
        match_v = "✓" if 0.99 < r_v < 1.01 else "✗"
        print(
            f"  {elem_label:<10} {n_val:>13.3e} {b_val:>13.3e} "
            f"{v_val:>13.3e} {r_b:>9.4f}{match_b} {r_v:>9.4f}{match_v}"
        )


m = ELECTRON_MASS
q = +ELECTRON_CHARGE


def calibrate_C(omega_z, eps, kind, gamma_tilde):
    """Choose the C_ijk magnitude that gives the requested
    dimensionless perturbation strength gamma_tilde.

    For a Cartesian monomial of order N (e.g. N=4 for C_400),
    gamma = q * C * x_zpf_plus^N / (ℏ ω_+).
    """
    quad = ElectrostaticPotential.from_quadrupole(omega_z, m, q, epsilon=eps)
    modes = linear_modes(quad, B, m, q)
    x_zpf_p = np.sqrt(HBAR / (2 * m * modes.omega_plus))
    n_total = sum(int(c) for c in kind[1:])  # e.g. 'c400' -> 4
    return gamma_tilde * HBAR * modes.omega_plus / (q * x_zpf_p**n_total)


def make_pot(omega_z, eps, gamma_tilde=1e-6, **higher_specs):
    pot = ElectrostaticPotential.from_quadrupole(omega_z, m, q, epsilon=eps)
    pert_dict = {}
    for kind, _placeholder in higher_specs.items():
        c = calibrate_C(omega_z, eps, kind, gamma_tilde)
        pert_dict[tuple(int(d) for d in kind[1:])] = c
    pert = ElectrostaticPotential(pert_dict)
    return pot + pert


# Sweep over different ω_+/ω_- ratios (controlled by ω_z at fixed B)
B = 0.140
for omega_z_GHz in [2.20, 1.50, 1.00, 0.60]:
    omega_z = TWO_PI * omega_z_GHz * 1e9
    pot = make_pot(omega_z, 0.0, c400=None)
    quad = pot.restrict_to_orders(min_order=2, max_order=2)
    modes = linear_modes(quad, B, m, q)
    label = f"ω_z={omega_z_GHz:.2f}GHz, eps=0, C400"
    label += f" (ω+/ω-={modes.omega_plus / modes.omega_minus:.2f})"
    run_test(label, pot, B, m, q)


# Different perturbations (C_220, C_040)
for kind in ["c220", "c040"]:
    pot = make_pot(TWO_PI * 2.20e9, 0.0, **{kind: None})
    label = f"ω_z=2.2GHz, eps=0, {kind.upper()}"
    run_test(label, pot, B, m, q)


# Elliptical (Kretzschmar ε > 0)
for eps in [0.1, 0.3, 0.5]:
    pot = make_pot(TWO_PI * 2.20e9, eps, c400=None)
    label = f"ω_z=2.2GHz, eps={eps}, C400"
    run_test(label, pot, B, m, q)

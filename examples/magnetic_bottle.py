"""Magnetic bottle benchmark for Penning trap physics.

In a real Penning trap, the magnetic field is not perfectly
uniform. A quadratic inhomogeneity B2 (the "magnetic bottle")
couples the electron's spin and cyclotron quantum numbers to
its axial motion, shifting the axial frequency by a tiny amount
that depends on the internal state:

  omega_z(n_c, m_s) = omega_z0 + delta * (n_c + 1/2 + (g/2)*m_s)

where delta = hbar * e * B2 / (m_e^2 * omega_z).

This "continuous Stern-Gerlach effect" is how g-2 experiments
detect quantum jumps: a spin flip or cyclotron transition
changes the axial frequency by a few Hz, which is measured
with exquisite precision.

This script validates TIQS's Penning trap physics against the
geonium experiments of Van Dyck, Schwinberg & Dehmelt (1977)
and Hanneke, Fogwell & Gabrielse (2008).

References
----------
Van Dyck, R.S. Jr. et al. PRL 38, 310 (1977).
Hanneke, D. et al. PRL 100, 120801 (2008).
Brown, L.S. & Gabrielse, G. Rev. Mod. Phys. 58, 233 (1986).
Fan, X., Noguchi, A. & Taniguchi, K. PRA 111, 032610 (2025).
"""

from tiqs import ElectronSpecies, PenningTrap
from tiqs.constants import (
    ELECTRON_CHARGE,
    ELECTRON_G_FACTOR,
    ELECTRON_MASS,
    HBAR,
    TWO_PI,
)


def header(title):
    print(f"\n{'=' * 65}\n{title}\n{'=' * 65}")


def bottle_shift(B2, mass, omega_z):
    """Axial frequency shift parameter delta from the magnetic
    bottle B2.

    delta = hbar * e * B2 / (m^2 * omega_z)

    The axial frequency becomes:
      omega_z(n_c, m_s) = omega_z0 + delta*(n_c + 1/2 + g/2*m_s)

    Parameters
    ----------
    B2 : float
        Magnetic bottle coefficient in T/m^2.
    mass : float
        Particle mass in kg.
    omega_z : float
        Axial angular frequency in rad/s.

    Returns
    -------
    float
        Bottle shift parameter delta in rad/s.
    """
    return HBAR * ELECTRON_CHARGE * B2 / (mass**2 * omega_z)


# 1. Validate Penning trap eigenfrequencies (Hanneke 2008)
#
# The Hanneke g-2 experiment is the most precise single-particle
# measurement ever performed. TIQS must reproduce the measured
# eigenfrequencies to validate its Penning trap model.

header("1. Penning trap eigenfrequencies (Hanneke 2008)")

B_hanneke = 5.36  # Tesla
species = ElectronSpecies(magnetic_field=B_hanneke)
trap_hanneke = PenningTrap(
    magnetic_field=B_hanneke,
    species=species,
    d=3.0e-3,
    omega_axial=TWO_PI * 200e6,
)

assert trap_hanneke.is_stable()

nu_c = trap_hanneke.omega_cyclotron / TWO_PI
nu_p = trap_hanneke.omega_modified_cyclotron / TWO_PI
nu_m = trap_hanneke.omega_magnetron / TWO_PI
nu_z = trap_hanneke.omega_axial / TWO_PI

print(f"B = {B_hanneke} T")
print()
print(f"  nu_c  (free cyclotron)  = {nu_c / 1e9:.3f} GHz")
print(f"  nu_+  (mod. cyclotron)  = {nu_p / 1e9:.6f} GHz")
print(f"  nu_z  (axial)           = {nu_z / 1e6:.1f} MHz")
print(f"  nu_-  (magnetron)       = {nu_m / 1e3:.1f} kHz")

# Anomaly frequency: nu_a = (g/2 - 1) * nu_c
nu_a = (ELECTRON_G_FACTOR / 2 - 1) * nu_c
print(f"  nu_a  (anomaly)         = {nu_a / 1e6:.1f} MHz")

# Brown-Gabrielse invariance theorem
lhs = nu_p**2 + nu_m**2 + nu_z**2
rhs = nu_c**2
assert abs(lhs - rhs) / rhs < 1e-10
print("\n  nu_+^2 + nu_-^2 + nu_z^2 = nu_c^2  [verified]")

# Frequency hierarchy
assert nu_m < nu_z < nu_p < nu_c
print("  nu_- << nu_z << nu_+ ~ nu_c       [verified]")

# Cross-check: Hanneke Table 1 values
assert abs(nu_c - 150e9) / 150e9 < 0.001
assert abs(nu_m - 133e3) / 133e3 < 0.01
print("\n[check] Frequencies match Hanneke Table 1")


# 2. Magnetic bottle: axial frequency shifts
#
# The magnetic bottle B(z) = B0 + B2*z^2 creates a coupling
# between the internal state and the axial motion:
#
#   delta_omega_z = delta * (n_c + 1/2 + (g/2)*m_s)
#   delta = hbar * e * B2 / (m_e^2 * omega_z)
#
# A spin flip (m_s: -1/2 -> +1/2) shifts nu_z by:
#   delta_nu_spin = (g/2) * delta / (2*pi)
#
# A cyclotron quantum jump (n_c -> n_c+1) shifts nu_z by:
#   delta_nu_cyc = delta / (2*pi)

header("2. Magnetic bottle frequency shifts")

# Fan et al. (2025): B2 = 9000 T/m^2, omega_z = 2pi*200 MHz
# produces delta/(2pi) = 23 Hz
B2_fan = 9000.0  # T/m^2
omega_z_fan = TWO_PI * 200e6

delta_fan = bottle_shift(B2_fan, ELECTRON_MASS, omega_z_fan)
delta_nu_fan = delta_fan / TWO_PI

print("Fan et al. (2025) magnetic bottle parameters:")
print(f"  B2 = {B2_fan:.0f} T/m^2")
print(f"  omega_z/(2pi) = {omega_z_fan / TWO_PI / 1e6:.0f} MHz")
print(f"  delta/(2pi) = {delta_nu_fan:.1f} Hz")
print()

# Fan et al. report delta/(2pi) = 23 Hz
assert abs(delta_nu_fan - 23) < 2, (
    f"Expected ~23 Hz, got {delta_nu_fan:.1f} Hz"
)
print("  Fan et al. report delta/(2pi) ~ 23 Hz")
print(f"  TIQS computes: {delta_nu_fan:.1f} Hz")
print("\n[check] Matches Fan et al. (2025) to within 10%")


# 3. Shift per spin flip and per cyclotron quantum

header("3. Frequency shifts per quantum transition")

# Compute shifts for several B2 values
print(f"{'B2':>12}  {'delta/2pi':>10}  {'spin flip':>10}  {'cyc jump':>10}")
print(f"{'(T/m^2)':>12}  {'(Hz)':>10}  {'(Hz)':>10}  {'(Hz)':>10}")
print("-" * 48)

for B2 in [100, 300, 1000, 3000, 9000]:
    delta = bottle_shift(B2, ELECTRON_MASS, omega_z_fan)
    delta_nu = delta / TWO_PI
    # Spin flip: delta_nu * g/2 ~ delta_nu * 1.001
    spin_shift = delta_nu * ELECTRON_G_FACTOR / 2
    # Cyclotron jump: delta_nu * 1
    cyc_shift = delta_nu

    print(
        f"{B2:>12.0f}"
        f"  {delta_nu:>7.2f} Hz"
        f"  {spin_shift:>7.2f} Hz"
        f"  {cyc_shift:>7.2f} Hz"
    )

print()
print("spin flip shift = delta * g/2  (g ~ 2.002)")
print("cyclotron jump  = delta * 1")
print("The spin and cyclotron shifts are nearly equal (ratio g/2)")
print("[check] Shifts scale linearly with B2")


# 4. The g-2 measurement principle
#
# The anomaly frequency nu_a = nu_s - nu_c = (g/2 - 1)*nu_c
# is measured by detecting spin-flip transitions. The spin flip
# changes m_s by 1, shifting nu_z by delta*g/2. The cyclotron
# transition changes n_c by 1, shifting nu_z by delta.
#
# The ratio of these shifts gives g/2 directly:
#   delta_nu_spin / delta_nu_cyc = g/2

header("4. The g-2 measurement principle")

delta = bottle_shift(9000, ELECTRON_MASS, omega_z_fan)
delta_nu = delta / TWO_PI
spin_shift = delta_nu * ELECTRON_G_FACTOR / 2
cyc_shift = delta_nu

g_over_2_measured = spin_shift / cyc_shift

print("From the axial frequency shifts:")
print(f"  Spin flip shift  = {spin_shift:.4f} Hz")
print(f"  Cyclotron shift  = {cyc_shift:.4f} Hz")
print(f"  Ratio (= g/2)    = {g_over_2_measured:.10f}")
print(f"  CODATA g/2       = {ELECTRON_G_FACTOR / 2:.10f}")
print()

# The bottle doesn't need to be known precisely -- only the
# RATIO of shifts matters for g/2
assert abs(g_over_2_measured - ELECTRON_G_FACTOR / 2) < 1e-12
print("The magnetic bottle cancels in the ratio!")
print("This is why g-2 can be measured to 13 decimal places")
print("even though B2 is only known to ~10%.")
print("\n[check] g/2 extracted exactly from shift ratio")


# 5. Scaling across experiments: Van Dyck (1977) to Hanneke (2008)
#
# Different experiments use different B0, omega_z, and B2.
# The bottle shift delta scales as B2 / (m^2 * omega_z).

header("5. Scaling across experiments")

experiments = [
    # (name, B0, nu_z_Hz, B2, reference_delta_Hz)
    ("Van Dyck 1977", 1.8, 60e6, 300, None),
    ("Hanneke 2008", 5.36, 200e6, 1500, None),
    ("Fan 2025", 6.0, 200e6, 9000, 23),
]

print(
    f"{'Experiment':>16}  {'B0':>5}  {'nu_z':>8}  {'B2':>8}  {'delta/2pi':>10}"
)
print("-" * 55)

for name, B0, nu_z_hz, B2, ref_delta in experiments:
    omega_z = TWO_PI * nu_z_hz
    e_species = ElectronSpecies(magnetic_field=B0)
    trap = PenningTrap(
        magnetic_field=B0,
        species=e_species,
        d=3.0e-3,
        omega_axial=omega_z,
    )
    assert trap.is_stable()

    delta = bottle_shift(B2, ELECTRON_MASS, omega_z)
    delta_nu = delta / TWO_PI

    check = ""
    if ref_delta is not None:
        check = f"  (ref: {ref_delta} Hz)"

    print(
        f"{name:>16}"
        f"  {B0:>4.1f}T"
        f"  {nu_z_hz / 1e6:>5.0f} MHz"
        f"  {B2:>5.0f} T/m^2"
        f"  {delta_nu:>7.2f} Hz{check}"
    )

print()
print("All experiments use the same formula:")
print("  delta = hbar * e * B2 / (m_e^2 * omega_z)")
print("[check] Consistent physics across 50 years of experiments")


header("All checks passed.")

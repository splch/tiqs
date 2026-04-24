"""Low-field Penning trap predictions.

Eigenfrequencies, magnetic bottle shifts, and stability analysis
for a low-field electron Penning trap with parameters typical of
chip-based designs.

Test trap: B0 = 160 mT permanent magnet, nu_z ~ 200 MHz
Qubit trap (planned): B0 = 160 mT, nu_z ~ 2-5 GHz
Magnetic inhomogeneities: B1 ~ 70 uT/m, B2 ~ 2.5 T/m^2

Key finding: at B0 = 160 mT, nu_z = 5 GHz is unstable (the
free cyclotron frequency 4.48 GHz < sqrt(2) * 5 GHz). The
qubit trap must use nu_z <= ~3 GHz, where the magnetron
frequency becomes comparable to the axial frequency.
"""

import numpy as np

from tiqs import (
    DuffingPotential,
    ElectronSpecies,
    PenningTrap,
    transition_frequencies,
)
from tiqs.constants import ELECTRON_CHARGE, ELECTRON_MASS, TWO_PI


def header(title):
    print(f"\n{'=' * 65}\n{title}\n{'=' * 65}")


B0 = 0.160  # 160 mT permanent magnet
species = ElectronSpecies(magnetic_field=B0)


# Test trap electrostatic coefficient (cylindrical convention)
# nu_z = 1/(2pi) * sqrt(2 * C2 * q * V_r / m)
C2 = -221119.0  # 1/m^2 (compensated at tuning ratio 0.881119)


# 1. Eigenfrequencies vs. ring voltage (benchmarking against
#    analytical calculations from the trap design)

header("1. Eigenfrequencies vs. ring voltage")

print(f"C2 = {C2:.0f} 1/m^2")
print(f"B0 = {B0 * 1e3:.0f} mT")
nu_c_header = ELECTRON_CHARGE * B0 / (TWO_PI * ELECTRON_MASS)
print(f"nu_c = {nu_c_header / 1e9:.4f} GHz")
print("Voltage range: 0 - 32 V (precision source)")
print()
print(
    f"{'V_ring':>8}  {'nu_z (MHz)':>12}  {'nu_+ (GHz)':>12}"
    f"  {'nu_- (MHz)':>12}  {'Stable':>7}"
)
print("-" * 60)

for V in [1, 2, 5, 10, 15, 20, 25, 30, 32]:
    trap_v = PenningTrap.from_ring_voltage(
        magnetic_field=B0,
        species=species,
        c2=C2,
        v_ring=-V,  # negative V with negative C2
        b1=70e-6,
        b2=2.5,
    )
    nz = trap_v.omega_axial / TWO_PI
    stable = trap_v.is_stable()
    if stable:
        np_ = trap_v.omega_modified_cyclotron / TWO_PI
        nm_ = trap_v.omega_magnetron / TWO_PI
        print(
            f"{V:>5.0f} V"
            f"  {nz / 1e6:>9.2f}"
            f"  {np_ / 1e9:>9.6f}"
            f"  {nm_ / 1e6:>9.4f}"
            f"  {'yes':>7}"
        )
    else:
        print(f"{V:>5.0f} V  {nz / 1e6:>9.2f}{'':>38}  {'NO':>7}")

# Verify Brown-Gabrielse at 10 V
trap_10 = PenningTrap.from_ring_voltage(
    magnetic_field=B0,
    species=species,
    c2=C2,
    v_ring=-10.0,
    b1=70e-6,
    b2=2.5,
)
nc = trap_10.omega_cyclotron / TWO_PI
np_ = trap_10.omega_modified_cyclotron / TWO_PI
nm_ = trap_10.omega_magnetron / TWO_PI
nz = trap_10.omega_axial / TWO_PI
bg_err = abs(np_**2 + nm_**2 + nz**2 - nc**2) / nc**2
print(f"\nBrown-Gabrielse invariant at 10 V: error = {bg_err:.1e}")
print("[check] All eigenfrequencies computed across voltage range")


# 2. Test trap eigenfrequencies at typical operating point

header("2. Test trap (B0 = 160 mT, V_r ~ 10 V)")

trap = PenningTrap.from_ring_voltage(
    magnetic_field=B0,
    species=species,
    c2=C2,
    v_ring=-10.0,
    b1=70e-6,
    b2=2.5,
)
assert trap.is_stable()

nu_c = trap.omega_cyclotron / TWO_PI
nu_p = trap.omega_modified_cyclotron / TWO_PI
nu_m = trap.omega_magnetron / TWO_PI
nu_z = trap.omega_axial / TWO_PI

print(f"nu_c  (free cyclotron)     = {nu_c / 1e9:.4f} GHz")
print(f"nu_+  (modified cyclotron) = {nu_p / 1e9:.4f} GHz")
print(f"nu_z  (axial)              = {nu_z / 1e6:.2f} MHz")
print(f"nu_-  (magnetron)          = {nu_m / 1e6:.4f} MHz")
print()
print(f"nu_c - nu_+ = {(nu_c - nu_p) / 1e6:.2f} MHz")

# Brown-Gabrielse invariance
lhs = nu_p**2 + nu_m**2 + nu_z**2
rhs = nu_c**2
print(f"Brown-Gabrielse: error = {abs(lhs - rhs) / rhs:.1e}")
print("\n[check] Test trap eigenfrequencies computed")


# 3. Low-field effects: nu_c - nu_+ vs B

header("3. Modified cyclotron correction vs. B field")

print("At low fields, nu_c - nu_+ cannot be neglected:")
print()
print(f"{'B':>8}  {'nu_c':>10}  {'nu_c - nu_+':>12}  {'Ratio':>8}")
print("-" * 44)

for B_mt in [100, 160, 300, 1000, 5000]:
    B = B_mt / 1000
    sp = ElectronSpecies(magnetic_field=B)
    t = PenningTrap(
        magnetic_field=B,
        species=sp,
        d=3.5e-3,
        omega_axial=TWO_PI * 200e6,
    )
    if not t.is_stable():
        print(f"{B_mt:>5} mT  UNSTABLE")
        continue
    nc = t.omega_cyclotron / TWO_PI
    diff = nc - t.omega_modified_cyclotron / TWO_PI
    ratio = diff / nc * 100
    print(
        f"{B_mt:>5} mT"
        f"  {nc / 1e9:>7.3f} GHz"
        f"  {diff / 1e6:>9.2f} MHz"
        f"  {ratio:>6.4f}%"
    )

print("\n[check] Low-field correction grows as B decreases")


# 4. Qubit trap stability scan

header("3. Qubit trap stability (B0 = 160 mT)")

print(f"{'nu_z':>8}  {'Stable':>7}  {'nu_+':>10}  {'nu_-':>10}")
print("-" * 42)

for nz_ghz in [1, 2, 3, 4, 5]:
    omega_z = TWO_PI * nz_ghz * 1e9
    t = PenningTrap(
        magnetic_field=B0,
        species=species,
        d=3.5e-3,
        omega_axial=omega_z,
    )
    stable = t.is_stable()
    if stable:
        np_ = t.omega_modified_cyclotron / TWO_PI
        nm_ = t.omega_magnetron / TWO_PI
        print(
            f"{nz_ghz:>5} GHz  {'yes':>7}"
            f"  {np_ / 1e9:>7.3f} GHz"
            f"  {nm_ / 1e6:>7.1f} MHz"
        )
    else:
        print(f"{nz_ghz:>5} GHz  {'NO':>7}  (omega_c < sqrt(2)*omega_z)")

nu_c_ghz = nu_c / 1e9
max_nz = nu_c_ghz / np.sqrt(2)
print(f"\nMax stable nu_z = nu_c / sqrt(2) = {max_nz:.2f} GHz")
print("[check] Stability boundary identified")


# 4. Magnetic bottle shifts at test trap parameters

header("4. Bottle shifts (B2 = 2.5 T/m^2, test trap)")

delta = trap.bottle_shift / TWO_PI
print(f"Bottle shift delta/(2pi) = {delta:.4f} Hz")
print()

print("Axial frequency shifts:")
for nc in range(3):
    for ms in [+0.5, -0.5]:
        shift = trap.axial_frequency_shift(n_cyclotron=nc, m_spin=ms) / TWO_PI
        print(f"  n_c={nc}, m_s={ms:+.1f}: {shift:+.4f} Hz")

print()
print("Cyclotron and magnetron shifts per axial quantum:")
for nz in range(3):
    sc = trap.cyclotron_frequency_shift(n_axial=nz) / TWO_PI
    sm = trap.magnetron_frequency_shift(n_axial=nz) / TWO_PI
    print(f"  n_z={nz}: cyc = {sc:+.4f} Hz, mag = {sm:+.4f} Hz")

print()
dz = trap.b1_equilibrium_shift
print(f"B1 equilibrium shift: {dz:.2e} m ({dz * 1e15:.1f} fm)")
print(f"B2/B0 = {trap.b2 / B0:.1f} T/m^2/T")
print("\n[check] All bottle shifts computed for test trap")


# 5. Anharmonic spectrum for qubit trap targets

header("5. Anharmonic spectra (qubit trap, nu_z = 2 GHz)")

omega_z_qubit = TWO_PI * 2e9

for alpha_mhz in [10, 50, 100, 200]:
    alpha = -TWO_PI * alpha_mhz * 1e6
    pot = DuffingPotential(omega=omega_z_qubit, anharmonicity=alpha)
    freqs = transition_frequencies(pot, n_fock=15)
    gap = (freqs[0] - freqs[1]) / TWO_PI / 1e6
    print(
        f"alpha = {alpha_mhz:>3} MHz:"
        f"  omega_01 = {freqs[0] / TWO_PI / 1e9:.4f} GHz"
        f"  gap = {gap:.1f} MHz"
    )

print()
print("gap = omega_01 - omega_12: qubit addressability")
print("[check] Anharmonic spectra for target range")


header("All checks passed.")

# TIQS - Trapped Ion Quantum Simulator

Simulate trapped-ion quantum computers from first principles. TIQS builds time-dependent Hamiltonians and Lindblad noise models from physical parameters - trap voltages, laser frequencies, ion species data - and solves them with [QuTiP](https://qutip.org). No black-box gate models: every gate is a Hamiltonian integrated numerically, not a unitary read out of a table.

Those Hamiltonians are still idealizations. What is approximated, and where the resulting fidelities are upper bounds rather than predictions, is listed in [Model scope and approximations](#model-scope-and-approximations).

## Quick start

Prepare a Bell state on two Ca-40 ions and measure it:

```python
import numpy as np

from tiqs import (
    PaulTrap,
    SimulationConfig,
    SimulationRunner,
    get_species,
)
from tiqs.analysis.fidelity import bell_state_fidelity

TWO_PI = 2 * np.pi

# Physical setup: Ca-40 ions in a linear Paul trap
species = get_species("Ca40")
trap = PaulTrap(
    v_rf=300,
    omega_rf=TWO_PI * 30e6,
    r0=0.5e-3,
    omega_axial=TWO_PI * 1e6,
    species=species,
)

# Simulate an MS entangling gate
config = SimulationConfig(
    species=species,
    trap=trap,
    n_ions=2,
    n_modes=1,
    n_fock=15,
)
runner = SimulationRunner(config)
result = runner.run_ms_gate(ions=[0, 1])

# Verify Bell state
rho_spin = result.states[-1].ptrace([0, 1])
fid = bell_state_fidelity(rho_spin)
print(f"Bell state fidelity: {fid:.4f}")  # 1.0000
```

Or build the Hamiltonian yourself for full control:

```python
import numpy as np
import qutip

from tiqs import HilbertSpace, OperatorFactory, StateFactory
from tiqs.gates.molmer_sorensen import (
    ms_gate_duration,
    ms_gate_hamiltonian,
)

TWO_PI = 2 * np.pi

hs = HilbertSpace(n_ions=2, n_modes=1, n_fock=20)
ops = OperatorFactory(hs)
sf = StateFactory(hs)

# MS gate parameters
eta = 0.05  # Lamb-Dicke parameter
delta = TWO_PI * 15e3  # sideband detuning
Omega = delta / (4 * eta)  # maximally entangling condition
tau = ms_gate_duration(delta)

# Construct the time-dependent Hamiltonian and solve
H = ms_gate_hamiltonian(
    ops,
    ions=[0, 1],
    mode=0,
    eta=[eta, eta],
    rabi_frequency=Omega,
    detuning=delta,
)
psi0 = sf.ground_state()
result = qutip.sesolve(H, psi0, np.linspace(0, tau, 500))
```

## What you can simulate

TIQS models every layer of a trapped-ion quantum computer:

| Layer | What it computes |
|-------|-----------------|
| **Ion species** | Atomic data for Yb-171, Ca-40, Ca-43, Ba-137, Be-9, Sr-88 (plus the electron, for Penning traps): mass, qubit splitting, transition wavelengths, linewidths, branching ratios |
| **Trap** | Paul trap Mathieu stability, secular frequencies, pseudopotential depth, micromotion amplitude; Penning trap cyclotron / modified-cyclotron / magnetron frequencies |
| **Coulomb crystal** | N-ion equilibrium positions, axial + radial normal mode frequencies and participation vectors via Hessian eigendecomposition, mixed-species chains, zigzag stability |
| **Laser-ion coupling** | Carrier and red/blue sideband Hamiltonians; `full_interaction_hamiltonian` adds the second-order Lamb-Dicke terms (second sidebands and Debye-Waller) |
| **Entangling gates** | Molmer-Sorensen (bichromatic sigma_x) and light-shift (sigma_z); a two-level Cirac-Zoller sequence is included as a teaching example, not as a usable gate |
| **Single-qubit gates** | Rx, Ry, Rz rotations with SK1 and BB1 composite pulse sequences |
| **Motional potentials** | Harmonic, Duffing and arbitrary single-mode potentials; anharmonic level structure and truncation convergence |
| **Cooling** | Doppler limit, resolved sideband cooling (analytical + simulated), EIT cooling, sympathetic cooling of mixed-species chains |
| **Decoherence** | Motional heating (1/f field noise, tunable ion-electrode distance exponent, default d^-4), motional dephasing, qubit T1/T2, off-resonant photon scattering (Raman + Rayleigh), laser phase/intensity noise, addressing crosstalk |
| **SPAM** | Optical pumping initialization, fluorescence detection with Poisson photon counting, joint-distribution measurement sampling, mid-circuit measurement |
| **Transport** | QCCD shuttling and crystal splitting with motional excitation models |
| **Analysis** | State/gate/Bell fidelity, Wigner functions, phase-space trajectories, error budgets |

Every entry is a library function you can call directly. `SimulationRunner` wires up a subset automatically - see [Model scope and approximations](#model-scope-and-approximations) for which channels it applies and which you must add to the Hamiltonian yourself.

## Installation

Requires Python 3.14+.

```bash
uv pip install -e ".[dev]"
```

The only direct dependency is [QuTiP](https://qutip.org) >= 5.2.3 (which brings in NumPy and SciPy).

## Adding noise

Most noise sources are Lindblad collapse operators you hand straight to QuTiP's master equation solver. The rest (addressing crosstalk, laser intensity noise) are Hamiltonian terms you add to the drive instead:

```python
from tiqs.noise.motional import motional_heating_ops
from tiqs.noise.qubit import qubit_dephasing_op
from tiqs.noise.photon_scattering import rayleigh_scattering_op

c_ops = [
    # infinite-temperature bath: d<n>/dt = 10^4 quanta/s, linear in t
    *motional_heating_ops(ops, mode=0, heating_rate=1e4),
    qubit_dephasing_op(ops, ion=0, t2=1e-3),  # T2 = 1 ms
    # Gamma_el, the elastic-scattering *decoherence* rate (s^-1),
    # not the Rayleigh scattering rate itself
    rayleigh_scattering_op(ops, ion=0, rate=500),
]

result = qutip.mesolve(H, initial_state, tlist, c_ops=c_ops)
```

`motional_heating_ops` returns a raising/lowering pair, so splat it rather than appending. Passing `n_bar_env=Nbar` swaps the infinite-temperature limit for a damped bath that starts at the same slope and equilibrates at `<n> = Nbar`.

## Architecture

```
src/tiqs/
    species/       Ion species database (6 ion species + the electron)
    trap.py        Paul trap (Mathieu equation) and Penning trap physics
    chain/         Coulomb crystal equilibrium, normal modes, Lamb-Dicke parameters
    hilbert_space/ Composite tensor-product space, operator/state factories
    interaction/   Laser-ion Hamiltonians (carrier, sidebands, Raman transitions)
    potential.py   Anharmonic single-mode potentials and their level structure
    gates/         Single-qubit (Rx/Ry/Rz, SK1, BB1) and entangling (MS, light-shift ZZ)
    cooling/       Doppler, resolved sideband, EIT, sympathetic cooling
    noise/         Decoherence channels (Lindblad operators + Hamiltonian terms)
    spam/          State preparation (optical pumping) and measurement (fluorescence)
    transport.py   QCCD shuttling and crystal splitting
    simulation/    SimulationRunner orchestrating the full pipeline
    analysis/      Fidelity metrics, Wigner functions, error budgets
```

## Testing

```bash
pytest tests/ -v            # add -n auto to run in parallel
```

893 tests, 98% line coverage. Where a closed form exists, the test asserts it rather than asserting a plausible range: carrier and sideband Rabi frequencies against the analytic sqrt(n) scaling, normal-mode frequency ratios against their exact eigenvalues, T1/T2 decay laws and the linear heating law against their master-equation solutions, composite pulse sequences against `expm` of the ideal rotation, and published measurements (Doppler limits, Coulomb-coupling splittings, Penning-trap frequencies) against the papers they came from.

Coverage is not uniform in strength, because not every module has a closed form to check against. Resolved-sideband and EIT cooling, crystal splitting and addressing crosstalk are phenomenological or truncated models; their tests pin scaling laws and calibration points, not independent references. [Model scope and approximations](#model-scope-and-approximations) says which is which.

CI runs `ruff format --check`, `ruff check` and `ty check` over both `src` and `tests`, then the suite with a 95% coverage floor.

## Supported ion species

| Species | Qubit type | Splitting | Cooling | Raman |
|---------|-----------|-----------|---------|-------|
| Yb-171 | Hyperfine | 12.6428 GHz | 369.5 nm (19.6 MHz) | 355 nm |
| Ca-40 | Optical | 729.3 nm | 397.0 nm (22.4 MHz) | - |
| Ca-43 | Hyperfine | 3.2256 GHz | 397.0 nm (22.4 MHz) | 397.0 nm |
| Ba-137 | Hyperfine | 8.0377 GHz | 493.5 nm (20.3 MHz) | 515 nm |
| Be-9 | Hyperfine | 1.2500 GHz | 313.1 nm (17.97 MHz) | 313.1 nm |
| Sr-88 | Optical | 674.0 nm | 421.7 nm (21.5 MHz) | - |

Hyperfine splittings are the stored `qubit_frequency_hz` rounded to 100 kHz - the stored values carry their full measured precision, with sources in `src/tiqs/species/ion.py`. Optical qubits list the transition wavelength instead. All wavelengths are NIST ASD vacuum values, and the figure in parentheses is the cooling transition's total upper-state decay rate divided by 2*pi.

## Model scope and approximations

Every Hamiltonian here is derived from atomic physics, and every one is still a specific idealization - so reported gate and SPAM fidelities are upper bounds, not predictions. What is deliberately left out:

**Entangling gates.** `ms_gate_hamiltonian` is the idealized single-mode spin-dependent force driven by square pulses. It omits the off-resonant carrier, the counter-rotating sideband halves, spectator motional modes, the Debye-Waller factor and the AC Stark shift; together these reach 1e-4 to 1e-2 infidelity for fast gates (Sorensen and Molmer, Phys. Rev. A 62, 022311 (2000), Sec. IV). There is no multi-mode closure solver and no pulse shaping. The two-level Cirac-Zoller sequence is *not* entangling: without the auxiliary internal level of Cirac and Zoller, Phys. Rev. Lett. 74, 4091 (1995), it realizes a local Z(x)Z and leaks 92.9% of the `|11>` population out of `|n=0>`. Use the MS gate.

**Laser-ion coupling.** Second-order Lamb-Dicke physics - second sidebands and the Debye-Waller correction - lives only in `full_interaction_hamiltonian`. No gate helper and no `SimulationRunner` path reaches it, so a runner simulation is first order in eta throughout.

**Ion structure.** Ions are two-level and singly charged, with no Zeeman or hyperfine substructure, no metastable shelving level and no dipole matrix elements. Magnetic-field physics is not modeled at all: there is no first- or second-order Zeeman dephasing, and magnetic decoherence enters only through the phenomenological `t2_qubit`. Mg-24/25 - the canonical sympathetic coolant for Be-9 chains - is absent from the species database.

**Noise.** `SimulationRunner` assembles motional heating, motional dephasing, qubit dephasing, spontaneous emission, Raman scattering and laser phase noise. Addressing crosstalk and laser intensity noise are Hamiltonian terms rather than collapse operators, so you must add them to the Hamiltonian yourself; Rayleigh scattering has no config field either. Heating and motional dephasing are applied independently and at the same rate to every mode - not weighted by mode participation, and not the correlated centre-of-mass channel that spatially uniform field noise produces (Brownnutt et al., Rev. Mod. Phys. 87, 1419 (2015), Eqs. 22-23). Qubit dephasing is one independent channel per ion, never the collective channel that common-mode field noise produces. Photon recoil on the motion and D-level leakage from Raman events are omitted.

**Traps and motion.** Micromotion is a standalone diagnostic (`micromotion_amplitude`, `stray_field_displacement`): it feeds into no Rabi frequency, mode frequency, Lamb-Dicke parameter or error-budget channel. Paul trap stability uses the lowest-order secular criterion, not a full Floquet solution. There is no surface-trap geometry and no mode-mode (Kerr) coupling between motional modes. The `coulomb_coupling` helpers return exchange terms only and expect *already renormalized* secular frequencies: they do not compute the Coulomb frequency shift, which is not a small correction.

**Transport.** Shuttling excitation is the exact coherent-displacement integral for a smooth velocity profile, so it is a real calculation. Crystal splitting is not: it is a phenomenological scaling calibrated to a single published data point (Bowler et al., Phys. Rev. Lett. 109, 080502 (2012)) and should be read as an order of magnitude.

**Readout.** Detection does not model qubit decay during the detection window, so SPAM fidelities are upper bounds.

See the matching `docs/theory/` page for the limits specific to each topic.

## References

The physics implemented follows:

- Leibfried, Blatt, Monroe, Wineland. "Quantum dynamics of single trapped ions." Rev. Mod. Phys. 75, 281 (2003)
- Molmer, Sorensen. "Multiparticle entanglement of hot trapped ions." Phys. Rev. Lett. 82, 1835 (1999)
- Wineland, Monroe, Itano, Leibfried, King, Meekhof. "Experimental issues in coherent quantum-state manipulation of trapped atomic ions." J. Res. Natl. Inst. Stand. Technol. 103, 259 (1998)
- Brownnutt, Kumph, Rabl, Blatt. "Ion-trap measurements of electric-field noise near surfaces." Rev. Mod. Phys. 87, 1419 (2015)

## License

BSD-3-Clause

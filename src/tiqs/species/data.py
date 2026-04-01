"""Ion species database with atomic properties for trapped-ion QC."""
from dataclasses import dataclass
from typing import Optional

import numpy as np

from tiqs.constants import AMU, BOLTZMANN, HBAR, SPEED_OF_LIGHT, TWO_PI
from tiqs.species.transitions import Transition


@dataclass(frozen=True)
class IonSpecies:
    """Complete atomic data for a single trapped-ion species.

    Parameters
    ----------
    symbol : str
        Species identifier, e.g. "Yb171".
    mass_amu : float
        Atomic mass in atomic mass units.
    nuclear_spin : float
        Nuclear spin quantum number I.
    qubit_type : str
        One of "hyperfine", "optical", or "zeeman".
    qubit_frequency_hz : float
        Qubit transition frequency in Hz.
    qubit_wavelength : float or None
        For optical qubits, the transition wavelength in meters.
    cooling_transition : Transition
        Primary Doppler cooling transition.
    repump_transitions : tuple[Transition, ...]
        Repumper transitions to clear metastable dark states.
    qubit_t1 : float
        T1 relaxation time in seconds (inf for ground-state hyperfine qubits).
    metastable_lifetime : float or None
        Lifetime of metastable D-state in seconds (for shelving/optical qubits).
    """

    symbol: str
    mass_amu: float
    nuclear_spin: float
    qubit_type: str
    qubit_frequency_hz: float
    qubit_wavelength: Optional[float]
    cooling_transition: Transition
    repump_transitions: tuple[Transition, ...]
    qubit_t1: float
    metastable_lifetime: Optional[float] = None
    raman_wavelength: Optional[float] = None

    @property
    def mass_kg(self) -> float:
        """Atomic mass in kilograms."""
        return self.mass_amu * AMU

    def doppler_limit_temperature(self) -> float:
        """Doppler cooling limit temperature T_D = hbar * Gamma / (2 * k_B) in Kelvin."""
        return HBAR * self.cooling_transition.linewidth / (2 * BOLTZMANN)

    def doppler_limit_nbar(self, trap_frequency_hz: float) -> float:
        """Mean phonon number at Doppler limit for a mode with given trap frequency."""
        gamma = self.cooling_transition.linewidth
        omega_trap = TWO_PI * trap_frequency_hz
        return gamma / (2 * omega_trap)


_SPECIES_DB: dict[str, IonSpecies] = {}


def _register(species: IonSpecies) -> None:
    _SPECIES_DB[species.symbol] = species


def get_species(name: str) -> IonSpecies:
    """Look up an ion species by name. Raises KeyError if not found."""
    return _SPECIES_DB[name]


# Ytterbium-171: Hyperfine qubit, 12.6428 GHz
_register(IonSpecies(
    symbol="Yb171",
    mass_amu=170.9363258,
    nuclear_spin=0.5,
    qubit_type="hyperfine",
    qubit_frequency_hz=12.6428e9,
    qubit_wavelength=None,
    cooling_transition=Transition(
        name="2S1/2 -> 2P1/2",
        wavelength=369.5e-9,
        linewidth=TWO_PI * 23e6,
        branching_ratio=0.995,
    ),
    repump_transitions=(
        Transition(name="2D3/2 -> 3D[3/2]1/2", wavelength=935.2e-9, linewidth=TWO_PI * 4.2e6),
        Transition(name="2F7/2 clearout", wavelength=760.0e-9, linewidth=TWO_PI * 0.05e6),
    ),
    qubit_t1=float('inf'),
    raman_wavelength=355e-9,
))

# Calcium-40: Optical qubit at 729 nm
_register(IonSpecies(
    symbol="Ca40",
    mass_amu=39.96259098,
    nuclear_spin=0.0,
    qubit_type="optical",
    qubit_frequency_hz=SPEED_OF_LIGHT / 729e-9,
    qubit_wavelength=729e-9,
    cooling_transition=Transition(
        name="4S1/2 -> 4P1/2",
        wavelength=397e-9,
        linewidth=TWO_PI * 22.4e6,
        branching_ratio=0.935,
    ),
    repump_transitions=(
        Transition(name="3D3/2 -> 4P1/2", wavelength=866e-9, linewidth=TWO_PI * 1.69e6),
        Transition(name="3D5/2 -> 4P3/2", wavelength=854e-9, linewidth=TWO_PI * 1.58e6),
    ),
    qubit_t1=1.168,
    metastable_lifetime=1.168,
))

# Calcium-43: Hyperfine qubit at 3.2256 GHz
_register(IonSpecies(
    symbol="Ca43",
    mass_amu=42.9587666,
    nuclear_spin=3.5,
    qubit_type="hyperfine",
    qubit_frequency_hz=3.22560829e9,
    qubit_wavelength=None,
    cooling_transition=Transition(
        name="4S1/2 -> 4P1/2",
        wavelength=397e-9,
        linewidth=TWO_PI * 22.4e6,
        branching_ratio=0.935,
    ),
    repump_transitions=(
        Transition(name="3D3/2 -> 4P1/2", wavelength=866e-9, linewidth=TWO_PI * 1.69e6),
    ),
    qubit_t1=float('inf'),
    metastable_lifetime=1.168,
    raman_wavelength=397e-9,
))

# Barium-137: Hyperfine qubit with all-visible wavelengths
_register(IonSpecies(
    symbol="Ba137",
    mass_amu=136.9058274,
    nuclear_spin=1.5,
    qubit_type="hyperfine",
    qubit_frequency_hz=8.038e9,
    qubit_wavelength=None,
    cooling_transition=Transition(
        name="6S1/2 -> 6P1/2",
        wavelength=493e-9,
        linewidth=TWO_PI * 15.1e6,
        branching_ratio=0.75,
    ),
    repump_transitions=(
        Transition(name="5D3/2 -> 6P1/2", wavelength=650e-9, linewidth=TWO_PI * 5.3e6),
    ),
    qubit_t1=float('inf'),
    metastable_lifetime=30.14,
    raman_wavelength=515e-9,
))

# Beryllium-9: Lightest ion qubit, hyperfine
_register(IonSpecies(
    symbol="Be9",
    mass_amu=9.0121831,
    nuclear_spin=1.5,
    qubit_type="hyperfine",
    qubit_frequency_hz=1.25e9,
    qubit_wavelength=None,
    cooling_transition=Transition(
        name="2S1/2 -> 2P3/2",
        wavelength=313e-9,
        linewidth=TWO_PI * 18e6,
        branching_ratio=1.0,
    ),
    repump_transitions=(),
    qubit_t1=float('inf'),
    raman_wavelength=313e-9,
))

# Strontium-88: Optical qubit at 674 nm
_register(IonSpecies(
    symbol="Sr88",
    mass_amu=87.9056121,
    nuclear_spin=0.0,
    qubit_type="optical",
    qubit_frequency_hz=SPEED_OF_LIGHT / 674e-9,
    qubit_wavelength=674e-9,
    cooling_transition=Transition(
        name="5S1/2 -> 5P1/2",
        wavelength=422e-9,
        linewidth=TWO_PI * 21.5e6,
        branching_ratio=0.944,
    ),
    repump_transitions=(
        Transition(name="4D3/2 -> 5P1/2", wavelength=1092e-9, linewidth=TWO_PI * 1.4e6),
    ),
    qubit_t1=0.390,
    metastable_lifetime=0.390,
))

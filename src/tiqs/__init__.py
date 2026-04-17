r"""TIQS: Trapped Ion Quantum Simulator.

Lowest-level trapped-ion quantum computing simulation built on QuTiP.

.. include:: ../../docs/theory/overview.md
"""

from tiqs.chain.equilibrium import equilibrium_positions
from tiqs.chain.lamb_dicke import lamb_dicke_parameters
from tiqs.chain.normal_modes import ModeGroup, NormalModeResult, normal_modes
from tiqs.cooling.sympathetic import (
    apply_sympathetic_cooling,
    coolant_participation,
    sympathetic_cooling_rate,
    sympathetic_doppler_nbar,
    sympathetic_sideband_nbar,
)
from tiqs.hilbert_space.builder import HilbertSpace
from tiqs.hilbert_space.operators import OperatorFactory
from tiqs.hilbert_space.states import StateFactory
from tiqs.interaction.coulomb_coupling import (
    beam_splitter_coupling,
    optomechanical_coupling,
)
from tiqs.potential import (
    ArbitraryPotential,
    DuffingPotential,
    HarmonicPotential,
    Potential,
    check_convergence,
    energy_levels,
    mode_hamiltonian,
    transition_frequencies,
)
from tiqs.simulation.config import SimulationConfig
from tiqs.simulation.runner import SimulationRunner
from tiqs.species.electron import ElectronSpecies
from tiqs.species.ion import IonSpecies, get_species
from tiqs.species.protocol import Species
from tiqs.trap import PaulTrap, PenningTrap, Trap

__all__ = [
    "ArbitraryPotential",
    "DuffingPotential",
    "ElectronSpecies",
    "HarmonicPotential",
    "HilbertSpace",
    "IonSpecies",
    "ModeGroup",
    "NormalModeResult",
    "OperatorFactory",
    "PaulTrap",
    "PenningTrap",
    "Potential",
    "SimulationConfig",
    "SimulationRunner",
    "Species",
    "StateFactory",
    "Trap",
    "apply_sympathetic_cooling",
    "beam_splitter_coupling",
    "check_convergence",
    "coolant_participation",
    "energy_levels",
    "equilibrium_positions",
    "get_species",
    "lamb_dicke_parameters",
    "mode_hamiltonian",
    "normal_modes",
    "optomechanical_coupling",
    "sympathetic_cooling_rate",
    "sympathetic_doppler_nbar",
    "sympathetic_sideband_nbar",
    "transition_frequencies",
]

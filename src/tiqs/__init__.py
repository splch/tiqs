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
from tiqs.elliptical import (
    AnharmonicCoeffs,
    OrbitParams,
    frequency_shifts_matrix,
    orbit_params,
)
from tiqs.hilbert_space.builder import HilbertSpace
from tiqs.hilbert_space.operators import OperatorFactory
from tiqs.hilbert_space.states import StateFactory
from tiqs.interaction.coulomb_coupling import (
    beam_splitter_coupling,
    coulomb_self_kerr,
    optomechanical_coupling,
)
from tiqs.multipole import (
    BirkhoffNormalForm,
    ElectrostaticPotential,
    LinearModeResult,
    Polynomial,
    birkhoff_normal_form,
    canonical_hessian,
    cartesian_polynomials,
    detect_resonances,
    frequency_shift_matrix_actions,
    frequency_shift_matrix_energy,
    linear_modes,
    potential_polynomial,
    quadratic_normal_form,
    shift_matrix_general,
    spectral_coefficient,
    split_kernel_image,
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
from tiqs.species.proton import ProtonSpecies
from tiqs.trap import PaulTrap, PenningTrap, Trap

__all__ = [
    "AnharmonicCoeffs",
    "ArbitraryPotential",
    "BirkhoffNormalForm",
    "DuffingPotential",
    "ElectronSpecies",
    "ElectrostaticPotential",
    "HarmonicPotential",
    "HilbertSpace",
    "IonSpecies",
    "LinearModeResult",
    "ModeGroup",
    "NormalModeResult",
    "OperatorFactory",
    "OrbitParams",
    "PaulTrap",
    "PenningTrap",
    "Polynomial",
    "Potential",
    "ProtonSpecies",
    "SimulationConfig",
    "SimulationRunner",
    "Species",
    "StateFactory",
    "Trap",
    "apply_sympathetic_cooling",
    "beam_splitter_coupling",
    "birkhoff_normal_form",
    "canonical_hessian",
    "cartesian_polynomials",
    "check_convergence",
    "coolant_participation",
    "coulomb_self_kerr",
    "detect_resonances",
    "energy_levels",
    "equilibrium_positions",
    "frequency_shift_matrix_actions",
    "frequency_shift_matrix_energy",
    "frequency_shifts_matrix",
    "get_species",
    "lamb_dicke_parameters",
    "linear_modes",
    "mode_hamiltonian",
    "normal_modes",
    "optomechanical_coupling",
    "orbit_params",
    "potential_polynomial",
    "quadratic_normal_form",
    "shift_matrix_general",
    "spectral_coefficient",
    "split_kernel_image",
    "sympathetic_cooling_rate",
    "sympathetic_doppler_nbar",
    "sympathetic_sideband_nbar",
    "transition_frequencies",
]

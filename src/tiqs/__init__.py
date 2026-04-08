r"""TIQS: Trapped Ion Quantum Simulator.

Lowest-level trapped-ion quantum computing simulation built on QuTiP.

.. include:: ../../docs/theory/overview.md
"""

from tiqs.chain.equilibrium import equilibrium_positions
from tiqs.chain.lamb_dicke import lamb_dicke_parameters
from tiqs.chain.normal_modes import NormalModeResult, normal_modes
from tiqs.hilbert_space.builder import HilbertSpace
from tiqs.hilbert_space.operators import OperatorFactory
from tiqs.hilbert_space.states import StateFactory
from tiqs.simulation.config import SimulationConfig
from tiqs.simulation.runner import SimulationRunner
from tiqs.species.data import IonSpecies, get_species
from tiqs.trap import PaulTrap

__all__ = [
    "HilbertSpace",
    "IonSpecies",
    "NormalModeResult",
    "OperatorFactory",
    "PaulTrap",
    "SimulationConfig",
    "SimulationRunner",
    "StateFactory",
    "equilibrium_positions",
    "get_species",
    "lamb_dicke_parameters",
    "normal_modes",
]

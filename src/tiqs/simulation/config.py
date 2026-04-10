"""Simulation configuration: approximation levels, solver options,
noise parameters.
"""

from dataclasses import dataclass, field

from tiqs.species.ion import IonSpecies
from tiqs.trap import PaulTrap


@dataclass
class SimulationConfig:
    """Complete configuration for a trapped-ion simulation.

    Attributes
    ----------
    species : IonSpecies
        Ion species for all qubits.
    trap : PaulTrap
        Trap configuration.
    n_ions : int
        Number of ions.
    n_modes : int
        Number of motional modes to include.
    n_fock : int
        Fock space cutoff per mode.
    solver : str
        QuTiP solver: ``"sesolve"``, ``"mesolve"``, or ``"mcsolve"``.
    lamb_dicke_order : int
        Order of Lamb-Dicke expansion (1 = standard, 2 = with corrections).
    heating_rate : float or None
        Motional heating rate in quanta/s. ``None`` = no heating.
    t2_qubit : float or None
        Qubit T2 dephasing time in seconds. ``None`` = no dephasing.
    t1_qubit : float or None
        Qubit T1 decay time. ``None`` = use species default.
    photon_scattering_rate : float or None
        Off-resonant photon scattering rate. ``None`` = no scattering.
    n_bar_initial : float
        Initial mean phonon number (after cooling). 0 = ground state.
    solver_options : dict[str, object]
        Additional options passed to the QuTiP solver.
    """

    species: IonSpecies
    trap: PaulTrap
    n_ions: int
    n_modes: int = 1
    n_fock: int = 15
    solver: str = "sesolve"
    lamb_dicke_order: int = 1
    heating_rate: float | None = None
    t2_qubit: float | None = None
    t1_qubit: float | None = None
    photon_scattering_rate: float | None = None
    n_bar_initial: float = 0.0
    solver_options: dict[str, object] = field(
        default_factory=lambda: {"max_step": 0.0, "nsteps": 5000}
    )

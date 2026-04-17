"""Simulation configuration: approximation levels, solver options,
noise parameters.
"""

from dataclasses import dataclass, field

from tiqs.potential import Potential
from tiqs.species.protocol import Species
from tiqs.trap import Trap


@dataclass
class SimulationConfig:
    """Complete configuration for a trapped-ion simulation.

    Attributes
    ----------
    species : Species or list[Species]
        Trapped particle species. A single ``Species`` applies to
        all ions. A list provides per-ion species for mixed-species
        chains (e.g. ``[get_species("Be9"), get_species("Ca40")]``).
        ``trap.species`` is the reference species for
        electrode-derived quantities (spring constant, Mathieu
        parameters).
    trap : Trap
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
    potentials : dict[int, Potential]
        Anharmonic potentials per mode index. Modes not in this dict
        default to harmonic. See ``DuffingPotential`` for transmon-like
        anharmonicity.
    coolant_indices : list[int] or None
        Indices of coolant ions for sympathetic cooling. ``None`` =
        no sympathetic cooling. When set, ``species`` must be a list
        with the coolant species at these indices.
    heating_rates : list[float] or None
        Per-mode heating rates in quanta/s. When set, overrides the
        scalar ``heating_rate``. Length must equal ``n_modes``.
    n_bar_initial_per_mode : list[float] or None
        Per-mode initial phonon numbers. When set, overrides the
        scalar ``n_bar_initial``. Length must equal ``n_modes``.
    solver_options : dict[str, object]
        Additional options passed to the QuTiP solver.
    """

    species: Species | list[Species]
    trap: Trap
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
    potentials: dict[int, Potential] = field(default_factory=dict)
    coolant_indices: list[int] | None = None
    heating_rates: list[float] | None = None
    n_bar_initial_per_mode: list[float] | None = None
    solver_options: dict[str, object] = field(
        default_factory=lambda: {"max_step": 0.0, "nsteps": 5000}
    )

    def __post_init__(self):
        for name in ("heating_rates", "n_bar_initial_per_mode"):
            value = getattr(self, name)
            if value is not None and len(value) != self.n_modes:
                raise ValueError(
                    f"{name} length {len(value)} != n_modes {self.n_modes}"
                )
        if self.coolant_indices is not None:
            if not self.coolant_indices:
                raise ValueError("coolant_indices must not be empty")
            for idx in self.coolant_indices:
                if idx < 0 or idx >= self.n_ions:
                    raise ValueError(
                        f"coolant index {idx} out of range"
                        f" [0, {self.n_ions})"
                    )

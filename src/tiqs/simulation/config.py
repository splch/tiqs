"""Simulation configuration: approximation levels, solver options,
noise parameters.
"""

from dataclasses import dataclass, field

from tiqs.potential import Potential
from tiqs.species.protocol import Species
from tiqs.trap import Trap

SOLVERS = ("sesolve", "mesolve", "mcsolve")
"""QuTiP solvers ``SimulationRunner`` knows how to dispatch to."""


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
        Number of ions (>= 1).
    n_modes : int
        Number of motional modes to include. A linear chain of
        ``n_ions`` ions has exactly ``n_ions`` modes per direction,
        so ``1 <= n_modes <= n_ions``.
    n_fock : int
        Fock space cutoff per mode (>= 2).
    solver : str
        QuTiP solver: ``"sesolve"``, ``"mesolve"``, or ``"mcsolve"``.
        ``"sesolve"`` is upgraded to ``"mesolve"`` automatically
        whenever the noise configuration produces collapse
        operators.
    ntraj : int
        Number of trajectories for ``"mcsolve"`` (default 500,
        matching QuTiP). Statistical error scales as
        $1/\\sqrt{N_\\text{traj}}$, so 500 trajectories give ~4.5%.
        Ignored by the deterministic solvers.
    k_eff : float or list[float] or None
        Effective laser wavevector in rad/m used for the
        Lamb-Dicke parameters. A single ``float`` applies to all
        ions; a list gives per-ion values. ``None`` derives it from
        each species' optical qubit or Raman wavelength, which is
        only defined for laser-driven ``IonSpecies``; species
        without a laser transition (e.g. ``ElectronSpecies``, whose
        spin-motion coupling comes from a magnetic-field gradient)
        must supply ``k_eff`` explicitly.
    heating_rate : float or None
        Motional heating rate in quanta/s. ``None`` = no heating.
    t2_qubit : float or None
        Qubit $T_2$ coherence time in seconds -- the *total*
        coherence time, not the pure-dephasing time. The runner
        subtracts the $1/(2 T_1)$ contribution of ``t1_qubit`` so
        that the simulated coherence decays at $1/T_2$. Requires
        ``t2_qubit <= 2 * t1_qubit``. ``None`` = no dephasing.
    t1_qubit : float or None
        Qubit $T_1$ decay time in seconds. ``None`` = no
        spontaneous emission (the species' own ``qubit_t1`` is
        deliberately *not* applied automatically).
    photon_scattering_rate : float or None
        Inelastic (Raman) photon scattering rate in events/s,
        applied per ion. ``None`` = no scattering. The elastic
        (Rayleigh) branch has a species- and detuning-dependent
        branching ratio (Ozeri et al., *Phys. Rev. A* **75**,
        042329 (2007)) and is not derived from this field; build it
        with ``rayleigh_scattering_op`` when needed.
    motional_dephasing_rates : list[float] or None
        Per-mode motional dephasing rates in rad/s (trap-frequency
        fluctuations, $L = \\sqrt{\\gamma}\\,\\hat{n}$). Length must
        equal ``n_modes``. ``None`` = no motional dephasing.
    laser_phase_linewidth : float or None
        FWHM linewidth of the driving laser / beat note in rad/s,
        applied as per-ion dephasing. ``None`` = no laser phase
        noise.
    n_bar_initial : float
        Initial mean phonon number (after cooling). 0 = ground state.
    potentials : dict[int, Potential]
        Anharmonic potentials per mode index. Modes not in this dict
        default to harmonic. Keys must be valid mode indices in
        ``range(n_modes)``. See ``DuffingPotential`` for
        transmon-like anharmonicity.
    coolant_indices : list[int] or None
        Indices of coolant ions for sympathetic cooling. ``None`` =
        no sympathetic cooling. When set, ``species`` must be a list
        with the coolant species at these indices, and all coolant
        ions must share one species (the cooling rate is derived
        from a single recoil frequency).
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
    ntraj: int = 500
    k_eff: float | list[float] | None = None
    heating_rate: float | None = None
    t2_qubit: float | None = None
    t1_qubit: float | None = None
    photon_scattering_rate: float | None = None
    motional_dephasing_rates: list[float] | None = None
    laser_phase_linewidth: float | None = None
    n_bar_initial: float = 0.0
    potentials: dict[int, Potential] = field(default_factory=dict)
    coolant_indices: list[int] | None = None
    heating_rates: list[float] | None = None
    n_bar_initial_per_mode: list[float] | None = None
    solver_options: dict[str, object] = field(
        default_factory=lambda: {"max_step": 0.0, "nsteps": 5000}
    )

    def __post_init__(self):
        """Validate the configuration eagerly.

        Every check here guards a value that would otherwise
        surface much later as a ``NaN`` state, an opaque integrator
        failure, or a silently different physical model.
        """
        self._validate_dimensions()
        self._validate_solver()
        self._validate_noise()
        self._validate_modes()
        self._validate_coolant()

    def _validate_dimensions(self):
        """Check Hilbert-space sizes."""
        if self.n_ions < 1:
            raise ValueError(f"n_ions must be >= 1, got {self.n_ions}")
        if self.n_modes < 1:
            raise ValueError(f"n_modes must be >= 1, got {self.n_modes}")
        if isinstance(self.n_fock, bool) or not isinstance(self.n_fock, int):
            raise TypeError(f"n_fock must be an int, got {self.n_fock!r}")
        if self.n_fock < 2:
            raise ValueError(
                f"n_fock must be >= 2, got {self.n_fock}. A single Fock "
                f"level freezes the motion (destroy(1) == 0)."
            )

    def _validate_solver(self):
        """Check the solver name and its trajectory count."""
        if self.solver not in SOLVERS:
            raise ValueError(
                f"Unknown solver {self.solver!r}. Expected one of "
                f"{', '.join(SOLVERS)}."
            )
        if self.ntraj < 1:
            raise ValueError(f"ntraj must be >= 1, got {self.ntraj}")

    def _validate_noise(self):
        """Check coherence times and non-negative rates."""
        for name in ("t1_qubit", "t2_qubit"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be > 0, got {value}")
        both_times = self.t1_qubit is not None and self.t2_qubit is not None
        if both_times and self.t2_qubit > 2 * self.t1_qubit:
            raise ValueError(
                f"t2_qubit {self.t2_qubit} exceeds 2 * t1_qubit "
                f"{2 * self.t1_qubit}: the implied pure-dephasing "
                f"rate 1/T2 - 1/(2*T1) would be negative."
            )
        scalars = (
            "heating_rate",
            "photon_scattering_rate",
            "laser_phase_linewidth",
            "n_bar_initial",
        )
        for name in scalars:
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be >= 0, got {value}")
        lists = (
            "heating_rates",
            "motional_dephasing_rates",
            "n_bar_initial_per_mode",
        )
        for name in lists:
            values = getattr(self, name)
            if values is None:
                continue
            for m, value in enumerate(values):
                if value < 0:
                    raise ValueError(f"{name}[{m}] must be >= 0, got {value}")

    def _validate_modes(self):
        """Check per-mode list lengths and mode-index ranges."""
        for name in (
            "heating_rates",
            "n_bar_initial_per_mode",
            "motional_dephasing_rates",
        ):
            value = getattr(self, name)
            if value is not None and len(value) != self.n_modes:
                raise ValueError(
                    f"{name} length {len(value)} != n_modes {self.n_modes}"
                )
        if self.n_modes > self.n_ions:
            raise ValueError(
                f"n_modes {self.n_modes} exceeds n_ions {self.n_ions}: a "
                f"chain of {self.n_ions} ions has only {self.n_ions} normal "
                f"modes per direction, so the extra modes would have no "
                f"frequency and no Lamb-Dicke parameter."
            )
        for mode_idx in self.potentials:
            if mode_idx < 0 or mode_idx >= self.n_modes:
                raise ValueError(
                    f"potentials key {mode_idx} out of range "
                    f"[0, {self.n_modes})"
                )

    def _validate_coolant(self):
        """Check the sympathetic-cooling coolant indices."""
        if self.coolant_indices is None:
            return
        if not self.coolant_indices:
            raise ValueError("coolant_indices must not be empty")
        if len(set(self.coolant_indices)) != len(self.coolant_indices):
            raise ValueError("coolant_indices contains duplicates")
        for idx in self.coolant_indices:
            if idx < 0 or idx >= self.n_ions:
                raise ValueError(
                    f"coolant index {idx} out of range [0, {self.n_ions})"
                )

"""Two-photon stimulated Raman transition parameters."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RamanPair:
    r"""A pair of laser beams driving a stimulated Raman transition.

    Every property below comes from adiabatic elimination of a
    **single** intermediate excited state (Wineland et al., J. Res.
    NIST 103, 259 (1998), Sec. 2.3.3, Eqs. (39)-(41)), which requires

    $$
    |\Delta| \gg \Omega_1,\ \Omega_2,\ \Gamma,\ \omega_\text{qubit}
    $$

    Only $\Delta \neq 0$ is enforced (see `__post_init__`); the
    inequality above is the caller's responsibility. Fine-structure
    interference between intermediate levels - which dominates real
    hyperfine-qubit Raman gates - is not modelled; see Ozeri et al.,
    Phys. Rev. A 75, 042329 (2007), Eqs. (1)-(6) for the
    fine-structure-resolved treatment.

    This class is a parameter calculator: nothing in TIQS consumes it
    yet. In particular `ac_stark_shift` is not applied to any gate
    Hamiltonian, so simulated MS/Raman gates behave as if the
    differential shift had been perfectly compensated.

    Parameters
    ----------
    omega_1, omega_2 : float
        Angular frequencies of the two beams (rad/s).
    rabi_1, rabi_2 : float
        Single-photon Rabi frequencies for each beam (rad/s).
    detuning_from_excited : float
        Single-photon detuning $\Delta$ from the intermediate excited
        state (rad/s). Signed: negative is below the intermediate
        state.
    excited_state_linewidth : float
        Natural linewidth $\Gamma$ of the intermediate state (rad/s).
    """

    omega_1: float
    omega_2: float
    rabi_1: float
    rabi_2: float
    detuning_from_excited: float
    excited_state_linewidth: float = 0.0

    def __post_init__(self):
        """Reject a zero single-photon detuning.

        The wider adiabatic-elimination condition
        $|\\Delta| \\gg \\Omega_1, \\Omega_2, \\Gamma$ is documented in
        the class docstring but not checked here, since the acceptable
        margin depends on the target gate error.
        """
        if self.detuning_from_excited == 0:
            raise ValueError(
                "detuning_from_excited must be non-zero for the "
                "adiabatic-elimination formula to apply"
            )

    @property
    def effective_rabi_frequency(self) -> float:
        r"""Two-photon effective Rabi frequency.

        $$
        \Omega_\text{eff} = \frac{\Omega_1 \Omega_2}{2\Delta}
        $$

        Signed, like the underlying coupling $-g_1^* g_2/\Delta$
        (Wineland 1998 Eq. (40)): tuning the beams to the other side
        of the intermediate state reverses the sign, which is
        equivalent to a $\pi$ shift of the drive phase. The flopping
        rate is $|\Omega_\text{eff}|$.
        """
        return self.rabi_1 * self.rabi_2 / (2 * self.detuning_from_excited)

    @property
    def frequency_difference(self) -> float:
        r"""Beat frequency $\omega_1 - \omega_2$.

        Should match the qubit splitting for resonance.
        """
        return self.omega_1 - self.omega_2

    @property
    def scattering_rate(self) -> float:
        r"""Off-resonant photon scattering rate (rad/s).

        $$
        \Gamma_\text{scatter} \sim
            \frac{(\Omega_1^2 + \Omega_2^2)\,\Gamma}
                 {4\Delta^2}
        $$

        This is the TOTAL photon-scattering rate (Rayleigh plus
        Raman) in the single-intermediate-level limit: Ozeri et al.,
        PRA 75, 042329 (2007), Eq. (2) with the fine-structure
        splitting $\omega_f \to \infty$. It is an upper bound on the
        decoherence rate, because elastic Rayleigh scattering that
        leaves the qubit state untouched does not necessarily
        decohere it (Ozeri 2007, Sec. IV). With the fine structure
        resolved, the scattering per $\pi$-pulse is minimised at
        $\Delta = (\sqrt{2} - 1)\,\omega_f$ (their Eq. (5)), a
        trade-off this formula cannot express.
        """
        if self.excited_state_linewidth == 0:
            return 0.0
        gamma = self.excited_state_linewidth
        delta = self.detuning_from_excited
        return (self.rabi_1**2 + self.rabi_2**2) * gamma / (4 * delta**2)

    @property
    def ac_stark_shift(self) -> float:
        r"""Differential AC Stark shift.

        $$
        \Delta_\text{AC} = \frac{\Omega_1^2 - \Omega_2^2}{4\Delta}
        $$

        The difference of the two level shifts $|g_i|^2/\Delta$
        (Wineland 1998 Eq. (41)), so it vanishes for balanced beams
        and reverses with the sign of $\Delta$. Enters a Hamiltonian
        as $(\Delta_\text{AC}/2)\sigma_z$; TIQS does not add that term
        anywhere, so a simulation using these parameters assumes the
        shift is compensated. Single-intermediate-level limit: in a
        real hyperfine qubit both beams shift both qubit levels
        through several intermediate states.
        """
        return (self.rabi_1**2 - self.rabi_2**2) / (
            4 * self.detuning_from_excited
        )

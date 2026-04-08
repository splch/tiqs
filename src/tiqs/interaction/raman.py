"""Two-photon stimulated Raman transition parameters."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RamanPair:
    """A pair of laser beams driving a stimulated Raman transition.

    Parameters
    ----------
    omega_1, omega_2 : float
        Angular frequencies of the two beams (rad/s).
    rabi_1, rabi_2 : float
        Single-photon Rabi frequencies for each beam (rad/s).
    detuning_from_excited : float
        Single-photon detuning Delta from the intermediate excited
        state (rad/s).
    excited_state_linewidth : float
        Natural linewidth Gamma of the intermediate state (rad/s).
    """

    omega_1: float
    omega_2: float
    rabi_1: float
    rabi_2: float
    detuning_from_excited: float
    excited_state_linewidth: float = 0.0

    @property
    def effective_rabi_frequency(self) -> float:
        """Two-photon effective Rabi frequency.

        Omega_eff = Omega_1 * Omega_2 / (2 * Delta)
        """
        return (
            self.rabi_1 * self.rabi_2 / (2 * abs(self.detuning_from_excited))
        )

    @property
    def frequency_difference(self) -> float:
        """Beat frequency omega_1 - omega_2.

        Should match the qubit splitting for resonance.
        """
        return self.omega_1 - self.omega_2

    @property
    def scattering_rate(self) -> float:
        """Off-resonant photon scattering rate (rad/s).

        Gamma_scatter ~ (Omega_1^2 + Omega_2^2) * Gamma / (4 * Delta^2)
        """
        if self.excited_state_linewidth == 0:
            return 0.0
        gamma = self.excited_state_linewidth
        delta = self.detuning_from_excited
        return (self.rabi_1**2 + self.rabi_2**2) * gamma / (4 * delta**2)

    @property
    def ac_stark_shift(self) -> float:
        """Differential AC Stark shift.

        (Omega_1^2 - Omega_2^2) / (4 * Delta)
        """
        return (self.rabi_1**2 - self.rabi_2**2) / (
            4 * self.detuning_from_excited
        )

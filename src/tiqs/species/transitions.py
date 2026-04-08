"""Atomic transition data for trapped-ion species."""

from dataclasses import dataclass

from tiqs.constants import SPEED_OF_LIGHT, TWO_PI


@dataclass(frozen=True)
class Transition:
    """An atomic transition between two energy levels.

    Attributes
    ----------
    name : str
        Human-readable label, e.g. ``"S1/2 -> P1/2"``.
    wavelength : float
        Transition wavelength in meters.
    linewidth : float
        Natural linewidth (angular frequency) in rad/s.
    branching_ratio : float
        Fraction of decays going through this channel (0 to 1).
    """

    name: str
    wavelength: float
    linewidth: float
    branching_ratio: float = 1.0

    @property
    def frequency(self) -> float:
        """Transition frequency in Hz."""
        return SPEED_OF_LIGHT / self.wavelength

    @property
    def wavevector(self) -> float:
        r"""Wavevector magnitude $|k| = 2\pi/\lambda$ in rad/m."""
        return TWO_PI / self.wavelength

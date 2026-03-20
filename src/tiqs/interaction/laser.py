"""Laser beam configuration for trapped-ion manipulation."""
from dataclasses import dataclass

from tiqs.constants import TWO_PI


@dataclass(frozen=True)
class LaserBeam:
    """A laser beam interacting with trapped ions.

    Parameters
    ----------
    wavelength : float
        Laser wavelength in meters.
    rabi_frequency : float
        Single-photon Rabi frequency (angular) in rad/s.
    detuning : float
        Detuning from qubit resonance in rad/s.
    phase : float
        Optical phase in radians.
    """

    wavelength: float
    rabi_frequency: float
    detuning: float = 0.0
    phase: float = 0.0

    @property
    def wavevector(self) -> float:
        return TWO_PI / self.wavelength

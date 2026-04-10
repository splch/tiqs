"""Electron species for trapped-electron quantum computing."""

from dataclasses import dataclass

from tiqs.constants import (
    BOHR_MAGNETON,
    ELECTRON_G_FACTOR,
    ELECTRON_MASS,
    HBAR,
    TWO_PI,
)


@dataclass(frozen=True)
class ElectronSpecies:
    """Electron spin-1/2 qubit in an applied magnetic field.

    Attributes
    ----------
    magnetic_field : float
        Applied magnetic field in Tesla.
    """

    magnetic_field: float

    @property
    def mass_kg(self) -> float:
        """Electron mass in kilograms."""
        return ELECTRON_MASS

    @property
    def qubit_frequency_hz(self) -> float:
        r"""Zeeman splitting frequency in Hz.

        $$
        f = \frac{g_e \, \mu_B \, B}{h}
        $$
        """
        return (
            ELECTRON_G_FACTOR * BOHR_MAGNETON * self.magnetic_field
            / (HBAR * TWO_PI)
        )

"""Proton species for Penning trap experiments."""

from dataclasses import dataclass

from tiqs.constants import (
    HBAR,
    NUCLEAR_MAGNETON,
    PROTON_G_FACTOR,
    PROTON_MASS,
    TWO_PI,
)


@dataclass(frozen=True)
class ProtonSpecies:
    """Proton spin-1/2 particle in an applied magnetic field.

    Used for Penning trap precision measurements (g-factor,
    charge-to-mass ratio) and as a cross-species benchmark
    for trapped-electron physics. The underlying Penning trap
    equations are identical to the electron case with different
    mass and g-factor.

    Attributes
    ----------
    magnetic_field : float
        Applied magnetic field in Tesla.
    """

    magnetic_field: float

    @property
    def mass_kg(self) -> float:
        """Proton mass in kilograms."""
        return PROTON_MASS

    @property
    def g_factor(self) -> float:
        """Proton spin g-factor (CODATA 2018)."""
        return PROTON_G_FACTOR

    @property
    def qubit_frequency_hz(self) -> float:
        r"""Larmor precession frequency in Hz.

        $$
        f = \frac{g_p \, \mu_N \, B}{h}
        $$
        """
        return (
            PROTON_G_FACTOR
            * NUCLEAR_MAGNETON
            * self.magnetic_field
            / (HBAR * TWO_PI)
        )

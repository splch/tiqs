"""Structural interface for trapped particle species."""

from typing import Protocol


class Species(Protocol):
    """Structural interface for any trapped particle species.

    Any class exposing ``mass_kg``, ``qubit_frequency_hz``, and
    ``g_factor`` as read-only properties satisfies this protocol.
    ``IonSpecies`` and ``ElectronSpecies`` conform without
    modification.
    """

    @property
    def mass_kg(self) -> float: ...

    @property
    def qubit_frequency_hz(self) -> float: ...

    @property
    def g_factor(self) -> float: ...

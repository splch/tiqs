"""Atomic transition data for trapped-ion species."""

from dataclasses import dataclass

from tiqs.constants import SPEED_OF_LIGHT, TWO_PI


@dataclass(frozen=True)
class Transition:
    r"""An atomic transition between two energy levels.

    Attributes
    ----------
    name : str
        Human-readable label, e.g. ``"S1/2 -> P1/2"``.
    wavelength : float
        Transition **vacuum** wavelength in meters.
    linewidth : float
        Decay rate of this line as an angular frequency in rad/s. See
        Notes: the database populates it with two different quantities
        depending on the role of the transition.
    branching_ratio : float
        Fraction of upper-state decays going through this channel
        (0 to 1). Populated only for the cooling transitions; see Notes.

    Notes
    -----
    ``linewidth`` carries two distinct meanings across
    :data:`~tiqs.species.ion._SPECIES_DB`, and consumers must know
    which one they are reading:

    * For a **cooling** transition it is the *total* natural linewidth
      of the upper state, $\Gamma = \sum_f A_{ki}^{(f)}$. This is the
      quantity the Doppler-limit formulas require, and it is what
      :meth:`~tiqs.species.ion.IonSpecies.doppler_limit_temperature`,
      :meth:`~tiqs.species.ion.IonSpecies.doppler_limit_nbar` and the
      sympathetic-cooling rates consume.
    * For a **repump** transition it is the *partial* Einstein
      coefficient of that single line, $A_{ki}/2\pi$ expressed as an
      angular frequency. It is therefore *not* the natural linewidth of
      the repumper's upper state, which is set by the (much faster)
      decay back to the ground state - e.g. the Ca-40 866 nm line has
      $A_{ki}/2\pi = 1.69$ MHz while its 4P1/2 upper state is
      $\Gamma/2\pi = 22.4$ MHz wide, a factor 13. Do not use a repump
      ``linewidth`` as a Lindblad rate or a saturation width.

    ``branching_ratio`` is the fraction of upper-state decays taking
    this line. It is measured and populated for the cooling
    transitions. For the repump entries it is left at its ``1.0``
    default, which is *not* physical (the real fractions are of order a
    few percent); treat it as "not characterised" there rather than as
    data.
    """

    name: str
    wavelength: float
    linewidth: float
    branching_ratio: float = 1.0

    @property
    def frequency(self) -> float:
        r"""Transition frequency in Hz.

        Returns
        -------
        float
            Frequency computed as $c / \lambda$.
        """
        return SPEED_OF_LIGHT / self.wavelength

    @property
    def wavevector(self) -> float:
        r"""Wavevector magnitude $|k| = 2\pi/\lambda$ in rad/m.

        Returns
        -------
        float
            Wavevector magnitude in rad/m.
        """
        return TWO_PI / self.wavelength

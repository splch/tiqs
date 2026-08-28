"""Ion species database with atomic properties for trapped-ion QC."""

import math
from dataclasses import dataclass

from tiqs.constants import (
    AMU,
    BOLTZMANN,
    ELECTRON_MASS,
    HBAR,
    SPEED_OF_LIGHT,
    TWO_PI,
)
from tiqs.species.transitions import Transition


@dataclass(frozen=True)
class IonSpecies:
    """Complete atomic data for a single trapped-ion species.

    Attributes
    ----------
    symbol : str
        Species identifier, e.g. ``"Yb171"``.
    mass_amu : float
        **Neutral-atom** relative atomic mass in atomic mass units,
        from the NIST/AME2020 atomic-mass evaluation. The ionic mass
        used by the dynamics is :attr:`mass_kg`, which subtracts one
        electron.
    nuclear_spin : float
        Nuclear spin quantum number *I*.
    qubit_type : str
        One of ``"hyperfine"``, ``"optical"``, or ``"zeeman"``.
    qubit_frequency_hz : float
        Qubit transition frequency in Hz.
    qubit_wavelength : float or None
        For optical qubits, the transition vacuum wavelength in meters.
    cooling_transition : Transition
        Primary Doppler cooling transition. Its ``linewidth`` is the
        *total* natural linewidth of the upper state.
    repump_transitions : tuple[Transition, ...]
        Repumper transitions to clear metastable dark states. Their
        ``linewidth`` is the *partial* Einstein coefficient
        $A_{ki}/2\\pi$ of that single line - see :class:`Transition`.
    qubit_t1 : float
        T1 relaxation time in seconds (``inf`` for ground-state
        hyperfine qubits).
    metastable_lifetime : float or None
        Lifetime of metastable D-state in seconds (for
        shelving/optical qubits).
    raman_wavelength : float or None
        Raman beam wavelength in meters (for hyperfine qubits).

    Notes
    -----
    Only singly charged ions are modelled: the charge is fixed at $+e$
    by every consumer, and :attr:`mass_kg` subtracts exactly one
    electron mass. There is no Zeeman or hyperfine sublevel structure,
    no dipole matrix elements and no magnetic-field response
    ($df/dB$, $d^2f/dB^2$, Lande $g_F$), so every ion in a chain
    necessarily shares one qubit frequency and magnetic dephasing can
    only be entered as a phenomenological T2.
    """

    symbol: str
    mass_amu: float
    nuclear_spin: float
    qubit_type: str
    qubit_frequency_hz: float
    qubit_wavelength: float | None
    cooling_transition: Transition
    repump_transitions: tuple[Transition, ...]
    qubit_t1: float
    metastable_lifetime: float | None = None
    raman_wavelength: float | None = None

    @property
    def mass_kg(self) -> float:
        r"""Mass of the singly charged ion in kilograms.

        $$
        m = m_\text{amu}\,u - m_e
        $$

        :attr:`mass_amu` is the *neutral* relative atomic mass, so one
        electron mass is subtracted. The ionisation-energy mass defect
        ($\mathrm{IE}/c^2 \sim 7\times10^{-9}\,u$) is neglected; it is
        four orders of magnitude below the electron term. Ignoring the
        electron entirely would overestimate the mass by 3.2e-6 (Yb-171)
        to 6.1e-5 (Be-9) relative, and any $\omega \propto m^{-1/2}$ by
        half that.
        """
        return self.mass_amu * AMU - ELECTRON_MASS

    def doppler_limit_temperature(self) -> float:
        r"""Doppler cooling limit temperature in Kelvin.

        $$
        T_D = \frac{\hbar\,\Gamma}{2\,k_B}
        $$
        """
        return HBAR * self.cooling_transition.linewidth / (2 * BOLTZMANN)

    def doppler_limit_nbar(self, trap_frequency_hz: float) -> float:
        r"""Mean phonon number at the Doppler limit.

        $$
        \bar{n}_D = \Gamma / (2\omega_\text{trap})
        $$
        """
        gamma = self.cooling_transition.linewidth
        omega_trap = TWO_PI * trap_frequency_hz
        return gamma / (2 * omega_trap)


# Atomic masses are the NIST/AME2020 *neutral* relative atomic masses
# (physics.nist.gov Atomic Weights and Isotopic Compositions);
# `IonSpecies.mass_kg` subtracts one electron. All wavelengths are
# VACUUM values; unless noted otherwise they and the Einstein A
# coefficients come from the NIST Atomic Spectra Database
# (physics.nist.gov/asd, "Wavelength in vacuum for all lines").
_SPECIES_DB: dict[str, IonSpecies] = {
    # Ytterbium-171: hyperfine qubit at 12.642812118 GHz.
    "Yb171": IonSpecies(
        symbol="Yb171",
        mass_amu=170.9363302,  # 170.9363302(22) u
        nuclear_spin=0.5,
        qubit_type="hyperfine",
        # 12 642 812 118.4690(8) Hz (fractional uncertainty 6.6e-14),
        # Appl. Phys. Lett. 125, 084002 (2024).
        qubit_frequency_hz=12.642812118e9,
        qubit_wavelength=None,
        cooling_transition=Transition(
            name="2S1/2 -> 2P1/2",
            # Trapped-ion literature vacuum value. NIST ASD gives
            # 369.524 nm, from a 2P1/2 energy known only to
            # 0.01 cm^-1; the two agree to 7e-6 relative.
            wavelength=369.5262e-9,
            linewidth=TWO_PI * 19.6e6,  # total 2P1/2 width, tau 8.12 ns
            branching_ratio=0.995,
        ),
        repump_transitions=(
            Transition(
                name="2D3/2 -> 3D[3/2]1/2",
                wavelength=935.187e-9,
                linewidth=TWO_PI * 4.2e6,
            ),
            Transition(
                name="2F7/2 clearout",
                wavelength=760.0e-9,
                # ORDER-OF-MAGNITUDE PLACEHOLDER. The 760 nm
                # 2F7/2 -> 1D[3/2]3/2 assignment is standard
                # (arXiv:1811.10451), but NIST ASD lists no A_ki for
                # any Yb II line near 760 nm and no published rate for
                # this channel was found. For scale, the sibling
                # 638.6 nm upper level has tau = 37.9(9) us, i.e.
                # 4.2 kHz (arXiv:2506.04320). Unused by any code path.
                linewidth=TWO_PI * 0.05e6,
            ),
        ),
        qubit_t1=math.inf,
        raman_wavelength=355e-9,  # tripled Nd:YVO4 laser, not a line
    ),
    # Calcium-40: optical qubit on the 729 nm 4S1/2 -> 3D5/2 line.
    "Ca40": IonSpecies(
        symbol="Ca40",
        mass_amu=39.962590863,  # 39.962590863(22) u
        nuclear_spin=0.0,
        qubit_type="optical",
        qubit_frequency_hz=SPEED_OF_LIGHT / 729.3473e-9,
        # Vacuum wavelength of the CIPM secondary representation of the
        # second at 411.0421297764 THz; NIST ASD rounds it to 729.348 nm.
        qubit_wavelength=729.3473e-9,
        cooling_transition=Transition(
            name="4S1/2 -> 4P1/2",
            wavelength=396.959e-9,
            # Total 4P1/2 width from tau = 7.098(20) ns,
            # Hettrich et al., PRL 115, 013004 (2015).
            linewidth=TWO_PI * 22.4e6,
            # 0.93565(7), Ramm et al., PRL 111, 023004 (2013).
            branching_ratio=0.93565,
        ),
        repump_transitions=(
            Transition(
                name="3D3/2 -> 4P1/2",
                wavelength=866.452e-9,
                linewidth=TWO_PI * 1.69e6,  # A_ki = 1.06e7 s^-1
            ),
            Transition(
                name="3D5/2 -> 4P3/2",
                wavelength=854.444e-9,
                linewidth=TWO_PI * 1.58e6,  # A_ki = 9.9e6 s^-1
            ),
        ),
        qubit_t1=1.168,
        metastable_lifetime=1.168,  # 3D5/2, Barton et al. (2000)
    ),
    # Calcium-43: hyperfine qubit at 3.2256 GHz.
    "Ca43": IonSpecies(
        symbol="Ca43",
        mass_amu=42.95876644,  # 42.95876644(24) u
        nuclear_spin=3.5,
        qubit_type="hyperfine",
        # 3 225 608 286.4(3) Hz zero-field 4S1/2 splitting.
        qubit_frequency_hz=3.22560829e9,
        qubit_wavelength=None,
        cooling_transition=Transition(
            name="4S1/2 -> 4P1/2",
            wavelength=396.959e-9,
            linewidth=TWO_PI * 22.4e6,
            branching_ratio=0.93565,
        ),
        repump_transitions=(
            Transition(
                name="3D3/2 -> 4P1/2",
                wavelength=866.452e-9,
                linewidth=TWO_PI * 1.69e6,  # A_ki = 1.06e7 s^-1
            ),
        ),
        qubit_t1=math.inf,
        metastable_lifetime=1.168,
        raman_wavelength=396.959e-9,
    ),
    # Barium-137: hyperfine qubit with all-visible wavelengths.
    "Ba137": IonSpecies(
        symbol="Ba137",
        mass_amu=136.90582714,  # 136.90582714(30) u
        nuclear_spin=1.5,
        qubit_type="hyperfine",
        # 8 037 741 667.7 Hz, Blatt and Werth, "Precision
        # determination of the ground-state hyperfine splitting in
        # 137Ba+ using the ion-storage technique", PRA 25, 1476 (1982).
        qubit_frequency_hz=8.037741668e9,
        qubit_wavelength=None,
        cooling_transition=Transition(
            name="6S1/2 -> 6P1/2",
            wavelength=493.5454e-9,
            linewidth=TWO_PI * 20.3e6,  # total 6P1/2 width, tau 7.9 ns
            # 1 - p(6P1/2 -> 5D3/2) with p = 0.268177(37)(20),
            # Arnold et al., PRA 100, 032503 (2019). The older 0.75
            # reflects only the NIST A_ki ratio 9.53e7/(9.53e7+3.10e7)
            # and the folkloric 3:1 rule of thumb.
            branching_ratio=0.7318,
        ),
        repump_transitions=(
            Transition(
                name="5D3/2 -> 6P1/2",
                wavelength=649.8693e-9,
                # 3.33e7 s^-1. NIST gives A_ki = 3.10e7 s^-1, but the
                # branching-derived partial rate 0.268177/7.9 ns
                # = 3.4e7 s^-1 supports the larger value.
                linewidth=TWO_PI * 5.3e6,
            ),
            # D5/2 clear-out; a real 137Ba+ system needs this laser
            # alongside 493/650 nm (e.g. arXiv:2511.05465).
            Transition(
                name="5D5/2 -> 6P3/2",
                wavelength=614.3413e-9,
                linewidth=TWO_PI * 6.56e6,  # A_ki = 4.12e7 s^-1
            ),
        ),
        qubit_t1=math.inf,
        # 5D5/2: 31.2(9) s, Auchter et al., PRA 90, 060501(R) (2014).
        # Mohanty et al., Hyperfine Interact. (2015),
        # doi:10.1007/s10751-015-1161-9 (arXiv:1504.03023) report
        # 26.4(1.7) s; theory gives 29.8(3) s. The literature spans
        # 25.6-31.2 s, so treat this as good to ~20%.
        metastable_lifetime=31.2,
        raman_wavelength=515e-9,  # doubled fiber laser, not a line
    ),
    # Beryllium-9: lightest ion qubit, hyperfine.
    "Be9": IonSpecies(
        symbol="Be9",
        mass_amu=9.012183065,  # 9.012183065(82) u
        nuclear_spin=1.5,
        qubit_type="hyperfine",
        # 2|A| of the 2S1/2 ground state. Cross-checked against
        # A = -625.008840(35) MHz -> 1.25001768(7) GHz
        # (arXiv:2601.14811), which agrees to 6 Hz.
        qubit_frequency_hz=1.250017674e9,
        qubit_wavelength=None,
        cooling_transition=Transition(
            name="2S1/2 -> 2P3/2",
            wavelength=313.133e-9,  # NIST ASD 313.13292 nm
            # NIST ASD A_ki = 1.1292e8 s^-1, and 2P3/2 has no other
            # decay channel, so Gamma/2pi = 17.97 MHz (tau 8.86 ns).
            # The trapped-ion literature propagates 19.4 MHz (tau
            # 8.20 ns), ~8% higher and unsupported by the transition
            # probability; it inflated the Doppler limit (466 uK vs
            # 431 uK, nbar 9.70 vs 8.99 at 1 MHz) and the
            # sympathetic-cooling rates that consume Gamma.
            linewidth=TWO_PI * 17.97e6,
            branching_ratio=1.0,
        ),
        repump_transitions=(),
        qubit_t1=math.inf,
        raman_wavelength=313.133e-9,
    ),
    # Strontium-88: optical qubit on the 674 nm 5S1/2 -> 4D5/2 line.
    "Sr88": IonSpecies(
        symbol="Sr88",
        mass_amu=87.9056125,  # 87.9056125(12) u
        nuclear_spin=0.0,
        qubit_type="optical",
        qubit_frequency_hz=SPEED_OF_LIGHT / 674.0256e-9,
        # Vacuum wavelength of the CIPM secondary representation of the
        # second at 444.7790440955 THz; NIST ASD gives 674.0252 nm.
        qubit_wavelength=674.0256e-9,
        cooling_transition=Transition(
            name="5S1/2 -> 5P1/2",
            wavelength=421.6711e-9,
            # Total 5P1/2 width: (1.279e8 + 7.46e6)/2pi = 21.5 MHz.
            linewidth=TWO_PI * 21.5e6,
            # 0.9453(+7/-5), Likforman et al., PRA 93, 052507 (2016).
            branching_ratio=0.9453,
        ),
        repump_transitions=(
            Transition(
                name="4D3/2 -> 5P1/2",
                wavelength=1091.786e-9,
                # A_ki = 7.46e6 s^-1. The previous TWO_PI * 1.4e6
                # (8.80e6 s^-1) was 18% high and matched no source.
                linewidth=TWO_PI * 1.19e6,
            ),
            # D5/2 clear-out: required to reset the 674 nm optical
            # qubit that this entry defines.
            Transition(
                name="4D5/2 -> 5P3/2",
                wavelength=1033.014e-9,
                linewidth=TWO_PI * 1.38e6,  # A_ki = 8.7e6 s^-1
            ),
        ),
        qubit_t1=0.390,
        metastable_lifetime=0.390,  # 4D5/2; NIST A_ki = 2.559 s^-1
    ),
}


def get_species(name: str) -> IonSpecies:
    """Look up an ion species by name.

    Parameters
    ----------
    name : str
        Species identifier. Available names: ``"Yb171"``, ``"Ca40"``,
        ``"Ca43"``, ``"Ba137"``, ``"Be9"``, ``"Sr88"``.

    Returns
    -------
    IonSpecies
        Atomic data for the requested species.

    Raises
    ------
    KeyError
        If *name* is not in the species database.
    """
    return _SPECIES_DB[name]

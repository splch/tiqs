r"""Magnetic field gradient configuration and gradient-mediated coupling.

A static (or oscillating) magnetic field gradient couples the
qubit spin to the shared motional bus, replacing the photon-recoil
momentum of laser-driven gates.  The gradient provides both
spin-motion coupling and individual ion addressing through
position-dependent qubit frequencies.
"""

from dataclasses import dataclass

import numpy as np

from tiqs.chain.lamb_dicke import lamb_dicke_parameters
from tiqs.chain.normal_modes import NormalModeResult
from tiqs.constants import TWO_PI
from tiqs.species.electron import ElectronSpecies
from tiqs.species.ion import IonSpecies


@dataclass(frozen=True)
class MagneticGradient:
    r"""Static magnetic field gradient configuration.

    Attributes
    ----------
    db_dz : float
        Axial gradient $dB/dz$ in Tesla/meter.
    b_field : float
        Background magnetic field magnitude in Tesla.
    """

    db_dz: float
    b_field: float

    def effective_k(self, species: IonSpecies | ElectronSpecies) -> float:
        r"""Effective wavevector for gradient-mediated spin-motion coupling.

        For an ion with differential Zeeman sensitivity
        $\partial\omega_q / \partial B$:

        $$
        k_\text{eff} = \frac{1}{\omega_q}
          \frac{\partial\omega_q}{\partial B}\,\frac{dB}{dz}
        $$

        For an electron this simplifies to $(dB/dz)/B$.

        Parameters
        ----------
        species : IonSpecies or ElectronSpecies

        Returns
        -------
        float
            Effective wavevector in rad/m.
        """
        if isinstance(species, ElectronSpecies):
            return self.db_dz / self.b_field
        if species.qubit_zeeman_sensitivity is None:
            raise ValueError(
                f"Species {species.symbol} has no "
                f"qubit_zeeman_sensitivity; cannot compute "
                f"gradient coupling."
            )
        return (
            species.qubit_zeeman_sensitivity
            * self.db_dz
            / (TWO_PI * species.qubit_frequency_hz)
        )


def gradient_lamb_dicke(
    modes: NormalModeResult,
    species: IonSpecies | ElectronSpecies,
    gradient: MagneticGradient,
    direction: str = "axial",
) -> np.ndarray:
    r"""Lamb-Dicke parameters from gradient coupling.

    Equivalent to calling :func:`~tiqs.chain.lamb_dicke.lamb_dicke_parameters`
    with ``k_eff = gradient.effective_k(species)``.

    Parameters
    ----------
    modes : NormalModeResult
    species : IonSpecies or ElectronSpecies
    gradient : MagneticGradient
    direction : str

    Returns
    -------
    np.ndarray
        Lamb-Dicke matrix, shape ``(N_ions, N_modes)``.
    """
    k_eff = gradient.effective_k(species)
    return lamb_dicke_parameters(modes, species, k_eff, direction)


def gradient_qubit_frequencies(
    species: IonSpecies,
    gradient: MagneticGradient,
    positions: np.ndarray,
) -> np.ndarray:
    r"""Per-ion qubit frequencies in the gradient field.

    $$
    \omega_q^{(j)} = \omega_0
      + \frac{\partial\omega_q}{\partial B}\,\frac{dB}{dz}\,z_j
    $$

    Parameters
    ----------
    species : IonSpecies
    gradient : MagneticGradient
    positions : np.ndarray
        Equilibrium positions in meters, shape ``(N,)``.

    Returns
    -------
    np.ndarray
        Qubit frequencies in Hz, shape ``(N,)``.
    """
    if species.qubit_zeeman_sensitivity is None:
        raise ValueError(
            f"Species {species.symbol} has no qubit_zeeman_sensitivity."
        )
    shift_rad_per_m = species.qubit_zeeman_sensitivity * gradient.db_dz
    return species.qubit_frequency_hz + shift_rad_per_m * positions / TWO_PI


def gradient_addressing_crosstalk(
    species: IonSpecies,
    gradient: MagneticGradient,
    positions: np.ndarray,
    microwave_rabi: float,
) -> np.ndarray:
    r"""Off-resonant crosstalk matrix from gradient addressing.

    When targeting ion *j*, the off-resonant excitation of ion *k*
    has effective Rabi frequency
    $\Omega_\text{eff} \approx \Omega^2 / (4\,\Delta_{jk}^2) \cdot \Omega$
    for large detuning.  The matrix element ``[j, k]`` is the ratio
    $\Omega_\text{eff} / \Omega$.

    Parameters
    ----------
    species : IonSpecies
    gradient : MagneticGradient
    positions : np.ndarray
        Equilibrium positions in meters, shape ``(N,)``.
    microwave_rabi : float
        Microwave Rabi frequency in rad/s.

    Returns
    -------
    np.ndarray
        Crosstalk matrix, shape ``(N, N)``.  Diagonal is 1.0.
    """
    freqs = gradient_qubit_frequencies(species, gradient, positions)
    n = len(positions)
    xt = np.eye(n)
    for j in range(n):
        for k in range(n):
            if j == k:
                continue
            delta_jk = TWO_PI * (freqs[j] - freqs[k])
            if abs(delta_jk) > 0:
                xt[j, k] = microwave_rabi**2 / (4 * delta_jk**2)
    return xt

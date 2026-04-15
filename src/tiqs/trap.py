r"""Trapped-particle confinement: Paul traps, Penning traps, and the
shared ``Trap`` protocol.

.. include:: ../../docs/theory/trapping.md
"""

from dataclasses import dataclass

import numpy as np

from tiqs.constants import ELECTRON_CHARGE
from tiqs.species.electron import ElectronSpecies
from tiqs.species.ion import IonSpecies


@dataclass
class PaulTrap:
    """Linear Paul trap with RF radial confinement and DC axial confinement.

    Construct directly with ``omega_axial``, or use
    ``PaulTrap.from_dc_voltage()`` if the DC voltage is known instead.

    Attributes
    ----------
    v_rf : float
        Peak RF voltage amplitude in volts.
    omega_rf : float
        RF drive angular frequency in rad/s.
    r0 : float
        Characteristic particle-to-electrode distance in meters.
    species : IonSpecies or ElectronSpecies
        The trapped particle species.
    omega_axial : float
        Axial secular angular frequency in rad/s.
    z0 : float
        Half-length of the trap for axial confinement in meters.
    kappa : float
        Geometric factor for axial potential (typical: 0.4 for linear traps).
    """

    v_rf: float
    omega_rf: float
    r0: float
    species: IonSpecies | ElectronSpecies
    omega_axial: float
    z0: float = 2.5e-3
    kappa: float = 0.4

    @classmethod
    def from_dc_voltage(
        cls,
        v_rf: float,
        omega_rf: float,
        r0: float,
        species: IonSpecies | ElectronSpecies,
        u_dc_axial: float,
        z0: float = 2.5e-3,
        kappa: float = 0.4,
    ) -> "PaulTrap":
        r"""Construct from DC axial voltage instead of axial frequency.

        $$
        \omega_z = \sqrt{\frac{\kappa\,e\,U_\mathrm{dc}}{m\,z_0^2}}
        $$
        """
        m = species.mass_kg
        omega_axial = np.sqrt(
            kappa * ELECTRON_CHARGE * u_dc_axial / (m * z0**2)
        )
        return cls(v_rf, omega_rf, r0, species, omega_axial, z0, kappa)

    @property
    def u_dc_axial(self) -> float:
        r"""DC axial endcap voltage in volts, derived from omega_axial.

        $$
        U_\mathrm{dc} = \frac{m\,\omega_z^2\,z_0^2}{\kappa\,e}
        $$
        """
        m = self.species.mass_kg
        return (
            m * self.omega_axial**2 * self.z0**2
            / (self.kappa * ELECTRON_CHARGE)
        )

    @property
    def mathieu_q(self) -> float:
        r"""Dimensionless Mathieu q parameter.

        $$
        q = \frac{2 e V_\mathrm{rf}}{m \Omega_\mathrm{rf}^2 r_0^2}
        $$
        """
        m = self.species.mass_kg
        return (
            2
            * ELECTRON_CHARGE
            * self.v_rf
            / (m * self.omega_rf**2 * self.r0**2)
        )

    @property
    def mathieu_a(self) -> float:
        r"""Dimensionless Mathieu a parameter from DC axial confinement.

        $$
        a \approx \frac{-2 \omega_\mathrm{axial}^2}{\Omega_\mathrm{rf}^2}
        $$
        """
        return -2 * self.omega_axial**2 / self.omega_rf**2

    def is_stable(self) -> bool:
        r"""Check if $(a, q)$ falls within the first Mathieu stability region."""
        q = self.mathieu_q
        a = self.mathieu_a
        if q <= 0 or q >= 0.908:
            return False
        beta_sq = a + q**2 / 2
        return beta_sq > 0

    @property
    def omega_radial(self) -> float:
        r"""Radial secular angular frequency in the pseudopotential
        approximation.

        $$
        \omega_r = \frac{\Omega_\mathrm{rf}}{2} \sqrt{a + \frac{q^2}{2}}
        $$
        """
        q = self.mathieu_q
        a = self.mathieu_a
        beta_sq = a + q**2 / 2
        if beta_sq <= 0:
            raise ValueError("Trap is unstable: beta^2 <= 0")
        return (self.omega_rf / 2) * np.sqrt(beta_sq)

    @property
    def pseudopotential_depth_eV(self) -> float:
        r"""Pseudopotential well depth in electron-volts.

        $$
        \Psi_0 = \frac{e^2 V_\mathrm{rf}^2}{4 m \Omega_\mathrm{rf}^2 r_0^2}
        $$
        """
        m = self.species.mass_kg
        depth_J = (ELECTRON_CHARGE**2 * self.v_rf**2) / (
            4 * m * self.omega_rf**2 * self.r0**2
        )
        return depth_J / ELECTRON_CHARGE

    def micromotion_amplitude(self, displacement_from_null: float) -> float:
        r"""Peak micromotion amplitude for a particle displaced from
        the RF null.

        $x_\mathrm{mm} = (q/2) \cdot x_\mathrm{displacement}$
        """
        return (self.mathieu_q / 2) * abs(displacement_from_null)

    def stray_field_displacement(self, stray_E_field: float) -> float:
        r"""Static displacement from RF null due to a stray DC field.

        $$
        x_\mathrm{displacement} = \frac{e E}{m \omega_r^2}
        $$
        """
        m = self.species.mass_kg
        return (
            ELECTRON_CHARGE * abs(stray_E_field) / (m * self.omega_radial**2)
        )

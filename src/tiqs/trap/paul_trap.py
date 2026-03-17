"""Radiofrequency Paul trap physics: Mathieu stability, secular frequencies, pseudopotential."""
from dataclasses import dataclass
from typing import Optional

import numpy as np

from tiqs.constants import ELECTRON_CHARGE
from tiqs.species.data import IonSpecies


@dataclass
class PaulTrap:
    """Linear Paul trap with RF radial confinement and DC axial confinement.

    Provide either u_dc_axial (DC endcap voltage) or omega_axial (desired axial
    secular frequency). If omega_axial is given, u_dc_axial is back-calculated.

    Parameters
    ----------
    v_rf : float
        Peak RF voltage amplitude in volts.
    omega_rf : float
        RF drive angular frequency in rad/s.
    r0 : float
        Characteristic ion-to-electrode distance in meters.
    species : IonSpecies
        The trapped ion species.
    omega_axial : float or None
        Axial secular angular frequency in rad/s. Specify this OR u_dc_axial.
    u_dc_axial : float or None
        DC axial endcap voltage in volts. Specify this OR omega_axial.
    z0 : float
        Half-length of the trap for axial confinement in meters (default 2.5 mm).
    kappa : float
        Geometric factor for axial potential (default 0.4, typical for linear traps).
    """

    v_rf: float
    omega_rf: float
    r0: float
    species: IonSpecies
    omega_axial: Optional[float] = None
    u_dc_axial: Optional[float] = None
    z0: float = 2.5e-3
    kappa: float = 0.4

    def __post_init__(self):
        m = self.species.mass_kg
        e = ELECTRON_CHARGE
        if self.omega_axial is not None and self.u_dc_axial is None:
            self.u_dc_axial = m * self.omega_axial**2 * self.z0**2 / (self.kappa * e)
        elif self.u_dc_axial is not None and self.omega_axial is None:
            self.omega_axial = np.sqrt(self.kappa * e * self.u_dc_axial / (m * self.z0**2))
        elif self.omega_axial is None and self.u_dc_axial is None:
            raise ValueError("Must specify either omega_axial or u_dc_axial")

    @property
    def mathieu_q(self) -> float:
        """Dimensionless Mathieu q parameter: q = 2*e*V_rf / (m * Omega_rf^2 * r0^2)."""
        m = self.species.mass_kg
        return 2 * ELECTRON_CHARGE * self.v_rf / (m * self.omega_rf**2 * self.r0**2)

    @property
    def mathieu_a(self) -> float:
        """Dimensionless Mathieu a parameter from DC axial confinement.

        The axial DC field also modifies the radial potential. For a linear trap:
        a = -4 * e * kappa * U_dc / (m * Omega_rf^2 * r0^2) * (r0/z0)^2
        In practice we use a simplified model: a ~ -2 * omega_axial^2 / omega_rf^2
        """
        return -2 * self.omega_axial**2 / self.omega_rf**2

    def is_stable(self) -> bool:
        """Check if (a, q) falls within the first Mathieu stability region.

        Approximate boundary: q < 0.908 and |a| < q^2/2 for the lowest region.
        More precisely, we check the secular frequency remains real and positive.
        """
        q = self.mathieu_q
        a = self.mathieu_a
        if q <= 0 or q >= 0.908:
            return False
        beta_sq = a + q**2 / 2
        return beta_sq > 0

    @property
    def omega_radial(self) -> float:
        """Radial secular angular frequency in the pseudopotential approximation.

        omega_r = (Omega_rf / 2) * sqrt(a + q^2/2)
        For |a| << q: omega_r ~ q * Omega_rf / (2*sqrt(2))
        """
        q = self.mathieu_q
        a = self.mathieu_a
        beta_sq = a + q**2 / 2
        if beta_sq <= 0:
            raise ValueError("Trap is unstable: beta^2 <= 0")
        return (self.omega_rf / 2) * np.sqrt(beta_sq)

    @property
    def pseudopotential_depth_eV(self) -> float:
        """Pseudopotential well depth in electron-volts.

        Psi_0 = e^2 * V_rf^2 / (4 * m * Omega_rf^2 * r0^2) converted to eV.
        """
        m = self.species.mass_kg
        depth_J = (ELECTRON_CHARGE**2 * self.v_rf**2) / (4 * m * self.omega_rf**2 * self.r0**2)
        return depth_J / ELECTRON_CHARGE

    def micromotion_amplitude(self, displacement_from_null: float) -> float:
        """Peak micromotion amplitude for an ion displaced from the RF null.

        x_mm = (q/2) * displacement, valid for q << 1.
        """
        return (self.mathieu_q / 2) * abs(displacement_from_null)

    def stray_field_displacement(self, stray_E_field: float) -> float:
        """Static displacement of ion from RF null due to a stray DC electric field.

        displacement = e * E / (m * omega_r^2)
        """
        m = self.species.mass_kg
        return ELECTRON_CHARGE * abs(stray_E_field) / (m * self.omega_radial**2)

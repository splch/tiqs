r"""Trapped-particle confinement: Paul traps, Penning traps, and the
shared ``Trap`` protocol.

.. include:: ../../docs/theory/trapping.md
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from tiqs.constants import (
    BOHR_MAGNETON,
    ELECTRON_CHARGE,
    ELECTRON_G_FACTOR,
    HBAR,
)
from tiqs.species.electron import ElectronSpecies
from tiqs.species.ion import IonSpecies
from tiqs.species.protocol import Species


class Trap(Protocol):
    """Structural interface for any charged-particle trap.

    Any class exposing ``omega_axial``, ``species``, and ``is_stable()``
    satisfies this protocol. ``PaulTrap`` and ``PenningTrap`` conform
    without modification.
    """

    @property
    def omega_axial(self) -> float: ...

    @property
    def species(self) -> Species: ...

    def is_stable(self) -> bool: ...


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
    ) -> PaulTrap:
        r"""Construct from DC axial voltage instead of axial frequency.

        $$
        \omega_z = \sqrt{\frac{\kappa\,e\,U_\mathrm{dc}}{m\,z_0^2}}
        $$
        """
        if u_dc_axial < 0:
            raise ValueError(
                f"u_dc_axial must be non-negative, got {u_dc_axial}"
            )
        m = species.mass_kg
        omega_axial = np.sqrt(
            kappa * ELECTRON_CHARGE * u_dc_axial / (m * z0**2)
        )
        return cls(
            v_rf=v_rf,
            omega_rf=omega_rf,
            r0=r0,
            species=species,
            omega_axial=omega_axial,
            z0=z0,
            kappa=kappa,
        )

    @property
    def u_dc_axial(self) -> float:
        r"""DC axial endcap voltage in volts, derived from omega_axial.

        $$
        U_\mathrm{dc} = \frac{m\,\omega_z^2\,z_0^2}{\kappa\,e}
        $$
        """
        m = self.species.mass_kg
        return (
            m
            * self.omega_axial**2
            * self.z0**2
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

        The axial DC field also modifies the radial potential. For a
        linear trap:

        $$
        a = \frac{-4 e \kappa U_\mathrm{dc}}{m \Omega_\mathrm{rf}^2 r_0^2}
        \left(\frac{r_0}{z_0}\right)^2
        $$

        In practice we use a simplified model:

        $$
        a \approx \frac{-2 \omega_\mathrm{axial}^2}{\Omega_\mathrm{rf}^2}
        $$
        """
        return -2 * self.omega_axial**2 / self.omega_rf**2

    def is_stable(self) -> bool:
        r"""Check if $(a, q)$ falls within the first Mathieu stability region.

        Approximate boundary: $q < 0.908$ and $|a| < q^2/2$ for the
        lowest region. More precisely, we check the secular frequency
        remains real and positive.
        """
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

        For $|a| \ll q$:

        $$
        \omega_r \approx \frac{q \, \Omega_\mathrm{rf}}{2\sqrt{2}}
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

        converted to eV.
        """
        m = self.species.mass_kg
        depth_J = (ELECTRON_CHARGE**2 * self.v_rf**2) / (
            4 * m * self.omega_rf**2 * self.r0**2
        )
        return depth_J / ELECTRON_CHARGE

    def micromotion_amplitude(self, displacement_from_null: float) -> float:
        r"""Peak micromotion amplitude for a particle displaced from
        the RF null.

        $x_\mathrm{mm} = (q/2) \cdot x_\mathrm{displacement}$,
        valid for $q \ll 1$.
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


@dataclass
class PenningTrap:
    r"""Static-field Penning trap with magnetic radial and electric axial
    confinement.

    Construct directly with ``omega_axial``, or use
    ``PenningTrap.from_dc_voltage()`` if the DC voltage is known instead.

    Attributes
    ----------
    magnetic_field : float
        Axial magnetic field strength in Tesla.
    species : IonSpecies or ElectronSpecies
        The trapped particle species.
    d : float
        Characteristic trap dimension in meters.
        For a hyperbolic trap, $d^2 = (z_0^2 + r_0^2/2) / 2$.
    omega_axial : float
        Axial angular frequency in rad/s.
    b1 : float
        Linear magnetic field gradient in T/m. Shifts the
        equilibrium position by an amount proportional to the
        particle's magnetic moment. Does not produce first-order
        frequency shifts but is an important error source.
    b2 : float
        Quadratic magnetic field gradient (magnetic bottle) in
        T/m^2. Couples the spin and cyclotron quantum numbers
        to all three mode frequencies via the continuous
        Stern-Gerlach effect.
    """

    magnetic_field: float
    species: IonSpecies | ElectronSpecies
    d: float
    omega_axial: float
    b1: float = 0.0
    b2: float = 0.0

    def __post_init__(self):
        if (
            isinstance(self.species, ElectronSpecies)
            and self.species.magnetic_field != self.magnetic_field
        ):
            raise ValueError(
                f"PenningTrap.magnetic_field ({self.magnetic_field}) "
                f"must match species.magnetic_field "
                f"({self.species.magnetic_field})"
            )

    @classmethod
    def from_dc_voltage(
        cls,
        magnetic_field: float,
        species: IonSpecies | ElectronSpecies,
        d: float,
        v_dc: float,
        b1: float = 0.0,
        b2: float = 0.0,
    ) -> PenningTrap:
        r"""Construct from DC trapping voltage instead of axial frequency.

        $$
        \omega_z = \sqrt{\frac{e\,V_\mathrm{dc}}{m\,d^2}}
        $$
        """
        if v_dc < 0:
            raise ValueError(f"v_dc must be non-negative, got {v_dc}")
        omega_axial = np.sqrt(
            ELECTRON_CHARGE * v_dc / (species.mass_kg * d**2)
        )
        return cls(
            magnetic_field=magnetic_field,
            species=species,
            d=d,
            omega_axial=omega_axial,
            b1=b1,
            b2=b2,
        )

    @property
    def v_dc(self) -> float:
        r"""DC trapping voltage in volts, derived from omega_axial.

        $$
        V_\mathrm{dc} = \frac{m\,\omega_z^2\,d^2}{e}
        $$
        """
        return (
            self.species.mass_kg
            * self.omega_axial**2
            * self.d**2
            / ELECTRON_CHARGE
        )

    @property
    def omega_cyclotron(self) -> float:
        r"""Free cyclotron angular frequency.

        $$
        \omega_c = \frac{eB}{m}
        $$
        """
        return ELECTRON_CHARGE * self.magnetic_field / self.species.mass_kg

    def _transverse_discriminant(self) -> float:
        """Shared discriminant for modified cyclotron and magnetron."""
        wc2 = self.omega_cyclotron / 2
        discriminant = wc2**2 - self.omega_axial**2 / 2
        if discriminant < 0:
            raise ValueError(
                "Trap is unstable: omega_c < sqrt(2)*omega_z. "
                "Check is_stable() before accessing transverse frequencies."
            )
        return discriminant

    @property
    def omega_modified_cyclotron(self) -> float:
        r"""Modified cyclotron angular frequency.

        $$
        \omega_+ = \frac{\omega_c}{2}
        + \sqrt{\left(\frac{\omega_c}{2}\right)^2
        - \frac{\omega_z^2}{2}}
        $$
        """
        wc2 = self.omega_cyclotron / 2
        return wc2 + np.sqrt(self._transverse_discriminant())

    @property
    def omega_magnetron(self) -> float:
        r"""Magnetron angular frequency.

        $$
        \omega_- = \frac{\omega_c}{2}
        - \sqrt{\left(\frac{\omega_c}{2}\right)^2
        - \frac{\omega_z^2}{2}}
        $$
        """
        wc2 = self.omega_cyclotron / 2
        return wc2 - np.sqrt(self._transverse_discriminant())

    def is_stable(self) -> bool:
        r"""Check Penning stability: $\omega_c > \sqrt{2}\,\omega_z$.

        When this condition is violated the discriminant
        $(\omega_c/2)^2 - \omega_z^2/2$ becomes negative and the
        modified cyclotron and magnetron frequencies are no longer real,
        meaning radial confinement is lost.
        """
        return self.omega_cyclotron > np.sqrt(2) * self.omega_axial

    @property
    def bottle_shift(self) -> float:
        r"""Axial frequency shift parameter from the magnetic bottle.

        $$
        \delta = \frac{\hbar\,e\,B_2}{m^2\,\omega_z}
        $$

        The axial frequency depends on the cyclotron quantum number
        $n_c$ and spin quantum number $m_s$ as:

        $$
        \omega_z(n_c, m_s) = \omega_{z,0}
          + \delta\bigl(n_c + \tfrac{1}{2}
          + \tfrac{g}{2}\,m_s\bigr)
        $$

        This is the continuous Stern-Gerlach effect used in g-2
        experiments to detect spin and cyclotron quantum jumps via
        tiny shifts in the axial frequency.

        Returns zero when ``b2 = 0`` (no magnetic bottle).

        References
        ----------
        Brown, L.S. & Gabrielse, G. Rev. Mod. Phys. 58, 233 (1986).
        Van Dyck, R.S. Jr. et al. PRL 38, 310 (1977).
        """
        if self.b2 == 0:
            return 0.0
        m = self.species.mass_kg
        return HBAR * ELECTRON_CHARGE * self.b2 / (m**2 * self.omega_axial)

    def axial_frequency_shift(
        self,
        n_cyclotron: int = 0,
        m_spin: float = -0.5,
        n_magnetron: int = 0,
    ) -> float:
        r"""Axial frequency shift from the magnetic bottle.

        The B2 term shifts the axial frequency depending on the
        cyclotron, magnetron, and spin quantum numbers:

        $$
        \Delta\omega_z = \delta\bigl(
          n_+ + \tfrac{1}{2}
          + \tfrac{\omega_-}{\omega_+}(n_- + \tfrac{1}{2})
          + \tfrac{g}{2}\,m_s\bigr)
        $$

        where $\delta = \hbar e B_2 / (m^2 \omega_z)$.

        Parameters
        ----------
        n_cyclotron : int
            Modified cyclotron quantum number (0 = ground state).
        m_spin : float
            Spin quantum number ($+1/2$ or $-1/2$).
        n_magnetron : int
            Magnetron quantum number (0 = ground state).

        Returns
        -------
        float
            Axial frequency shift in rad/s.
        """
        if self.b2 == 0:
            return 0.0
        omega_p = self.omega_modified_cyclotron
        omega_m = self.omega_magnetron
        return self.bottle_shift * (
            n_cyclotron
            + 0.5
            + (omega_m / omega_p) * (n_magnetron + 0.5)
            + (ELECTRON_G_FACTOR / 2) * m_spin
        )

    def cyclotron_frequency_shift(
        self,
        n_axial: int = 0,
        m_spin: float = -0.5,
    ) -> float:
        r"""Modified cyclotron frequency shift from the magnetic
        bottle.

        The B2 term shifts the modified cyclotron frequency
        depending on the axial quantum number and spin:

        $$
        \Delta\omega_+ = \frac{B_2}{2 m \omega_- B_0}
          \bigl[\hbar\omega_+(2 n_z + 1)\bigr]
          + \frac{\hbar e\,g\,B_2}{4 m^2 \omega_+}\,m_s
        $$

        In the Penning trap frequency hierarchy
        $\omega_- \ll \omega_z \ll \omega_+$, this simplifies to
        a shift proportional to the axial quantum number.

        Parameters
        ----------
        n_axial : int
            Axial quantum number.
        m_spin : float
            Spin quantum number ($+1/2$ or $-1/2$).

        Returns
        -------
        float
            Modified cyclotron frequency shift in rad/s.
        """
        if self.b2 == 0:
            return 0.0
        m = self.species.mass_kg
        omega_p = self.omega_modified_cyclotron
        omega_m = self.omega_magnetron
        B0 = self.magnetic_field
        # Axial contribution
        axial_term = (
            self.b2
            * HBAR
            * omega_p
            * (2 * n_axial + 1)
            / (2 * m * omega_m * B0)
        )
        # Spin contribution
        spin_term = (
            HBAR
            * ELECTRON_CHARGE
            * ELECTRON_G_FACTOR
            * self.b2
            * m_spin
            / (4 * m**2 * omega_p)
        )
        return axial_term + spin_term

    def magnetron_frequency_shift(
        self,
        n_axial: int = 0,
        m_spin: float = -0.5,
    ) -> float:
        r"""Magnetron frequency shift from the magnetic bottle.

        The B2 term shifts the magnetron frequency depending on
        the axial quantum number:

        $$
        \Delta\omega_- = -\frac{B_2}{2 m \omega_- B_0}
          \bigl[\hbar\omega_-(2 n_z + 1)\bigr]
          - \frac{\hbar e\,g\,B_2}{4 m^2 \omega_-}\,m_s
        $$

        Note the opposite sign compared to the cyclotron shift:
        the magnetron mode softens where the cyclotron stiffens.

        Parameters
        ----------
        n_axial : int
            Axial quantum number.
        m_spin : float
            Spin quantum number ($+1/2$ or $-1/2$).

        Returns
        -------
        float
            Magnetron frequency shift in rad/s.
        """
        if self.b2 == 0:
            return 0.0
        m = self.species.mass_kg
        omega_m = self.omega_magnetron
        B0 = self.magnetic_field
        # Axial contribution (opposite sign from cyclotron)
        axial_term = -self.b2 * HBAR * (2 * n_axial + 1) / (2 * m * B0)
        # Spin contribution
        spin_term = (
            -HBAR
            * ELECTRON_CHARGE
            * ELECTRON_G_FACTOR
            * self.b2
            * m_spin
            / (4 * m**2 * omega_m)
        )
        return axial_term + spin_term

    @property
    def b1_equilibrium_shift(self) -> float:
        r"""Axial equilibrium position shift from B1 gradient.

        A linear magnetic field gradient $B_1$ exerts a force
        on the particle's magnetic moment, displacing the
        equilibrium position by:

        $$
        \Delta z_0 = \frac{g\,\mu_B\,B_1}{2\,m\,\omega_z^2}
        $$

        for an electron with spin along the field. This shift
        is important as an error source because it moves the
        particle away from the electric field minimum, coupling
        to electric anharmonicities.

        Returns
        -------
        float
            Equilibrium displacement in meters.
        """
        if self.b1 == 0:
            return 0.0
        m = self.species.mass_kg
        return (
            ELECTRON_G_FACTOR
            * BOHR_MAGNETON
            * self.b1
            / (2 * m * self.omega_axial**2)
        )

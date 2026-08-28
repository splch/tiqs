r"""Trapped-particle confinement: Paul traps, Penning traps, and the
shared ``Trap`` protocol.

.. include:: ../../docs/theory/trapping.md
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from tiqs.constants import ELECTRON_CHARGE
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
    r"""Linear Paul trap with RF radial confinement and DC axial
    confinement.

    Construct directly with ``omega_axial``, or use
    ``PaulTrap.from_dc_voltage()`` if the DC voltage is known instead.

    Attributes
    ----------
    v_rf : float
        Peak RF voltage amplitude in volts (non-negative).
    omega_rf : float
        RF drive angular frequency in rad/s.
    r0 : float
        *Effective* particle-to-electrode distance in meters. Every
        formula below assumes an ideal hyperbolic quadrupole whose
        electrodes lie on equipotentials of
        $\Phi \propto (x^2 - y^2)/r_0^2$. Real electrodes do not, so
        the field must be scaled by a geometric efficiency of order
        unity (Wineland et al., J. Res. NIST 103, 259 (1998),
        parenthetical after Eq. 1; Leibfried et al., Rev. Mod. Phys.
        75, 281 (2003), Eq. 7 carries it explicitly as
        $\alpha, \alpha'$). Absorb that factor into ``r0``: for the
        PCB slot trap of tests/test_electron.py the ideal-geometry
        value underestimates the measured radial frequency by 18%.
    species : IonSpecies or ElectronSpecies
        The trapped particle species.
    omega_axial : float
        Axial secular angular frequency in rad/s.
    z0 : float
        Half-length of the trap for axial confinement in meters.
    kappa : float
        Dimensionless geometric factor for the axial DC potential
        (typical: 0.4 for linear traps), defined by
        $\Phi_\mathrm{dc} = (\kappa U_\mathrm{dc}/z_0^2)
        [z^2 - (x^2+y^2)/2]$.
    """

    v_rf: float
    omega_rf: float
    r0: float
    species: IonSpecies | ElectronSpecies
    omega_axial: float
    z0: float = 2.5e-3
    kappa: float = 0.4

    def __post_init__(self):
        if self.omega_rf <= 0:
            raise ValueError(f"omega_rf must be positive, got {self.omega_rf}")
        if self.r0 <= 0:
            raise ValueError(f"r0 must be positive, got {self.r0}")
        if self.z0 <= 0:
            raise ValueError(f"z0 must be positive, got {self.z0}")
        if self.kappa <= 0:
            raise ValueError(f"kappa must be positive, got {self.kappa}")
        if self.v_rf < 0:
            raise ValueError(
                f"v_rf is a peak amplitude and must be non-negative, "
                f"got {self.v_rf}"
            )
        if self.omega_axial < 0:
            raise ValueError(
                f"omega_axial must be non-negative, got {self.omega_axial}"
            )

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
        \omega_z = \sqrt{\frac{2\,\kappa\,e\,U_\mathrm{dc}}{m\,z_0^2}}
        $$

        The factor 2 is fixed by Laplace's equation: the static
        endcap potential
        $\Phi_\mathrm{dc} = (\kappa U_\mathrm{dc}/z_0^2)
        [z^2 - (x^2+y^2)/2]$
        is the only harmonic quadrupole with axial curvature
        $\kappa U_\mathrm{dc}/z_0^2$, so
        $m\ddot{z} = -e\,\partial_z\Phi_\mathrm{dc}
        = -2 e \kappa U_\mathrm{dc} z / z_0^2$.
        See Wineland et al., J. Res. NIST 103, 259 (1998), Eq. (2) and
        Berkeland et al., J. Appl. Phys. 83, 5025 (1998), Eqs. (5) and
        (9).
        """
        if u_dc_axial < 0:
            raise ValueError(
                f"u_dc_axial must be non-negative, got {u_dc_axial}"
            )
        m = species.mass_kg
        omega_axial = np.sqrt(
            2 * kappa * ELECTRON_CHARGE * u_dc_axial / (m * z0**2)
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

        Exact inverse of ``from_dc_voltage``:

        $$
        U_\mathrm{dc} = \frac{m\,\omega_z^2\,z_0^2}{2\,\kappa\,e}
        $$
        """
        m = self.species.mass_kg
        return (
            m
            * self.omega_axial**2
            * self.z0**2
            / (2 * self.kappa * ELECTRON_CHARGE)
        )

    @property
    def mathieu_q(self) -> float:
        r"""Dimensionless Mathieu q parameter.

        $$
        q = \frac{2 e V_\mathrm{rf}}{m \Omega_\mathrm{rf}^2 r_0^2}
        $$

        Valid for an ideal quadrupole; ``r0`` must absorb the geometric
        efficiency of a real electrode layout (see the class
        docstring).
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

        The axial DC potential is radially *defocusing*
        ($a_x = a_y = -a_z/2 < 0$ by Laplace's equation), and with
        $\omega_z^2 = 2 e \kappa U_\mathrm{dc}/(m z_0^2)$ (see
        ``from_dc_voltage``) the voltage form and the frequency form
        are the same number, not an approximation of one another:

        $$
        a = \frac{-4 e \kappa U_\mathrm{dc}}{m \Omega_\mathrm{rf}^2 z_0^2}
        = \frac{-2 \omega_\mathrm{axial}^2}{\Omega_\mathrm{rf}^2}
        $$

        (Berkeland et al., J. Appl. Phys. 83, 5025 (1998), Eq. (5).)
        """
        return -2 * self.omega_axial**2 / self.omega_rf**2

    def is_stable(self) -> bool:
        r"""Check if $(a, q)$ falls within the first Mathieu stability
        region, to lowest order.

        Two approximate conditions are applied: $0 < q < 0.908$ and
        $\beta^2 = a + q^2/2 > 0$.

        Both are lowest-order results, not refinements of the exact
        criterion. The exact condition on the Mathieu characteristic
        exponent is $0 \le \beta \le 1$ with $\beta$ from the
        continued fraction of Leibfried et al., Rev. Mod. Phys. 75,
        281 (2003), Eqs. (11)-(13); $\beta^2 = a + q^2/2$ is only its
        leading term, and $q < 0.908$ is the exact boundary only at
        $a = 0$. Near either boundary the classification can be wrong
        in both directions (Floquet monodromy reference values): at
        $q = 0.4,\ a = -0.0793$ the exact motion is unstable while
        $\beta^2 > 0$; at $a = -0.10,\ q = 0.95$ the exact motion is
        stable while the $q$ cut rejects it. The static radial
        defocusing that opens this gap is discussed after RMP Eq. (14).
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

        For $|a| \ll q^2/2 \ll 1$:

        $$
        \omega_r \approx \frac{q \, \Omega_\mathrm{rf}}{2\sqrt{2}}
        $$

        (Wineland et al., J. Res. NIST 103, 259 (1998), text after
        Eq. (5); Berkeland et al., J. Appl. Phys. 83, 5025 (1998),
        Sec. II. Note $|a| \ll q$ is *not* sufficient: the standard
        Yb-171 fixture has $|a|/q = 0.017$ but $|a|/(q^2/2) = 0.28$,
        and dropping $a$ there overestimates $\omega_r$ by 17%.)

        Raises
        ------
        ValueError
            If the trap is outside the first Mathieu stability region.
            The pseudopotential secular frequency is meaningless there
            - at $q = 14298$ the unguarded formula returns
            $\omega_r = 5055\,\Omega_\mathrm{rf}$ - so ``is_stable()``
            gates this property, as ``PenningTrap`` does for its
            transverse frequencies.
        """
        q = self.mathieu_q
        a = self.mathieu_a
        beta_sq = a + q**2 / 2
        if beta_sq <= 0:
            raise ValueError(
                f"Trap is unstable: beta^2 <= 0 (a = {a:.6g}, "
                f"q = {q:.6g}); DC axial defocusing exceeds the RF "
                f"pseudopotential."
            )
        if not self.is_stable():
            raise ValueError(
                f"Trap is unstable: q = {q:.6g} is outside the first "
                f"Mathieu stability region (0 < q < 0.908 at a = 0). "
                f"Check is_stable() before accessing omega_radial."
            )
        return (self.omega_rf / 2) * np.sqrt(beta_sq)

    @property
    def pseudopotential_depth_eV(self) -> float:
        r"""RF pseudopotential well depth at $r = r_0$, in
        electron-volts.

        $$
        \Psi_0 = \frac{e^2 V_\mathrm{rf}^2}
        {4 m \Omega_\mathrm{rf}^2 r_0^2} = \frac{e\,q\,V_\mathrm{rf}}{8}
        $$

        converted to eV. This is the *RF-only* depth evaluated at the
        electrode radius (Wineland et al., J. Res. NIST 103, 259
        (1998), Eq. (6)), i.e. the identity
        $\Psi_0 = \tfrac{1}{2} m \omega_r^2 r_0^2$ holds with the
        RF-only secular frequency $q\Omega_\mathrm{rf}/(2\sqrt{2})$,
        *not* with ``omega_radial``. The DC axial confinement is
        radially defocusing ($a < 0$), so the true radial well depth is
        lower: at the standard Yb-171 fixture ($V_\mathrm{rf} = 1000$ V,
        $\Omega_\mathrm{rf} = 2\pi\cdot30$ MHz, $r_0 = 0.5$ mm,
        $\omega_z = 2\pi\cdot1$ MHz) this property returns 15.89 eV
        while $\tfrac{1}{2} m \omega_\mathrm{rad}^2 r_0^2 / e$ gives
        11.52 eV -- 27% lower.
        """
        m = self.species.mass_kg
        depth_J = (ELECTRON_CHARGE**2 * self.v_rf**2) / (
            4 * m * self.omega_rf**2 * self.r0**2
        )
        return depth_J / ELECTRON_CHARGE

    def micromotion_amplitude(self, displacement_from_null: float) -> float:
        r"""Peak micromotion amplitude for a particle displaced from
        the RF null.

        $x_\mathrm{mm} = (q/2) \cdot |x_\mathrm{displacement}|$,
        valid for $q \ll 1$ (Berkeland et al., J. Appl. Phys. 83, 5025
        (1998), Eq. (15)).

        Notes
        -----
        Diagnostic only. TIQS works in the pseudopotential
        approximation, which time-averages micromotion away, so no
        Hamiltonian, Rabi frequency, or Lamb-Dicke parameter in this
        package carries a micromotion correction. To include it by
        hand, scale a carrier Rabi frequency by
        $J_0(k_\mathrm{eff} x_\mathrm{mm})$ and the $n$-th RF sideband
        by $J_n(k_\mathrm{eff} x_\mathrm{mm})$ (Berkeland Sec. II-III);
        the resulting error also does not appear in
        ``compute_error_budget``.
        """
        return (self.mathieu_q / 2) * abs(displacement_from_null)

    def stray_field_displacement(self, stray_E_field: float) -> float:
        r"""Static displacement from RF null due to a stray DC field.

        $$
        x_\mathrm{displacement} = \frac{e E}{m \omega_r^2}
        $$

        (Berkeland et al., J. Appl. Phys. 83, 5025 (1998), Eq. (16).)
        Diagnostic only, like ``micromotion_amplitude``; raises through
        ``omega_radial`` for an unstable trap.
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
        Axial magnetic field strength in Tesla (positive).
    species : IonSpecies or ElectronSpecies
        The trapped particle species. The charge magnitude is fixed at
        $e$: singly charged ions and the electron only.
    d : float
        Characteristic trap dimension in meters.
        For a hyperbolic trap, $d^2 = (z_0^2 + r_0^2/2) / 2$.
        Enters ``v_dc`` and ``from_dc_voltage`` only - never an
        eigenfrequency.
    omega_axial : float
        Axial angular frequency in rad/s (strictly positive: with no
        axial well the particle is not bound).
    """

    magnetic_field: float
    species: IonSpecies | ElectronSpecies
    d: float
    omega_axial: float

    def __post_init__(self):
        if self.magnetic_field <= 0:
            raise ValueError(
                f"magnetic_field must be positive, got {self.magnetic_field}"
            )
        if self.d <= 0:
            raise ValueError(f"d must be positive, got {self.d}")
        if self.omega_axial <= 0:
            raise ValueError(
                f"omega_axial must be positive, got {self.omega_axial}; "
                f"a Penning trap with no axial well does not confine."
            )
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
    ) -> PenningTrap:
        r"""Construct from DC trapping voltage instead of axial frequency.

        $$
        \omega_z = \sqrt{\frac{e\,V_\mathrm{dc}}{m\,d^2}}
        $$

        ``v_dc`` is the magnitude $|V_0|$; the polarity that produces
        axial confinement is set by the sign of the charge (opposite
        for the electron), as in Gabrielse's convention.
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
        r"""Free (unshifted) cyclotron angular frequency.

        $$
        \omega_c = \frac{eB}{m}
        $$

        This is the free-space value, *not* the trap-shifted
        $\bar\omega_c \equiv \omega_+$ that precision experiments
        report (see ``omega_modified_cyclotron``); the two are related
        by the invariance theorem
        $\omega_c^2 = \omega_+^2 + \omega_-^2 + \omega_z^2$
        (Brown & Gabrielse, Rev. Mod. Phys. 58, 233 (1986)).
        The charge magnitude is fixed at $e$: only singly charged ions
        and the electron are representable.
        """
        return ELECTRON_CHARGE * self.magnetic_field / self.species.mass_kg

    def _transverse_discriminant_value(self) -> float:
        r"""Raw radial discriminant $(\omega_c/2)^2 - \omega_z^2/2$.

        Single source of truth for radial stability, shared by
        ``is_stable()`` and ``_transverse_discriminant()``.
        """
        return (self.omega_cyclotron / 2) ** 2 - self.omega_axial**2 / 2

    def _transverse_discriminant(self) -> float:
        """Radial discriminant, guarded for the transverse frequencies."""
        discriminant = self._transverse_discriminant_value()
        if discriminant <= 0:
            raise ValueError(
                "Trap is unstable: omega_c <= sqrt(2)*omega_z. "
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
        r"""Magnetron angular frequency (a NEGATIVE-energy mode).

        $$
        \omega_- = \frac{\omega_c}{2}
        - \sqrt{\left(\frac{\omega_c}{2}\right)^2
        - \frac{\omega_z^2}{2}}
        $$

        Warnings
        --------
        The returned float is a positive *frequency*, but the magnetron
        mode enters the energy with the opposite sign to the other two:

        $$
        H = \hbar\omega_+\left(n_+ + \tfrac{1}{2}\right)
          + \hbar\omega_z\left(n_z + \tfrac{1}{2}\right)
          - \hbar\omega_-\left(n_- + \tfrac{1}{2}\right)
        $$

        because the radial electrostatic force is defocusing. The
        magnetron motion is therefore only metastable - its total
        energy *decreases* as the orbit radius grows - and the roles of
        the upper and lower motional sidebands are exchanged relative
        to a normal oscillator: removing a magnetron quantum requires
        the BLUE-detuned tone. Cooling it needs an axialization drive
        coupling it to the modified-cyclotron mode.

        Consequently this frequency must not be handed unsigned to code
        that assumes an ascending ladder - ``HarmonicPotential``,
        ``mode_hamiltonian``, ``full_interaction_hamiltonian``'s
        ``mode_frequency``, or ``sideband_cooling_nbar`` - without
        flipping the sideband/energy sign yourself.

        References
        ----------
        Brown & Gabrielse, Rev. Mod. Phys. 58, 233 (1986), Sec. II;
        Dehmelt, Nobel lecture (1989), Figs. 2 and 4; Jain et al.,
        Nature 627, 510 (2024) (magnetron Doppler cooling via
        axialization).
        """
        wc2 = self.omega_cyclotron / 2
        return wc2 - np.sqrt(self._transverse_discriminant())

    def is_stable(self) -> bool:
        r"""Check Penning stability: $\omega_c > \sqrt{2}\,\omega_z$.

        Equivalent to a strictly positive radial discriminant
        $(\omega_c/2)^2 - \omega_z^2/2$, which is the same predicate the
        transverse frequencies are gated on. At or below zero the
        modified cyclotron and magnetron frequencies are degenerate or
        complex and radial confinement is lost. Axial confinement
        ($\omega_z > 0$) is guaranteed by ``__post_init__``.
        """
        return self._transverse_discriminant_value() > 0

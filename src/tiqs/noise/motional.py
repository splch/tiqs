r"""Motional decoherence: anomalous heating and motional dephasing."""

import numpy as np
import qutip

from tiqs.constants import ELECTRON_CHARGE, HBAR
from tiqs.hilbert_space.operators import OperatorFactory
from tiqs.species.ion import get_species

_CA40_MASS_KG = get_species("Ca40").mass_kg


def motional_heating_ops(
    ops: OperatorFactory,
    mode: int,
    heating_rate: float,
    n_bar_env: float | None = None,
) -> list[qutip.Qobj]:
    r"""Lindblad collapse operators for motional heating of a single mode.

    ``heating_rate`` is the ground-state heating rate
    $\dot{\bar{n}}$ in quanta/s -- the quantity measured by sideband
    thermometry and returned by `heating_rate_from_noise`.

    **Damped thermal bath** (finite ``n_bar_env``
    $= \bar{n}_\text{env}$), Turchette et al., PRA 62, 053807 (2000)
    Eq. (4) and Brownnutt et al., Rev. Mod. Phys. 87, 1419 (2015)
    Eq. (14):

    $$
    L_\uparrow = \sqrt{\Gamma\,\bar{n}_\text{env}}\; a^\dagger,
    \qquad
    L_\downarrow = \sqrt{\Gamma\,(\bar{n}_\text{env} + 1)}\; a,
    \qquad
    \Gamma = \dot{\bar{n}} / \bar{n}_\text{env}
    $$

    These give $d\langle n\rangle/dt
    = \dot{\bar{n}} - \Gamma\langle n\rangle$
    (Brownnutt Eq. 18): the initial slope from the ground state is
    $\dot{\bar{n}}$ and the steady state is
    $\langle n\rangle = \bar{n}_\text{env}$, so from vacuum
    $\langle n\rangle(t)
    = \bar{n}_\text{env}\,(1 - e^{-\Gamma t})$.

    **Infinite-temperature limit** (``n_bar_env = None``, the
    default) -- the regime anomalous electric-field noise actually
    occupies, where $\bar{n}_\text{env} \sim 10^5$--$10^7$ for
    $\omega/2\pi = 1$ MHz at 4--300 K (Brownnutt Sec. III). Taking
    $\Gamma \to 0$ at fixed $\Gamma\bar{n}_\text{env}
    = \dot{\bar{n}}$:

    $$
    L_\uparrow = \sqrt{\dot{\bar{n}}}\; a^\dagger,
    \qquad
    L_\downarrow = \sqrt{\dot{\bar{n}}}\; a
    $$

    so $d\langle n\rangle/dt = \dot{\bar{n}}$ exactly and
    $\langle n\rangle(t) = \langle n\rangle_0 + \dot{\bar{n}}\,t$ --
    the linear growth from the ground state measured by Turchette
    et al. Sec. III.A.3.

    Model scope: this builds the bath for one mode in isolation with a
    single rate. Spatially uniform (long-wavelength) field noise heats
    only the center-of-mass mode, at $N_\text{ion}\,\dot{\bar{n}}$
    (Brownnutt Eqs. 22--23), so per-mode rates should be derived from
    the mode participation rather than shared across modes.

    Parameters
    ----------
    ops : OperatorFactory
        Factory for constructing multi-body operators.
    mode : int
        Motional mode index.
    heating_rate : float
        Ground-state heating rate $\dot{\bar{n}}$ in quanta/second.
    n_bar_env : float or None, optional
        Mean phonon number of the thermal environment. ``None``
        (default) selects the infinite-temperature limit. A finite
        positive value gives a bath that equilibrates at ``n_bar_env``
        with damping rate ``heating_rate / n_bar_env``.

    Returns
    -------
    list[qutip.Qobj]
        Collapse operators $[L_\uparrow, L_\downarrow]$, or an empty
        list when ``heating_rate`` is zero.

    Raises
    ------
    ValueError
        If ``heating_rate`` is negative, or ``n_bar_env`` is negative
        or exactly zero (a $T = 0$ bath only cools).
    """
    if heating_rate < 0:
        raise ValueError(
            f"heating_rate must be non-negative, got {heating_rate}"
        )
    if n_bar_env is not None:
        if n_bar_env < 0:
            raise ValueError(
                f"n_bar_env must be positive or None, got {n_bar_env}"
            )
        if n_bar_env == 0:
            raise ValueError(
                "n_bar_env=0 is a T=0 bath, which only cools and cannot "
                "heat. Use n_bar_env=None for the infinite-temperature "
                "limit (pure heating at heating_rate)."
            )
    if heating_rate == 0:
        return []
    amp_up = np.sqrt(heating_rate)
    if n_bar_env is None:
        amp_down = amp_up
    else:
        amp_down = np.sqrt(heating_rate * (n_bar_env + 1) / n_bar_env)
    return [amp_up * ops.create(mode), amp_down * ops.annihilate(mode)]


def motional_dephasing_op(
    ops: OperatorFactory,
    mode: int,
    rate: float,
) -> qutip.Qobj:
    r"""Collapse operator for motional dephasing.

    $L = \sqrt{\gamma}\, \hat{n}$.

    Models fluctuations in the trap frequency that cause dephasing
    of motional superposition states without changing the phonon
    number. Fock coherences decay as
    $\langle n|\rho|n'\rangle \propto
    e^{-\gamma (n - n')^2 t / 2}$.

    Parameters
    ----------
    ops : OperatorFactory
        Factory for constructing multi-body operators.
    mode : int
        Motional mode index.
    rate : float
        Motional dephasing rate $\gamma$ in $\text{s}^{-1}$ (it
        multiplies $(n - n')^2 t / 2$, a dimensionless exponent, so
        it is a decay rate and not an angular frequency).

    Returns
    -------
    qutip.Qobj
        Collapse operator for motional dephasing.
    """
    if rate < 0:
        raise ValueError(f"rate must be non-negative, got {rate}")
    return np.sqrt(rate) * ops.number(mode)


def heating_rate_from_noise(
    spectral_density: float,
    distance: float,
    frequency_hz: float,
    mass_kg: float = _CA40_MASS_KG,
    alpha: float = 1.0,
    reference_distance: float = 100e-6,
    reference_frequency_hz: float = 1e6,
    *,
    beta: float = 4.0,
) -> float:
    r"""Estimate heating rate from electric field noise spectral density.

    $$
    \dot{\bar{n}} = \frac{e^2 \, S_E(\omega)}{4 m \hbar \omega},
    \qquad \omega = 2\pi f
    $$

    (Brownnutt et al., Rev. Mod. Phys. 87, 1419 (2015) Eq. 12), with
    $S_E$ scaled from its reference point as
    $S_E \propto d^{-\beta} f^{-\alpha}$.

    Note the unit break with the rest of the package: ``frequency_hz``
    and ``reference_frequency_hz`` are ordinary frequencies in Hz,
    not angular frequencies (the $2\pi$ is applied internally).

    Parameters
    ----------
    spectral_density : float
        Electric field noise spectral density $S_E$ at the reference
        distance and frequency, in
        $\text{V}^2\,\text{m}^{-2}\,\text{Hz}^{-1}$.
    distance : float
        Ion-electrode distance in meters.
    frequency_hz : float
        Motional mode frequency in Hz.
    mass_kg : float
        Ion mass in kg. **Defaults to the Ca-40 ion mass** from the
        species table; pass the actual species mass otherwise, since
        $\dot{\bar{n}} \propto 1/m$ (using the default for Yb-171
        overestimates the rate by 4.3x).
    alpha : float
        Frequency scaling exponent (typically 1--2 for $1/f$ noise).
    reference_distance : float
        Reference distance for the spectral density, in meters.
    reference_frequency_hz : float
        Reference frequency for the spectral density, in Hz.
    beta : float, keyword-only
        Ion-electrode distance scaling exponent. Defaults to 4.0, the
        planar patch-potential prediction; Brownnutt et al. Sec. IV
        report measured values between about 2 and 4.

    Returns
    -------
    float
        Heating rate in quanta/second.

    Raises
    ------
    ValueError
        If any of ``spectral_density``, ``distance``,
        ``frequency_hz``, ``mass_kg``, ``reference_distance`` or
        ``reference_frequency_hz`` is non-positive
        (``spectral_density`` may be zero).
    """
    if spectral_density < 0:
        raise ValueError(
            f"spectral_density must be non-negative, got {spectral_density}"
        )
    for name, value in (
        ("distance", distance),
        ("frequency_hz", frequency_hz),
        ("mass_kg", mass_kg),
        ("reference_distance", reference_distance),
        ("reference_frequency_hz", reference_frequency_hz),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    d_scaling = (reference_distance / distance) ** beta
    f_scaling = (reference_frequency_hz / frequency_hz) ** alpha
    S_E = spectral_density * d_scaling * f_scaling
    omega = 2 * np.pi * frequency_hz
    return ELECTRON_CHARGE**2 * S_E / (4 * mass_kg * HBAR * omega)

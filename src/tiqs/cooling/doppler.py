"""Doppler cooling: rate-equation estimate of final motional occupation."""

from tiqs.species.ion import IonSpecies


def doppler_cooled_nbar(
    species: IonSpecies, trap_frequency_hz: float
) -> float:
    r"""Estimate mean phonon number after Doppler cooling.

    $$
    \bar{n}_\text{Doppler} = \frac{\Gamma}{2\omega_\text{trap}}
    $$

    where $\Gamma$ is the cooling transition linewidth
    and $\omega_\text{trap}$ is the secular frequency.
    This is the weak-binding limit
    ($\Gamma \gg \omega_\text{trap}$), which is the
    relevant regime for most trapped-ion experiments.

    Model scope: the textbook form above corresponds to
    $k_B T_D = \hbar\Gamma/2$ with $\bar{n} = k_B T/(\hbar\omega)$,
    and drops three $O(1)$ effects of Leibfried RMP **75**, 281
    (2003) Eqs. (105)-(106):

    - the optimal detuning is $\Delta = -\Gamma\sqrt{1+s}/2$, so the
      form above assumes low saturation $s \ll 1$;
    - the emission-recoil geometry factor $\xi$ ($2/5$ for dipole
      radiation) gives
      $k_B T_\text{min} = \hbar\Gamma(1+\xi)\sqrt{1+s}/4$, i.e. $0.7
      \times$ the textbook value at $s \to 0$;
    - $\bar{n} = k_B T/(\hbar\omega)$ is a classical
      ($\bar{n} \gg 1$) result, so it overestimates when
      $\Gamma \lesssim 2\omega$ - for Leibfried's $^9$Be$^+$ example
      ($\omega/2\pi = 11.2$ MHz, $\Gamma/2\pi = 19.4$ MHz) it gives
      0.87 against a measured 0.47(5).

    Unit convention: this function takes an ordinary frequency in
    Hz, unlike its siblings ``sideband_cooling_nbar`` and
    ``eit_cooling_nbar``, which take angular frequencies in rad/s
    like the rest of the library. The ``_hz`` suffix is the only
    marker - passing rad/s here understates $\bar{n}$ by $2\pi$.

    Parameters
    ----------
    species : IonSpecies
        Ion species with cooling transition data.
    trap_frequency_hz : float
        Trap secular frequency in Hz (not angular frequency).

    Returns
    -------
    float
        Mean phonon number after Doppler cooling.

    Raises
    ------
    ValueError
        If ``trap_frequency_hz`` is not positive.
    """
    if trap_frequency_hz <= 0:
        raise ValueError(
            f"trap_frequency_hz must be > 0, got {trap_frequency_hz}"
        )
    return species.doppler_limit_nbar(trap_frequency_hz)

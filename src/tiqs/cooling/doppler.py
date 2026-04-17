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
    """
    return species.doppler_limit_nbar(trap_frequency_hz)

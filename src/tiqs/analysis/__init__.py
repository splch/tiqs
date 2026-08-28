r"""Analysis tools: fidelity metrics, phase-space visualization,
and error budgets.

## Analysis Overview

This package provides tools for characterizing simulation results:

- **Fidelity metrics**: State fidelity
  $F = |\langle\psi_\text{target}|\psi\rangle|^2$,
  process fidelity, and Bell state fidelity for
  benchmarking gate performance. `bell_state_fidelity` takes the
  sign of the MS geometric phase, which flips with the sign of the
  detuning and with the relative sign of the two ions' Lamb-Dicke
  parameters.
- **Phase-space visualization**: Wigner functions

  $$
  W(\alpha) = \frac{2}{\pi}\text{Tr}[
  D^\dagger(\alpha)\rho\,
  D(\alpha)(-1)^{a^\dagger a}]
  $$

  and motional-state trajectories for visualizing
  gate dynamics.
- **Error budgets**: Aggregation of individual error contributions (heating,
  dephasing, scattering, crosstalk) into a total gate infidelity estimate.
"""

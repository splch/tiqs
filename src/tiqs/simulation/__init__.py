r"""Simulation orchestration: configuration and top-level runner.

## Simulation Framework

`SimulationRunner` integrates the Lindblad master equation

$$
\frac{d\rho}{dt} = -i[H(t), \rho] + \sum_k \gamma_k \left(L_k \rho L_k^\dagger
  - \frac{1}{2}\{L_k^\dagger L_k, \rho\}\right)
$$

using QuTiP's `mesolve` (density-matrix solver) or
`sesolve` (pure-state solver for noiseless evolution).
Time-dependent Hamiltonians for laser pulses and gates
are assembled from the `tiqs.interaction` and `tiqs.gates` modules, while
collapse operators are constructed from `tiqs.noise`.
"""

__docformat__ = "numpy"

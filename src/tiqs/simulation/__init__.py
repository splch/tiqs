r"""Simulation orchestration: configuration and top-level runner.

## Simulation Framework

`SimulationRunner` integrates the Lindblad master equation

$$
\dot\rho = -i[H, \rho]
  {} + \sum_k \bigl(
  L_k \rho L_k^\dagger
  {} - \tfrac{1}{2}
  \lbrace L_k^\dagger L_k, \rho\rbrace
  \bigr)
$$

using QuTiP's `mesolve` (density-matrix solver), `mcsolve`
(quantum-trajectory solver) or `sesolve` (pure-state solver for
noiseless evolution). Every collapse operator built by `tiqs.noise`
already carries its own rate inside $L_k$ (e.g.
$L = \sqrt{\gamma_\phi/2}\,\sigma_z$), so no separate $\gamma_k$
prefactor appears above.

Time-dependent Hamiltonians for laser pulses and gates
are assembled from the `tiqs.interaction` and `tiqs.gates` modules, while
collapse operators are constructed from `tiqs.noise`.
"""

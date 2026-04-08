r"""Composite Hilbert space: tensor-product construction,
operators, and states.

## Hilbert Space Structure

TIQS simulates trapped-ion systems in the composite Hilbert space

$$
\mathcal{H} = \mathcal{H}_\text{qubit}^{\otimes N}
  \otimes \mathcal{H}_\text{motion}^{\otimes M}
$$

where $N$ is the number of ions (each a two-level qubit with $\dim = 2$) and
$M$ is the number of motional modes (each a truncated harmonic oscillator with
$\dim = n_\text{max} + 1$). The total Hilbert space dimension is
$2^N \times (n_\text{max} + 1)^M$.

Operators acting on individual subsystems are embedded into the full space via
tensor products:

$$
\sigma_x^{(j)} = I \otimes \cdots
  \otimes \sigma_x \otimes \cdots \otimes I
$$

where $\sigma_x$ acts on the $j$-th qubit.
"""
__docformat__ = "numpy"

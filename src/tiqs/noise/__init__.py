r"""Noise models: motional heating, qubit decoherence,
photon scattering, and crosstalk.

Symbol convention for the theory page below: it uses the textbook
labels $\sigma_+ = |e\rangle\langle g|$ (excitation) and
$\sigma_- = |g\rangle\langle e|$ (de-excitation). The code labels are
inverted with respect to those symbols because $|0\rangle$ is the
ground state, so
`tiqs.hilbert_space.operators.OperatorFactory.sigma_plus`
$= |0\rangle\langle 1|$ is de-excitation and ``sigma_minus``
$= |1\rangle\langle 0|$ is excitation. The decay operators written
$\propto \sigma_-$ below are therefore built with ``ops.sigma_plus``.

.. include:: ../../../docs/theory/noise.md
"""

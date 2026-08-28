r"""Quantum gate implementations: single-qubit, Molmer-Sorensen,
Cirac-Zoller, and light-shift.

``molmer_sorensen`` and ``light_shift`` are the usable entangling
gates. ``cirac_zoller`` builds the historical three-pulse sequence
without the auxiliary level the 1995 protocol requires, so it is
**not** an entangling gate; see its docstring for the map it really
produces.

.. include:: ../../../docs/theory/gates.md
"""

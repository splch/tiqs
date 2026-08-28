"""Shared test fixtures for TIQS test suite."""

import numpy as np
import pytest

RNG_SEED = 42
"""Seed for the shared `rng` fixture."""


@pytest.fixture
def rng():
    """Deterministic random number generator for reproducible tests.

    Function scoped, so every test that requests it gets a generator
    freshly seeded with `RNG_SEED`. Used by the sampling tests in
    tests/test_end_to_end.py; tests that need a *specific* different
    stream (e.g. tests/test_spam.py, which checks several independent
    seeds) construct their own generator instead.
    """
    return np.random.default_rng(RNG_SEED)

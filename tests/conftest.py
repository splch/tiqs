"""Shared test fixtures for TIQS test suite."""
import numpy as np
import pytest


@pytest.fixture
def rng():
    """Deterministic random number generator for reproducible tests."""
    return np.random.default_rng(42)

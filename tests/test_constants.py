from tiqs.constants import (
    AMU,
    BOLTZMANN,
    ELECTRON_CHARGE,
    EPSILON_0,
    HBAR,
    SPEED_OF_LIGHT,
)


def test_hbar_value():
    assert abs(HBAR - 1.054571817e-34) < 1e-43


def test_electron_charge_value():
    assert abs(ELECTRON_CHARGE - 1.602176634e-19) < 1e-28


def test_amu_value():
    assert abs(AMU - 1.66053906660e-27) < 1e-36


def test_boltzmann_value():
    assert abs(BOLTZMANN - 1.380649e-23) < 1e-29


def test_speed_of_light_value():
    assert abs(SPEED_OF_LIGHT - 299792458.0) < 1.0


def test_epsilon_0_value():
    assert abs(EPSILON_0 - 8.8541878128e-12) < 1e-21

from tiqs.constants import (
    AMU,
    BOHR_MAGNETON,
    BOLTZMANN,
    ELECTRON_CHARGE,
    ELECTRON_G_FACTOR,
    ELECTRON_MASS,
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


def test_electron_mass_value():
    assert abs(ELECTRON_MASS - 9.1093837015e-31) < 1e-40


def test_bohr_magneton_value():
    assert abs(BOHR_MAGNETON - 9.2740100783e-24) < 1e-33


def test_electron_g_factor_value():
    assert abs(ELECTRON_G_FACTOR - 2.00231930436256) < 1e-14

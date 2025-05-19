import numpy as np
import pytest

from kinetic_model.core.eyring import Eyring_k


def test_eyring_rate_positive():
    """
    The Eyring rate constant should always be positive for any Gibbs energy and temperature.
    """
    k = Eyring_k(G=50, T=300)
    assert k > 0


def test_eyring_rate_increases_with_temperature():
    """
    At fixed Gibbs energy, the rate constant should increase with temperature.
    """
    G = 50  # kJ/mol
    k_low = Eyring_k(G, T=300)
    k_high = Eyring_k(G, T=350)
    assert k_high > k_low


def test_eyring_rate_decreases_with_gibbs_energy():
    """
    At fixed temperature, a higher activation free energy should yield a smaller rate constant.
    """
    T = 300
    k_low_G = Eyring_k(G=10, T=T)
    k_high_G = Eyring_k(G=100, T=T)
    assert k_low_G > k_high_G


def test_eyring_rate_return_type():
    """
    The function should return a float or numpy floating scalar.
    """
    k = Eyring_k(G=75, T=298.15)
    assert isinstance(k, float) or isinstance(k, np.floating)

def test_eyring_rate_negative_gibbs_energy():
    """
    The function should handle negative Gibbs energy correctly.
    """
    G = -50  # kJ/mol
    T = 300
    k = Eyring_k(G, T)
    assert k < 1e13  # Ensure the rate constant is reasonable

def test_eyring_right_calculation():
    """
    Test the Eyring equation with known values.
    """
    G = 50  # kcal/mol
    T = 300  # K
    expected_k = 12293.0924546  # Replace with the expected value based on your calculations
    k = Eyring_k(G, T)
    assert np.isclose(k, expected_k, rtol=1e-3)  # Adjust tolerance as needed
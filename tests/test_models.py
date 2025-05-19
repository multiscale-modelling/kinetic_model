import os
import warnings
import pandas as pd
import numpy as np
import pytest

from pathlib import Path
from kinetic_model.constants.diffusion import DiffusionLinear
from kinetic_model.constants.adsorption import AdsorptionKNN
from kinetic_model.simulation.reaction_list_creator import compute_rate
from kinetic_model.core.eyring import Eyring_k

def test_diffusion_linear_reproduces_synthetic():
    """
    DiffusionLinear should reproduce D_TST = alpha * sqrt(T) perfectly.
    """
    df = pd.DataFrame({
        "temperature": [100, 400, 900],
        "n": [1, 1, 1],
        "D_TST": [5 * np.sqrt(100), 5 * np.sqrt(400), 5 * np.sqrt(900)]
    })
    model = DiffusionLinear().fit(df)
    preds = model.predict(df)
    assert preds.shape == (3,)
    assert np.allclose(preds, df["D_TST"].values)


def test_diffusion_linear_unseen_group():
    """
    Predictions for unseen n should be NaN.
    """
    df = pd.DataFrame({"temperature": [100], "n": [1], "D_TST": [5 * np.sqrt(100)]})
    model = DiffusionLinear().fit(df)
    df2 = pd.DataFrame({"temperature": [200], "n": [2]})
    preds = model(df2)
    assert np.isnan(preds[0])


def test_adsorption_knn_basic():
    """
    AdsorptionKNN should predict known values for n_neighbors=1.
    """
    data = pd.DataFrame({
        "pressure_bar": [1, 2, 3, 4],
        "temperature": [100, 200, 300, 400],
        "n": [1, 1, 2, 2],
        "adsorption_type_num": [0, 0, 0, 0],
        "mol_uc_avg": [10, 20, 30, 40]
    })
    model = AdsorptionKNN(n_neighbors=1).fit(data)
    df = pd.DataFrame({
        "pressure_bar": [1, 3],
        "temperature": [100, 300],
        "n": [1, 2],
        "adsorption_type_num": [0, 0]
    })
    preds = model.predict(df)
    assert preds[0] == pytest.approx(10)
    assert preds[1] == pytest.approx(30)


def test_adsorption_knn_unseen_group():
    """
    Predictions for unseen (n, adsorption_type_num) should be NaN.
    """
    data = pd.DataFrame({
        "pressure_bar": [1],
        "temperature": [100],
        "n": [1],
        "adsorption_type_num": [0],
        "mol_uc_avg": [10]
    })
    model = AdsorptionKNN().fit(data)
    df = pd.DataFrame({
        "pressure_bar": [1],
        "temperature": [100],
        "n": [2],
        "adsorption_type_num": [1]
    })
    preds = model(df)
    assert np.isnan(preds[0])


def test_compute_rate_behavior(monkeypatch):
    """
    compute_rate should adjust Ea correctly and call Eyring_k.
    """
    import kinetic_model.simulation.reaction_list_creator as rlc
    calls = []
    def fake_Eyring_k(Ea, T):
        calls.append((Ea, T))
        return 'rate'
    monkeypatch.setattr(rlc, 'Eyring_k', fake_Eyring_k)
    # no delta_E
    rate = compute_rate(10, T=300)
    assert rate == 'rate'
    assert calls[-1] == (10, 300)
    # positive Ea
    calls.clear()
    rate = compute_rate(10, T=400, delta_E=5)
    assert calls[-1] == (15, 400)
    # negative Ea
    calls.clear()
    rate = compute_rate(-1, T=500, delta_E=7)
    assert calls[-1] == (7, 500)

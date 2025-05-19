import re
import numpy as np
import pandas as pd
import pytest

import kinetic_model.simulation.run_reactor as rr_mod
from kinetic_model.simulation.run_reactor import ReactorSimulation

# --- Stubs & Fixtures --------------------------------------------------------

class DummyEnergy:
    """Fake energy‐model factory with minimal .get_params()."""
    def __init__(self, pressure, temperature):
        self.pressure = pressure
        self.temperature = temperature

    def get_params_basic(self):
        # prepare_constants is stubbed, so these keys are ignored
        return {'foo': 1.0, 'bar': 2.0}


class DummySolver:
    """Replaces Reaction_Diffusion; returns a simple constant time‐series."""
    def __init__(self, reactions=None, capacity=None):
        # ignore inputs
        pass

    def solve(self, initial_concs_dict, time_points):
        # Build a DataFrame with two species: C2/C2_out = 0, C4/C4_out = 1
        n_steps = len(time_points)
        df = pd.DataFrame({
            'C2':       np.zeros(n_steps),
            'C2_out':   np.zeros(n_steps),
            'C4':       np.ones(n_steps),
            'C4_out':   np.ones(n_steps),
        })
        return df


@pytest.fixture(autouse=True)
def patch_all(monkeypatch):
    # 1) Stub out joblib.load → returns DummyModel.predict(X) = const array
    class DummyModel:
        def __init__(self, val):
            self.val = val
        def predict(self, X):
            return np.full(len(X), self.val)

    def fake_load(path):
        if 'diffusion' in path:
            return DummyModel(0.5)
        elif 'adsorption' in path:
            return DummyModel(1.0)
        else:
            raise ValueError(f"Unexpected model path: {path!r}")

    monkeypatch.setattr(rr_mod, 'load', fake_load)

    # 2) Stub out prepare_constants & make_reaction_list
    monkeypatch.setattr(rr_mod, 'prepare_constants_basic', lambda *args, **kwargs: {})
    monkeypatch.setattr(rr_mod, 'make_reaction_list_basic', lambda *args, **kwargs: [])

    # 3) Replace the actual solver with our DummySolver
    monkeypatch.setattr(rr_mod, 'Reaction_Diffusion', DummySolver)


# --- Tests -------------------------------------------------------------------

def test_reactor_simulation_distribution_and_ts():
    """
    With DummySolver producing C2/C2_out=0 and C4/C4_out=1,
    final distribution should block n=2 (NaN) and give p=1.0 at n=4,
    and the time‐series should normalize to [0,1] at every time.
    """
    # small problem size for speed
    N = 4
    T = 300.0
    pressure = 1.0
    L = 10.0
    delta_ins = 1.0
    delta_ads = 1.0
    time_hours = 0.001
    n_steps = 5
    initial_concs = {'Cat': 0.1, 'C2': 0.2}

    sim = ReactorSimulation(
        N=N,
        T=T,
        pressure=pressure,
        L=L,
        delta_ins=delta_ins,
        delta_ads=delta_ads,
        time_hours=time_hours,
        n_steps=n_steps,
        initial_concs=initial_concs,
        diff_model_path='diffusion_model.pkl',
        ads_model_path='knn_adsorption_model.pkl',
        energy_model=DummyEnergy
    )

    # Ensure our DummyEnergy was instantiated correctly
    assert isinstance(sim.energy_model, DummyEnergy)
    assert sim.energy_model.pressure == pressure
    assert sim.energy_model.temperature == T

    dist, ts = sim.run()

    # --- final distribution checks ---
    assert isinstance(dist, pd.DataFrame)
    # We should have exactly two rows for n=2 and n=4
    assert list(dist['n']) == [2, 4]

    # Concentrations come from DummySolver: 
    #   C2+C2_out = 0+0,  C4+C4_out = 1+1 = 2
    concs = dict(zip(dist['n'], dist['concentration']))
    assert concs[2] == pytest.approx(0.0)
    assert concs[4] == pytest.approx(2.0)

    # Probabilities: n=2 → NaN, n=4 → 2/2 = 1.0
    probs = dict(zip(dist['n'], dist['probability']))
    assert np.isnan(probs[2])
    assert probs[4] == pytest.approx(1.0)

    # --- time‐series checks ---
    assert isinstance(ts, pd.DataFrame)
    # Index should be the time grid in seconds:
    assert ts.index[0] == pytest.approx(0.0)
    assert ts.index[-1] == pytest.approx(time_hours * 3600.0)
    # Columns should be the two oligomer sizes:
    assert list(ts.columns) == [2, 4]
    # Every row should normalize to [0,1]:
    for _, row in ts.iterrows():
        assert row[2] == pytest.approx(0.0)
        assert row[4] == pytest.approx(1.0)


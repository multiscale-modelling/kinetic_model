import pytest
from kinetic_model.core.reactor import Reaction
from kinetic_model.core.diffusion import Reaction_Diffusion
import numpy as np


def test_parse_reaction_forward():
    r = Reaction([])
    reactants, products, is_back = r.parse_reaction("A + B -> C + D")
    assert reactants == ["A", "B"]
    assert products == ["C", "D"]
    assert is_back is False


def test_parse_reaction_backward():
    r = Reaction([])
    reactants, products, is_back = r.parse_reaction("A + B <-> C")
    assert reactants == ["A", "B"]
    assert products == ["C"]
    assert is_back is True


def test_rate_of_change_simple():
    k = 2.5
    reactions = [("A -> B", k, 0.0)]
    r = Reaction(reactions)
    r.create_reaction_dict()
    # sorted_keys should be ['A', 'B']
    concs = [1.0, 0.0]
    rates = r.rate_of_change(concs, t=0)
    assert rates == pytest.approx([-k * 1.0, k * 1.0])


def test_solve_initial_conditions():
    k = 1.0
    reactions = [("A -> B", k, 0.0)]
    r = Reaction(reactions)
    time_points = [0, 10]
    df = r.solve({"A": 5.0, "B": 0.0}, time_points)
    # columns order sorted alphabetically
    assert list(df.columns) == ["A", "B"]
    # initial conditions preserved at t=0
    assert df.loc[0, "A"] == pytest.approx(5.0)
    assert df.loc[0, "B"] == pytest.approx(0.0)


def test_normalize_capacity():
    cap = {"X": 2.0, "Y": 3.0}
    rd = Reaction_Diffusion([], capacity=cap.copy())
    total = sum(rd.capacity.values())
    assert total == pytest.approx(1.0)
    assert rd.capacity["X"] == pytest.approx(2.0 / 5.0)
    assert rd.capacity["Y"] == pytest.approx(3.0 / 5.0)


def test_parse_diffusion_forward():
    rd = Reaction_Diffusion([])
    reactants, products, is_back = rd.parse_diffusion("M + N D> O + P")
    assert reactants == ["M", "N"]
    assert products == ["O", "P"]
    assert is_back is False


def test_parse_diffusion_backward():
    rd = Reaction_Diffusion([])
    reactants, products, is_back = rd.parse_diffusion("M + N <D> O")
    assert reactants == ["M", "N"]
    assert products == ["O"]
    assert is_back is True


def test_parse_reaction_diffusion_flag_forward():
    rd = Reaction_Diffusion([])
    parsed, flag = rd.parse_reaction("A + B -> C")
    reactants, products, is_back = parsed
    assert reactants == ["A", "B"]
    assert products == ["C"]
    assert flag is False


def test_parse_reaction_diffusion_flag_diffusion():
    rd = Reaction_Diffusion([])
    parsed, flag = rd.parse_reaction("A + B <D> C")
    reactants, products, is_back = parsed
    assert reactants == ["A", "B"]
    assert products == ["C"]
    assert flag is True

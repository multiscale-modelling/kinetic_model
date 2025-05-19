import pytest
from kinetic_model.core.diffusion import Reaction_Diffusion


def test_capacity_normalization_in_init():
    """
    Initializing with a capacity dict should normalize its values to sum to 1.
    """
    cap = {'A': 2.0, 'B': 3.0}
    rd = Reaction_Diffusion([], capacity=cap.copy())
    # Sum of capacities should be 1.0
    assert pytest.approx(sum(rd.capacity.values())) == 1.0
    # Individual entries normalized correctly
    assert rd.capacity['A'] == pytest.approx(2.0/5.0)
    assert rd.capacity['B'] == pytest.approx(3.0/5.0)


def test_create_reaction_dict_diffusion_computes_correct_coefficients():
    """
    create_reaction_dict should apply capacity-based coefficients for diffusion reactions.
    """
    kf, kr = 2.0, 3.0
    cap = {'A': 1.0, 'B': 1.0}
    rd = Reaction_Diffusion([('A <D> B', kf, kr)], capacity=cap.copy())
    rd.create_reaction_dict()
    # With equal capacities, each normalized to 0.5 => coefficient = 1/0.5 = 2.0
    forward_coeff = kf * 2.0
    back_coeff = kr * 2.0
    # For 'A': loss from A and gain from B
    assert 'A' in rd.reaction_dict
    assert rd.reaction_dict['A'] == [ (('A',), -forward_coeff), (('B',), back_coeff) ]
    # For 'B': gain from A and loss from B
    assert 'B' in rd.reaction_dict
    assert rd.reaction_dict['B'] == [ (('A',), forward_coeff), (('B',), -back_coeff) ]


def test_rate_of_change_includes_diffusion_flux():
    """
    rate_of_change should combine forward and backward diffusion terms correctly.
    """
    kf, kr = 2.0, 1.0
    cap = {'A': 2.0, 'B': 2.0}
    rd = Reaction_Diffusion([('A <D> B', kf, kr)], capacity=cap.copy())
    rd.create_reaction_dict()
    # Ensure sorted_keys is set for rate_of_change
    rd.sorted_keys = sorted(rd.reaction_dict.keys())
    # Equal concentrations for both species
    concs = [1.0, 1.0]
    rates = rd.rate_of_change(concs, t=0)
    # With capacities normalized to 0.5, coefficient = 2.0
    # forward rate = kf * 2.0 * [A] = 2.0 * 2.0 * 1.0 = 4.0
    # backward rate = kr * 2.0 * [B] = 1.0 * 2.0 * 1.0 = 2.0
    # net rate A = -4.0 + 2.0 = -2.0, net rate B = +4.0 - 2.0 = 2.0
    assert rates == pytest.approx([-2.0, 2.0])


def test_unidirectional_diffusion_has_no_backward_term():
    """
    A D> B should represent one-way diffusion without a backward term.
    """
    kf, kr = 2.0, 5.0
    cap = {'A': 1.0, 'B': 1.0}
    rd = Reaction_Diffusion([('A D> B', kf, kr)], capacity=cap.copy())
    rd.create_reaction_dict()
    rd.sorted_keys = sorted(rd.reaction_dict.keys())
    concs = [1.0, 1.0]
    rates = rd.rate_of_change(concs, t=0)
    # With no backward term, only forward contributes: forward_coeff = kf * 2.0 = 4.0
    # net rate A = -4.0, net rate B = +4.0
    assert rates == pytest.approx([-4.0, 4.0])


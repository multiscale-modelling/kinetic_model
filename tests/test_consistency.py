"""audit: fast unit checks of the corrected model (no production grids).

    cd scripts && python -m pytest tests -q
"""
import re

import numpy as np
import pytest

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
import km_common as km  # noqa: E402

TEMPS = [300.0, 360.0, 420.0, 480.0, 540.0]
R = 8.314462618e-3


def reactions(T, p=1.0, elim_map='per_step', **kw):
    """Final reaction list (after elim_map / crowding / lattice) of a tiny run, before rescaling."""
    cap = {}
    orig = km.normilize_constants

    def grab(rx, r_max=1e6):
        cap['rx'] = list(rx)
        return orig(rx, r_max)
    km.normilize_constants = grab
    try:
        km.simulate(T, p, mode='batch', particle_nm=None, model='pore', hours=1e-9, n_steps=3,
                    elim_map=elim_map, **kw)
    finally:
        km.normilize_constants = orig
    return {rx[0]: rx for rx in cap['rx']}


def dG(rx, T):
    """Reaction free energy of one step from K = kf/kr (kJ/mol)."""
    return -R * T * np.log(rx[1] / rx[2])


def net_growth(rx, T, n):
    """dG of the gas-phase reaction Cn + C2 -> Cn+2 through the network cycle
    (CatC2 + Cn -> CatCn + C2, CatCn + C2 -> CatCn+2, CatCn+2 + C2 -> CatC2 + Cn+2)."""
    el = lambda m: dG(rx[f'CatC{m}(C2) + C2 <-> CatC2(C2) + C{m}'], T)
    ins = dG(rx[f'CatC{n}(C2) + C2 <-> CatC{n + 2}(C2)'], T)
    return (-el(n) if n > 2 else 0.0) + ins + el(n + 2)


@pytest.mark.parametrize('T', TEMPS)
def test_dft_barriers_positive(T):
    """Eyring_k clips negative barriers to 0; that would silently break detailed balance."""
    for p in (1.0, 50.0):
        for model in ('pore', 'old'):
            bp = km.energy_model(T, p, model).get_energy_barriers()
            assert min(bp.values()) > 0, (model, p, bp)


@pytest.mark.parametrize('T', TEMPS)
def test_pressure_counted_once(T):
    """pore: rate constants independent of p; pressure enters only through pore [C2]."""
    a, b = reactions(T, 1.0), reactions(T, 50.0)
    for k in a:
        if '<D>' not in k:
            assert a[k][1:3] == pytest.approx(b[k][1:3], rel=1e-12)
    c1, c50 = (float(km.C2_MODEL.predict_absolute(np.array([p * 101325.0]), np.array([T]))[0]) for p in (1, 50))
    assert c50 > c1


@pytest.mark.parametrize('T', TEMPS)
def test_bep_reproduces_dft(T):
    bp = km.energy_model(T, 1.0, 'pore').get_energy_barriers()
    bep, fit = km.bep_barriers(bp, 16)
    for step, pts in km.BEP_POINTS.items():
        assert 0 <= fit[step][1] <= 1
        for n, s in pts.items():
            assert bep[step][n][0] == pytest.approx(bp[f'{step}_{s}'], abs=1.0)
        assert all(gr > 0 for _, gr in bep[step].values())


@pytest.mark.parametrize('T', TEMPS)
def test_per_step_bep_thermodynamically_consistent(T):
    """Cn + C2 -> Cn+2 through the network must not depend on the barrier model where DFT exists
    (n = 2, 4, 6: own DFT steps). 'per_step' is the reference; BEP must match it."""
    f, b = reactions(T, elim_map='per_step'), reactions(T, elim_map='bep')
    for n in (2, 4, 6):
        assert net_growth(b, T, n) == pytest.approx(net_growth(f, T, n), abs=1.0)


@pytest.mark.parametrize('T', TEMPS)
def test_crowding_shifts_K_by_gamma(T):
    p = 30.0
    a, c = reactions(T, p), reactions(T, p, crowding='pocket', s_pocket=3.0)
    g = km.product_activity(8, T, p, 'pocket', 3.0)
    k = 'CatC8(C2) + C2 <-> CatC2(C2) + C8'
    assert (a[k][1] / a[k][2]) / (c[k][1] / c[k][2]) == pytest.approx(g, rel=1e-9)


def test_batch_conserves_molecules_in_ldf():
    """No reaction (no sites): pore + gas totals of C2 constant, final split by capacity."""
    _, ts = km.simulate(400.0, 10.0, mode='batch', particle_nm=300.0, model='pore', hours=1.0,
                        n_steps=50, elim_map='per_step', site_frac=1e-12)
    tot = ts['C2'] + ts['C2_out']
    assert tot.iloc[-1] == pytest.approx(tot.iloc[0], rel=1e-6)


def test_network_consistency_flags_high_T_low_p():
    assert km.network_consistency(300.0, 1.0)[0]
    assert km.network_consistency(420.0, 1.0)[0]
    ok, gap, msgs = km.network_consistency(540.0, 1.0)
    assert not ok and gap == pytest.approx(19.1, abs=0.2) and msgs
    assert not km.network_consistency(450.0, 1.0)[0]        # the non-converging 450 K / 1 bar points
    assert km.network_consistency(450.0, 50.0)[0]            # pressure stabilises ethylene binding


def test_filter_flag_refuses_inconsistent_points():
    with pytest.raises(km.InconsistentNetworkError, match='540 K, 1 bar'):
        km.simulate(540.0, 1.0, mode='flow', particle_nm=300.0, model='pore', hours=1e-6, n_steps=3,
                    elim_map='per_step', check_network='filter')
    with pytest.warns(UserWarning, match='inconsistent'):
        km.simulate(540.0, 1.0, mode='flow', particle_nm=300.0, model='pore', hours=1e-6, n_steps=3,
                    elim_map='per_step')

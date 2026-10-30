"""Shared helpers for running the kinetic model: reactors, simulate(), run_case().

Inputs are read relative to the working directory ('../data/structures', ...), so we chdir into
data/, where '../data' points back to the same folder.

run_case() (used by run_model.py) exposes the two consistent variants under plain names:
counting='pore' -> model='pore', counting='gas' -> model='ideal'; barriers='tetra' /
'individual' / 'legacy' -> elim_map='tetra' / 'per_step' / 'legacy'. Details: docs/model.md.

Ways of letting ethylene pressure enter in simulate() (see docs/model.md):
  model='pore'  Gibbs energies at p0 = 1 bar; pressure enters only through the
                     GCMC pore concentration of ethylene (counted once).
  model='old'        Gibbs energies at the operating pressure AND pore concentration
                     (ethylene counted twice; previous version).
                     Do not use: the released olefins are also put at p, which
                     slows every elimination by p/p0 (docs/model.md).
  model='gas'        Gibbs energies at the operating pressure; pore ethylene fixed at its
                     1 bar GCMC value. Not consistent: the profile also puts the released
                     olefins at p, and their concentration enters the rates as well (kept
                     only for comparison).
  model='ideal'      Ideal-gas ethylene: Gibbs energies at p0, then every state is shifted by
                     (n_ethylene - 5) RT ln(p/p0), so only ethylene is at p and the released
                     olefins stay at the standard state; ethylene removed from the rate laws
                     (CatCn <-> CatCn+2), so its activity p/p0 enters once, through dG.
                     Flow only: in batch, ethylene would not be consumed.
"""
import functools
import os
import re
import sys
import warnings

warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
INPUTS = os.path.join(ROOT, 'data')
sys.path.insert(0, HERE)
os.chdir(INPUTS)

import joblib
import numpy as np
import pandas as pd
import yaml

from kinetic_model.constants.energy_profile import EnergyBarriers, EnergyBarriersSimplifyed
from kinetic_model.core.eyring import Eyring_k
from kinetic_model.core.diffusion import Reaction_Diffusion
from kinetic_model.simulation.reaction_list_creator import (
    make_reaction_list_chemisorp_simplified, prepare_constants_chemisorp_simplified)
from kinetic_model.simulation.run_reactor import ReactorSimulationChemisorptionSimplified

DIFF_MODEL = '../data/processed/diffusion_extrap_model.pkl'
ADS_MODEL = '../data/processed/knn_adsorption_slope_model.pkl'
P0_CODE = 1e5 / 101325.0     # the code multiplies 'pressure' by 101325 Pa, so this is 1 bar
CAT = 1e-2                   # catalytic sites per unit cell
N_MAX = 16
C2_MODEL = joblib.load('../data/adsorption/pt_poly_model.pkl')
GAS_FORMULAE = yaml.safe_load(open('formulae.yaml'))['1']   # free gas molecules per state (ref '1')


class _EthylenePressureProfile(EnergyBarriers):
    """Profile at p0 for all gases, then only ethylene moved to p: each state gets
    (n_ethylene - 5) RT ln(p/p0), 5 = free ethylene in the reference state 1."""

    def __init__(self, temperature, pressure):
        EnergyBarriers.__init__(self, temperature, P0_CODE)
        rt_ln_p = 8.314462618e-3 * temperature * np.log(pressure * 101325.0 / 1e5)   # kJ/mol
        for order in self.energy_profile:
            key = 'TS_' + order if '-' in order else order
            self.energy_profile[order] += (GAS_FORMULAE[key]['ethylene'] - 5) * rt_ln_p


class EnergyBarriersEthylenePressure(EnergyBarriersSimplifyed, _EthylenePressureProfile):
    """Reduced-network barriers (EnergyBarriersSimplifyed) on the ethylene-only shifted profile."""


class ReactorBatch(ReactorSimulationChemisorptionSimplified):
    """Batch reactor (ethylene consumed) with optional log-spaced sampling times."""

    def __init__(self, *args, log_times=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_times = log_times

    def _times(self):
        t_end = self.time_hours * 3600
        if not self.log_times:
            return np.linspace(0, t_end, self.n_steps)
        return np.concatenate(([0.0], np.geomspace(t_end * 1e-12, t_end, self.n_steps - 1)))

    def run(self, build_solver=True):
        if build_solver:
            self.solver = self._build_solver()
        self._result = self.solver.solve(initial_concs_dict=self.initial_concs,
                                         time_points=self._times())
        dist = self._calc_distribution(self._result)
        cols = [c for c in self._result.columns if re.match(r'C\d+(_out)?$', c)]
        return dist, self._result[cols]


class ReactorFlow(ReactorBatch):
    """Flow reactor: ethylene held fixed, products leave into a large outer volume."""

    def _build_solver(self):
        df, idx = self._make_dataframe()
        self._diffusion_rates = self._calc_diffusion_constants(df, idx)
        self._capacity = self._calc_capacity(df, idx)
        for key in self._capacity:
            if 'out' in key:
                self._capacity[key] = 1e6
        self.consts = prepare_constants_chemisorp_simplified(self.barrier_params, temperature=self.T)
        self._reactions = make_reaction_list_chemisorp_simplified(
            n_max=self.N, consts=self.consts, diffusion_rates=self._diffusion_rates,
            include_diffusion=self.include_diffusion)
        for i, r in enumerate(self._reactions):
            self._reactions[i] = (r[0].replace('<D>', 'D>'), r[1], r[2])
        return Reaction_Diffusion(reactions=self._reactions, capacity=self._capacity,
                                  fixed_concentrations=['C2', 'C2_out'])


class AdsInterpT:
    """Pore capacities linear in T between the GCMC grid temperatures (every 30 K).
    The kNN alone (2 distance-weighted neighbours in scaled (p, T)) swaps neighbours
    between grid points, which makes the capacities, and so the diffusion-limited flow
    curves, jump (e.g. 1000 µm, 1 bar, near 375 and 405 K). On the grid the values are
    the kNN's own (exact training points), in between a straight line."""
    T_GRID = np.arange(300.0, 601.0, 30.0)

    def __init__(self, knn):
        self.knn = knn

    def predict(self, df):
        T = float(df['temperature'].iloc[0])
        i = int(np.clip(np.searchsorted(self.T_GRID, T) - 1, 0, len(self.T_GRID) - 2))
        t0, t1 = self.T_GRID[i], self.T_GRID[i + 1]
        y0 = self.knn.predict(df.assign(temperature=t0))
        y1 = self.knn.predict(df.assign(temperature=t1))
        return y0 + (y1 - y0) * (T - t0) / (t1 - t0)


def normilize_constants(reactions, r_max=1e6):
    """Same rescaling as in the notebooks (stiffness control); returns the factor."""
    factor = min(1, r_max / np.max([r[1:] for r in reactions]))
    for i in range(len(reactions)):
        reactions[i] = (reactions[i][0], reactions[i][1] * factor, reactions[i][2] * factor)
    return factor


def energy_check(T, p):
    """Validity of the reduced network : E8 > E10 with ethylene at the
    operating pressure. This is a thermodynamic stability test, so the operating
    pressure belongs here in both models."""
    eb = EnergyBarriersSimplifyed(temperature=T, pressure=p)
    return bool(eb.energy_profile['8'] - eb.energy_profile['10'] > 0)


# product crowding in the ethylene-loaded pore (docs/model.md)
CROWDING = ('lattice', 'multisite', 'pocket', 'pressure')
# 'pocket': ethylene-sized positions to clear at the Ni site for olefin release (independent of
# chain length). s = 3 reproduces the 'old'/'gas' penalty p/p0 at 300 K: s_eff =
# ln(p/p0) / ln(1/theta_v) = 3.5 / 3.0 / 2.8 at 15 / 30 / 50 bar. See product_activity().
S_POCKET = 3


def ethylene_filling(T, p):
    """Fractional ethylene pore filling theta = b f / (1 + b f) of the Langmuir fit in C2_MODEL
    (f = Peng-Robinson fugacity at p bar)."""
    Tk = np.array([float(T)])
    bf = C2_MODEL.b_model.predict(Tk) * C2_MODEL.eos.fugacity(np.array([p * 1e5]), Tk)
    return float((bf / (1.0 + bf))[0])


def product_activity(n, T, p, crowding, s_pocket=None, theta=None):
    """Activity coefficient gamma_n of a released olefin Cn in the ethylene-loaded pore.
    'lattice'   single-site Langmuir lattice gas: 1 / theta_vacant
    'multisite' each Cn blocks n/2 ethylene-sized sites: (1 / theta_vacant)^(n/2)
    'pocket'    (1 / theta_vacant)^S_POCKET, the same for every chain length
    'pressure'  olefin treated as a pure gas at the operating pressure: p / p0
    theta_vacant = 1 - theta_C2(p, T) from the Langmuir fit already used for ethylene
    (ethylene_filling); lattice-gas activity of the released olefin gamma = theta_v^-s.

    'pocket': olefin release is limited by clearing the local pocket at the Ni site, about
    3 ethylene-sized positions. Once the double bond has left the node the rest of the chain
    no longer matters, so s does not depend on n (unlike 'multisite' with s = n/2, which
    blows up exponentially with chain length). s = 3 reproduces the 'old'/'gas'
    penalty gamma = p/p0 at 300 K: s_eff = ln(p/p0) / ln(1/theta_v) = 3.5 / 3.0 / 2.8 at
    15 / 30 / 50 bar. At higher T it deliberately differs from 'old': the pore empties
    (theta_C2 at 50 bar 0.75 at 300 K -> 0.35 at 480 K), so crowding fades, while RT ln p
    grows with T; matching 'old' would need s ~ 6 at 420 K and ~ 9 at 480 K. That growth
    with T is the unphysical part of the double counting.

    Caveat (audit M5): gamma divides the lumped forward elimination constant, whose barrier
    is the post-release TS. This is a demonstration-level approximation; the consistent model
    would split the elimination (CatCn(C2) <-> Ni-H(C2) + Cn; Ni-H(C2) + C2 <-> CatC2(C2)).
    Classical DFT would replace gamma_n with real excess chemical potentials."""
    if crowding == 'pressure':
        return float(p)                                     # p0 = 1 bar
    # theta (default None = ethylene-only Langmuir filling) lets product_crowding
    # pass the total site fraction (ethylene + products)
    vac = 1.0 - (ethylene_filling(T, p) if theta is None else theta)
    s = {'lattice': 1, 'multisite': n / 2,
         'pocket': S_POCKET if s_pocket is None else s_pocket}[crowding]
    return vac ** -s


class InconsistentNetworkError(ValueError):
    """The reduced reaction network does not represent the full DFT profile at (T, p)."""


NETWORK_TOL = 0.0      # kJ/mol: largest allowed barrier underestimate before a point counts as inconsistent


@functools.lru_cache(maxsize=None)
def network_consistency(T, p, tol=NETWORK_TOL):
    """Is the reduced network (resting states CatCn(C2), ethylene bound) kinetically
    consistent with the full DFT profile at the operating (T, p)?
    The reduced steps start from the ethylene-bound alkyls (states 2, 7, 10 and 11 + ethylene). When
    ethylene binding becomes endergonic (high T, low p) the bare alkyl, or an intermediate on the
    elimination path, lies lower and is the real resting state; the reduced barriers are then
    measured from too high a state and come out too low by that gap (energetic span). The
    thermodynamics is exact by construction (each reduced cycle connects the same profile states),
    so only these kinetic gaps are checked. The full profile is taken at the operating pressure,
    because ethylene binding to Ni physically depends on it, whatever pressure treatment the run uses.
    Returns (ok, max_gap_kJ_mol, list of messages for gaps > tol)."""
    G = EnergyBarriersSimplifyed(float(T), float(p)).energy_profile
    bound8 = G['11'] + G['10'] - G['8']           # octyl + ethylene, as in elimination_tetra
    gaps = {
        'insertion C2->C4 (ethylene on Ni-ethyl, G2 - G1)': G['2'] - G['1'],
        'insertion C4->C6 (ethylene on Ni-butyl, G7 - G3)': G['7'] - G['3'],
        'insertion C6->C8+ (ethylene on Ni-hexyl, G10 - G8)': G['10'] - G['8'],
        'elimination C4 (G7 - min(G3, G4))': G['7'] - min(G['3'], G['4']),
        'elimination C6 (G10 - min(G8, G9))': G['10'] - min(G['8'], G['9']),
        'elimination C8+ (bound C8 - min(G11, G12))': bound8 - min(G['11'], G['12']),
    }
    msgs = [f'{k}: barrier too low by {v:.1f} kJ/mol' for k, v in gaps.items() if v > tol]
    return not msgs, max(0.0, max(gaps.values())), msgs


def energy_model(T, p, model):
    """fit: barrier model of simulate() for the given pressure treatment (factored out)."""
    if model == 'ideal':
        return EnergyBarriersEthylenePressure(temperature=T, pressure=p)
    return EnergyBarriersSimplifyed(temperature=T, pressure=P0_CODE if model == 'pore' else p)


# BEP barrier model (elim_map='bep'): DFT points per step, the chain-length index n of the
# reacting CatCn(C2) for each set (insertion CatCn -> CatCn+2, elimination CatCn -> Cn)
BEP_POINTS = {'insertion': {2: 'di', 4: 'tri', 6: 'tetra'}, 'elimination': {4: 'di', 6: 'tri', 8: 'tetra'}}


def bep_barriers(bp, n_max):
    """Brønsted-Evans-Polanyi barriers for every step, from the three DFT points per step type.
    bp: barrier_params (dG_TS in kJ/mol, keys '<step>_<set>' and 'de<step>_<set>').
    1. Reaction free energy dG_n = dG_TS_f - dG_TS_r at the DFT points. Beyond the last point the
       increments decay geometrically with the ratio of the two DFT increments (group additivity:
       the end-group effect fades and dG_n saturates); a ratio outside (0, 1) means no trend, and
       dG_n stays at the last DFT value.
    2. BEP line dG_TS_f = a + alpha dG (least squares over the three DFT points, alpha clipped to [0, 1]);
       every step, the DFT ones included, takes the line value, and dG_TS_r = dG_TS_f - dG_n.
    Returns {step: {n: (dG_TS_f, dG_TS_r)}} and {step: (a, alpha, {n: dG_n})}."""
    out, fit = {}, {}
    last = {'insertion': n_max - 2, 'elimination': n_max}
    for step, pts in BEP_POINTS.items():
        ns = sorted(pts)
        gf = np.array([bp[f'{step}_{pts[n]}'] for n in ns])
        gr = np.array([bp[f'de{step}_{pts[n]}'] for n in ns])
        dg = dict(zip(ns, gf - gr))
        d1, d2 = dg[ns[1]] - dg[ns[0]], dg[ns[2]] - dg[ns[1]]
        q = d2 / d1 if d1 != 0 else 0.0
        q = q if 0 < q < 1 else 0.0
        inc = d2
        for n in range(ns[-1] + 2, last[step] + 1, 2):
            inc *= q
            dg[n] = dg[n - 2] + inc
        alpha, a = np.polyfit(gf - gr, gf, 1)
        alpha = float(np.clip(alpha, 0.0, 1.0))
        a = float(np.mean(gf - alpha * (gf - gr)))
        out[step] = {n: (a + alpha * g, a + alpha * g - g) for n, g in dg.items()}
        fit[step] = (a, alpha, dg)
    return out, fit


def rate_consts(T, p, model):
    """fit: raw rate constants k_<step>_<set> = (forward, reverse) as simulate() builds
    them (before elim_map, crowding and normalisation)."""
    return prepare_constants_chemisorp_simplified(energy_model(T, p, model).get_energy_barriers(),
                                                  temperature=T)


# product_crowding: self-consistent total site fraction (ethylene + products).
# Units: C2_MODEL.predict_absolute = nmax(T) * theta, i.e. the pore concentrations (C2 and every
# Cn pore species in the rate laws) are in the same units as the Langmuir saturation capacity
# nmax(T) (C2_MODEL.nmax_model). Hence sites occupied = [C2] + sum_n (n/2) [Cn]_pore (an olefin
# Cn occupies n/2 ethylene-sized sites) and theta_tot = sites / nmax(T), clipped at THETA_TOT_MAX.
# The Ni-bound CatCn(C2) species (CAT = 1e-2 vs nmax ~ 1e2) are negligible and ignored.
THETA_TOT_MAX = 0.98
PC_MAX_ITER = 4          # reruns after the plain (ethylene-only) run
PC_TOL = 0.02            # converged when <n> changes less than this between successive runs
PC_BATCH_WEIGHT = 'release'   # batch: 'release' = theta averaged with olefin-release weights,
                              # 'final' = last time point (ethylene depleted -> theta ~ 0)
LAST_PC_INFO = {}        # filled by simulate(product_crowding=True): iterations, converged, ...


def _theta_total(ts, T, mode):
    """total pore site fraction (ethylene + n/2 per olefin Cn) from a run's time series.
    flow: last time point (steady state). batch: ethylene is consumed, so the final pore holds
    almost no ethylene and the crowding that acts while olefins are released is lost; the site
    fraction is therefore averaged over time with the olefin formation rate as weight
    (PC_BATCH_WEIGHT='release'; 'final' = last point only)."""
    nmax = float(C2_MODEL.nmax_model.predict(np.array([float(T)]))[0])
    pore = [c for c in ts.columns if re.fullmatch(r'C\d+', c)]
    sites = sum(int(c[1:]) / 2.0 * ts[c].to_numpy(float) for c in pore) / nmax
    if mode == 'flow' or PC_BATCH_WEIGHT == 'final':
        th = sites[-1]
    else:
        tot = sum(ts[c].to_numpy(float) for c in ts.columns
                  if re.fullmatch(r'C\d+(_out)?', c) and c not in ('C2', 'C2_out'))
        w = np.clip(np.diff(tot), 0, None)
        th = float(np.dot(w, 0.5 * (sites[1:] + sites[:-1])) / w.sum()) if w.sum() > 0 else sites[-1]
    return float(min(max(th, 0.0), THETA_TOT_MAX))


def simulate(T, p, mode='batch', particle_nm=None, model='pore', hours=1.0,
             n_steps=10_000, log_times=False, ads_interp=False, elim_map='tetra', crowding=None,
             s_pocket=None, ldf_alpha=None, n_max=None, site_frac=1.0, c2_lattice=False,
             product_crowding=False, check_network='warn'):
    """Run one case (all knobs: see _simulate_once).
    check_network: 'warn' (default) runs and warns when the reduced network is
    kinetically inconsistent with the full DFT profile at (T, p) (network_consistency); 'filter'
    does not run such a point and raises InconsistentNetworkError with the reason; None skips the check.
    product_crowding=True: self-consistent iteration. Run with the ethylene-only
    theta, take the total pore site fraction theta_tot from that run (_theta_total), recompute
    gamma (pocket, and gamma_C2 if c2_lattice) with theta_v = 1 - theta_tot and rerun, up to
    PC_MAX_ITER reruns or until <n> changes by less than PC_TOL; the update is damped (factor
    0.5) once theta oscillates. Diagnostics of the last call in LAST_PC_INFO."""
    kw = dict(mode=mode, particle_nm=particle_nm, model=model, hours=hours, n_steps=n_steps,
              log_times=log_times, ads_interp=ads_interp, elim_map=elim_map, crowding=crowding,
              s_pocket=s_pocket, ldf_alpha=ldf_alpha, n_max=n_max, site_frac=site_frac,
              c2_lattice=c2_lattice)
    if check_network is not None:
        ok, gap, msgs = network_consistency(float(T), float(p))
        if not ok:
            text = (f'reduced network inconsistent with the full DFT profile at {T:g} K, {p:g} bar '
                    f'(resting state not ethylene-bound): ' + '; '.join(msgs))
            if check_network == 'filter':
                raise InconsistentNetworkError(text)
            warnings.warn(text, stacklevel=2)
    if not product_crowding:
        return _simulate_once(T, p, **kw)
    if crowding is None and not c2_lattice:
        raise ValueError('product_crowding needs crowding and/or c2_lattice (nothing to recompute)')
    theta = ethylene_filling(T, p)
    dist, ts = _simulate_once(T, p, theta=None, **kw)
    n_prev, hist, w, last_step, conv, it = mean_length(dist), [theta], 1.0, 0.0, False, 0
    for it in range(1, PC_MAX_ITER + 1):
        calc = _theta_total(ts, T, mode)
        step = calc - theta
        if last_step * step < 0:
            w = 0.5                                    # oscillating: damp
        last_step = step
        theta = float(min(theta + w * step, THETA_TOT_MAX))
        hist.append(theta)
        dist, ts = _simulate_once(T, p, theta=theta, **kw)
        n_now = mean_length(dist)
        conv = abs(n_now - n_prev) < PC_TOL
        n_prev = n_now
        if conv:
            break
    LAST_PC_INFO.clear()
    LAST_PC_INFO.update(iterations=it, converged=bool(conv), theta=hist, n=n_prev)
    return dist, ts


def _simulate_once(T, p, mode='batch', particle_nm=None, model='pore', hours=1.0,
                   n_steps=10_000, log_times=False, ads_interp=False, elim_map='tetra', crowding=None,
                   s_pocket=None, ldf_alpha=None, n_max=None, site_frac=1.0, c2_lattice=False,
                   theta=None):
    """Run one case. particle_nm=None means no diffusion (infinite crystal).
    calibration knobs (fit): s_pocket overrides S_POCKET for
    crowding='pocket'; ldf_alpha replaces the LDF constant 15 (sphere) in k_D = alpha D / R^2
    (slab 3, cylinder 8); n_max overrides N_MAX. elim_map='tri': like 'tetra' with the _tri
    (hexene-path) constants for every step.
    more knobs (all default to the previous behaviour):
      site_frac (1.0): active-site fraction, initial CatC2(C2) = CAT * site_frac.
      c2_lattice (False): ethylene as lattice gas too, gamma_C2 = 1/theta_v (same theta_v as the
        pocket); the forward constant of every reaction that consumes C2 (insertions and
        eliminations, CatCn(C2) + C2 <-> ...) is multiplied by gamma_C2, reverse unchanged, so K
        changes by gamma_C2. Only model='pore'.
      product_crowding (False, set in simulate()): theta_v includes the products; theta (internal
        argument) is the total site fraction handed in by that iteration.
    ads_interp=True: pore capacities interpolated linearly in T (AdsInterpT).
    elim_map='legacy': elimination constants as in the legacy reaction list (C4 gets _tri,
    C6+ _tetra); 'per_step': C4 gets _di, C6 _tri, C8+ _tetra (each olefin its own barrier);
    'tetra' (default since 2026-10-26): every insertion and elimination uses the _tetra
    constants, i.e. one barrier set for all chain lengths.
    'bep': every insertion and elimination from a BEP line through the three DFT points of its
    step type, chain-length dependence of dG_n by group additivity (bep_barriers()).
    crowding (model='ideal' or 'pore'): product release penalised by RT ln gamma_n,
    gamma_n from product_activity(); the forward elimination constant of each
    CatCn(C2) [+ C2] -> CatC2(C2) + Cn is divided by gamma_n (the _tetra elimination TS lies
    after the olefin release), so K_elim changes by exactly 1 / gamma_n. With 'pore' the
    legacy-like network is kept: [C2] and [Cn] stay in the rate laws as pore concentrations
    and only the released olefin gets an activity coefficient (ethylene keeps gamma = 1).
    Returns (distribution DataFrame [n, concentration, probability], time series with
    the index in real seconds)."""
    if model == 'ideal' and mode != 'flow':
        raise ValueError("model='ideal' needs mode='flow' (ethylene is not consumed)")
    if c2_lattice and model != 'pore':
        raise ValueError("c2_lattice needs model='pore' (ethylene in the rate laws)")
    eb = energy_model(T, p, model)
    # 'gas': pressure only through the DFT Gibbs profile; the pore ethylene is held at its
    # 1 bar GCMC value, so 'gas' and 'pore' are identical at 1 bar
    pa = np.array([(1.0 if model == 'gas' else p) * 101325.0])
    tk = np.array([float(T)])
    init = {'CatC2(C2)': CAT * site_frac, 'C2': float(C2_MODEL.predict_absolute(pa, tk)[0]),
            'C2_out': float(C2_MODEL.predict_out(pa, tk)[0])}
    cls = ReactorFlow if mode == 'flow' else ReactorBatch
    r = cls(N=n_max or N_MAX, T=T, pressure=p, L=particle_nm or 300, time_hours=hours, n_steps=n_steps,
            diff_model_path=DIFF_MODEL, ads_model_path=ADS_MODEL, energy_model=eb,
            initial_concs=init, include_diffusion=particle_nm is not None, log_times=log_times)
    if ads_interp:
        r.ads_model = AdsInterpT(r.ads_model)
    r.solver = r._build_solver()
    if model == 'ideal':
        # ethylene counted once.
        # ethylene leaves the network: CatCn + C2 <-> ... becomes CatCn <-> ...; its activity
        # p/p0 is already in the DFT Gibbs profile, so pressure is counted once
        r._reactions = [(re.sub(r' \+ C2 (?=<->)', ' ', rx[0]),) + tuple(rx[1:]) for rx in r._reactions]
        fixed = ['C2', 'C2_out'] if mode == 'flow' else None
        r.solver = Reaction_Diffusion(reactions=r._reactions, capacity=r._capacity,
                                      fixed_concentrations=fixed)
    if elim_map in ('tetra', 'tri'):
        # current choice, see docs/model.md.
        # one barrier set for every step (insertion and elimination both _tetra, or both _tri)
        b = elim_map
        for i, rx in enumerate(r._reactions):
            if re.match(r'CatC\d+\(C2\)( \+ C2)? <-> CatC\d+\(C2\)$', rx[0]):
                r._reactions[i] = (rx[0], r.consts[f'k_insertion_{b}'], r.consts[f'k_deinsertion_{b}'])
            elif re.match(r'CatC\d+\(C2\)( \+ C2)? <-> CatC2\(C2\) \+ C\d+$', rx[0]):
                r._reactions[i] = (rx[0], r.consts[f'k_elimination_{b}'], r.consts[f'k_deelimination_{b}'])
        r.solver = Reaction_Diffusion(reactions=r._reactions, capacity=r._capacity,
                                      fixed_concentrations=['C2', 'C2_out'] if mode == 'flow' else None)
    if elim_map == 'per_step':
        # elimination off-by-one in the legacy reaction list (docs/model.md).
        # the legacy list gives CatC4 the _tri (hexene, from state 10) and CatC6 the _tetra
        # (octene) elimination constants; _di is the butene path from state 7 = CatC4(C2)
        own = {4: 'di', 6: 'tri'}
        for i, rx in enumerate(r._reactions):
            m = re.match(r'CatC(\d+)\(C2\)( \+ C2)? <-> CatC2\(C2\) \+ C\d+$', rx[0])
            if m and int(m.group(1)) in own:
                s = own[int(m.group(1))]
                r._reactions[i] = (rx[0], r.consts[f'k_elimination_{s}'], r.consts[f'k_deelimination_{s}'])
        r.solver = Reaction_Diffusion(reactions=r._reactions, capacity=r._capacity,
                                      fixed_concentrations=['C2', 'C2_out'] if mode == 'flow' else None)
    if elim_map == 'bep':
        # BEP line through the DFT points, dG_n extrapolated by group additivity (bep_barriers)
        bep, _ = bep_barriers(r.barrier_params, r.N)
        for i, rx in enumerate(r._reactions):
            m = (re.match(r'CatC(\d+)\(C2\)( \+ C2)? <-> CatC\d+\(C2\)$', rx[0]) and 'insertion') or                 (re.match(r'CatC(\d+)\(C2\)( \+ C2)? <-> CatC2\(C2\) \+ C\d+$', rx[0]) and 'elimination')
            if m:
                gf, gr = bep[m][int(re.match(r'CatC(\d+)', rx[0]).group(1))]
                r._reactions[i] = (rx[0], Eyring_k(gf, T), Eyring_k(gr, T))
        r.solver = Reaction_Diffusion(reactions=r._reactions, capacity=r._capacity,
                                      fixed_concentrations=['C2', 'C2_out'] if mode == 'flow' else None)
    if crowding is not None:
        # product crowding: forward elimination divided by gamma_n
        if model not in ('ideal', 'pore') or crowding not in CROWDING:
            raise ValueError(f"crowding needs model='ideal' or 'pore' and one of {CROWDING}, "
                             f"got {model!r}, {crowding!r}")
        for i, rx in enumerate(r._reactions):
            m = re.match(r'CatC\d+\(C2\)(?: \+ C2)? <-> CatC2\(C2\) \+ C(\d+)$', rx[0])
            if m:
                g = product_activity(int(m.group(1)), T, p, crowding, s_pocket, theta)
                r._reactions[i] = (rx[0], rx[1] / g, rx[2])
        r.solver = Reaction_Diffusion(reactions=r._reactions, capacity=r._capacity,
                                      fixed_concentrations=['C2', 'C2_out'] if mode == 'flow' else None)
    if c2_lattice:
        # ethylene activity coefficient gamma_C2 = 1/theta_v on the forward constant of
        # every reaction that consumes C2 (insertions, eliminations); reverse unchanged
        vac = 1.0 - (ethylene_filling(T, p) if theta is None else theta)
        for i, rx in enumerate(r._reactions):
            if re.match(r'CatC\d+\(C2\) \+ C2 <->', rx[0]):
                r._reactions[i] = (rx[0], rx[1] / vac, rx[2])
        r.solver = Reaction_Diffusion(reactions=r._reactions, capacity=r._capacity,
                                      fixed_concentrations=['C2', 'C2_out'] if mode == 'flow' else None)
    if ldf_alpha is not None and particle_nm is not None:
        # LDF geometry constant: original code uses 15 (sphere) in run_reactor.py
        for i, rx in enumerate(r._reactions):
            if re.match(r'C\d+ <?D> C\d+_out$', rx[0]):
                r._reactions[i] = (rx[0], rx[1] * ldf_alpha / 15, rx[2] * ldf_alpha / 15)
        r.solver = Reaction_Diffusion(reactions=r._reactions, capacity=r._capacity,
                                      fixed_concentrations=['C2', 'C2_out'] if mode == 'flow' else None)
    f = normilize_constants(r._reactions)
    r.time_hours /= f
    dist, ts = r.run(build_solver=False)
    ts = ts.copy()
    ts.index = ts.index * f
    return dist, ts


PRESSURE_COUNTING = {'pore': 'pore', 'gas': 'ideal'}          # run_case name -> simulate model
BARRIER_SCHEMES = {'tetra': 'tetra', 'individual': 'per_step', 'legacy': 'legacy'}   # -> elim_map


def run_case(T, p, counting='pore', barriers='tetra', mode='batch', particle_nm=None, hours=1.0,
             n_steps=10_000, log_times=False):
    """One model run with the option names of run_model.py (see docs/model.md).
    counting: 'pore' = DFT profile at p0, ethylene enters the rate laws through its GCMC pore
              concentration at (p, T); 'gas' = ideal-gas ethylene, only ethylene moved to p in the
              DFT profile (RT ln p/p0 per free ethylene) and removed from the rate laws (flow only).
    barriers: 'tetra' = _tetra constants for every insertion and elimination; 'individual' = each
              step its own DFT barrier (C4 elimination _di, C6 _tri, C8+ _tetra); 'legacy' = the
              legacy list (eliminations shifted by one olefin).
    Returns (dist, ts) as simulate()."""
    if counting not in PRESSURE_COUNTING:
        raise ValueError(f"counting must be one of {list(PRESSURE_COUNTING)}, got {counting!r}")
    if barriers not in BARRIER_SCHEMES:
        raise ValueError(f"barriers must be one of {list(BARRIER_SCHEMES)}, got {barriers!r}")
    if counting == 'gas' and mode != 'flow':
        raise ValueError("counting='gas' needs mode='flow': ethylene is a constant-pressure reservoir "
                         "and is not consumed")
    return simulate(T, p, mode=mode, particle_nm=particle_nm, model=PRESSURE_COUNTING[counting],
                    hours=hours, n_steps=n_steps, log_times=log_times,
                    elim_map=BARRIER_SCHEMES[barriers])


def mean_length(dist):
    d = dist[dist.n > 2].set_index('n')['probability'].clip(lower=0)
    d = d / d.sum()
    return float((d.index * d).sum())

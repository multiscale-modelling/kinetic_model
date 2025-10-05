import numpy as np
import pandas as pd
import re
from joblib import load

from kinetic_model.constants.diffusion import *
from kinetic_model.constants.adsorption import *
from kinetic_model.simulation.reaction_list_creator import (
    make_reaction_list_basic,
    prepare_constants_basic,
    make_reaction_list_chemisorp,
    make_reaction_list_chemisorp_recomb,
    prepare_constants_chemisorp,
    make_reaction_list_chemisorp_simplified,
    prepare_constants_chemisorp_simplified
)
from kinetic_model.core.diffusion import Reaction_Diffusion
from kinetic_model.constants.energy_profile import EnergyPlaceholder


class ReactorSimulation:
    """
    Basic ReactorSimulation (ancestor interface).
    run() returns (distribution_df, probability_time_series_df).
    """
    def __init__(
        self,
        N: int,
        T: float,
        pressure: float,
        L: float,
        delta_ins: float,
        delta_ads: float,
        time_hours: float,
        n_steps: int,
        initial_concs: dict,
        diff_model_path: str,
        ads_model_path: str,
        energy_model: object,
    ):
        self.N = N
        self.T = T
        self.pressure = pressure
        self.L = L
        self.delta_ins = delta_ins
        self.delta_ads = delta_ads
        self.time_hours = time_hours
        self.n_steps = n_steps
        self.initial_concs = initial_concs

        # load ML models & basic barrier params
        self.diff_model = load(diff_model_path)
        self.ads_model  = load(ads_model_path)
        self.energy_model = energy_model#(pressure=pressure, temperature=T)
        self.barrier_params = self.energy_model.get_energy_barriers()

    def _make_dataframe(self):
        idx = list(range(2, self.N + 1, 2))
        df = pd.DataFrame({
            'pressure_bar': [self.pressure] * len(idx) * 2,
            'temperature':  [self.T]       * len(idx) * 2,
            'n':             idx + idx,
            'adsorption_type_num': [0]*len(idx) + [1]*len(idx),
        })
        return df, idx

    def _calc_diffusion_constants(self, df, idx):
        dvals = self.diff_model.predict(df)  # m²/s
        rates = 15 * dvals / (self.L**2 * 1e-9)      # s⁻¹
        return {n: rates[i] for i, n in enumerate(idx)}

    def _calc_capacity(self, df, idx):
        ads = self.ads_model.predict(df)
        cap = {f'C{n}': ads[i] for i, n in enumerate(idx)}
        half = len(idx)
        for i, n in enumerate(idx):
            cap[f'C{n}_out'] = ads[i] - ads[i + half]
        return cap

    def _build_solver(self):
        df, idx = self._make_dataframe()
        diffusion_rates = self._calc_diffusion_constants(df, idx)
        capacity       = self._calc_capacity(df, idx)

        consts = prepare_constants_basic(
            self.barrier_params,
            n_max=self.N,
            diffusion_rates=diffusion_rates,
            T=self.T,
            delta_ins=self.delta_ins,
            delta_ads=self.delta_ads,
        )
        reactions = make_reaction_list_basic(
            self.N,
            consts,
            include_diffusion=True
        )
        return Reaction_Diffusion(reactions=reactions, capacity=capacity)

    @staticmethod
    def _calc_distribution(result):
        pat = re.compile(r'\bC(\d+)')
        cols = [c for c in result.columns if pat.match(c)]
        sub = result[cols].iloc[-1]
        dfm = (
            sub.reset_index(name='concentration')
               .assign(n=lambda d: d['index'].str.extract(r'(\d+)').astype(int))
               .groupby('n', as_index=False)['concentration'].sum()
        )
        total = dfm.loc[dfm['n'] > 2, 'concentration'].sum()
        dfm['probability'] = dfm['concentration'] / total
        dfm.loc[dfm['n'] == 2, 'probability'] = np.nan
        return dfm

    def run(self, build_solver: bool = True):
        if build_solver:
            self.solver = self._build_solver()
        times  = np.linspace(0, self.time_hours * 3600, self.n_steps)
        self._result = self.solver.solve(
            initial_concs_dict=self.initial_concs,
            time_points=times
        )
        # self._result.index = times
        # final distribution
        dist = self._calc_distribution(self._result)
        # time-series probabilities
        olefin_cols = [c for c in self._result.columns if re.match(r'C\d+(_out)?$', c)]
        ts = self._result[olefin_cols]
        # new:
        # combined = ts.T \
        #             .groupby(lambda col: int(re.search(r'(\d+)', col).group(1))) \
        #             .sum() \
        #             .T

        #prob_ts = combined.div(combined.sum(axis=1), axis=0)
        
        return dist, ts


class ReactorSimulationChemisorption(ReactorSimulation):
    """
    Chemisorption variant, inherits ReactorSimulation interface.
    Uses chemisorption barrier params and reaction list.
    """
    def __init__(
        self,
        N: int,
        T: float,
        pressure: float,
        L: float,
        delta_ins: float = 0.0,
        delta_ads: float = 0.0,              # this will be delta_ads_coord_cat_cn
        delta_recomb: float = 0.0,
        time_hours: float = 2.0,
        n_steps: int = 10000,
        initial_concs: dict = {'Cat': 1e-2, 'C2': 1},
        diff_model_path: str = '../data/processed/diffusion_model.pkl',
        ads_model_path: str = '../data/processed/knn_adsorption_model.pkl',
        energy_model: object = EnergyPlaceholder,
        include_diffusion: bool = True
    ):
        # call ancestor init with delta_ads = delta_ads_coord_cat_cn
        super().__init__(
            N, T, pressure, L,
            delta_ins,
            delta_ads,      # will serve as delta_ads_coord_cat_cn
            time_hours,
            n_steps,
            initial_concs,
            diff_model_path,
            ads_model_path,
            energy_model
        )
        self.delta_recomb = delta_recomb
        # override barrier_params to use chemisorption ones
        self.barrier_params = self.energy_model.get_energy_barriers()
        self.include_diffusion = include_diffusion

    def _build_solver(self):
        # reuse dataframe, diffusion, capacity from parent
        df, idx = self._make_dataframe()
        self._diffusion_rates = self._calc_diffusion_constants(df, idx)
        self._capacity       = self._calc_capacity(df, idx)

        self.consts = prepare_constants_chemisorp(
            self.barrier_params,
            n_max=self.N,
            delta_ins=self.delta_ins,
            delta_ads_coord_cat_cn=self.delta_ads,
            delta_recomb=self.delta_recomb,
            diffusion_rates=self._diffusion_rates,
            temperature=self.T
        )
        self._reactions = make_reaction_list_chemisorp_recomb(
            n_max=self.N,
            consts=self.consts,
            diffusion_rates=self._diffusion_rates,
            include_diffusion=self.include_diffusion
        )
        return Reaction_Diffusion(reactions=self._reactions, capacity=self._capacity)



class ReactorSimulationChemisorptionSimplified(ReactorSimulation):
    """
    Simplified chemisorption variant, inherits ReactorSimulation interface.
    Uses simplified chemisorption barrier params and reaction list.
    """
    def __init__(
        self,
        N: int,
        T: float, 
        pressure: float,
        L: float,
        time_hours: float = 2.0,
        n_steps: int = 10000,
        initial_concs: dict = {'CatC2(C2)': 1e-2, 'C2': 1},
        diff_model_path: str = '../data/processed/diffusion_model.pkl',
        ads_model_path: str = '../data/processed/knn_adsorption_model.pkl',
        energy_model: object = EnergyPlaceholder,
        include_diffusion: bool = True
    ):
        super().__init__(
            N, T, pressure, L,
            None,
            None,
            time_hours,
            n_steps,
            initial_concs,
            diff_model_path,
            ads_model_path,
            energy_model
        )
        self.include_diffusion = include_diffusion

    def _build_solver(self):
        df, idx = self._make_dataframe()
        self._diffusion_rates = self._calc_diffusion_constants(df, idx)
        self._capacity = self._calc_capacity(df, idx)

        self.consts = prepare_constants_chemisorp_simplified(
            self.barrier_params,
            temperature=self.T
        )
        self._reactions = make_reaction_list_chemisorp_simplified(
            n_max=self.N,
            consts=self.consts,
            diffusion_rates=self._diffusion_rates,
            include_diffusion=self.include_diffusion

        )
        return Reaction_Diffusion(reactions=self._reactions, capacity=self._capacity)
    


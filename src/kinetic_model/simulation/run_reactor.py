import numpy as np
import pandas as pd
import re
from joblib import load
from typing import Tuple

from kinetic_model.constants.diffusion import DiffusionLinear
from kinetic_model.constants.adsorption import AdsorptionKNN
from kinetic_model.simulation.reaction_list_creator import make_reaction_list, prepare_constants
from kinetic_model.core.diffusion import Reaction_Diffusion
from kinetic_model.constants.energy_profile import EnergyPlaceholder


class ReactorSimulation:
    """
    Encapsulates end-to-end dimerization simulation:
      - loads diffusion & adsorption models
      - prepares kinetics & diffusion constants
      - builds & runs master-equation solver
      - computes product distribution at final time and over time

    Inputs:
      N (int): Maximum oligomer length (# of states)
      T (float): Temperature (K)
      pressure (float): Pressure (bar)
      L (float): Particle size (nm)
      delta_ins (float): Additional insertion barrier (kJ/mol)
      delta_ads (float): Additional adsorption barrier (kJ/mol)
      time_hours (float): Total simulation time (hours)
      n_steps (int): Number of time points
      initial_concs (dict): Initial concentrations, e.g. {'Cat': 1e-2, 'C2': 1.5}
      diff_model_path (str): Path to joblib'd diffusion model
      ads_model_path (str): Path to joblib'd adsorption model

    Outputs:
      distribution (pd.DataFrame): Final-state probability vs. n
      time_series (pd.DataFrame): Time-evolution of olefin distributions
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
        
        # load ML models
        self.diff_model = load(diff_model_path)
        self.ads_model = load(ads_model_path)
        # energy barrier placeholder
        self.energy_model = EnergyPlaceholder(pressure=self.pressure, temperature=self.T)
        self.barrier_params = self.energy_model.get_params()
        self.results = None

    def _make_dataframe(self) -> Tuple[pd.DataFrame, list]:
        # prepare feature table for ML predictions
        idx = list(range(2, self.N + 1, 2))
        df = pd.DataFrame({
            'pressure_bar': [self.pressure] * len(idx) * 2,
            'temperature': [self.T] * len(idx) * 2,
            'n': idx + idx,
            'adsorption_type_num': [0] * len(idx) + [1] * len(idx),
        })
        return df, idx

    def _calc_diffusion_constants(self, df: pd.DataFrame, idx: list) -> dict:
        # predict D_TST and convert to rates (m2/s -> s^-1 via L)
        dvals = self.diff_model.predict(df)
        # convert nm to m: L*1e-9
        rates = dvals / (self.L * 1e-9)
        return {idx[i]: rates[i] for i in range(len(idx))}

    def _calc_capacity(self, df: pd.DataFrame, idx: list) -> dict:
        ads = self.ads_model.predict(df)
        cap = {f'C{n}': ads[i] for i, n in enumerate(idx)}
        # outside capacity = difference excess - absolute
        half = len(idx)
        for i, n in enumerate(idx):
            cap[f'C{n}_out'] = ads[i] - ads[i + half]
        return cap

    def _build_solver(self) -> Reaction_Diffusion:
        # build solver with diffusion and adsorption capacities
        df, idx = self._make_dataframe()
        diffusion_rates = self._calc_diffusion_constants(df, idx)
        capacity = self._calc_capacity(df, idx)
        # prepare reaction constants
        consts = prepare_constants(
            self.barrier_params,
            delta_ins=self.delta_ins,
            delta_ads=self.delta_ads,
            n_max=self.N,
            diffusion_rates=diffusion_rates,
            T=self.T,
        )
        # full reaction list
        reactions = make_reaction_list(
            self.N,
            consts,
            include_diffusion=True
        )
        # instantiate reactor solver
        return Reaction_Diffusion(reactions=reactions, capacity=capacity)

    @staticmethod
    def _calc_distribution(df_last: pd.DataFrame) -> pd.DataFrame:
        # extract Cn and Cn_out columns
        pat = re.compile(r'\bC(\d+)')
        cols = [c for c in df_last.columns if pat.match(c)]
        # use last row for final distribution
        sub = df_last[cols].iloc[-1]
        # group by n
        dfm = (
            sub.reset_index(name='concentration')
               .assign(n=lambda d: d['index'].str.extract(r'(\d+)').astype(int))
               .groupby('n', as_index=False)['concentration'].sum()
        )
        # normalize probabilities (exclude n=2 if blocked)
        total = dfm.loc[dfm['n'] > 2, 'concentration'].sum()
        dfm['probability'] = dfm['concentration'] / total
        dfm.loc[dfm['n'] == 2, 'probability'] = np.nan
        return dfm

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # build solver
        solver = self._build_solver()
        # time grid in seconds
        time = np.linspace(0, self.time_hours * 3600, self.n_steps)
        # solve
        self.result = solver.solve(
            initial_concs_dict=self.initial_concs,
            time_points=time
        )
        # final distribution
        dist = self._calc_distribution(self.result)
        # time-series of olefin distributions
        olefins = [c for c in self.result.columns if re.match(r'C\d+(_out)?$', c)]
        ts = self.result[olefins].copy()
        # combine inside and outside per n and normalize per time
        combined = []
        for row in ts.itertuples(index=False):
            # row is a namedtuple; convert to pandas Series for grouping
            s = pd.Series(row, index=ts.columns)
            ser = s.groupby(lambda c: int(re.search(r'(\d+)', c).group(1))).sum()
            combined.append(ser)
        ts_df = pd.DataFrame(combined, index=time)
        prob_ts = ts_df.div(ts_df.sum(axis=1), axis=0)

        self.solver = solver

        return dist, prob_ts


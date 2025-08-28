# src/kinetic_model/constants/energy_profile.py
import os
import pandas as pd
from kinetic_model.energy_profile.energy_processing import calculate_relative_energies
import numpy as np

class EnergyBarriers:
    """
    Compute and store energy profiles and barriers for a given kinetic model.
    """

    def __init__(
        self,
        temperature: float,
        pressure: float,
    ):
        # Store inputs
        models = ['QM-QM_TPSSh-PBED3']   # Options: ['QM-QM_M06L-PBED3', 'QM-QM_TPPSh-PBED3]  
        pathways = ['trigonal']         # Options: ['trigonal', 'hexagonal']
        ref_struc = '1'                 # Choose either 1 or 5
        energy_unit = 'kJ/mol'              # Options: ['eV', 'kJ/mol', 'kcal/mol']

        custom_order = ["1", "2", "3", "4", "5", "5a", "6", "1p", "7", "8", "9", "5p", "6p", "1pp", "10", "11", "12", "5pp", "6pp", "1ppp", "2-3", "3-4", "6-1p", "7-8", "8-9", "6p-1pp", "10-11", "11-12", "6pp-1ppp"]  # Use for sorting the intermediates in a table (or pd.DataFrame)

        # Calculate relative energies
        self.rel = calculate_relative_energies(
            structures_dir=os.path.abspath('../data/structures'),
            molecules_dir=os.path.abspath('../data/gas-phase-molecules'),
            temperatures=[temperature],
            models=models,
            ref_struc=ref_struc,
            pathways=pathways,
            pressures=[pressure * 101325.0]  # Convert bar to Pa
        )

        # Filter and sort
        filt = self.rel[self.rel['order'].isin(custom_order)].copy()
        filt['order'] = pd.Categorical(
            filt['order'], categories=custom_order, ordered=True
        )
        self._rel_filtered = filt.sort_values('order')

        # Build energy profile DataFrame
        df = self._rel_filtered.loc[:, ['order', 'rel_G_eV']].copy()
        if energy_unit == 'kJ/mol':
            df['rel_G'] = df['rel_G_eV'] * 96.485
        elif energy_unit == 'kcal/mol':
            df['rel_G'] = df['rel_G_eV'] * 23.061
        else:
            df['rel_G'] = df['rel_G_eV']
        # df = df.set_index('system_name')['rel_G']
        df = df.set_index('order')['rel_G']
        self.energy_profile = df.to_dict()
        # Compute barriers
        self.energy_barriers = {
            # 1) Coordination at catalysis
            'coordination_Cat_C2':        self.energy_profile['6'] - self.energy_profile['5'],
            'decoordination_Cat_C2':     - (self.energy_profile['6'] - self.energy_profile['5']),
            'coordination_Cat_Cn':        self.energy_profile['4'] - self.energy_profile['5'],
            'decoordination_Cat_Cn':     - (self.energy_profile['4'] - self.energy_profile['5']),
            'coordination_CatC2_C2':     self.energy_profile['2'],
            'decoordination_CatC2_C2':  - self.energy_profile['2'],
            'coordination_CatCn_C2':     self.energy_profile['7'] - self.energy_profile['3'],
            'decoordination_CatCn_C2':  - (self.energy_profile['7'] - self.energy_profile['3']),
            'coordination_CatC2_Cn':     self.energy_profile['5a'] - self.energy_profile['4'],
            'decoordination_CatC2_Cn':  - (self.energy_profile['5a'] - self.energy_profile['4']),

            # 2) Chemisorption
            'chemisorption_Cat_C2':      self.energy_profile['6-1p'] - self.energy_profile['6'],
            'dechemisorption_Cat_C2':    self.energy_profile['6-1p'] - self.energy_profile['1p'],
            'chemisorption_Cat_Cn':      self.energy_profile['3-4'] - self.energy_profile['4'],
            'dechemisorption_Cat_Cn':    self.energy_profile['3-4'] - self.energy_profile['3'],

            # 3) Insertion of C2 into CatCn
            'insertion_CatCn_C2':        self.energy_profile['2-3'] - self.energy_profile['2'],
            'deinsertion_CatCn_C2':      self.energy_profile['2-3'] - self.energy_profile['3'],

            # 4) Recombination (CatCn(C2) ↔ Cat(C2)Cn)
            # 'recombination_CatCn_C2':    self.energy_profile['TS_7-8'] - self.energy_profile['7'],
            # 'derecombination_CatCn_C2':  self.energy_profile['TS_7-8'] - self.energy_profile['but-Ni-ethyl'],
            # 5) alternative c2cn adsorbtion throw recombination
            'recombination_Cat_Cn_C2': self.energy_profile['5a'] - self.energy_profile['6'],
            'derecombination_Cat_Cn_C2': - (self.energy_profile['5a'] - self.energy_profile['6']),
            'coordination_Cat_Cn_C2': self.energy_profile['5a'] - self.energy_profile['4'],
            'decoordination_Cat_Cn_C2': - (self.energy_profile['5a'] - self.energy_profile['4']),
        }

    def get_energy_profile(self) -> dict[str, float]:
        """
        Returns a copy of the computed energy profile.
        """
        return dict(self.energy_profile)

    def get_energy_barriers(self) -> dict[str, float]:
        """
        Returns a copy of the computed energy barriers.
        """
        return dict(self.energy_barriers)


class EnergyBarriersSimplifyed(EnergyBarriers):
    def __init__(self, temperature: float, pressure: float):
        super().__init__(temperature, pressure)
        # Override energy barriers with custom values
        lowest_way = np.min([            self.energy_profile['5'],
            self.energy_profile['5a'],
            self.energy_profile['5']])
        
        highest_elimination_barrier = np.max([
            lowest_way,
            self.energy_profile['3-4'],
            self.energy_profile['6-1p'],
        ])
        self.energy_barriers = {            # 3) Insertion of C2 into CatCn
        'insertion':        self.energy_profile['2-3'] - self.energy_profile['2'],
        'deinsertion':      self.energy_profile['2-3'] - self.energy_profile['7'],
        'elimination':      highest_elimination_barrier - self.energy_profile['7'],
        'deelimination':   highest_elimination_barrier - self.energy_profile['1p'] + \
        (self.energy_profile['1'] - self.energy_profile['2'])
        }




        
class EnergyPlaceholder:
    def __init__(self, pressure: float, temperature: float):
        # store inputs in case you extend this later
        self.pressure = pressure
        self.temperature = temperature

    def get_params_basic(self) -> dict:
        """
        Return the barrier parameters.
        """
        # return a copy to prevent accidental mutation
        return {
            'C2_ins':  84.382, 'C2_den': 100.151,
            'C2_ads': -73.930, 'C2_des':  73.930,
            'C4_ads': -78.151, 'C4_des':  78.151,
        }

    def get_chemisorb_params(self) -> dict[str, tuple[float, float]]:
        """
        Detailed barrier parameters for each unique reversible step.
        Keys are descriptive; values are (ΔG‡_fwd, ΔG‡_rev) in kJ/mol.
        """
        return {
            # 1) Coordination at catalysis
            'coordination_Cat_C2':       -48.0,
          'decoordination_Cat_C2':        48.0,
            'coordination_Cat_Cn':       -56.0,
           'decoordination_Cat_Cn':       56.0,
            'coordination_CatC2_C2':     -53.0,
          'decoordination_CatC2_C2':      53.0,
            'coordination_CatCn_C2':     -44.0,
          'decoordination_CatCn_C2':      44.0,
            'coordination_CatC2_Cn':     -61.0,
          'decoordination_CatC2_Cn':      61.0,

            # 2) Chemisorption
            'chemisorption_Cat_C2':      6.0,
            'dechemisorption_Cat_C2':    32.0,
            'chemisorption_Cat_Cn':      6.0,
            'dechemisorption_Cat_Cn':    29.0,

            # 3) Insertion of C2 into CatCn
            'insertion_CatCn_C2':        85.0,
            'deinsertion_CatCn_C2':      101.0,

            # 4) Recombination (CatCn(C2) ↔ Cat(C2)Cn)
            'recombination_CatCn_C2':    85.0,
            'derecombination_CatCn_C2':  96.0,
        }

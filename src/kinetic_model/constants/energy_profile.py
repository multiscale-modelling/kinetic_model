# src/kinetic_model/constants/energy_profile.py
import os
import pandas as pd
from kinetic_model.energy_profile.energy_processing import calculate_relative_energies


class EnergyBarriers:
    """
    Compute and store energy profiles and barriers for a given kinetic model.
    """

    def __init__(
        self,
        structures_dir: str,
        molecules_dir: str,
        model: str,
        pathway: str,
        ref_struc: str,
        temperature: float,
        pressure: float,
        custom_order: list[str],
        energy_type: str = 'Gibbs free energy',
        energy_unit: str = 'kJ/mol'
    ):
        # Store inputs
        self.structures_dir = os.path.abspath(structures_dir)
        self.molecules_dir = os.path.abspath(molecules_dir)
        self.model = model
        self.pathway = pathway
        self.ref_struc = ref_struc
        self.temperatures = [temperature]
        self.pressures = [pressure * 101325.0]  # Convert bar to Pa
        self.custom_order = custom_order
        self.energy_type = energy_type
        self.energy_unit = energy_unit

        # Calculate relative energies
        rel = calculate_relative_energies(
            structures_dir=self.structures_dir,
            molecules_dir=self.molecules_dir,
            temperatures=self.temperatures,
            model=self.model,
            ref_struc=self.ref_struc,
            pathway=self.pathway,
            pressures=self.pressures
        )

        # Filter and sort
        filt = rel[rel['order'].isin(self.custom_order)].copy()
        filt['order'] = pd.Categorical(
            filt['order'], categories=self.custom_order, ordered=True
        )
        self._rel_filtered = filt.sort_values('order')

        # Build energy profile DataFrame
        df = self._rel_filtered.loc[:, ['system_name', 'rel_G_eV']].copy()
        if self.energy_unit == 'kJ/mol':
            df['rel_G'] = df['rel_G_eV'] * 96.485
        elif self.energy_unit == 'kcal/mol':
            df['rel_G'] = df['rel_G_eV'] * 23.061
        else:
            df['rel_G'] = df['rel_G_eV']
        df = df.set_index('system_name')['rel_G']
        self.energy_profile = df.to_dict()

        # Compute barriers
        self.energy_barriers = {
            # 1) Coordination at catalysis
            'coordination_Cat_C2':        self.energy_profile['et-Ni-H'] - self.energy_profile['Ni-H'],
            'decoordination_Cat_C2':     - (self.energy_profile['et-Ni-H'] - self.energy_profile['Ni-H']),
            'coordination_Cat_Cn':        self.energy_profile['H-Ni-but'] - self.energy_profile['Ni-H'],
            'decoordination_Cat_Cn':     - (self.energy_profile['H-Ni-but'] - self.energy_profile['Ni-H']),
            'coordination_CatC2_C2':     self.energy_profile['et-Ni-ethyl'],
            'decoordination_CatC2_C2':  - self.energy_profile['et-Ni-ethyl'],
            'coordination_CatCn_C2':     self.energy_profile['et-Ni-butyl'] - self.energy_profile['Ni-butyl'],
            'decoordination_CatCn_C2':  - (self.energy_profile['et-Ni-butyl'] - self.energy_profile['Ni-butyl']),
            'coordination_CatC2_Cn':     self.energy_profile['but-Ni-ethyl'] - self.energy_profile['Ni-ethyl'],
            'decoordination_CatC2_Cn':  - (self.energy_profile['but-Ni-ethyl'] - self.energy_profile['Ni-ethyl']),

            # 2) Chemisorption
            'chemisorption_Cat_C2':      self.energy_profile['TS_6-1p'] - self.energy_profile['et-Ni-H'],
            'dechemisorption_Cat_C2':    self.energy_profile['TS_6-1p'] - self.energy_profile['Ni-ethyl'],
            'chemisorption_Cat_Cn':      self.energy_profile['TS_3-4'] - self.energy_profile['H-Ni-but'],
            'dechemisorption_Cat_Cn':    self.energy_profile['TS_3-4'] - self.energy_profile['Ni-butyl'],

            # 3) Insertion of C2 into CatCn
            'insertion_CatCn_C2':        self.energy_profile['TS_2-3'] - self.energy_profile['et-Ni-ethyl'],
            'deinsertion_CatCn_C2':      self.energy_profile['TS_2-3'] - self.energy_profile['Ni-butyl'],

            # 4) Recombination (CatCn(C2) ↔ Cat(C2)Cn)
            'recombination_CatCn_C2':    self.energy_profile['TS_7-8'] - self.energy_profile['et-Ni-butyl'],
            'derecombination_CatCn_C2':  self.energy_profile['TS_7-8'] - self.energy_profile['but-Ni-ethyl'],
            # 5) alternative c2cn adsorbtion throw recombination
            'recombination_Cat_Cn_C2': self.energy_profile['H-Ni-et-but'] - self.energy_profile['H-Ni-but'],
            'derecombination_Cat_Cn_C2': - (self.energy_profile['H-Ni-et-but'] - self.energy_profile['H-Ni-but']),
            'coordination_Cat_Cn_C2': self.energy_profile['H-Ni-et-but'] - self.energy_profile['et-Ni-H'],
            'decoordination_Cat_Cn_C2': - (self.energy_profile['H-Ni-et-but'] - self.energy_profile['et-Ni-H']),
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

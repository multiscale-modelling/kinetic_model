# src/kinetic_model/constants/energy_profile.py

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
        Keys are descriptive; values are (ΔG‡_fwd, ΔG‡_rev) in kcal/mol.
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

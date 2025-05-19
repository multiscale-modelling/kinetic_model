# Refactored reaction list creator for olefin chain growth
# Insertion uses C2 barriers for all n; adsorption/desorption uses C2 or C4 based on n

import numpy as np
from kinetic_model.core.eyring import Eyring_k


def compute_rate(Ea, T=300, delta_E=None):
    """
    Compute rate constant from activation energy Ea (kJ/mol) at temperature T.
    Optionally adjust Ea by an energy shift delta_E.
    """
    if delta_E is not None:
        Ea = Ea + delta_E if Ea > 0 else delta_E
    return Eyring_k(Ea, T)


def prepare_constants_basic(barrier_params, n_max, diffusion_rates=None, T=300,
                      clamp=(1e-7, 1e7), delta_ins=None, delta_ads=None):
    """
    Build a constants dict:
      - insertion/deinsertion: same for all n, from C2_ins/C2_des
      - adsorption/desorption: C2 for n=2, C4 for n>=4
      - diffusion: per-n if provided
    barrier_params: dict with C2_ins, C2_des, C2_ads, C2_des,
                    C4_ins, C4_des, C4_ads, C4_des
    diffusion_rates: dict mapping even n to diffusion rate
    Returns dict:
      k_ins, k_deins,
      k_ads_2, k_des_2, k_ads_4, k_des_4,
      k_diff_n (optional)
    """
    consts = {}
    # Global insertion from C2
    Ei2 = barrier_params['C2_ins']
    Eden = barrier_params['C2_den']
    consts['k_ins'] = np.clip(compute_rate(Ei2, T, delta_ins), *clamp)
    consts['k_deins'] = np.clip(compute_rate(Eden, T, delta_ins), *clamp)

    # Ads/des for C2 and C4
    Ea2 = barrier_params['C2_ads']
    Eds2 = barrier_params['C2_des']
    consts['k_ads_2'] = np.clip(compute_rate(Ea2, T, delta_ads), *clamp)
    consts['k_des_2'] = np.clip(compute_rate(Eds2, T, delta_ads), *clamp)

    Ea4 = barrier_params.get('C4_ads', Ea2)
    Eds4 = barrier_params.get('C4_des', Eds2)
    consts['k_ads_4'] = np.clip(compute_rate(Ea4, T, delta_ads), *clamp)
    consts['k_des_4'] = np.clip(compute_rate(Eds4, T, delta_ads), *clamp)

    # Diffusion constants per n
    if diffusion_rates:
        for n, kdiff in diffusion_rates.items():
            if 2 <= n <= n_max:
                consts[f'k_diff_{n}'] = np.clip(kdiff, *clamp)

    return consts


def make_reaction_list_basic(n_max, constants, include_diffusion=False):
    """
    Generate reaction list up to chain length n_max (even):
      - insertion: uses constants['k_ins'], ['k_deins']
      - adsorption: uses k_ads_2/k_des_2 for n=2, else k_ads_4/k_des_4
      - diffusion if include_diffusion and k_diff_n present
    Returns list of [reaction_str, k_f, k_r].
    """
    rxns = []
    for n in range(2, n_max+2, 2):
        # adsorption step
        if n == 2:
            k_ads = constants['k_ads_2']
            k_des = constants['k_des_2']
        else:
            k_ads = constants['k_ads_4']
            k_des = constants['k_des_4']
        rxns.append([f'Cat + C{n} <-> CatC{n}', k_ads, k_des])

        # insertion step (same for all n)
        k_ins = constants['k_ins']
        k_deins = constants['k_deins']
        rxns.append([f'CatC{n} + C2 <-> CatC{n+2}', k_ins, k_deins])

        # diffusion step
        diff_key = f'k_diff_{n}'
        if include_diffusion and diff_key in constants:
            kdiff = constants[diff_key]
            rxns.append([f'C{n} <D> C{n}_out', kdiff, kdiff])

    # remove the last insertion beyond n_max
    cutoff = f'C{n_max+2}'
    rxns = [r for r in rxns if cutoff not in r[0]]
    return rxns


from typing import Dict, Tuple, List
from kinetic_model.constants.energy_profile import EnergyPlaceholder

from typing import Dict, Tuple, List


from typing import Dict, Tuple, List
from kinetic_model.constants.energy_profile import EnergyPlaceholder

from typing import Dict, Tuple, List

def prepare_constants_chemisorp(
    barrier_params: Dict[str, Tuple[float, float]],
    n_max: int, delta_ins: float = 0.0, 
    delta_ads_coord_cat_cn: float = 0.0,
    delta_recomb: float = 0.0,
    diffusion_rates: Dict[int, float] = None,
) -> Dict[str, Tuple[float, float]]:
    """
    For each n=2,4,…,N, grab the appropriate ΔG‡_fwd/ΔG‡_rev tuple from barrier_params
    and map it to the exact reaction string.

    Expects barrier_params to contain:
      - 'coordination_Cat_C2'
      - 'coordination_CatC2_C2'
      - 'coordination_CatCn_C2'
      - 'coordination_CatC2_Cn'
      - 'chemisorption_Cat_C2'
      - 'chemisorption_Cat_Cn'
      - 'insertion_CatCn_C2'
      - 'recombination_CatCn_C2'
    """
    consts: Dict[str, Tuple[float, float]] = {}
    

    # 1) Coordination at Cat
    
    for key, val in barrier_params.items():
        if 'insertion' in key:
            consts[f'k_{key}'] = compute_rate(val, T=300, delta_E=delta_ins)
        elif 'recombination' in key:
            consts[f'k_{key}'] = compute_rate(val, T=300, delta_E=delta_recomb)
        elif 'coordination_Cat_Cn' in key:
            consts[f'k_{key}'] = compute_rate(val, T=300, delta_E=delta_ads_coord_cat_cn)
        else:
            consts[f'k_{key}'] = compute_rate(val, T=300)
    # Diffusion constants per n
    if diffusion_rates:
        for n, kdiff in diffusion_rates.items():
            if 2 <= n <= len(diffusion_rates):
                consts[f'k_diff_{n}'] = kdiff
    return consts

def make_reaction_list_chemisorp(
        n_max: int,
    consts: Dict[str, Tuple[float, float]],
    diffusion_rates: Dict[int, float],
    include_diffusion: bool = True
) -> List[Tuple[str, float, float]]:
    """
    Turn the detailed constants dict into a full reaction list,
    appending diffusion steps Cn <D> Cn_out if requested.
    """
    reactions: List[Tuple[str, float, float]] = []

    # 1) coordination
    reactions.append((f"Cat + C2 <-> Cat(C2)", consts['k_coordination_Cat_C2'], consts['k_decoordination_Cat_C2']))
    reactions.append((f"CatC2 + C2 <-> CatC2(C2)", consts['k_coordination_CatC2_C2'], consts['k_decoordination_CatC2_C2']))

    for n in range(4, n_max + 1, 2):
        reactions.append((f"CatC{n} + C2 <-> CatC{n}(C2)", consts[f'k_coordination_CatCn_C2'], consts['k_decoordination_CatCn_C2']))
        reactions.append((f"CatC2 + C{n} <-> CatC2(C{n})", consts[f'k_coordination_CatC2_Cn'], consts['k_decoordination_CatC2_Cn']))
        reactions.append((f"Cat + C{n} <-> Cat(C{n})", consts[f'k_coordination_Cat_Cn'], consts['k_decoordination_Cat_Cn']))
    # 2) chemisorption
    reactions.append((f"Cat(C2) <-> CatC2", consts['k_chemisorption_Cat_C2'], consts['k_dechemisorption_Cat_C2']))
    for n in range(4, n_max + 1, 2):
        reactions.append((f"Cat(C{n}) <-> CatC{n}", consts[f'k_chemisorption_Cat_Cn'], consts['k_dechemisorption_Cat_Cn']))
    
    # 3) insertion
    for n in range(2, n_max, 2):
        reactions.append((f"CatC{n}(C2) <-> CatC{n+2}", consts[f'k_insertion_CatCn_C2'], consts['k_deinsertion_CatCn_C2']))
    
    # 4) recombination
    for n in range(4, n_max + 1, 2):
        reactions.append((f"CatC{n}(C2) <-> CatC2(C{n})", consts[f'k_recombination_CatCn_C2'], consts['k_derecombination_CatCn_C2']))

    # 2) diffusion in/out
    if include_diffusion:
        for n, rate in diffusion_rates.items():
            reactions.append((f"C{n} <D> C{n}_out", rate, rate))

    return reactions

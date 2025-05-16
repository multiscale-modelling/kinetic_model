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


def prepare_constants(barrier_params, n_max, diffusion_rates=None, T=300,
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


def make_reaction_list(n_max, constants, include_diffusion=False):
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


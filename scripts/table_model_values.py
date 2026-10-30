"""Model numbers -> JSON.

B-3 (Li et al. 2016 row): product shares C4/C6/C8 at 300 K, 1 bar, flow (from flow_1bar_300-500K.csv).
B-4: TOF at the experimental conditions (293 K / 20 bar, 308 K / 50 bar; flow, 100 µm), apparent
activation energy from an Arrhenius fit of tof_vs_T_1bar.csv, and the local pressure order
d ln TOF / d ln p at the ends of tof_vs_p_300K.csv (1 and 50 bar).

  python scripts/table_model_values.py [--results-dir DIR] [--no-sim] [--out FILE]

--results-dir reads the scan CSVs from another folder (e.g. results/legacy_barriers, to check that
the old table values are reproduced); --no-sim skips the two extra TOF runs.
"""
import argparse
import json
import os
import sys

CALLER_CWD = os.getcwd()              # km_common changes the working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import km_common as km  # noqa: E402
from run_tof import tof  # noqa: E402

np, pd = km.np, km.pd
RES = os.path.join(km.ROOT, 'results')
SIZE, SIZE_NM = '100 µm', 1e5
R = 8.314462618e-3   # kJ/mol/K


def b3_shares(res):
    d = pd.read_csv(os.path.join(res, 'flow_1bar_300-500K.csv'))
    rows = d[(d['T'] == 300)].set_index('particle')
    return {p: {f'C{n}': round(100 * rows.loc[p, f'P_C{n}'], 1) for n in (4, 6, 8)} for p in rows.index}


def b4_fits(res):
    t = pd.read_csv(os.path.join(res, 'tof_vs_T_1bar.csv'))
    t = t[(t.particle == SIZE) & (t.model == 'pore')].drop_duplicates('T')
    ea = -np.polyfit(1 / t['T'], np.log(t.tof_per_h), 1)[0] * R
    p = pd.read_csv(os.path.join(res, 'tof_vs_p_300K.csv'))
    p = p[(p.particle == SIZE) & (p.model == 'pore')].drop_duplicates('pressure_bar').sort_values('pressure_bar')
    lp, lt = np.log(p.pressure_bar.values), np.log(p.tof_per_h.values)
    return {'Ea_app_kJmol': round(float(ea), 1),
            'order_low_p': round(float((lt[1] - lt[0]) / (lp[1] - lp[0])), 2),
            'order_high_p': round(float((lt[-1] - lt[-2]) / (lp[-1] - lp[-2])), 2),
            'tof_300K_1bar': float(t.set_index('T').tof_per_h[300])}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results-dir', default=RES)
    ap.add_argument('--no-sim', action='store_true')
    ap.add_argument('--out', default=None)
    a = ap.parse_args(argv)
    res = os.path.join(CALLER_CWD, a.results_dir)
    out = {'b3_300K_1bar_flow': b3_shares(res), **b4_fits(res)}
    if not a.no_sim:
        out['tof_293K_20bar'] = float(tof(293, 20, SIZE_NM, 'pore'))
        out['tof_308K_50bar'] = float(tof(308, 50, SIZE_NM, 'pore'))
    print(json.dumps(out, indent=1, ensure_ascii=False))
    if a.out:
        with open(os.path.join(CALLER_CWD, a.out), 'w', encoding='utf8') as f:
            json.dump(out, f, indent=1, ensure_ascii=False)
        print('saved', a.out)


if __name__ == '__main__':
    main()

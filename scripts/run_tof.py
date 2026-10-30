"""Turnover frequency (TOF) of the kinetic model, flow mode,.

TOF = ethylene molecules built into products (C4+) per Ni site per hour:
  d/dt sum_n (n/2) (C_n + C_n_out) / CAT, slope over the second half of the 1 h run.
At 300 K, 1 bar this gives 3.5e3 h^-1 for all particle sizes, the value in the paper
(npj Comput. Mater. 2026, SI Fig. S14a).

Two scans (model='pore' unless --models says otherwise):
  --scan p : 300 K, pressures log-spaced 1-50 bar
  --scan T : 1 bar, 300-430 K (validity limit of the reduced network at 1 bar)

  python scripts/run_tof.py --scan p --out results/tof_vs_p_300K.csv
  python scripts/run_tof.py --scan T --out results/tof_vs_T_1bar.csv
"""
import argparse
import os
import sys

CALLER_CWD = os.getcwd()              # km_common changes the working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import km_common as km  # noqa: E402
from run_distributions import size_label  # noqa: E402

np, pd = km.np, km.pd
PRESSURES = [1, 1.5, 2, 3, 5, 7, 10, 15, 20, 30, 50]
TEMPS = np.arange(300, 431, 10)


def tof(T, p, nm, model):
    _, ts = km.simulate(T, p, 'flow', nm, model=model, n_steps=2000)
    t = ts.index.values
    ns = sorted({int(c[1:].split('_')[0]) for c in ts.columns} - {2})
    eth = sum((n / 2) * (ts.get(f'C{n}', 0) + ts.get(f'C{n}_out', 0)) for n in ns)
    k = len(t) // 2
    return np.polyfit(t[k:], np.asarray(eth)[k:], 1)[0] * 3600 / km.CAT


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scan', choices=['p', 'T'], required=True)
    ap.add_argument('--sizes', type=float, nargs='+', default=[300, 1e5, 1e6], help='nm')
    ap.add_argument('--models', nargs='+', default=['pore'], choices=['pore', 'old'])
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    a.out = os.path.join(CALLER_CWD, a.out)

    cases = [(300.0, p) for p in PRESSURES] if a.scan == 'p' else [(float(T), 1.0) for T in TEMPS]
    rows = []
    for model in a.models:
        for nm in a.sizes:
            for T, p in cases:
                if not km.energy_check(T, p):
                    print(f'skip (outside validity): {T} K {p} bar', flush=True)
                    continue
                try:
                    v = tof(T, p, nm, model)
                except RuntimeError as exc:
                    print(f'solver failed: {T} K {p} bar {nm} nm: {exc}', flush=True)
                    continue
                rows.append(dict(model=model, particle=size_label(nm), particle_nm=nm, T=T,
                                 pressure_bar=p, tof_per_h=v))
                print(f'{model:9s} {size_label(nm):8s} {T:5.0f} K {p:5g} bar  TOF = {v:.3g} h^-1', flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    pd.DataFrame(rows).to_csv(a.out, index=False, float_format='%.6g')
    print('saved', a.out)


if __name__ == '__main__':
    main()

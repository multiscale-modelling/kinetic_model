"""Average olefin length vs temperature for batch / flow mode and several particle sizes.

Writes one CSV row per case (T, p, mode, particle size, model) with the mean chain length,
the share of C4, C6, C8+ and the full distribution. Cases outside the validity range of
the reduced network (energy check E8 > E10, see km_common.energy_check) are skipped.

Examples (run from anywhere):
  python scripts/run_distributions.py                      
  python scripts/run_distributions.py --pressures 1 25 50 --modes flow \
         --sizes 1e5 --tmin 300 --tmax 420 --tstep 30 --models corrected old --out my.csv
"""
import argparse
import os
import sys

CALLER_CWD = os.getcwd()              # km_common changes the working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import km_common as km  # noqa: E402  (sets paths and working directory)


def size_label(nm):
    if nm is None:
        return 'No diffusion'
    return f'{nm:g} nm' if nm < 1e3 else f'{nm / 1e3:g} µm'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pressures', type=float, nargs='+', default=[50.0], help='bar')
    ap.add_argument('--modes', nargs='+', default=['batch', 'flow'], choices=['batch', 'flow'])
    ap.add_argument('--sizes', nargs='+', default=['300', '1e5', '1e6', 'none'],
                    help="particle size in nm, or 'none' for no diffusion")
    ap.add_argument('--tmin', type=float, default=300)
    ap.add_argument('--tmax', type=float, default=540)
    ap.add_argument('--tstep', type=float, default=15)
    ap.add_argument('--models', nargs='+', default=['corrected'], choices=['corrected', 'old', 'gas', 'ideal'])
    ap.add_argument('--elim-map', default='tetra', choices=['legacy', 'per_step', 'tetra'],
                    help="elimination constants: original list, or each olefin its own barrier (km.simulate)")
    ap.add_argument('--crowding', nargs='+', default=['none'], choices=['none', *km.CROWDING],
                    help="product activity in the loaded pore (km.product_activity)")
    ap.add_argument('--hours', type=float, default=1.0, help='simulated time')
    ap.add_argument('--steps', type=int, default=10_000, help='output time points')
    ap.add_argument('--ads-interp', action='store_true',
                    help='pore capacities linear in T between GCMC grid points (km.AdsInterpT)')
    ap.add_argument('--out', default=os.path.join(km.ROOT, 'results',
                                                  'avg_olefin_vs_T_corrected_pressure_count.csv'))
    a = ap.parse_args(argv)
    a.out = os.path.join(CALLER_CWD, a.out)          # relative paths: from where you called it

    sizes = [None if s.lower() == 'none' else float(s) for s in a.sizes]
    temps = km.np.arange(a.tmin, a.tmax + 1e-9, a.tstep)
    rows = []
    for model, crowd in [(m, c) for m in a.models for c in a.crowding]:
        for p in a.pressures:
            for mode in a.modes:
                for nm in sizes:
                    for T in temps:
                        case = dict(model=model, crowding=crowd, elim_map=a.elim_map, mode=mode, pressure_bar=p,
                                    particle=size_label(nm),
                                    particle_nm=nm if nm is not None else float('nan'), T=float(T))
                        if not km.energy_check(T, p):
                            print(f'skip (outside reduced-network validity): {case}', flush=True)
                            continue
                        dist = None
                        for steps in (a.steps, 2_000):     # LSODA occasionally fails; retry once
                            try:
                                dist, _ = km.simulate(T, p, mode, nm, model=model, hours=a.hours,
                                                      n_steps=steps, ads_interp=a.ads_interp,
                                                      elim_map=a.elim_map,
                                                      crowding=None if crowd == 'none' else crowd)
                                break
                            except RuntimeError as exc:
                                print(f'solver failed ({steps} steps): {case}: {exc}', flush=True)
                        if dist is None:
                            continue
                        d = dist[dist.n > 2].set_index('n')['probability'].clip(lower=0)
                        d = d / d.sum()
                        case.update(mean_n=km.mean_length(dist), C4=d[4], C6=d[6],
                                    C8plus=d[d.index >= 8].sum(),
                                    **{f'P_C{n}': d[n] for n in d.index})
                        rows.append(case)
                        print(f"{model:9s} {crowd:9s} {mode:5s} {p:5g} bar {case['particle']:12s} "
                              f"{T:5.0f} K  <n> = {case['mean_n']:.2f}", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    km.pd.DataFrame(rows).to_csv(a.out, index=False, float_format='%.6g')
    print('saved', a.out)


if __name__ == '__main__':
    main()

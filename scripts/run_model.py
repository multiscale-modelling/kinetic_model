"""Run the Ni-NU-1000 ethylene-oligomerization kinetic model for chosen conditions and model options.

Every combination of temperature, pressure, reactor mode and particle size is one case. For each
case the script prints the average olefin size and the product distribution C4-C16, and can
write them to a CSV and the time series of every olefin to one CSV per case.

Model options (README.md, section "run_model.py"):
  --counting  pore | gas          how ethylene pressure enters (pore: GCMC pore concentration in
                                  the rate laws; gas: ideal-gas activity p/p0 in the DFT profile,
                                  ethylene removed from the rate laws, flow only)
  --barriers  tetra | individual | legacy
                                  which DFT barriers the insertion and elimination steps use
  --mode      batch | flow        reactor mode (both allowed: --mode batch flow)

Examples (run from anywhere):
  python scripts/run_model.py --T 300 340 380 420 --p 1
  python scripts/run_model.py --T 300 420 --tstep 15 --p 50 --mode batch flow \
         --size 1e6 --barriers individual --out my_scan.csv
  python scripts/run_model.py --T 390 --p 50 --counting gas --mode flow --size 1e6 \
         --timeseries ts_out
"""
import argparse
import os
import sys

CALLER_CWD = os.getcwd()              # km_common changes the working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import km_common as km  # noqa: E402  (sets paths and working directory)
from run_distributions import size_label  # noqa: E402


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--T', type=float, nargs='+', required=True,
                    help='temperature(s) in K; with --tstep, two values give a range tmin tmax')
    ap.add_argument('--tstep', type=float, default=None, help='step (K) for a range given by --T tmin tmax')
    ap.add_argument('--p', type=float, nargs='+', default=[1.0], help='ethylene pressure(s) in bar (default 1)')
    ap.add_argument('--counting', choices=list(km.PRESSURE_COUNTING), default='pore',
                    help='pressure counting: pore (GCMC pore concentration, default) or gas (ideal gas)')
    ap.add_argument('--barriers', choices=list(km.BARRIER_SCHEMES), default='tetra',
                    help='reaction scheme barriers: tetra (all steps _tetra, default), individual '
                         '(each step its own), legacy (original list)')
    ap.add_argument('--mode', nargs='+', choices=['batch', 'flow'], default=['batch'],
                    help='reactor mode(s) (default batch)')
    ap.add_argument('--size', nargs='+', default=['none'],
                    help="particle size(s) in nm, or 'none' for no diffusion (default none)")
    ap.add_argument('--hours', type=float, default=1.0, help='simulated time in h (default 1)')
    ap.add_argument('--steps', type=int, default=2000, help='output time points (default 2000)')
    ap.add_argument('--linear-time', action='store_true',
                    help='linear instead of log-spaced output times (time series only)')
    ap.add_argument('--no-check', action='store_true',
                    help='also run cases that fail the reduced-network validity check (E8 > E10)')
    ap.add_argument('--out', help='CSV file for the summary (one row per case)')
    ap.add_argument('--timeseries', metavar='DIR',
                    help='folder for one time-series CSV per case (concentrations in and outside the pores)')
    a = ap.parse_args(argv)

    if a.tstep is not None:
        if len(a.T) != 2:
            ap.error('--tstep needs exactly two values in --T (tmin tmax)')
        a.T = list(km.np.arange(a.T[0], a.T[1] + 1e-9, a.tstep))
    if a.counting == 'gas' and 'batch' in a.mode:
        ap.error("--counting gas needs --mode flow: ethylene is a constant-pressure reservoir and is "
                 "not consumed, so a batch run would not conserve carbon")
    a.size = [None if s.lower() == 'none' else float(s) for s in a.size]
    for attr in ('out', 'timeseries'):
        if getattr(a, attr):
            setattr(a, attr, os.path.join(CALLER_CWD, getattr(a, attr)))
    return a


def case_name(mode, p, T, nm, counting, barriers):
    size = 'nodiff' if nm is None else f'{nm:g}nm'
    return f'{mode}_{p:g}bar_{T:g}K_{size}_{counting}_{barriers}'


def main(argv=None):
    a = parse_args(argv)
    rows = []
    for p in a.p:
        for mode in a.mode:
            for nm in a.size:
                for T in a.T:
                    case = dict(counting=a.counting, barriers=a.barriers, mode=mode, pressure_bar=p,
                                particle=size_label(nm), particle_nm=nm if nm is not None else float('nan'),
                                T=float(T), hours=a.hours)
                    if not a.no_check and not km.energy_check(T, p):
                        print(f'skip (outside reduced-network validity, E8 < E10): {mode} {p:g} bar '
                              f'{T:g} K', flush=True)
                        continue
                    try:
                        dist, ts = km.run_case(T, p, counting=a.counting, barriers=a.barriers, mode=mode,
                                               particle_nm=nm, hours=a.hours, n_steps=a.steps,
                                               log_times=not a.linear_time)
                    except RuntimeError as exc:
                        print(f'solver failed: {mode} {p:g} bar {T:g} K: {exc}', flush=True)
                        continue
                    d = dist[dist.n > 2].set_index('n')['probability'].clip(lower=0)
                    d = d / d.sum()
                    case.update(mean_n=km.mean_length(dist), **{f'P_C{n}': d[n] for n in d.index})
                    rows.append(case)
                    shares = '  '.join(f'C{n} {d[n]:.2f}' for n in d.index)
                    print(f"{mode:5s} {p:5g} bar {case['particle']:12s} {T:5.0f} K  <n> = "
                          f"{case['mean_n']:5.2f}   {shares}", flush=True)
                    if a.timeseries:
                        os.makedirs(a.timeseries, exist_ok=True)
                        f = os.path.join(a.timeseries, case_name(mode, p, T, nm, a.counting, a.barriers) + '.csv')
                        ts.rename_axis('time_s').to_csv(f, float_format='%.6g')
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        km.pd.DataFrame(rows).to_csv(a.out, index=False, float_format='%.6g')
        print('saved', a.out)
    if a.timeseries:
        print('time series in', a.timeseries)


if __name__ == '__main__':
    main()

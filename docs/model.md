# Model notes

How the reactor runs in `scripts/` (`km_common.py`) and what changed relative to the previous
version. The `src/kinetic_model` package is unchanged; the changes are in the driver layer.

## Run

```bash
python scripts/run_model.py --T 300 340 380 420 --p 1                        # distribution, batch
python scripts/run_model.py --T 390 --p 50 --counting gas --mode flow --size 1e6
python scripts/run_distributions.py --models pore --pressures 1 50     # grid -> results/*.csv
python -m pytest tests/test_consistency.py -q                                # fast checks
```

From Python, temperature, pressure and the longest chain are enough:

```python
import km_common as km                     # scripts/
dist, ts = km.simulate(380, 50, n_max=16)  # K, bar
dist, ts = km.simulate(380, 50, n_max=16, elim_map='bep')
```

Inputs are read from `data/` (the working directory is switched there on import). Output goes to
`results/`.

## Pressure enters once

| | `--counting pore` (default, `model='pore'`) | `--counting gas` (`model='ideal'`) |
|---|---|---|
| Gibbs profile | all gases at p0 = 1 bar | p0, then only ethylene shifted to p: (n_C2 - 5) RT ln(p/p0) |
| ethylene in rate laws | GCMC pore concentration c(p, T) | removed (activity p/p0 is in dG) |
| modes | batch, flow | flow only |

Previous behaviour (`model='old'`): profile evaluated at p for *all* gases (including released
olefins) and the pore concentration also multiplied the rates. Kept for comparison only.
At 1 bar `old` and `pore` coincide.

## Barriers (`--barriers`, `elim_map`)

Three DFT sets: `_di` (C4), `_tri` (C6), `_tetra` (C8+).

| | `flory-tetra` (default) | `individual` (`per_step`) | `legacy` |
|---|---|---|---|
| growth C2->C4, C4->C6, C6+ | `_tetra` | `_di`, `_tri`, `_tetra` | `_di`, `_tri`, `_tetra` |
| elimination C4 / C6 / C8+ | `_tetra` | `_di` / `_tri` / `_tetra` | `_tri` / `_tetra` / `_tetra` |
| elimination in `bep` | BEP line through the DFT points per step type, chain-length extrapolation by group additivity |

`legacy` is the original reaction list (eliminations shifted by one olefin). `flory-tri` is
`flory-tetra` with the `_tri` set for every step. `tetra` / `tri` are accepted as earlier names.

## Product crowding (`crowding=`)

Forward elimination constant is divided by the activity coefficient of the released olefin in the
ethylene-loaded pore (`km.product_activity`, theta_v = 1 - theta_C2 from the ethylene Langmuir fit).

| value | gamma |
|---|---|
| `none` | 1 |
| `lattice` | 1/theta_v |
| `multisite` | (1/theta_v)^(n/2) |
| `pocket` | (1/theta_v)^s (default) |
| `pressure` | p/p0 (reproduces `old`) |

Default `s`: 3 for `flory-tetra` / `flory-tri`, 5 for `bep`, 3 otherwise (`km.DEFAULT_S_POCKET`);
`s_pocket=` overrides it. `simulate(T, p, n_max=...)` alone uses pocket crowding; `crowding=None`
switches it off.

`product_crowding=True` iterates the total site fraction (ethylene + products) self-consistently.

## Network validity

`network_consistency(T, p)` flags (T, p) where the reduced network (ethylene-bound resting state)
does not hold (E8 > E10 violated). `simulate(check_network='warn'|'filter'|None)`; default `warn`.

## Other additions

- `data/processed/diffusion_extrap_model.pkl`: diffusion surrogate used by `km_common.py`.
- `tests/test_consistency.py`: barriers > 0, pressure counted once, BEP fit, thermodynamic cycle,
  crowding shift, LDF conservation.
- `scripts/run_tof.py`: turnover frequency scans (flow mode).
- `scripts/table_model_values.py`: summary values from scan CSVs.

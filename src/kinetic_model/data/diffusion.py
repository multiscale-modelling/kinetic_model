# src/kinetic_model/data/diffusion.py

from pathlib import Path
import re
import warnings
import numpy as np
import pandas as pd

# pattern to capture BiasingSpline_<OLEFIN>_<AXIS>.dat
_DAT_RE = re.compile(r"^BiasingSpline_([A-Za-z0-9]+)_(\d+)\.dat$")

def parse_diffusion_dat(fp: Path) -> float:
    """Extract the D^TST value (m^2/s) from a .dat file."""
    pat = re.compile(r"#\s*D\^TST:\s*([0-9eE\+\-\.]+)\s*\[m\^2/s\]")
    with fp.open() as f:
        for line in f:
            m = pat.match(line)
            if m:
                return float(m.group(1))
    raise ValueError(f"No ‘D^TST’ line found in {fp}")

def process_diffusion_files(
    raw_dir: Path = Path("data/diffusion"),
    processed_file: Path = Path("data/processed/diffusion.csv"),
) -> pd.DataFrame:
    """
    Read all BiasingSpline_*.dat files in raw_dir, group by olefin,
    parse axis‐specific D values, compute average D, and save to CSV.

    Result has columns: olefin, Dx, Dy, Dz, D
    """
    raw_dir = Path(raw_dir)
    # temp storage: olefin -> {axis_index: D_value}
    rates: dict[str, dict[int, float]] = {}

    for fp in sorted(raw_dir.glob("*.dat")):
        m = _DAT_RE.match(fp.name)
        if not m:
            warnings.warn(f"Skipping unrecognized file name: {fp.name}")
            continue
        olefin, axis_str = m.groups()
        axis = int(axis_str)
        try:
            Dval = parse_diffusion_dat(fp)
        except ValueError as e:
            warnings.warn(str(e))
            continue

        rates.setdefault(olefin, {})[axis] = Dval

    records = []
    for olefin, axis_map in rates.items():
        missing = set((0, 1, 2)) - set(axis_map.keys())
        if missing:
            warnings.warn(
                f"Skipping {olefin}: missing axis files {sorted(missing)}"
            )
            continue
        Dx, Dy, Dz = axis_map[0], axis_map[1], axis_map[2]
        Dmean = float(np.mean([Dx, Dy, Dz]))
        records.append({"olefin": olefin, "Dx": Dx, "Dy": Dy, "Dz": Dz, "D": Dmean})

    df = pd.DataFrame(records, columns=["olefin", "Dx", "Dy", "Dz", "D"])
    processed_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_file, index=False)
    return df

def load_processed_diffusion(
    processed_file: Path = Path("data/processed/diffusion.csv"),
) -> pd.DataFrame:
    """
    Load the processed diffusion table with columns:
      olefin, Dx, Dy, Dz, D
    """
    return pd.read_csv(processed_file)

def diffusion_dict(df: pd.DataFrame) -> dict[str, float]:
    """
    Map olefin -> average D value in m^2/s.
    """
    return df.set_index("olefin")["D"].to_dict()

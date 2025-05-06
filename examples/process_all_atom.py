# python - <<'PYCODE'
from pathlib import Path
from kinetic_model.data.diffusion import (
    process_diffusion_files,
    load_processed_diffusion,
    diffusion_dict,
)

# 1) Parse and write CSV
df = process_diffusion_files(
    raw_dir=Path("data/diffusion"),
    processed_file=Path("data/processed/diffusion.csv"),
)
print(df)

# 2) Load it later
df2 = load_processed_diffusion(Path("data/processed/diffusion.csv"))
print(df2)
print("As dict:", diffusion_dict(df2))
# PYCODE

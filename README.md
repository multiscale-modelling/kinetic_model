<div align="center">

# Kinetic Model

_Multiscale toolkit for simulating Ni-catalysed olefin chain growth, diffusion, and adsorption._

</div>

## Overview
- Deterministic ODE reactor solver (`kinetic_model.core`) that handles coupled reaction networks and capacity-weighted diffusion between compartments.
- Machine-learning surrogates for adsorption capacities and intracrystalline diffusion (KNN/Gaussian-process regressors stored under `data/processed`).
- Thermochemistry and energy-profile tooling (`kinetic_model.energy_profile`) to turn DFT results into relative free-energy landscapes and barrier dictionaries that plug directly into the reactor simulations.
- Supplementary notebooks (`notebooks/`) for training surrogates and analysing experimental/DFT data, plus reproducible examples in `examples/`.

## Repository Layout
| Path | Purpose |
| --- | --- |
| `src/kinetic_model/core` | Base `Reaction` ODE solver, diffusion-aware reactor (`Reaction_Diffusion`), and Eyring rate helper. |
| `src/kinetic_model/constants` | ML models and utilities for adsorption/diffusion, energy-barrier loaders, and thermodynamic constants. |
| `src/kinetic_model/energy_profile` | Parsers for DFT folders, thermochemistry (harmonic, anharmonic, and free-rotor treatments), relative energy calculators, and plotting helpers. |
| `src/kinetic_model/simulation` | High-level simulation drivers (`ReactorSimulation`, chemisorption variants), reaction-list builders, plotting utilities, and synthetic-data generators. |
| `data/` | Raw DFT structures, gas-phase molecules, YAML metadata, and pretrained `.pkl` surrogates (`data/processed`). |
| `examples/` | Ready-to-run settings files and notebooks demonstrating energy-profile extraction. |
| `notebooks/` | Jupyter notebooks for adsorption/diffusion model development and loading studies. |
| `tests/` | Pytest suite that covers kinetic core logic, diffusion handling, rate calculations, and runner utilities. |

## Installation
1. **Create an environment (optional but recommended):**
   ```bash
   conda env create -f environment.yml
   conda activate kinetic_model
   ```
   or manually install Python ≥3.12.
2. **Install the package and dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```
   The editable install exposes `kinetic_model` for notebooks, scripts, and tests.

## Driver scripts
`scripts/` holds the command-line runners (`run_model.py`, `run_distributions.py`, `run_tof.py`). Pressure counting, barrier sets and crowding options: [docs/model.md](docs/model.md).

## Quick Start
```python
from kinetic_model.simulation.run_reactor import ReactorSimulationChemisorptionSimplified
from kinetic_model.constants.energy_profile import EnergyBarriersSimplifyed

energy = EnergyBarriersSimplifyed(pressure=5.0, temperature=320.0)
sim = ReactorSimulationChemisorptionSimplified(
    N=16,
    T=320.0,
    pressure=5.0,
    L=8.0,
    time_hours=2.0,
    n_steps=500,
    diff_model_path='../data/processed/diffusion_extrap_place_holder_model.pkl',
    ads_model_path='../data/processed/knn_adsorption_slope_model.pkl',
    energy_model=energy,
    initial_concs={'CatC2(C2)': 1e-2, 'C2': 1}
)
distribution_df, probability_traces = sim.run()
```
- `distribution_df` summarises the final olefin chain-length distribution and probabilities.
- `probability_traces` (time-series DataFrame) tracks intra-/extra-crystalline concentrations for each chain length.

## Energy Profiles and Barriers
1. **Prepare DFT folders:** populate `data/structures/<model_method>/<pathway>/<system>` and `data/gas-phase-molecules/<model>/…` with `POSCAR`/`mol.xyz`, `energy.dat`, `freq.txt`, and optional `rotAtoms.yaml`.
2. **Compute thermochemistry:**
   ```python

    from kinetic_model.energy_profile.energy_processing import calculate_relative_energies
    import os 
    import numpy as np

    temperatures = [300]        # Can be varied
    pressures = [101325.]           # Can be carried
    models = ['QM-QM_TPSSh-PBED3']   # Options: ['QM-QM_M06L-PBED3', 'QM-QM_TPPSh-PBED3]  
    pathways = ['trigonal']         # Options: ['trigonal', 'hexagonal']
    ref_struc = '1'                 # Choose either 1 or 5
    energy_type = 'Gibbs free energy'   # Options: ['Electronic energy', 'Enthalpy', 'Gibbs free energy']
    energy_unit = 'kJ/mol'              # Options: ['eV', 'kJ/mol', 'kcal/mol']

    custom_order = ["1", "2", "3", "4", "5", "5a", "6", "1p", "7", "8", "9", "5p", "6p", "1pp", "10", "11", "12", "5pp", "6pp", "1ppp", "2-3", "3-4", "6-1p", "7-8", "8-9", "6p-1pp", "10-11", "11-12", "6pp-1ppp"]  # Use for sorting the intermediates in a table (or pd.DataFrame)

    rel_results = calculate_relative_energies(
        structures_dir=os.path.abspath('../data/structures'),
        molecules_dir=os.path.abspath('../data/gas-phase-molecules'),
        temperatures=temperatures,
        models=models,
        ref_struc=ref_struc,
        pathways=pathways,
        pressures=pressures
    )

   ```
3. **Convert to barriers:** instantiate `EnergyBarriers` or `EnergyBarriersSimplifyed` with the target temperature/pressure; call `get_energy_barriers()` to obtain ΔG‡ dictionaries that feed `prepare_constants_*` or a custom simulation setup.

## Surrogate Models & Data
- **Adsorption:** `data/processed/knn_adsorption_model.pkl` contains the grouped/scaled KNN (`AdsorptionKNN` / `AdsorptionKNNSlopeExtrap`). Re-train using `notebooks/adsorbtion_process_ml.ipynb`, then `joblib.dump` the estimator.
- **Diffusion:** `data/processed/diffusion_model*.pkl` stores LightGBM-based regressors (see `notebooks/diffusion_process_ml.ipynb`). Rates are later converted to s⁻¹ via pore-geometry scaling inside `ReactorSimulation`.
- **DFT sources:** `data/structures`, `data/gas-phase-molecules`, and `data/formulae.yaml` describe the thermochemistry inputs. The mini `data/README` highlights shared intermediates between pathways.

## Examples & Notebooks
- `test_soft.ipynb` - the main example file where you can find more detailed examples how to use current soft. `examples/settings.yaml` and `examples/c8_energy_profiles_settings.yaml` demonstrate how to configure pathway, model, and plotting options for energy-profile reports.
- Notebooks in `notebooks/` reproduce adsorption/diffusion training, sanity-check Eyring prefactors, and explore C₂ loading. They assume you installed the package in editable mode so imports resolve.

## Testing & Development
- Run the full suite with `pytest` from the repository root. Key tests (`tests/test_core.py`, `tests/test_diffusion_behavior.py`, etc.) validate ODE integration, diffusion bookkeeping, barrier-to-rate conversions, and runner wiring.
- When adding features, prefer dropping new modules under `src/kinetic_model` and extend the pytest coverage alongside; notebooks and examples can reference the same public API.

## License
The project is released under the MIT License 

---

Questions or ideas? Open an issue or start a discussion in your preferred tracker—contributions that improve the kinetic workflow, data parsers, or surrogate accuracy are very welcome.



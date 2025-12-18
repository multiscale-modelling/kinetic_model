
# Dataset for Multiscale Modeling of Olefin Adsorption, Diffusion, and Kinetics in NU-1000

## Overview

This dataset accompanies the publication describing a multiscale modeling framework for olefin adsorption, diffusion, and reaction kinetics in the NU-1000 metal–organic framework. The dataset provides all raw simulation inputs, processed data, and reference electronic-structure calculations required to reproduce the results reported in the paper.

The dataset is organized into three main components:

- **Adsorption**: Grand Canonical Monte Carlo (GCMC) simulations performed with RASPA
- **Diffusion**: Widom insertion–based diffusion free-energy profiles for olefins in NU-1000
- **Kinetic model**: Python software and DFT reference data for kinetic and reactor-level simulations

---

## Dataset Structure


```
.
├── adsorption/
│   ├── C2H4_single/
│   └── mixture_validation_set/
│
├── diffusion/
│   ├── set_0/
│   ├── set_1/
│   └── set_2/
│
└── kinetic_model/
    ├── build/
    ├── data/
    ├── examples/
    ├── notebooks/
    ├── src/
    ├── tests/
    ├── pyproject.toml
    ├── environment.yml
    ├── requirements.txt
    ├── requirements-clean.txt
    ├── clean_requirements.txt
    └── README.md
```


---

## Adsorption (`adsorption/`)

This directory contains **RASPA GCMC simulations** for olefin adsorption in NU-1000.

### `C2H4_single/`
- Adsorption simulations of **ethylene** in NU-1000
- Performed over a range of temperatures and pressures
- Used to parameterize adsorption thermodynamics in the kinetic model

### `mixture_validation_set/`
- Adsorption data for **C₂–C₁₀ olefins**
- Includes simulations at diverse thermodynamic conditions
- Used to **validate surrogate machine-learning models** for adsorption prediction, as reported in the publication

---

## Diffusion (`diffusion/`)

This directory contains **Widom insertion–based diffusion calculations** for olefins in NU-1000.

### `set_0/`, `set_1/`, `set_2/`
- Three statistically independent Widom diffusion simulation sets
- Each set samples the diffusion free-energy landscape independently
- The three datasets are combined in post-processing to improve statistical robustness and reduce noise

The resulting diffusion profiles are used to extract effective diffusion barriers and rates for kinetic modeling.

---

## Kinetic Model (`kinetic_model/`)

This directory contains the **Python-based kinetic modeling framework** used in the publication, together with all **DFT reference data** required to parameterize the model.

The kinetic model integrates:
- DFT-derived reaction energetics and vibrational data
- Gas-phase thermochemistry
- Adsorption thermodynamics
- Diffusion limitations
- Batch and flow (stream) reactor formulations

The kinetic model is distributed as a standalone Python package.

Detailed installation instructions, usage examples, and API documentation are provided in:

```

kinetic_model/README.md

```

---

## DFT Reference Data (`kinetic_model/data/`)

The `data/` directory contains results from **density functional theory (DFT) calculations** within a hierarchical cluster approach used to construct accurate free-energy profiles and reaction rate constants.

These data include electronic energies, dispersion corrections, vibrational frequencies, and optimized geometries for gas-phase molecules, adsorbed intermediates, and transition states.

---

### Gas-Phase Molecules

```

kinetic_model/data/gas-phase-molecules/

```

Gas-phase reference results for **linear olefins (C₂–C₈)** used to compute free energies.

#### Periodic reference calculations

- `pbc_PBED3BJ/`
  - Periodic DFT calculations using the **PBE-D3(BJ)** functional
  - Typical files:
    - `energy.dat` – total electronic energy in eV
    - `freq.txt` – harmonic vibrational frequencies
    - `POSCAR` – optimized molecular structure

#### QM/QM-corrected energies

- `QM-QM_PBED3-PBED3/`
- `QM-QM_M06L-PBED3/`
- `QM-QM_TPSSh-PBED3/`

These directories contain QM/QM-corrected gas-phase energies, where:
- Periodic PBE-D3 calculations provide reference geometries
- Cluster-level calculations (PBE-D3, M06-L, TPSSh) are used to refine energetics

Typical files include:
- `energy.dat` – periodic electronic energy in eV
- `energy_cluster.dat` – cluster-level corrected energy in Eh
- `freq.txt` – vibrational frequencies
- `mol.xyz` – molecular geometry
- `POSCAR` – periodic reference structure

---

### Adsorbed Intermediates and Transition States

```

kinetic_model/data/structures/

```

This directory contains results from **DFT calculations** for **adsorbed intermediates and transition states** involved in olefin oligomerization inside NU-1000.

#### Directory hierarchy

```

kinetic_model/data/structures/pbc_PBED3BJ/
├── trigonal/
│   └── README
└── hexagonal/
├── Int_1_Ni-ethyl/
├── Int_7_et-Ni-butyl/
├── Int_8_Ni-hexyl/
├── Int_9_H-Ni-hex/
├── Int_10_et-Ni-hexyl/
├── Int_11_Ni-octyl/
├── Int_12_H-Ni-oct/
├── TS_10-11/
└── ...

```

Each directory corresponds to a specific **adsorbed intermediate (`Int_*`) or transition state (`TS_*`)**.

Typical contents:
- `energy.dat` – electronic energy in eV
- `Edisp.dat` – dispersion contribution in eV
- `freq.txt` – vibrational frequencies
- `POSCAR` – optimized periodic structure
- `rotAtoms.yaml`, `rotAtoms.dat` – definition of internal rotors for entropy corrections

Both **hexagonal and trigonal NU-1000 unit cells** are provided where relevant.

---

## Purpose of the DFT Data

The DFT datasets are used to:
- Construct accurate free energy profiles using a hierarchical cluster approach
- Compute reaction rate constants via transition-state theory

All DFT data are directly consumed by the kinetic modeling framework without manual preprocessing.

---

## Reproducibility

- All raw simulation inputs and processed data used in the publication are provided
- Directory names encode the employed functional and correction scheme
- Input geometries, energies, and vibrational data are explicitly included
- The kinetic model can be executed using only the contents of this dataset

---

## Citation

If you use this dataset or the accompanying kinetic modeling software, please cite the associated publication.

---

## Contact

For questions regarding the dataset or the kinetic modeling framework, please contact the corresponding authors listed in the associated publication.
```


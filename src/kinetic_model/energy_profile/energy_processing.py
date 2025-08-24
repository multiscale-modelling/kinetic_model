from kinetic_model.energy_profile.thermochemistry import IdealGasThermo, HarmonicThermo, HA1DFRThermo
from kinetic_model.energy_profile.utils import get_symmetry_number
import numpy as np
import pandas as pd
import os, sys
from ase import units
from ase.io import *
import yaml 


def _read_freq_orca(filename):
    vibs = np.loadtxt(filename)
    return np.sort(vibs)[::-1]

def _read_freq_vasp(filename):
    vibs = []
    with open(filename, 'r') as fil:
        lines = fil.readlines()

    for line in lines:
        row = line.split()
        if len(row) == 11:
            if row[1] == 'f/i':
                vib = -float(row[3])
            else:
                vib = float(row[3])
            vibs.append(vib)
    return np.sort(vibs)[::-1]

## Extract the data from the intermediates and transition states
def parse_structure_data(base_dir: str, model_method='formate_PBED3BJ', pathway='agostic') -> pd.DataFrame:
    data = []
    model, method = model_method.split(sep='_')
    model_path = os.path.join(base_dir, model_method)
    pathway_path = os.path.join(model_path, pathway)
    if os.path.isdir(pathway_path):
        for system in os.listdir(pathway_path):
            if system.startswith('TS_'):
                type = 'TS'
                order = system.split(sep='_')[-1]
                name = system
            elif system.startswith('Int_'):
                type = 'Intermediate'
                _, order, name = system.split(sep='_')
            system_path = os.path.join(pathway_path, system)
            poscar_file = os.path.join(system_path, 'POSCAR')
            atoms = read(poscar_file)
            energy_file = os.path.join(system_path, 'energy.dat')
            freq_file = os.path.join(system_path, 'freq.txt')
            rot_atoms_file = os.path.join(system_path, 'rotAtoms.yaml')
            if_free_rotor = False
            if os.path.exists(rot_atoms_file):
                if_free_rotor = True
            if os.path.exists(energy_file) and os.path.exists(freq_file):
                electronic_energy = float(np.loadtxt(energy_file))
                if model in ['pbc', 'QM-QM']:
                    freqs = _read_freq_vasp(freq_file)
                elif model in ['benzoate', 'nonRigidBenzoate', 'formate']:
                    freqs = _read_freq_orca(freq_file)
                data.append({
                    'model': model,
                    'method': method,
                    'pathway': pathway,
                    'system': system,
                    'order': order,
                    'system_name': name,
                    'type': type,
                    'electronic_energy': electronic_energy,
                    'freqs': freqs,
                    'if_free_rotor': if_free_rotor,
                    'rot_atoms_file': rot_atoms_file,
                    'ase_atoms': atoms,
                })
    return pd.DataFrame(data)

## Extract data for gas-phase molecules
def parse_molecule_data(base_dir: str, structure_model='formate_PBED3BJ') -> pd.DataFrame:
    structure2molecule_model = {
        'formate_PBED3BJ': 'cluster_PBED3BJ',
        'benzoate_PBED3BJ': 'cluster_PBED3BJ',
        'nonRigidBenzoate_PBED3BJ': 'cluster_PBED3BJ',
        'formate_DLPNO-CCSDT': 'cluster_DLPNO-CCSDT',
        'pbc_PBED3BJ': 'pbc_PBED3BJ'
    }
    data = []
    if structure_model in structure2molecule_model.keys():
        molecule_model = structure2molecule_model[structure_model]
    else:
        molecule_model = structure_model

    model, method = molecule_model.split(sep='_')
    model_path = os.path.join(base_dir, molecule_model)
    if os.path.isdir(model_path):
        for system in os.listdir(model_path):
            system_path = os.path.join(model_path, system)
            energy_file = os.path.join(system_path, 'energy.dat')
            freq_file = os.path.join(system_path, 'freq.txt')
            poscar_file = os.path.join(system_path, 'POSCAR')
            xyz_file = os.path.join(system_path, 'mol.xyz')
            if os.path.exists(energy_file) and os.path.exists(freq_file):
                electronic_energy = float(np.loadtxt(energy_file))
                if model in ['pbc', 'mixed', 'QM-QM']:
                    freqs = _read_freq_vasp(freq_file)
                    atoms = read(poscar_file)
                elif model == 'cluster':
                    freqs = _read_freq_orca(freq_file)
                    atoms = read(xyz_file)
                data.append({
                    'system': system,
                    'model': model,
                    'method': method,
                    'electronic_energy': electronic_energy,
                    'freqs': freqs,
                    'ase_atoms': atoms,
                })
    return pd.DataFrame(data)

def calculate_harmonic_thermodynamics(base_dir: str,
                                      temperatures: list[float], 
                                      model: str = 'formate_PBED3BJ', 
                                      pathway: str = 'agostic',
                                      truhlar: bool = False, 
                                      shift_freq: float =50., 
                                      real_freqs: bool = False, 
                                      pressures: list[float] =[101325.]) -> pd.DataFrame:
    df = parse_structure_data(base_dir, model, pathway)
    results = []
    for temp in temperatures:
        for pressure in pressures:
            for _, row in df.iterrows():
                num_vibs = len(row['freqs'])
                if row['model'] in ['pbc', 'QM-QM']:
                    # Partial Hessian analysis was done in VASP
                    # no translational and rotational degrees of freedom
                    if row['type'] == 'TS':
                        freqs = row['freqs'][:num_vibs-1]
                    else:
                        freqs = row['freqs']
                elif row['model'] in ['formate', 'benzoate', 'nonRigidBenzoate']:
                    # Full Hessian analysis was done in ORCA
                    # Need to exclude the first 6 degrees of freedom
                    if row['type'] == 'TS':
                        freqs = row['freqs'][:num_vibs-7]
                    else:
                        freqs = row['freqs'][:num_vibs-6]
                
                if truhlar:
                    freqs = [max(freq, shift_freq) for freq in freqs]
                else:
                    if real_freqs:
                        freqs = [np.abs(freq) for freq in freqs]
                    else:
                        freqs = [freq for freq in freqs if freq > 0]
                
                ## The unit for energies is eV
                freqs = np.array(freqs)
                vib_energies = freqs / 8065.54429

                if row['model'] in ['pbc', 'QM-QM']:
                    electronic_energy = row['electronic_energy']
                else:
                    electronic_energy = row['electronic_energy'] * 27.2114
                
                if row['if_free_rotor'] == True:
                    atoms = row['ase_atoms']
                    with open(row['rot_atoms_file'], 'r') as f:
                        rot_atoms_data = yaml.safe_load(f)
                    axis_atoms_idx = rot_atoms_data['rot_axis']
                    rot_atoms_idx = rot_atoms_data['rot_atoms']

                    ha1dfr = HA1DFRThermo(
                        atoms = atoms,
                        axis_atoms_idx=axis_atoms_idx,
                        rot_atoms_idx=rot_atoms_idx,
                        vib_energies=vib_energies,
                        potentialenergy=electronic_energy,
                        sigma=3,
                    )
                    enthalpy = ha1dfr.get_HA1DFR_internal_energy(temp)
                    free_energy = ha1dfr.get_HA1DFR_helmholtz_energy(temp)
                else:
                    ha = HarmonicThermo(
                        vib_energies=vib_energies, 
                        potentialenergy=electronic_energy,
                        ignore_imag_modes = True,
                        )
                    _, _, enthalpy = ha.get_internal_energy(temp)
                    free_energy = ha.get_helmholtz_energy(temp)

                results.append({
                    'model': row['model'],
                    'method': row['method'],
                    'pathway': row['pathway'],
                    'system': row['system'],
                    'system_name': row['system_name'],
                    'order': row['order'],
                    'type': row['type'],
                    'pressure': pressure,
                    'temperature': temp,
                    'electronic_energy_eV': electronic_energy,
                    'enthalpy_eV': enthalpy,
                    'free_energy_eV': free_energy,
                })
    return pd.DataFrame(results)

def calculate_ideal_gas_thermodynamics(base_dir: str, 
                                       temperatures: list[float],
                                       structure_model: str = 'formate_PBED3BJ',
                                       pressures: list[float] = [101325.]) -> pd.DataFrame:
    df = parse_molecule_data(base_dir, structure_model)
    results = []
    for temp in temperatures:
        for pressure in pressures:
            for _, row in df.iterrows():
                atoms = row['ase_atoms']
                sigma = get_symmetry_number(atoms)
                freqs = np.array(row['freqs'])
                vib_energies = freqs / 8065.54429

                if row['model'] in ['pbc', 'QM-QM']:
                    electronic_energy = row['electronic_energy']
                else:
                    electronic_energy = row['electronic_energy'] * 27.2114
                
                ig = IdealGasThermo(vib_energies=vib_energies,
                                    geometry='nonlinear',
                                    potentialenergy=electronic_energy,
                                    atoms=atoms,
                                    symmetrynumber=sigma,
                                    spin=0,
                                    )
                _, _, _, _, enthalpy = ig.get_enthalpy(temperature=temp)
                free_energy = ig.get_gibbs_energy(temperature=temp, pressure=pressure)

                results.append({
                    'system': row['system'],
                    'model': row['model'],
                    'method': row['method'],
                    'temperature': temp,
                    'pressure': pressure,
                    'electronic_energy_eV': electronic_energy,
                    'enthalpy_eV': enthalpy,
                    'free_energy_eV': free_energy,
                })
    return pd.DataFrame(results)

def calculate_energy_with_references(row, molecule_results, formulas):
    if row['type'] == 'Intermediate':
        formula = formulas.get(row['order'], {'ethylene': 0, 'but-1-ene': 0, 'hex-1-ene': 0, 'oct-1-ene': 0})
    elif row['type'] == 'TS':
        formula = formulas.get(row['system'], {'ethylene': 0, 'but-1-ene': 0, 'hex-1-ene': 0, 'oct-1-ene': 0})

    ref_energies = {
        'ref_E_eV': row['electronic_energy_eV'],
        'ref_H_eV': row['enthalpy_eV'],
        'ref_G_eV': row['free_energy_eV']
    }
    for molecule, weight in formula.items():
        if weight > 0:
            if row['model'] in ['formate', 'benzoate', 'nonRigidBenzoate']:
                model = 'cluster'
            elif row['model'] == 'QM-QM':
                model = 'QM-QM'
            else:
                model = 'pbc'

            molecule_row = molecule_results[
                (molecule_results['system'] == molecule) &
                (molecule_results['model'] == model) &
                (molecule_results['method'] == row['method']) &
                (molecule_results['pressure'] == row['pressure']) &
                (molecule_results['temperature'] == row['temperature'])
            ]
            if not molecule_row.empty:
                molecule_row = molecule_row.iloc[0]
                ref_energies['ref_E_eV'] += weight * molecule_row['electronic_energy_eV']
                
                ref_energies['ref_H_eV'] += weight * molecule_row['enthalpy_eV']
                ref_energies['ref_G_eV'] += weight * molecule_row['free_energy_eV']
    return ref_energies

def calculate_energies_with_references(structures_dir: str, 
                                       molecules_dir: str, 
                                       temperatures: list[float],
                                       model: str = 'formate_PBED3BJ',
                                       pathway: str = 'agostic',
                                       ref_struc: str ='1', 
                                       pressures: list[float] =[101325.], truhlar: bool = False, 
                                       shift_freq: float = 50.,
                                       real_freqs: bool = False):
    structures_results_df = calculate_harmonic_thermodynamics(
                    base_dir=structures_dir, 
                    temperatures=temperatures,
                    model=model,
                    pathway=pathway,
                    pressures=pressures,
                    truhlar=truhlar,
                    shift_freq=shift_freq,
                    real_freqs=real_freqs,
                    )
    molecules_results_df = calculate_ideal_gas_thermodynamics(
        base_dir=molecules_dir,
        temperatures=temperatures,
        structure_model=model,
        pressures=pressures
    )
    # TODO: replace the path to formulae file 
    formulae_file = os.path.abspath('../data/formulae.yaml')
    with open(formulae_file, 'r') as f:
        formulae = yaml.safe_load(f)
    formula = formulae[ref_struc]
    # Step 3: Apply the corrections
    corrected_data = []
    for _, row in structures_results_df.iterrows():
        corrected_values = calculate_energy_with_references(row, molecules_results_df, formula)
        corrected_row = {**row.to_dict(), **corrected_values}
        corrected_data.append(corrected_row)
    
    

    # Step 4: Create corrected_results DataFrame
    return pd.DataFrame(corrected_data)

def calculate_relative_energies(structures_dir: str,
                                molecules_dir: str,
                                temperatures: list[float],
                                models: list[str],
                                pathways: list[str],  
                                ref_struc: str = '1', 
                                pressures: list[float] = [101325.],
                                truhlar: bool = False, 
                                shift_freq: float = 50., 
                                real_freqs: bool =False) -> pd.DataFrame:
    from itertools import product
    all_results = []

    for model, pathway in product(models, pathways):
        try:
            corrected = calculate_energies_with_references(
                structures_dir=structures_dir,
                molecules_dir=molecules_dir,
                temperatures=temperatures,
                model=model,
                pathway=pathway,
                ref_struc=ref_struc,
                pressures=pressures,
                truhlar=truhlar,
                shift_freq=shift_freq,
                real_freqs=real_freqs,
            )
            all_results.append(corrected)
        except Exception as e:
            print(f"Failed for model='{model}', pathway='{pathway}': {e}")
    
    if not all_results:
        raise ValueError("No results were successfully computed.")
        
    corrected_results = pd.concat(all_results, ignore_index=True)
    
    # Group by relevant columns
    grouped = corrected_results.groupby(['model', 'method', 'pressure', 'temperature', 'pathway'])
    
    # Create relative columns
    corrected_results['rel_E_eV'] = None
    corrected_results['rel_H_eV'] = None
    corrected_results['rel_G_eV'] = None

    # Process each group
    for (model, method, pressure, temp, pathway), group in grouped:
        # Find the reference row for structure '1'
        ref_row = group[group['order'] == ref_struc]
        if not ref_row.empty:
            ref_electronic = ref_row['ref_E_eV'].values[0]
            ref_enthalpy = ref_row['ref_H_eV'].values[0]
            ref_free = ref_row['ref_G_eV'].values[0]
            
            # Calculate relative values for the group
            for idx, row in group.iterrows():
                corrected_results.at[idx, 'rel_E_eV'] = row['ref_E_eV'] - ref_electronic
                corrected_results.at[idx, 'rel_H_eV'] = row['ref_H_eV'] - ref_enthalpy
                corrected_results.at[idx, 'rel_G_eV'] = row['ref_G_eV'] - ref_free
    
    return corrected_results

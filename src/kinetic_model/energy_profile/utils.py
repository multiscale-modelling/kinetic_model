import os
import numpy as np
from ase.io import *
from ase import units
from pointgroup import PointGroup

def read_freq_vasp(filename, tol=1., scale_factor=1.):
    vibs = []
    vib_energies = []
    with open(filename, 'r') as fil:
        lines = fil.readlines()
    for line in lines:
        row = line.split()
        ## read wavenumbers in cm**-1
        if len(row) == 11:
            if row[1] == 'f' and row[2] == '=':
                vib = scale_factor * float(row[3])
                if vib > tol:
                    ## convert to vibrational energy in eV
                    vib_energy = vib / 8065.54429
                    vibs.append(vib)
                    vib_energies.append(vib_energy)
    vibs = np.array(vibs)
    vib_energies = np.array(vib_energies)
    return vibs, vib_energies

def read_freq_orca(filename, tol=1., scale_factor=1):
    vibs = []
    vib_energies = []
    init_vibs = np.loadtxt(filename).T
    for vib in init_vibs:
        if vib > tol:
            # convert to vibrational energy in eV
            vib_energy = vib / 8065.54429
            vibs.append(vib)
            vib_energies.append(vib_energy)
    vibs = np.array(vibs)
    vib_energies = np.array(vib_energies)
    return vibs, vib_energies

def get_point_group(atoms):
    pg = PointGroup(positions=atoms.get_positions(),
                    symbols=atoms.get_chemical_symbols())
    point_group = pg.get_point_group()
    return point_group

def get_symmetry_number(atoms):
    sym_numbers = {
    # Cyclic groups
    "C1": 1,
    "Ci": 1,
    "Cs": 1,
    "C2": 2,
    "C3": 3,
    "C4": 4,
    "C5": 5,
    "C6": 6,
    # Dihedral groups
    "C2v": 2,
    "C3v": 3,
    "C4v": 4,
    "C5v": 5,
    "C6v": 6,
    "C2h": 2,
    "C3h": 3,
    "C4h": 4,
    "C5h": 5,
    "C6h": 6,
    "D2": 4,
    "D3": 6,
    "D4": 8,
    "D5": 10,
    "D6": 12,
    "D2h": 4,
    "D3h": 6,
    "D4h": 8,
    "D5h": 10,
    "D6h": 12,
    "D2d": 4,
    "D3d": 6,
    "D4d": 8,
    "D5d": 10,
    "D6d": 12,
    # Special groups
    "S2": 1,
    "S4": 2,
    "S6": 3,
    "S8": 4,
    "T": 12,
    "Td": 12,
    "Th": 12,
    "O": 24,
    "Oh": 24,
    "I": 60,
    "Ih": 60,
    "Cinfv": 1,
    "Dinfh": 2,
}


    point_group = get_point_group(atoms)
    return sym_numbers[point_group]


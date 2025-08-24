#!/usr/bin/env python

"""Modules for calculating thermochemical information from computational
outputs. Modify from ase.thermochemistry module"""

import os
import sys
from warnings import warn
import numpy as np
import pandas as pd
from pointgroup import PointGroup
from ase import units
from ase.io import *

class ThermoChem:
    """Base class containing common methods used in thermochemistry
    calculations."""

    def get_ZPE_correction(self):
        """Returns the zero-point vibrational energy correction in eV."""
        return 0.5 * np.sum(self.vib_energies)

    def _vibrational_energy_contribution(self, temperature):
        """Calculates the change in internal energy due to vibrations from
        0K to the specified temperature for a set of vibrations given in
        eV and a temperature given in Kelvin. Returns the energy change
        in eV."""
        kT = units.kB * temperature
        dU = 0.
        for energy in self.vib_energies:
            dU += energy / (np.exp(energy / kT) - 1.)
        return dU

    def _vibrational_entropy_contribution(self, temperature):
        """Calculates the entropy due to vibrations for a set of vibrations
        given in eV and a temperature given in Kelvin.  Returns the entropy
        in eV/K."""
        kT = units.kB * temperature
        S_v = 0.
        for energy in self.vib_energies:
            x = energy / kT
            S_v += x / (np.exp(x) - 1.) - np.log(1. - np.exp(-x))
        S_v *= units.kB
        return S_v
    
def _clean_vib_energies(vib_energies, ignore_imag_modes=False):
    """Checks for presence of imaginary vibrational modes and removes
    them if ignore_imag_modes is True. Also removes +0.j from real
    vibrational energies.

    Inputs:

    vib_energies : list
        a list of the vibrational energies

    ignore_imag_modes : bool
        If True, any imaginary frequencies will be removed. If False,
        (default), an error will be raised if imaginary frequencies are
        present.

    Outputs:

    vib_energies : list
        a list of the cleaned vibrational energies with imaginary frequencies
        removed if ignore_imag_modes is True.

    n_imag : int
        the number of imaginary frequencies removed
    """
    if ignore_imag_modes:
        n_vib_energies = len(vib_energies)
        vib_energies = [v for v in vib_energies if np.real(v) > 0]
        n_imag = n_vib_energies - len(vib_energies)
        if n_imag > 0:
            warn(f"{n_imag} imag modes removed", UserWarning)
    else:
        if sum(np.iscomplex(vib_energies)):
            raise ValueError('Imaginary vibrational energies are present.')
        n_imag = 0
    vib_energies = np.real(vib_energies)  # clear +0.j

    return vib_energies, n_imag

class HarmonicThermo(ThermoChem):
    """Class for calculating thermodynamic properties in the approximation
    that all degrees of freedom are treated harmonically. Often used for
    adsorbates.

    Inputs:

    vib_energies : list
        a list of the harmonic energies of the adsorbate (e.g., from
        ase.vibrations.Vibrations.get_energies). The number of
        energies should match the number of degrees of freedom of the
        adsorbate; i.e., 3*n, where n is the number of atoms. Note that
        this class does not check that the user has supplied the correct
        number of energies. Units of energies are eV.
    potentialenergy : float
        the potential energy in eV (e.g., from atoms.get_potential_energy)
        (if potentialenergy is unspecified, then the methods of this
        class can be interpreted as the energy corrections)
    ignore_imag_modes : bool
        If True, any imaginary frequencies will be ignored in the calculation
        of the thermochemical properties. If False (default), an error will
        be raised if any imaginary frequencies are present.
    """

    def __init__(self, vib_energies, potentialenergy=0.,
                 ignore_imag_modes=False):
        self.ignore_imag_modes = ignore_imag_modes

        # Check for imaginary frequencies.
        vib_energies, n_imag = _clean_vib_energies(
            vib_energies, ignore_imag_modes=ignore_imag_modes
        )
        self.vib_energies = vib_energies
        self.n_imag = n_imag

        self.potentialenergy = potentialenergy

    def get_internal_energy(self, temperature):
        """Returns the internal energy, in eV, in the harmonic approximation
        at a specified temperature (K)."""

        U = 0.

        U += self.potentialenergy

        zpe = self.get_ZPE_correction()
        U += zpe

        dU_v = self._vibrational_energy_contribution(temperature)
        U += dU_v

        return zpe, dU_v, U


    def get_entropy(self, temperature):
        """Returns the entropy, in eV/K, in the harmonic approximation
        at a specified temperature (K)."""

        S = 0.

        S_v = self._vibrational_entropy_contribution(temperature)
        S += S_v

        return S


    def get_helmholtz_energy(self, temperature):
        """Returns the Helmholtz free energy, in eV, in the harmonic
        approximation at a specified temperature (K)."""


        zpe, dU_v, U = self.get_internal_energy(temperature)
        S = self.get_entropy(temperature)
        F = U - temperature * S
        
        return F
    

class HA1DFRThermo(HarmonicThermo):
    """Class for calculating thermodynamic properties in the approximation
    that all degrees of freedom are treated harmonically, but one treated as 
    1D free rotor. Often used for adsorbates.

    Inputs:

    vib_energies : list
        a list of the harmonic energies of the adsorbate (e.g., from
        ase.vibrations.Vibrations.get_energies). The number of
        energies should match the number of degrees of freedom of the
        adsorbate; i.e., 3*n, where n is the number of atoms. Note that
        this class does not check that the user has supplied the correct
        number of energies. Units of energies are eV.
    potentialenergy : float
        the potential energy in eV (e.g., from atoms.get_potential_energy)
        (if potentialenergy is unspecified, then the methods of this
        class can be interpreted as the energy corrections)
    ignore_imag_modes : bool
        If True, any imaginary frequencies will be ignored in the calculation
        of the thermochemical properties. If False (default), an error will
        be raised if any imaginary frequencies are present.
    """

    def __init__(self, atoms, axis_atoms_idx, rot_atoms_idx, vib_energies, potentialenergy=0., sigma = 1, ignore_imag_modes=True):
        self.ignore_imag_modes = ignore_imag_modes
        self.atoms = atoms
        self.axis_atoms_idx = axis_atoms_idx
        self.rot_atoms_idx = rot_atoms_idx
        self.sigma = sigma

        # Check for imaginary frequencies.
        vib_energies, n_imag = _clean_vib_energies(
            vib_energies, ignore_imag_modes=ignore_imag_modes
        )
        self.vib_energies = vib_energies
        self.n_imag = n_imag

        self.potentialenergy = potentialenergy

    @staticmethod
    def distance_atom_to_line(atoms, idx_1, idx_2, idx_3):
        A = atoms[idx_1].position
        B = atoms[idx_2].position
        C = atoms[idx_3].position
        AB = B - A
        AC = C - A
        cross = np.cross(AB, AC)
        distance = np.linalg.norm(cross) / np.linalg.norm(AB)
        return distance
    
    def calculate_free_rotor_thermo(self, temperature):
        from ase.units import _amu, _k, kB, _hplanck
        angstrom_to_m = 1e-10
        # calculate moment of inertia in amu * angstrom ** 2 $I = \sum mr^2$ around atom indexed 0
        I = 0
        for idx in self.rot_atoms_idx:
            atom = self.atoms[idx]
            mr2 = atom.mass * self.distance_atom_to_line(self.atoms, self.axis_atoms_idx[0], self.axis_atoms_idx[1], idx) ** 2
            I += mr2
        # Convert to amu * m ** 2
        I = I * _amu * angstrom_to_m ** 2

        free_rotor_partition = (8 * np.pi ** 3 * I * _k * temperature) ** 0.5 / (self.sigma * _hplanck)

        U = 0.5 * kB * temperature
        S = kB * (np.log(free_rotor_partition) + 0.5)
        G_freeRot = U - temperature * S
        return U, S, G_freeRot
    
    def get_HA1DFR_internal_energy(self, temperature):
        _, _, U_HA = self.get_internal_energy(temperature)
        U_FR, S_FR, G_FR = self.calculate_free_rotor_thermo(temperature)
        U = U_HA + U_FR
        return U
    
    def get_HA1DFR_entropy(self, temperature):
        S_HA = self.get_entropy(temperature)
        U_FR, S_FR, G_FR = self.calculate_free_rotor_thermo(temperature)
        S = S_HA + S_FR
        return S
    
    def get_HA1DFR_helmholtz_energy(self, temperature):
        G_HA = self.get_helmholtz_energy(temperature)
        U_FR, S_FR, G_FR = self.calculate_free_rotor_thermo(temperature)
        G = G_HA + G_FR
        return G


class IdealGasThermo(ThermoChem):
    """Class for calculating thermodynamic properties of a molecule
    based on statistical mechanical treatments in the ideal gas
    approximation.

    Inputs for enthalpy calculations:

    vib_energies : list
        a list of the vibrational energies of the molecule (e.g., from
        ase.vibrations.Vibrations.get_energies). The number of vibrations
        used is automatically calculated by the geometry and the number of
        atoms. If more are specified than are needed, then the lowest
        numbered vibrations are neglected. If either atoms or natoms is
        unspecified, then uses the entire list. Units are eV.
    geometry : 'monatomic', 'linear', or 'nonlinear'
        geometry of the molecule
    potentialenergy : float
        the potential energy in eV (e.g., from atoms.get_potential_energy)
        (if potentialenergy is unspecified, then the methods of this
        class can be interpreted as the energy corrections)
    natoms : integer
        the number of atoms, used along with 'geometry' to determine how
        many vibrations to use. (Not needed if an atoms object is supplied
        in 'atoms' or if the user desires the entire list of vibrations
        to be used.)

    Extra inputs needed for entropy / free energy calculations:

    atoms : an ASE atoms object
        used to calculate rotational moments of inertia and molecular mass
    symmetrynumber : integer
        symmetry number of the molecule. See, for example, Table 10.1 and
        Appendix B of C. Cramer "Essentials of Computational Chemistry",
        2nd Ed.
    spin : float
        the total electronic spin. (0 for molecules in which all electrons
        are paired, 0.5 for a free radical with a single unpaired electron,
        1.0 for a triplet with two unpaired electrons, such as O_2.)
    ignore_imag_modes : bool
        If True, any imaginary frequencies present after the 3N-5/3N-6 cut
        will not be included in the calculation of the thermochemical
        properties. If False (default), a ValueError will be raised if
        any imaginary frequencies remain after the 3N-5/3N-6 cut.
    """

    def __init__(self, vib_energies, geometry, potentialenergy=0.,
                 atoms=None, symmetrynumber=None, spin=None, natoms=None,
                 ignore_imag_modes=False):
        self.potentialenergy = potentialenergy
        self.geometry = geometry
        self.atoms = atoms
        self.sigma = symmetrynumber
        self.spin = spin
        self.ignore_imag_modes = ignore_imag_modes
        if natoms is None and atoms:
            natoms = len(atoms)
        self.natoms = natoms

        # Sort the vibrations
        vib_energies = list(vib_energies)
        vib_energies.sort(key=np.abs)

        # Cut the vibrations to those needed from the geometry.
        if natoms:
            if geometry == 'nonlinear':
                vib_energies = vib_energies[-(3 * natoms - 6):]
            elif geometry == 'linear':
                vib_energies = vib_energies[-(3 * natoms - 5):]
            elif geometry == 'monatomic':
                vib_energies = []
            else:
                raise ValueError(f"Unsupported geometry: {geometry}")

        # Check for imaginary frequencies.
        vib_energies, n_imag = _clean_vib_energies(
            vib_energies, ignore_imag_modes=ignore_imag_modes
        )
        self.vib_energies = vib_energies
        self.n_imag = n_imag

        self.referencepressure = 1.0e5  # Pa

    def get_enthalpy(self, temperature):
        """Returns the enthalpy, in eV, in the ideal gas approximation
        at a specified temperature (K)."""

        H = 0.
        H += self.potentialenergy

        zpe = self.get_ZPE_correction()
        H += zpe

        Cv_t = 5. / 2. * units.kB  # translational heat capacity (3-d gas)
        H += Cv_t * temperature

        if self.geometry == 'nonlinear':  # rotational heat capacity
            Cv_r = 3. / 2. * units.kB
        elif self.geometry == 'linear':
            Cv_r = units.kB
        elif self.geometry == 'monatomic':
            Cv_r = 0.
        H += Cv_r * temperature

        dH_v = self._vibrational_energy_contribution(temperature)
        H += dH_v

        return zpe, Cv_t * temperature, Cv_r * temperature, dH_v, H


    def get_entropy(self, temperature, pressure):
        """Returns the entropy, in eV/K, in the ideal gas approximation
        at a specified temperature (K) and pressure (Pa)."""

        if self.atoms is None or self.sigma is None or self.spin is None:
            raise RuntimeError('atoms, symmetrynumber, and spin must be '
                               'specified for entropy and free energy '
                               'calculations.')

        S = 0.0

        # Translational entropy (term inside the log is in SI units).
        mass = sum(self.atoms.get_masses()) * units._amu  # kg/molecule
        S_t = (2 * np.pi * mass * units._k *
               temperature / units._hplanck**2)**(3.0 / 2)
        S_t *= units._k * temperature / self.referencepressure
        S_t = units.kB * (np.log(S_t) + 5.0 / 2.0)
        S += S_t

        # Rotational entropy (term inside the log is in SI units).
        if self.geometry == 'monatomic':
            S_r = 0.0
        elif self.geometry == 'nonlinear':
            inertias = (self.atoms.get_moments_of_inertia() * units._amu /
                        (10.0**10)**2)  # kg m^2
            S_r = np.sqrt(np.pi * np.prod(inertias)) / self.sigma
            S_r *= (8.0 * np.pi**2 * units._k * temperature /
                    units._hplanck**2)**(3.0 / 2.0)
            S_r = units.kB * (np.log(S_r) + 3.0 / 2.0)
        elif self.geometry == 'linear':
            inertias = (self.atoms.get_moments_of_inertia() * units._amu /
                        (10.0**10)**2)  # kg m^2
            inertia = max(inertias)  # should be two identical and one zero
            S_r = (8 * np.pi**2 * inertia * units._k * temperature /
                   self.sigma / units._hplanck**2)
            S_r = units.kB * (np.log(S_r) + 1.)
        S += S_r

        # Electronic entropy.
        S_e = units.kB * np.log(2 * self.spin + 1)
        S += S_e

        # Vibrational entropy.
        S_v = self._vibrational_entropy_contribution(temperature)
        S += S_v

        # Pressure correction to translational entropy.
        S_p = - units.kB * np.log(pressure / self.referencepressure)
        S += S_p

        return S_t, S_r, S_e, S_v, S_p, S


    def get_gibbs_energy(self, temperature, pressure):
        """Returns the Gibbs free energy, in eV, in the ideal gas
        approximation at a specified temperature (K) and pressure (Pa)."""

        zpe, Hv_t, Hv_r, dH_v, H = self.get_enthalpy(temperature)
        S_t, S_r, S_e, S_v, S_p, S = self.get_entropy(temperature, pressure)
        G = H - temperature * S

        return G

class ClassicalHarmonicThermo(ThermoChem):
    """Class for calculating thermodynamic properties in the approximation
    that all degrees of freedom are treated harmonically. Here, the 
    semi-classical harmonic formalism is used. 

    Inputs:

    vib_energies : list
        a list of the harmonic energies of the adsorbate (e.g., from
        ase.vibrations.Vibrations.get_energies). The number of
        energies should match the number of degrees of freedom of the
        adsorbate; i.e., 3*n, where n is the number of atoms. Note that
        this class does not check that the user has supplied the correct
        number of energies. Units of energies are eV.
    potentialenergy : float
        the potential energy in eV (e.g., from atoms.get_potential_energy)
        (if potentialenergy is unspecified, then the methods of this
        class can be interpreted as the energy corrections)
    ignore_imag_modes : bool
        If True, any imaginary frequencies will be ignored in the calculation
        of the thermochemical properties. If False (default), an error will
        be raised if any imaginary frequencies are present.
    """

    def __init__(self, vib_energies, potentialenergy=0.,
                 ignore_imag_modes=False):
        self.ignore_imag_modes = ignore_imag_modes

        # Check for imaginary frequencies.
        vib_energies, n_imag = _clean_vib_energies(
            vib_energies, ignore_imag_modes=ignore_imag_modes
        )
        self.vib_energies = vib_energies
        self.n_imag = n_imag

        self.potentialenergy = potentialenergy

    def get_internal_energy(self, temperature):
        """Returns the internal energy, in eV, in the harmonic approximation
        at a specified temperature (K) with the assumption that each degree of
        freedom contributes k_B T (half from potential and half from kinetic
        energy)."""

        U = 0.
        zpe = 0.

        U += self.potentialenergy

        U_v = units.kB * temperature * len(self.vib_energies) 
        U += U_v

        return zpe, U_v, U


    def get_entropy(self, temperature):
        """Returns the entropy, in eV/K, in the harmonic approximation
        at a specified temperature (K)."""

        kT = units.kB * temperature
        S = 0.
        _, U_v, U = self.get_internal_energy(temperature)

        F_v = 0.
        q_v = 1.
        for vib_energy in self.vib_energies:
            F_v += np.log(kT/vib_energy)
        F_v *= -kT
        S_v = (U_v - F_v)/temperature

        S += S_v

        return S


    def get_helmholtz_energy(self, temperature):
        """Returns the Helmholtz free energy, in eV, in the harmonic
        approximation at a specified temperature (K)."""


        zpe, dU_v, U = self.get_internal_energy(temperature)
        S = self.get_entropy(temperature)
        F = U - temperature * S
        
        return F

class ClassicalIdealGasThermo(ThermoChem):
    """Class for calculating thermodynamic properties of a molecule
    based on statistical mechanical treatments in the ideal gas
    approximation.

    Inputs for enthalpy calculations:

    vib_energies : list
        a list of the vibrational energies of the molecule (e.g., from
        ase.vibrations.Vibrations.get_energies). The number of vibrations
        used is automatically calculated by the geometry and the number of
        atoms. If more are specified than are needed, then the lowest
        numbered vibrations are neglected. If either atoms or natoms is
        unspecified, then uses the entire list. Units are eV.
    geometry : 'monatomic', 'linear', or 'nonlinear'
        geometry of the molecule
    potentialenergy : float
        the potential energy in eV (e.g., from atoms.get_potential_energy)
        (if potentialenergy is unspecified, then the methods of this
        class can be interpreted as the energy corrections)
    natoms : integer
        the number of atoms, used along with 'geometry' to determine how
        many vibrations to use. (Not needed if an atoms object is supplied
        in 'atoms' or if the user desires the entire list of vibrations
        to be used.)

    Extra inputs needed for entropy / free energy calculations:

    atoms : an ASE atoms object
        used to calculate rotational moments of inertia and molecular mass
    symmetrynumber : integer
        symmetry number of the molecule. See, for example, Table 10.1 and
        Appendix B of C. Cramer "Essentials of Computational Chemistry",
        2nd Ed.
    spin : float
        the total electronic spin. (0 for molecules in which all electrons
        are paired, 0.5 for a free radical with a single unpaired electron,
        1.0 for a triplet with two unpaired electrons, such as O_2.)
    ignore_imag_modes : bool
        If True, any imaginary frequencies present after the 3N-5/3N-6 cut
        will not be included in the calculation of the thermochemical
        properties. If False (default), a ValueError will be raised if
        any imaginary frequencies remain after the 3N-5/3N-6 cut.
    """

    def __init__(self, vib_energies, geometry, potentialenergy=0.,
                 atoms=None, symmetrynumber=None, spin=None, natoms=None,
                 ignore_imag_modes=False):
        self.potentialenergy = potentialenergy
        self.geometry = geometry
        self.atoms = atoms
        self.sigma = symmetrynumber
        self.spin = spin
        self.ignore_imag_modes = ignore_imag_modes
        if natoms is None and atoms:
            natoms = len(atoms)
        self.natoms = natoms

        # Sort the vibrations
        vib_energies = list(vib_energies)
        vib_energies.sort(key=np.abs)

        # Cut the vibrations to those needed from the geometry.
        if natoms:
            if geometry == 'nonlinear':
                vib_energies = vib_energies[-(3 * natoms - 6):]
            elif geometry == 'linear':
                vib_energies = vib_energies[-(3 * natoms - 5):]
            elif geometry == 'monatomic':
                vib_energies = []
            else:
                raise ValueError(f"Unsupported geometry: {geometry}")

        # Check for imaginary frequencies.
        vib_energies, n_imag = _clean_vib_energies(
            vib_energies, ignore_imag_modes=ignore_imag_modes
        )
        self.vib_energies = vib_energies
        self.n_imag = n_imag

        self.referencepressure = 1.0e5  # Pa

    def get_enthalpy(self, temperature):
        """Returns the enthalpy, in eV, in the ideal gas approximation
        at a specified temperature (K)."""

        kT = units.kB * temperature
        H = 0.
        H += self.potentialenergy

        zpe = 0.

        Cv_t = 5. / 2. * units.kB  # translational heat capacity (3-d gas)
        H += Cv_t * temperature

        if self.geometry == 'nonlinear':  # rotational heat capacity
            Cv_r = 3. / 2. * units.kB
        elif self.geometry == 'linear':
            Cv_r = units.kB
        elif self.geometry == 'monatomic':
            Cv_r = 0.
        H += Cv_r * temperature

        dH_v = 0.5 * kT * len(self.vib_energies)
        H += dH_v

        return zpe, Cv_t * temperature, Cv_r * temperature, dH_v, H


    def get_entropy(self, temperature, pressure):
        """Returns the entropy, in eV/K, in the ideal gas approximation
        at a specified temperature (K) and pressure (Pa)."""

        if self.atoms is None or self.sigma is None or self.spin is None:
            raise RuntimeError('atoms, symmetrynumber, and spin must be '
                               'specified for entropy and free energy '
                               'calculations.')
        kT = units.kB * temperature
        S = 0.0

        # Translational entropy (term inside the log is in SI units).
        mass = sum(self.atoms.get_masses()) * units._amu  # kg/molecule
        S_t = (2 * np.pi * mass * units._k *
               temperature / units._hplanck**2)**(3.0 / 2)
        S_t *= units._k * temperature / self.referencepressure
        S_t = units.kB * (np.log(S_t) + 5.0 / 2.0)
        S += S_t

        # Rotational entropy (term inside the log is in SI units).
        if self.geometry == 'monatomic':
            S_r = 0.0
        elif self.geometry == 'nonlinear':
            inertias = (self.atoms.get_moments_of_inertia() * units._amu /
                        (10.0**10)**2)  # kg m^2
            S_r = np.sqrt(np.pi * np.prod(inertias)) / self.sigma
            S_r *= (8.0 * np.pi**2 * units._k * temperature /
                    units._hplanck**2)**(3.0 / 2.0)
            S_r = units.kB * (np.log(S_r) + 3.0 / 2.0)
        elif self.geometry == 'linear':
            inertias = (self.atoms.get_moments_of_inertia() * units._amu /
                        (10.0**10)**2)  # kg m^2
            inertia = max(inertias)  # should be two identical and one zero
            S_r = (8 * np.pi**2 * inertia * units._k * temperature /
                   self.sigma / units._hplanck**2)
            S_r = units.kB * (np.log(S_r) + 1.)
        S += S_r

        # Electronic entropy.
        S_e = units.kB * np.log(2 * self.spin + 1)
        S += S_e

        # Vibrational entropy.
        zpe, H_t, H_r, dH_v, H = self.get_enthalpy(temperature)
        F_v = 0.
        for vib_energy in self.vib_energies:
            F_v += np.log(kT/vib_energy)
        F_v *= -kT
        S_v = (dH_v- F_v)/temperature
        S += S_v

        # Pressure correction to translational entropy.
        S_p = - units.kB * np.log(pressure / self.referencepressure)
        S += S_p

        return S_t, S_r, S_e, S_v, S_p, S


    def get_gibbs_energy(self, temperature, pressure):
        """Returns the Gibbs free energy, in eV, in the ideal gas
        approximation at a specified temperature (K) and pressure (Pa)."""

        zpe, Hv_t, Hv_r, dH_v, H = self.get_enthalpy(temperature)
        S_t, S_r, S_e, S_v, S_p, S = self.get_entropy(temperature, pressure)
        G = H - temperature * S

        return G

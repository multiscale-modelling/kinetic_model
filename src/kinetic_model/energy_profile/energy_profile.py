import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os, sys
import yaml
from ase.units import kJ, kcal, eV, _Nav
from kinetic_model.energy_profile.plotter import energy_bar, inter_connect, ts_curve

# Apply general settings
with open(os.path.abspath('./settings.yaml'), "r") as f:
            settings = yaml.safe_load(f)

plt.rcParams.update({
    'lines.linewidth': settings['linewidth'],
    'font.size': settings['fontsize'],
    'figure.figsize': (settings['fig_width'], settings['fig_height']),
    'figure.dpi': settings['dpi']
})

# Optional: Apply color palette globally (to cycle through)
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=settings['color_palette'])

class EnergyProfile:
    def __init__(self, data: pd.DataFrame, energy_type, energy_unit, fig_size=None):
        """
        Initialize with a pandas DataFrame containing energetics data.
        """
        self.data = data
        if fig_size:
            self.fig, self.ax = plt.subplots(fig_size=fig_size)
        else:
            self.fig, self.ax = plt.subplots()
        self.energy_type = energy_type
        self.energy_unit = energy_unit
        self.levels = {}

        unit_conversion = {
            'eV': 1,
            'kJ/mol': 1 * eV / kJ * _Nav,
            'kcal/mol': 1 * eV / kcal * _Nav
        }

        energyType2Keywords = {
            'Electronic energy': 'rel_E_eV',
            'Enthalpy': 'rel_H_eV',
            'Gibbs free energy': 'rel_G_eV'
        }

        self.energy_keyword = energyType2Keywords[energy_type]
        self.unit_factor = unit_conversion[energy_unit]
    
    def plot(self):
        """
        Prepares and displays the initial plots.
        """
        self.ax.set_xlabel("Reaction Coordinate")
        self.ax.set_ylabel(f"{self.energy_type} ({self.energy_unit})")
    
    def get_energy_range(self, systems):
        """
        Compute and return the minimum and maximum energy values (after applying unit factor)
        from all intermediates and any transition states (TSs) between them.

        Parameters:
        - systems_ls: list of lists, each inner list contains ordered system labels for one pathway

        Returns:
        - (min_energy, max_energy): tuple of floats
        """
        energy_values = []

        # Get energy for each intermediate
        for system in systems:
            system_data = self.data[self.data['order'] == system]
            if not system_data.empty:
                energy = system_data[self.energy_keyword].iloc[0] * self.unit_factor
                energy_values.append(energy)

        # Get transition state energy if available between adjacent pairs
        for system1, system2 in zip(systems[:-1], systems[1:]):
            ts_order_candidates = [f"{system1}-{system2}", f"{system2}-{system1}"]
            for ts_candidate in ts_order_candidates:
                if ts_candidate in self.data['order'].values:
                    ts_energy = self.data[self.data['order'] == ts_candidate][self.energy_keyword].iloc[0] * self.unit_factor
                    energy_values.append(ts_energy)
                    break  # use only the first valid TS

        if energy_values:
            return min(energy_values), max(energy_values)
        else:
            raise ValueError("No energy values found in the provided systems.")
    
    def add_level(self, system, position, label="", name="", color="#E69F00", alpha=1.0, linewidth=6):
        """
        Add a horizontal level (for reaction intermediate) at the given 
        position and energy.
        """
        system_data = self.data[self.data['order'] == system]
        name = system_data['system_name'].iloc[0]
        energy = system_data[self.energy_keyword].iloc[0] * self.unit_factor
        x, y = energy_bar(position, energy)
        self.ax.plot(x, y, label=label, color=color, alpha=alpha, linewidth=linewidth)
        self.levels[system] = position
    
    def add_linear_connection(self, system1, system2, color="#E69F00", alpha=1.0, linewidth=6):
        """
        Connect two intermediates via a straight line (to represent adsorption/
        desorption step)
        """
        system1_data = self.data[self.data['order'] == system1]
        energy1 = system1_data[self.energy_keyword].iloc[0] * self.unit_factor

        system2_data = self.data[self.data['order'] == system2]
        energy2 = system2_data[self.energy_keyword].iloc[0] * self.unit_factor

        position1 = self.levels[system1] + 1
        position2 = self.levels[system2]

        x, y = inter_connect(position1, position2, energy1, energy2)
        self.ax.plot(x, y, color=color, alpha=alpha, linewidth=linewidth)
    
    def add_ts_connection(self, system1, system2, color="#E69F00", alpha=1.0, linewidth=6):
        """
        Connect two intermediates with a parabolic curve (to represent a 
        transition state).
        """

        system1_data = self.data[self.data['order'] == system1]
        energy1 = system1_data[self.energy_keyword].iloc[0] * self.unit_factor

        system2_data = self.data[self.data['order'] == system2]
        energy2 = system2_data[self.energy_keyword].iloc[0] * self.unit_factor

        position1 = self.levels[system1] + 1
        position2 = self.levels[system2]

        ts_data = self.data[self.data['order'] == f"{system1}-{system2}"]
        ts_energy = ts_data[self.energy_keyword].iloc[0] * self.unit_factor

        x, y = ts_curve([position1, energy1], [position2, energy2], ts_energy)
        self.ax.plot(x, y, color=color, alpha=alpha, linewidth=linewidth)

    def add_connection(self, system1, system2, color="#E69F00", alpha=1.0, linewidth=6):
        """
        Automatically connect two systems via a transition state (curved) if available,
        otherwise via a straight line.
        """

        # Get energies
        energy1 = self.data[self.data['order'] == system1][self.energy_keyword].iloc[0] * self.unit_factor
        energy2 = self.data[self.data['order'] == system2][self.energy_keyword].iloc[0] * self.unit_factor

        # Get positions
        position1 = self.levels[system1] + 1
        position2 = self.levels[system2]

        # Attempt to identify TS entry
        ts_order_candidates = [
            f"{system1}-{system2}",
            f"{system2}-{system1}"
        ]
        ts_energy = None
        for ts_candidate in ts_order_candidates:
            if ts_candidate in self.data['order'].values:
                ts_energy = self.data[self.data['order'] == ts_candidate][self.energy_keyword].iloc[0] * self.unit_factor
                break

        # Draw curved line if TS exists, otherwise straight
        if ts_energy is not None:
            x, y = ts_curve([position1, energy1], [position2, energy2], ts_energy)
        else:
            x, y = inter_connect(position1, position2, energy1, energy2)

        self.ax.plot(x, y, color=color, alpha=alpha, linewidth=linewidth)


    # def add_intermediate_energy_label(self, system, position, fontsize=18, round_digit=None):
    #     """
    #     Add the energy value above the line, centered at (position + 0.5).

    #     Parameters:
    #     - system: str, the key in self.data['order'] used to look up energy
    #     - position: float, the x-axis start of the level
    #     - fontsize: int, font size of the label
    #     - round: int, number of decimal places to round based on self.unit (if None, auto-set)
    #     """
    #     # Auto-set rounding based on unit
    #     if round_digit is None:
    #         if self.energy_unit == "eV":
    #             round_digit = 2
    #             offset = 0.05
    #         elif self.energy_unit == "kJ/mol":
    #             round_digit = 0
    #             offset = 2
    #         elif self.energy_unit == "kcal/mol":
    #             round_digit = 1
    #             offset = 0.5
    #         else:
    #             round_digit = 2  # default fallback
    #             offset = 0.05

    #     # Get energy
    #     system_data = self.data[self.data['order'] == system]
    #     energy = system_data[self.energy_keyword].iloc[0] * self.unit_factor

    #     # Format label
    #     energy_str = f"{round(energy, round_digit):.{round_digit}f}"

    #     # Add text centered at (position + 0.5, energy + small offset)
    #     self.ax.text(
    #         position + 0.5,
    #         energy + offset,  # you may adjust the vertical offset if needed
    #         energy_str,
    #         fontsize=fontsize,
    #         ha='center',
    #         va='bottom'
    #     )

    def add_intermediate_energy_label(self, system, position, fontsize=18, round_digit=None):
        """
        Add the energy value of an intermediate above its horizontal level.
        The label is centered at (position + 0.5) on the x-axis and slightly 
        offset above the energy level on the y-axis.

        Parameters:
        - system: str, identifier for the intermediate (used to query self.data)
        - position: float, x-axis start position of the energy level
        - fontsize: int, font size of the energy label
        - round_digit: int or None, number of decimal places to round (auto-set by energy_unit if None)
        """

        # Determine decimal precision and y-offset based on energy unit
        round_by_unit = {"eV": 2, "kJ/mol": 0, "kcal/mol": 1}
        offset_by_unit = {"eV": 0.02, "kJ/mol": 2, "kcal/mol": 0.5}
        round_digit = round_by_unit.get(self.energy_unit, 2) if round_digit is None else round_digit
        offset = offset_by_unit.get(self.energy_unit, 0.02)

        # Retrieve and format energy
        system_data = self.data[self.data['order'] == system]
        energy = system_data[self.energy_keyword].iloc[0] * self.unit_factor
        energy_str = f"{round(energy, round_digit):.{round_digit}f}"

        # Draw energy label
        self.ax.text(
            position + 0.5,
            energy + offset,
            energy_str,
            fontsize=fontsize,
            ha='center',
            va='bottom'
        )

    def add_ts_energy_label(self, system1, system2, fontsize=14, round_digit=None):
        """
        Add the transition state (TS) energy label above the curved connection 
        between two systems. The label is centered at the midpoint of the curve 
        (x = position1 + 0.5) and offset slightly above the TS energy.

        Parameters:
        - system1: str, label of the first intermediate
        - system2: str, label of the second intermediate
        - fontsize: int, font size of the energy label
        - round_digit: int or None, number of decimal places (auto-set by unit if None)
        """

        # Default rounding and vertical offset based on energy unit
        round_by_unit = {"eV": 2, "kJ/mol": 0, "kcal/mol": 1}
        offset_by_unit = {"eV": 0.02, "kJ/mol": 2, "kcal/mol": 0.5}
        round_digit = round_by_unit.get(self.energy_unit, 2) if round_digit is None else round_digit
        offset = offset_by_unit.get(self.energy_unit, 0.02)

        # Attempt to locate TS energy via naming conventions
        ts_order_candidates = [f"{system1}-{system2}", f"{system2}-{system1}"]
        ts_energy = None
        for ts_candidate in ts_order_candidates:
            if ts_candidate in self.data['order'].values:
                ts_energy = self.data[self.data['order'] == ts_candidate][self.energy_keyword].iloc[0] * self.unit_factor
                break

        if ts_energy is None:
            return  # silently skip if no TS is found

        # Compute label coordinates
        position1 = self.levels[system1] + 1
        label_x = position1 + 0.5
        label_y = ts_energy + offset

        # Draw the label
        self.ax.text(
            label_x,
            label_y,
            f"{round(ts_energy, round_digit):.{round_digit}f}",
            fontsize=fontsize,
            ha='center',
            va='bottom'
        )

    # def add_ts_energy_label(self, system1, system2, fontsize=14, round_digit=None):
    #     """
    #     Add a TS energy label centered at (position1 + 0.5, ts_energy + offset).
        
    #     Parameters:
    #     - system1: str, starting intermediate
    #     - system2: str, ending intermediate
    #     - fontsize: int, font size for the label
    #     - round: int or None, decimal rounding based on unit
    #     """

    #     # Auto-set rounding based on unit
    #     if round_digit is None:
    #         if self.energy_unit == "eV":
    #             round_digit = 2
    #             offset = 0.05
    #         elif self.energy_unit == "kJ/mol":
    #             round_digit = 0
    #             offset = 2
    #         elif self.energy_unit == "kcal/mol":
    #             round_digit = 1
    #             offset = 0.5
    #         else:
    #             round_digit = 2  # default fallback
    #             offset = 0.05

    #     # Try finding TS data using both naming conventions
    #     ts_order_candidates = [
    #         f"{system1}-{system2}",
    #         f"{system2}-{system1}"
    #     ]

    #     ts_energy = None
    #     for ts_candidate in ts_order_candidates:
    #         if ts_candidate in self.data['order'].values:
    #             ts_energy = self.data[self.data['order'] == ts_candidate][self.energy_keyword].iloc[0] * self.unit_factor
    #             break

    #     if ts_energy is None:
    #         return

    #     # Determine x-axis label position (middle of connector)
    #     position1 = self.levels[system1] + 1
    #     label_x = position1 + 0.5

    #     # Format energy string
    #     ts_energy_str = f"{round(ts_energy, round_digit):.{round_digit}f}"

    #     # Place the label
    #     self.ax.text(
    #         label_x,
    #         ts_energy + offset,  # vertical offset
    #         ts_energy_str,
    #         fontsize=fontsize,
    #         ha='center',
    #         va='bottom'
    #     )

    def add_structure_name(self, system, position, fontsize=18, color='blue'):
        """
        Add the structure (system) name below the energy level line,
        centered horizontally at (position + 0.5) and offset slightly below the energy value.

        Parameters:
        - system: str, the key in self.data['order'] used to look up energy and name
        - position: float, the x-axis start position of the level
        - fontsize: int, font size of the label
        - color: str, color of the label text
        """

        # Define vertical offset based on energy unit
        unit_offsets = {"eV": 0.02, "kJ/mol": 2, "kcal/mol": 0.5}
        offset = unit_offsets.get(self.energy_unit, 0.02)

        # Get system data
        system_data = self.data[self.data['order'] == system]
        energy = system_data[self.energy_keyword].iloc[0] * self.unit_factor
        name = system_data['system_name'].iloc[0]

        # Format and add label
        self.ax.text(
            position + 0.5,
            energy - offset,
            f"[{name}]",
            fontsize=fontsize,
            color=color,
            ha='center',
            va='top'
        )

    def add_level_from_subset(self, subset, system, position, color, linewidth, alpha, label=None):
        system_data = subset[subset['order'] == system]
        self.levels[system] = position
        if system_data.empty:
            return
        energy = system_data[self.energy_keyword].iloc[0] * self.unit_factor
        x, y = energy_bar(position, energy)
        self.ax.plot(x, y, label=label, color=color, alpha=alpha, linewidth=linewidth)

    def add_connection_from_subset(self, subset, system1, system2, color, linewidth, alpha):
        ts_names = [f"{system1}-{system2}", f"{system2}-{system1}"]
        ts_energy = None
        for ts in ts_names:
            ts_data = subset[subset['order'] == ts]
            if not ts_data.empty:
                ts_energy = ts_data[self.energy_keyword].iloc[0] * self.unit_factor
                break

        e1 = subset[subset['order'] == system1][self.energy_keyword].iloc[0] * self.unit_factor
        e2 = subset[subset['order'] == system2][self.energy_keyword].iloc[0] * self.unit_factor
        p1 = self.levels[system1] + 1
        p2 = self.levels[system2]

        if ts_energy is not None:
            x, y = ts_curve([p1, e1], [p2, e2], ts_energy)
        else:
            x, y = inter_connect(p1, p2, e1, e2)

        self.ax.plot(x, y, color=color, linewidth=linewidth, alpha=alpha)

    def add_intermediate_energy_label_from_subset(self, subset, system, position, fontsize=14):
        energy = subset[subset['order'] == system][self.energy_keyword].iloc[0] * self.unit_factor
        offset = 0.05 if self.energy_unit == "eV" else 2
        self.ax.text(position + 0.5, energy + offset,
                     f"{energy:.2f}", fontsize=fontsize, ha='center', va='bottom')

    def add_ts_energy_label_from_subset(self, subset, system1, system2, fontsize=14):
        ts_names = [f"{system1}-{system2}", f"{system2}-{system1}"]
        for ts in ts_names:
            ts_data = subset[subset['order'] == ts]
            if not ts_data.empty:
                ts_energy = ts_data[self.energy_keyword].iloc[0] * self.unit_factor
                p1 = self.levels[system1] + 1
                offset = 0.05 if self.energy_unit == "eV" else 2
                self.ax.text(p1 + 0.5, ts_energy + offset,
                             f"{ts_energy:.2f}", fontsize=fontsize, ha='center', va='bottom')
                break
    
    def plot_pathway(self,
                     model: str,
                     pathway: str,
                     systems: list[str],
                     label: str = None,
                     color: str = "#E69F00",
                     add_energies: bool = False,
                     add_names: bool = False,
                     linewidth=6,
                     alpha=1.0):
        """
        Plot one reaction sequence (systems) on the same axes.
        
        systems:    ordered list of 'order' keys, e.g. ['1','2','3']
        label:      legend label for the first level (pass None to skip legend)
        color:      color for levels & connections
        add_energies: if True, draw intermediate TS & intermediate energy labels
        add_names:    if True, draw structure names beneath each level
        """
        from collections import OrderedDict
        subset = self.data[(self.data['model'] == model) & (self.data['pathway'] == pathway)]
        if subset.empty:
            raise ValueError(f"No data for model='{model}', pathway='{pathway}'")
        
        # initialize shared position map if needed
        if not hasattr(self, 'position_map'):
            self.position_map = OrderedDict()
        
        # assign x-positions (2*idx of first appearance)
        for syss in systems:
            for i, sys in enumerate(syss):
                position = 2 * i
                if sys not in self.position_map:
                    self.position_map[sys] = position
        
        # draw levels
        count = 0
        for idx, syss in enumerate(systems):
            for sys in syss:
                pos = self.position_map[sys]
                if count == 0 and label is not None:
                    self.add_level_from_subset(subset=subset,
                                            system=sys, 
                                            position=pos, 
                                            label=label, 
                                            color=color, 
                                            linewidth=linewidth,
                                            alpha=alpha)
                else:
                    self.add_level_from_subset(subset=subset,
                                            system=sys, 
                                            position=pos, 
                                            color=color, 
                                            linewidth=linewidth,
                                            alpha=alpha)
                count += 1
        
            # draw connections + TS labels
            for s1, s2 in zip(syss[:-1], syss[1:]):
                self.add_connection_from_subset(subset=subset,
                                                system1=s1, 
                                                system2=s2, 
                                                color=color,
                                                linewidth=linewidth,
                                                alpha=alpha)
                # self.add_ts_energy_label_from_subset(subset=subset,
                #                                      system1=s1, 
                #                                      system2=s2)
        
        # optional energy labels
        if add_energies:
            for syss in systems:
                for sys in syss:
                    pos = self.position_map[sys]
                    self.add_intermediate_energy_label_from_subset(subset=subset,
                                                                system=sys,
                                                                position=pos)
                for s1, s2 in zip(syss[:-1], syss[1:]):
                    self.add_ts_energy_label_from_subset(subset=subset,
                                                        system1=s1, 
                                                        system2=s2)
            
        # optional structure names
        # if add_names:
        #     for sys in systems:
        #         pos = self.position_map[sys]
        #         self.add_structure_name_from_subset(system=sys, position=pos)
    

    def show(self, xmin=None, xmax=None, ymin=None, ymax=None, padding=None,  legend=False, save_fig=False, outfile=None):
        """
        Display the completed figure.
        """
        if legend:
            self.ax.legend(loc='best')
        
        if all(v is not None for v in (padding, ymin, ymax)):
            self.ax.set_ylim(ymin - padding, ymax + padding)

        if all(v is not None for v in (xmin, xmax)):
            self.ax.set_xlim(xmin, xmax)
        
        self.ax.set_xticklabels([])
        self.ax.grid(True, axis='y', linestyle='--')
        self.ax.relim()
        self.ax.autoscale_view()

        if save_fig:
            if outfile == None:
                outfile = 'out.pdf'
            plt.savefig(outfile, dpi=600)
        plt.show()
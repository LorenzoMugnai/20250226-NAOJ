from re import S
from astropy.io import ascii
from astropy.table import Table
import astropy.units as u
import h5py
import os
import numpy as np
from tqdm import tqdm
from mol_to_ratios import calculate_ratios_with_uncertainties
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from astropy.table import vstack
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Update global font settings for all plots
plt.rc('font', size=14)          # controls default text sizes
plt.rc('axes', titlesize=16)     # fontsize of the axes title
plt.rc('axes', labelsize=14)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
plt.rc('legend', fontsize=12)    # fontsize of the legend
plt.rc('figure', titlesize=18)   # fontsize of the figure title

class Inspector:

    def __init__(self, input_table):
        """
        Inspector(input_table)

        A class for aggregating, processing, and visualising exoplanet atmospheric retrieval results, forward models, and derived elemental ratios.

        The `Inspector` class facilitates the comparison between retrieved atmospheric parameters and expected forward model values. It supports loading and managing retrieval outputs from HDF5 files, computing molecular and elemental ratios (with uncertainties), integrating additional observational data (e.g., number of Ariel observations), and generating summary statistics and plots. It also supports identifying missing data and visualising discrepancies between predicted and retrieved parameters.

        Parameters
        ----------
        input_table : str
            Path to the input CSV or ASCII table listing planet metadata and identifiers.

        Attributes
        ----------
        targetlist : astropy.table.Table
            Table containing the initial list of planetary targets.
        out_targetlist : astropy.table.Table
            Output table that aggregates retrieved parameters, computed ratios, and observational metadata for valid planets.

        Notes
        -----
        - Retrievals are expected in a standard HDF5 format following the `Output/Solutions/solution0/fit_params` convention.
        - Computation of elemental ratios requires specific keys to be present (e.g., log-scaled abundances and uncertainties).
        - This class supports extensive plotting for analysis and validation of atmospheric parameters across a sample of planets.
        """

        self.targetlist = self._load_targetlist(input_table)
        
        self.out_targetlist = self._prepare_out_table()
    
    def _load_targetlist(self, input_table):
        def _is_float(x):
            try:
                float(x)
                return True
            except:
                return False

        def _standardise_table(t):
            numeric_columns = ['Star V Mag', 'Star Mass', 'Star Radius']
            for col in numeric_columns:
                if col in t.colnames:
                    t[col] = [float(x) if _is_float(x) else np.nan for x in t[col]]
                    print("Converting column", col, "to float64 from", t[col].dtype)
                    t[col] = t[col].astype('float64')

            for col in t.colnames:
                if t[col].dtype.kind in {'U', 'S'}:
                    print("Converting column", col, "to U128 from ", t[col].dtype)
                    t[col] = t[col].astype('U128')

            return t

        if isinstance(input_table, list):
            tables = []
            for tbl in input_table:
                print("loading table", tbl)
                t = ascii.read(tbl, encoding='utf-8-sig')
                print("found", len(t), "rows")
                t = _standardise_table(t)
                tables.append(t)

            # Fai vstack ora che tutto è omogeneo
            new_tab = vstack(tables, join_type='outer')
            print("combined in", len(new_tab), "rows")
            return new_tab
        else:
            t = ascii.read(input_table, encoding='utf-8-sig')
            return _standardise_table(t)

    def _prepare_out_table(self):
        """
        Initialise an empty output table with the same structure as the input target list.

        Creates a new `astropy.table.Table` object using the same column names and data types
        as the original input table (`self.targetlist`), but with no rows. This table is used 
        to store processed data for planets with available retrievals.

        Returns
        -------
        astropy.table.Table
            An empty table with column names and types matching the input table.
        """
        return Table(names=self.targetlist.colnames, dtype=[self.targetlist[col].dtype for col in self.targetlist.colnames])
        
    def save_table(self, out_tab_fname):
        """
        Save the processed output table to a CSV file.

        Writes the contents of `self.out_targetlist` to disk in CSV format using `astropy.io.ascii`.
        Existing files with the same name will be overwritten.

        Parameters
        ----------
        out_tab_fname : str
            Path to the output file where the table should be saved.
        """
        ascii.write(self.out_targetlist, out_tab_fname, format='csv', overwrite=True)   
        print(f"Table saved to {out_tab_fname}") 
    
    def load_retrievals(self, retrieval_folders):
        """
        Load atmospheric retrieval results from HDF5 files and populate the output table.

        Parameters
        ----------
        retrieval_folders : list of str
            List of directories to search for retrieval files.

        Notes
        -----
        See previous docstring. This version searches multiple folders.
        """
        print("Loading retrieval data from multiple folders...")

        additional_columns = {}
        valid_planets = []

        for i, planet in tqdm(enumerate(self.targetlist["Planet Name"]), total=len(self.targetlist)):
            file_found = False
            for folder in retrieval_folders:
                path = os.path.join(folder, planet)
                file = os.path.join(path, f"{planet}_retrieval.hdf5")

                if os.path.exists(file):
                    file_found = True
                    break  # Stop at the first folder where the file is found

            if not file_found:
                print("-> File not found:", planet)
                continue

            valid_planets.append((i, file))  # Store index and path to file

            with h5py.File(file, 'r') as f:
                solution_path = "Output/Solutions/solution0/fit_params"
                if solution_path in f:
                    additional_columns["H2/He"] = f["ModelParameters/Chemistry/ratio"][()].dtype
                    for key in f[solution_path].keys():
                        additional_columns[key] = f[f"{solution_path}/{key}/value"][()].dtype
                        additional_columns[key + "_sigma_p"] = f[f"{solution_path}/{key}/sigma_p"][()].dtype
                        additional_columns[key + "_sigma_m"] = f[f"{solution_path}/{key}/sigma_m"][()].dtype

        print(f"Found {len(valid_planets)} valid planets")

        self.out_targetlist = Table(
            names=self.targetlist.colnames,
            dtype=[self.targetlist[col].dtype for col in self.targetlist.colnames]
        )

        for col_name, col_dtype in additional_columns.items():
            if col_name not in self.out_targetlist.colnames:
                if np.issubdtype(col_dtype, np.number):
                    self.out_targetlist[col_name] = np.full(0, np.nan, dtype=col_dtype)
                else:
                    self.out_targetlist[col_name] = np.full(0, "", dtype="U20")

        for i, file in tqdm(valid_planets, total=len(valid_planets)):
            planet = self.targetlist["Planet Name"][i]
            planet_data = {col: self.targetlist[i][col] for col in self.targetlist.colnames}

            with h5py.File(file, 'r') as f:
                solution_path = "Output/Solutions/solution0/fit_params"
                if solution_path in f:
                    planet_data["H2/He"] = f["ModelParameters/Chemistry/ratio"][()]
                    for key in f[solution_path].keys():
                        planet_data[key] = f[f"{solution_path}/{key}/value"][()]
                        planet_data[key + "_sigma_p"] = f[f"{solution_path}/{key}/sigma_p"][()]
                        planet_data[key + "_sigma_m"] = f[f"{solution_path}/{key}/sigma_m"][()]

            for col in self.out_targetlist.colnames:
                if col not in planet_data:
                    if np.issubdtype(self.out_targetlist[col].dtype, np.integer):
                        planet_data[col] = -1
                    elif np.issubdtype(self.out_targetlist[col].dtype, np.number):
                        planet_data[col] = np.nan
                    else:
                        planet_data[col] = ""

            self.out_targetlist.add_row([planet_data[col] for col in self.out_targetlist.colnames])

    def missing_planets(self):
        """
        Print the list and number of planets from the input table that are missing in the output table.

        Compares the list of planet names in `self.targetlist` with those in `self.out_targetlist`,
        identifying any planets that were not successfully processed during the retrieval loading step.

        Prints
        ------
        - Total number of missing planets.
        - Indices of missing planets in the input table.
        - Names of missing planets.
        """
        idx = [i for i, name in enumerate(self.targetlist["Planet Name"]) if name not in self.out_targetlist["Planet Name"]]
        print(f"Missing planets: {len(idx)}")
        print(idx)
        print(self.targetlist["Planet Name"][idx])
        
    def create_missing_planets_table(self):
        """
        Create a subtable of planets that are missing from the output table.

        Identifies all planets in the input table (`self.targetlist`) that do not appear in the output
        table (`self.out_targetlist`), and returns them as a new `astropy.table.Table`.

        Returns
        -------
        missing_planets : astropy.table.Table
            A table containing only the rows for planets that are missing in `self.out_targetlist`.
        """
        idx = [i for i, name in enumerate(self.targetlist["Planet Name"]) if name not in self.out_targetlist["Planet Name"]]
        missing_planets = self.targetlist[idx]
        return missing_planets

        
    def load_ariel_data(self, forwards_folders):
        """
        Load Ariel observation data (number of observations and total observing time) for each planet.

        Parameters
        ----------
        forwards_folders : list of str
            List of directories where forward model HDF5 files may be found.

        Outputs
        -------
        Updates the columns `nobs` and `obs_time` in `self.out_targetlist` for planets with available data.
        """
        print("Loading Ariel data")

        # Ensure the required columns exist
        for col in ["nobs", "obs_time"]:
            if col not in self.out_targetlist.colnames:
                self.out_targetlist[col] = np.nan  # Initialise missing columns

        for i, planet_data in tqdm(enumerate(self.out_targetlist), total=len(self.out_targetlist)):
            planet_dict = {col: planet_data[col] for col in self.out_targetlist.colnames}
            planet_name = planet_dict["Planet Name"]

            for folder in forwards_folders:
                fname = os.path.join(folder, f"{planet_name}/{planet_name}.hdf5")
                if os.path.isfile(fname):
                    break
            else:
                print("File not found:", planet_name)
                continue

            with h5py.File(fname, 'r') as f:
                ariel_path = f["Output/Spectra"]
                planet_dict["nobs"] = ariel_path["instrument_nobs"][()]
                planet_dict["obs_time"] = planet_dict['Transit Duration [s]'] * planet_dict["nobs"] * u.s.to(u.hr)

            # Aggiorna la riga nella tabella
            self.out_targetlist[i] = [planet_dict[col] for col in self.out_targetlist.colnames]
             
    def load_profiles(self, comp_folders, pt_folders):
        """
        Load chemical composition and temperature profiles from external CSV files for each planet.

        Parameters
        ----------
        comp_folders : list of str
            List of directories containing composition profile CSV files.
        pt_folders : list of str
            List of directories containing pressure-temperature profile CSV files.
        """
        print("Loading composition and PT profiles...")

        for col in ["H2_profile", "He_profile", "H2O_profile", "CO_profile", "CO2_profile", "CH4_profile", "T_profile"]:
            if col not in self.out_targetlist.colnames:
                self.out_targetlist[col] = np.nan

        def find_file(folders, keyword, selection):
            for path in folders:
                if not os.path.isdir(path):
                    continue
                file_list = os.listdir(path)
                matching_files = [file for file in file_list if keyword in file and selection in file and file.endswith(".csv")]
                if matching_files:
                    return os.path.join(path, matching_files[0])
            return None

        for i, planet_data in tqdm(enumerate(self.out_targetlist), total=len(self.out_targetlist)):
            planet_dict = {col: planet_data[col] for col in self.out_targetlist.colnames}
            planet_key = planet_data['Planet Name'].split("_")[0]
            planet_key += "_"

            comp_file = find_file(comp_folders, planet_key, selection="Comp")
            if not comp_file:
                print(f"[WARN] Composition file not found for: {planet_data['Planet Name']}")
            else:
                data = ascii.read(comp_file, format='no_header', delimiter=',', comment="#")
                column_map = ["H2", "H2O", "CO", "CO2", "CH4", "He"]
                data.rename_columns(data.colnames, column_map)

                for column in column_map:
                    planet_dict[f"{column}_profile"] = np.median(data[column])

            pt_file = find_file(pt_folders, planet_key, selection="PT")
            if not pt_file:
                print(f"[WARN] PT file not found for: {planet_data['Planet Name']}")
            else:
                data = ascii.read(pt_file, format='no_header', delimiter=',', comment="#")
                planet_dict[f"T_profile"] = np.median(data["col2"])

            self.out_targetlist[i] = [planet_dict[col] for col in self.out_targetlist.colnames]

    def compute_elemental_ratios(self):
        """
        Compute elemental abundance ratios and associated uncertainties for each planet.

        This method converts retrieved log-molecular abundances (e.g., `log_H2O`, `log_CO`, etc.)
        into linear space, propagates their uncertainties, and calculates key elemental ratios
        using the `calculate_ratios_with_uncertainties` function.

        The elemental ratios include:
        - O/H (oxygen-to-hydrogen)
        - C/O (carbon-to-oxygen)
        - He/H (helium-to-hydrogen)
        as well as intermediate quantities such as total H, C, O, H₂, He, and their uncertainties.

        Required molecular inputs:
        - log-scaled abundances: `log_H2O`, `log_CO`, `log_CO2`, `log_CH4`
        - corresponding uncertainties: `log_*_sigma_p` and `log_*_sigma_m`
        - H₂/He ratio from retrievals: `H2/He`

        Notes
        -----
        - All computed values and uncertainties are added to the output table (`self.out_targetlist`).
        - Columns are created if not already present.
        - log-abundances are assumed to be base-10 logarithms.

        Raises
        ------
        None

        Outputs
        -------
        Updates `self.out_targetlist` with computed molecular abundances, elemental ratios,
        and their propagated uncertainties.
        """
        print("computing elemental ratios")
        """Compute elemental ratios for each planet in out_targetlist."""
        
        required_columns = ["O/H", "O/H_sigma", "C/O", "C/O_sigma", "He/H", "He/H_sigma", 
                            "He", "H2", "H2O", "CO", "CO2", "CH4", "H2O_sigma", "CO_sigma", 
                            "CO2_sigma", "CH4_sigma", "He_sigma", "H2_sigma", "H", "C", "O", 
                            "H_sigma", "C_sigma", "O_sigma"]
        
        # Ensure all necessary columns exist
        for col in required_columns:
            if col not in self.out_targetlist.colnames:
                self.out_targetlist[col] = np.nan

        # Iterate through each planet in the target list
        for i, planet_data in tqdm(enumerate(self.out_targetlist), total=len(self.out_targetlist)):
            planet_dict = {col: planet_data[col] for col in self.out_targetlist.colnames}
            
            ln10 = np.log(10)
            
            # Retrieve molecular fractions and uncertainties
            H2O = 10**planet_dict.get("log_H2O", np.nan)
            CO = 10**planet_dict.get("log_CO", np.nan)
            CO2 = 10**planet_dict.get("log_CO2", np.nan)
            CH4 = 10**planet_dict.get("log_CH4", np.nan)
            
            sigma_H2O = ln10 * H2O * np.mean([planet_dict.get("log_H2O_sigma_p", np.nan), planet_dict.get("log_H2O_sigma_m", np.nan)])
            sigma_CO = ln10 * CO * np.mean([planet_dict.get("log_CO_sigma_p", np.nan), planet_dict.get("log_CO_sigma_m", np.nan)])
            sigma_CO2 = ln10 * CO2 * np.mean([planet_dict.get("log_CO2_sigma_p", np.nan), planet_dict.get("log_CO2_sigma_m", np.nan)])
            sigma_CH4 = ln10 * CH4 * np.mean([planet_dict.get("log_CH4_sigma_p", np.nan), planet_dict.get("log_CH4_sigma_m", np.nan)])
            
            planet_dict.update({
                "H2O": H2O, "CO": CO, "CO2": CO2, "CH4": CH4,
                "H2O_sigma": sigma_H2O, "CO_sigma": sigma_CO,
                "CO2_sigma": sigma_CO2, "CH4_sigma": sigma_CH4,
            })
            
            H2_He = planet_dict.get("H2/He", np.nan)
            
            # Compute elemental ratios
            ratios = calculate_ratios_with_uncertainties(H2_He, H2O, CO, CO2, CH4,
                                                        sigma_H2O, sigma_CO, sigma_CO2, sigma_CH4)
            
            # Update planet_dict with computed ratios
            for key in ["O/H", "C/O", "He/H", "He", "H2", "H", "C", "O"]:
                planet_dict[key] = ratios[key][0]
                planet_dict[key + "_sigma"] = ratios[key][1]
                

            # Update the table row with computed values
            self.out_targetlist[i] = [planet_dict[col] for col in self.out_targetlist.colnames]

    def compute_profile_ratios(self):
        """
        Compute elemental abundance ratios from median profile values for each planet.

        This method uses the median values of molecular abundances obtained from external 
        composition profile files (e.g., `H2O_profile`, `CO_profile`, etc.) to estimate 
        elemental ratios and atomic quantities. It employs the 
        `calculate_ratios_with_uncertainties` function with a placeholder value for H₂/He 
        (assumed negligible) since actual uncertainties are not provided for profile data.

        The computed quantities include:
        - Elemental ratios: `O/H_profile`, `C/O_profile`, `He/H_profile`
        - Atomic quantities: `He_profile`, `H2_profile`, `H_profile`, `C_profile`, `O_profile`

        Notes
        -----
        - Columns are created in `self.out_targetlist` if not already present.
        - Uncertainties are not computed in this method due to lack of error propagation input.
        - The H₂/He ratio is fixed to an extremely low value to minimise its influence 
        in the ratio calculation.

        Raises
        ------
        None

        Outputs
        -------
        Updates `self.out_targetlist` with profile-based elemental ratios and atomic quantities.
        """
        print("computing profile ratios")
        
        profile_columns = ["O/H_profile", "C/O_profile", "He/H_profile", "He_profile", "H2_profile", "H_profile", "C_profile", "O_profile"]
        
        # Ensure all necessary profile columns exist
        for col in profile_columns:
            if col not in self.out_targetlist.colnames:
                self.out_targetlist[col] = np.nan
        
        # Iterate through each planet in the target list
        for i, planet_data in tqdm(enumerate(self.out_targetlist), total=len(self.out_targetlist)):
            planet_dict = {col: planet_data[col] for col in self.out_targetlist.colnames}
            
            # Retrieve molecular fractions from planet_dict
            H2O = planet_dict.get("H2O_profile", np.nan)
            CO = planet_dict.get("CO_profile", np.nan)
            CO2 = planet_dict.get("CO2_profile", np.nan)
            CH4 = planet_dict.get("CH4_profile", np.nan)
            H2_He = 1e-100  # Placeholder value
            
            ratios = calculate_ratios_with_uncertainties(H2_He, H2O, CO, CO2, CH4)
            
            # Add computed ratios to the planet dictionary
            planet_dict.update({
                "O/H_profile": ratios["O/H"],
                "C/O_profile": ratios["C/O"],
                "He/H_profile": ratios["He/H"],
                "He_profile": ratios["He"],
                "H2_profile": ratios["H2"],
                "H_profile": ratios["H"],
                "C_profile": ratios["C"],
                "O_profile": ratios["O"],
            })
            
            # Update the table row with computed values
            self.out_targetlist[i] = [planet_dict[col] for col in self.out_targetlist.colnames]
            
    def plot_nobs(self, selections=None, labels=None, colours=None, threshold=20):
        """
        Plot histogram(s) of the number of Ariel observations per planet, with optional selections.

        Parameters
        ----------
        selections : list of array-like, optional
            Boolean or index arrays to select subsets of the output table. If None, full dataset is used.
        labels : list of str, optional
            Labels corresponding to each selection for the legend.
        colours : list of str, optional
            Colours to use for each selection's histogram.
        threshold : int, default=20
            Threshold to draw the vertical reference line and split count annotations.

        Outputs
        -------
        Displays a matplotlib histogram.
        """
        if selections is None:
            selections = [None]

        if labels is None:
            labels = [f"Selection {i+1}" for i in range(len(selections))]

        if colours is None:
            base_colours = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
            colours = base_colours[:len(selections)]

        nobs_data_full = np.array(self.out_targetlist["nobs"])
        bins = np.concatenate([np.linspace(0, threshold, 31), [np.max(nobs_data_full)]])

        fig = plt.figure(figsize=(10, 6))

        for sel, label, colour in zip(selections, labels, colours):
            nobs_data = nobs_data_full if sel is None else nobs_data_full[sel]

            below = np.sum(nobs_data < threshold)
            above = np.sum(nobs_data >= threshold)
            # Plot histogram
            counts, bin_edges, _ = plt.hist(nobs_data, bins=bins, edgecolor="black", alpha=0.5, color=colour, label=label)

            plt.annotate(
                f"< {threshold}: {below}\n≥ {threshold}: {above}", 
                xy=(0.85, 0.9 - 0.1 * labels.index(label)), xycoords='axes fraction',
                ha='left', va='top', fontsize=10,
                bbox=dict(facecolor='white', edgecolor=colour, alpha=0.5)
            )

        # Styling
        plt.xlabel("Number of Observations (nobs)", fontsize=14)
        plt.ylabel("Frequency", fontsize=14)
        plt.title("Distribution of Observations (nobs)", fontsize=16)
        # plt.xticks(list(range(0, threshold, 10)) + [threshold+10], labels=list(range(0, threshold, 10)) + [f"{threshold}+"])
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        plt.xlim(0, threshold + 20)
        plt.legend()
        plt.tight_layout()
        plt.show()
        return fig


    def plot_obs_time(self, selections=None, labels=None, colours=None, show_cumulative=True, threshold=200, all_combined=True, title=None):
        """
        Plot histogram(s) of total observing time with optional cumulative curves.

        Parameters
        ----------
        selections : list of array-like, optional
            A list of boolean or integer index arrays to select subsets from the output table.
            If None, the full dataset is used as a single selection.
        labels : list of str, optional
            A list of labels corresponding to each selection for the legend.
        colours : list of str, optional
            A list of colours for the histograms.
        show_cumulative : bool, default=True
            Whether to show cumulative observing time curves on a secondary axis.
        threshold : float, default=200
            Value in hours used to count and annotate planets with obs_time > threshold.
        """
        if selections is None:
            selections = [None]

        if labels is None:
            labels = [f"Selection {i+1}" for i in range(len(selections))]

        if colours is None:
            base_colours = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple', 'tab:cyan']
            colours = base_colours[:len(selections)]

        fig, ax1 = plt.subplots(figsize=(10, 8))
        ax2 = ax1.twinx() if show_cumulative else None

        obs_all = np.array(self.out_targetlist["obs_time"])
        bins = np.concatenate([
            np.linspace(0, threshold, 31),
            [np.max(obs_all)]
        ])

        offset = 1
        for sel, label, colour in zip(selections, labels, colours):
            obs_time_data = obs_all if sel is None else obs_all[sel]

            # Histogram
            counts, _, _ = ax1.hist(obs_time_data, bins=bins, edgecolor="black", alpha=0.3, label=label, color=colour)
            offset += 2
            
            below = np.sum(obs_time_data < threshold)
            above = np.sum(obs_time_data >= threshold)
            plt.annotate(
                f"< {threshold}: {below}\n≥ {threshold}: {above}", 
                xy=(0.85, 0.9 - 0.1 * labels.index(label)), xycoords='axes fraction',
                ha='left', va='top', fontsize=10,
                bbox=dict(facecolor='white', edgecolor=colour, alpha=0.5)
            )


            # Cumulative
            if show_cumulative and ax2:
                bin_indices = np.digitize(obs_time_data, bins, right=False)
                cumulative_hours = [np.sum(obs_time_data[bin_indices <= i]) for i in range(1, len(bins))]
                ax2.plot(bins[:-1], cumulative_hours, linestyle="--", marker="o", label=f"Cumulative {label}", color=colour)

        # --- Combined selection ---
        if len(selections) > 1 and all_combined:
            combined_mask = np.zeros(len(obs_all), dtype=bool)
            for sel in selections:
                if sel is not None:
                    sel_bool = np.zeros(len(obs_all), dtype=bool)
                    sel_bool[sel] = True if np.issubdtype(np.array(sel).dtype, np.integer) else np.array(sel)
                    combined_mask |= sel_bool
                else:
                    combined_mask = np.ones(len(obs_all), dtype=bool)
                    break  

            combined_data = obs_all[combined_mask]

            counts, _, _ = ax1.hist(combined_data, bins=bins, edgecolor="black", alpha=0.2, color='dimgray', label="All combined")
            if show_cumulative and ax2:
                bin_indices = np.digitize(combined_data, bins, right=False)
                cumulative_hours = [np.sum(combined_data[bin_indices <= i]) for i in range(1, len(bins))]
                ax2.plot(bins[:-1], cumulative_hours, linestyle="-.", marker=None, label="Cumulative All", color='dimgray')

        # Labels and appearance
        ax1.set_xlabel("Observing Time (hours)", fontsize=14)
        ax1.set_ylabel("Frequency", fontsize=14)
        if ax2:
            ax2.set_ylabel("Cumulative Observing Time (hours)", fontsize=14, color="darkred")
            ax2.set_yscale("log")
            ax2.tick_params(axis="y", labelcolor="darkred")

        ax1.set_xlim(0, threshold + 20)
        ax1.grid(axis="y", linestyle="--", alpha=0.5)
        if title is not None:
            plt.title(title, fontsize=16)
        else:
            plt.title("Histogram of Observing Time with Cumulative Distribution", fontsize=16)

        # Combined legend
        lines_labels = [ax.get_legend_handles_labels() for ax in [ax1, ax2] if ax]
        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        fig.legend(lines, labels, loc="upper left", bbox_to_anchor=(0.1, 0.95), fontsize=12)

        plt.tight_layout()
        plt.show()
        
        return fig




    def plot_comparison(self, x_col, y_col, y_err_plus_col, y_err_minus_col, conversion_factor=None, title=None):
        """
        Plot a comparison between two columns with error bars and sigma-based colour coding.

        This method visualises the relationship between a reference column (`x_col`) and a retrieved
        column (`y_col`) with symmetric error bars. The deviation from the identity line (y = x) 
        is measured in units of sigma and colour-coded using a quantised colormap.

        Parameters
        ----------
        x_col : str
            Name of the column to be used as the x-axis reference values.
        y_col : str
            Name of the column containing retrieved values to be plotted on the y-axis.
        y_err_plus_col : str
            Column name for the positive uncertainty of `y_col`.
        y_err_minus_col : str
            Column name for the negative uncertainty of `y_col`.
        conversion_factor : float, optional
            Factor to multiply `x_col` values before plotting (e.g., to convert units).
        title : str, optional
            Custom title for the plot.

        Notes
        -----
        - Points are coloured by the floor of their sigma deviation, quantised from 0 to 5.
        - A dashed identity line (y = x) is plotted for reference.
        - The number of points deviating by more than 3σ is annotated on the plot.

        Raises
        ------
        KeyError
            If any required column is missing from `self.out_targetlist`.

        Outputs
        -------
        Displays a matplotlib plot comparing retrieved vs. expected values.
        """

        # Retrieve x values and apply conversion if needed.
        if conversion_factor is not None:
            x_values = self.out_targetlist[x_col] * conversion_factor
        else:
            x_values = self.out_targetlist[x_col]
        
        # Retrieve y values.
        y_values = self.out_targetlist[y_col]
        
        # Compute the symmetric y error as the mean of the positive and negative errors.
        y_error = np.mean([self.out_targetlist[y_err_plus_col], self.out_targetlist[y_err_minus_col]], axis=0)
        
        # Compute the sigma difference from the bisector (y = x)
        sigma_diff = (y_values - x_values) / y_error
        
        # Count the number of points with an absolute sigma difference greater than 3.
        num_over3 = np.sum(np.abs(sigma_diff) > 3)
        
        # Quantise the sigma difference: floor the absolute value and cap at 5.
        quant_sigma = np.clip(np.floor(np.abs(sigma_diff)), 0, 5)
        
        # Create a discrete colormap with 6 bins (0, 1, 2, 3, 4, 5).
        cmap = plt.get_cmap('viridis', 6)
        
        fig, ax = plt.subplots()
        
        if title is not None:
            plt.title(title)
        # Plot the error bars in grey.
        ax.errorbar(x_values, y_values, yerr=y_error, fmt='none', color='grey', alpha=0.5)
        
        # Plot the scatter points, coloured by the quantised sigma difference.
        sc = ax.scatter(x_values, y_values, c=quant_sigma, cmap=cmap, s=50,)
        
        # Define limits for the bisector line.
        min_val = min(np.min(x_values), np.min(y_values))
        max_val = max(np.max(x_values), np.max(y_values))
        
        # Plot the bisector line (y = x).
        ax.plot([min_val, max_val], [min_val, max_val], linestyle='--', color='black', label='Bisector (y = x)')
        
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.legend()
        
        # Add an annotation indicating the number of points beyond 3σ.
        ax.annotate(f'{num_over3} over {len(x_values)} points > 3σ', xy=(0.55, 0.1), xycoords='axes fraction',
                    ha='left', va='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
        
        # Create a quantised colourbar.
        cbar = plt.colorbar(sc, ax=ax, ticks=range(6))
        plt.show()

    def plot_comparison_panels(self, keys, y_label="Retrieved / Expected", title="Comparison Plot",
                                color_map="viridis", normalize_range=(0, 5), marker="s", alpha=0.7, log_scale=True,
                                add_annotations=True, fig_size=(15, 5), shared_y=False, shared_x=False,
                                background_regions=None, ylim=None):
        """
        Create multiple side-by-side comparison plots for a list of retrieved parameters.
        Optionally, add coloured background regions to separate sections of the x-axis.

        Parameters
        ----------
        keys : list of str
            List of column base names for which to compare `key` vs `key_profile`.
        y_label : str, optional
            Label for the y-axis shared across all subplots.
        title : str, optional
            Overall figure title.
        color_map : str, optional
            Matplotlib colormap used for sigma-based colouring.
        normalize_range : tuple, optional
            Minimum and maximum values used to normalise the colour range.
        marker : str, optional
            Marker style for scatter points.
        alpha : float, optional
            Transparency level for points and error bars.
        log_scale : bool, optional
            Whether to use a logarithmic y-axis.
        add_annotations : bool, optional
            Whether to annotate the number of >3σ outliers in each subplot.
        fig_size : tuple, optional
            Size of the full figure (width, height) in inches.
        shared_y : bool, optional
            Whether to share the y-axis across all subplots.
        shared_x : bool, optional
            Whether to share the x-axis across all subplots.
        background_regions : list of dict, optional
            Each dictionary specifies a background region with keys:
                - 'start': starting x-index,
                - 'end': ending x-index,
                - 'color': (optional) colour for the region (default 'lightgrey'),
                - 'alpha': (optional) transparency (default 0.3),
                - 'label': (optional) legend label for the region.
            If None, no background regions are added.
        ylim : tuple, optional
            The limits for the y-axis. If specified, out-of-bound values will be annotated with arrows.
            The tuple should be (lower_limit, upper_limit).

        Notes
        -----
        - Each subplot compares retrieved values (`key`) against expected profiles (`key_profile`).
        - Sigma differences are colour-coded and capped to 5σ.
        - A global colourbar for sigma differences is added to the right side of the figure.
        - In case `background_regions` is provided, the coloured bands are added on each subplot 
        and the global legend is updated with the corresponding labels.

        Raises
        ------
        KeyError
            If any of the required columns (`key`, `key_profile`, `key_sigma`) are missing.

        Outputs
        -------
        Displays a matplotlib figure with one subplot per parameter.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D

        num_plots = len(keys)
        fig, axes = plt.subplots(1, num_plots, figsize=fig_size, sharey=shared_y, sharex=shared_x)

        if num_plots == 1:
            axes = [axes]  # Ensure axes is always iterable

        cmap = plt.get_cmap(color_map, 6)
        norm = mcolors.Normalize(vmin=normalize_range[0], vmax=normalize_range[1])

        # Create a custom handle for the "Expected" marker for the legend
        expected_handle = Line2D([], [], marker=marker, color='none', markerfacecolor='w', 
                                markeredgecolor='k', markersize=8, linestyle='None', label='Expected')
        region_handles = []
        if background_regions is not None:
            # Create custom patch handles for the background regions
            for reg in background_regions:
                handle = Patch(facecolor=reg.get('color', 'lightgrey'),
                            alpha=reg.get('alpha', 0.3),
                            label=reg.get('label', f"{reg['start']}-{reg['end']}"))
                region_handles.append(handle)

        for i, key in enumerate(keys):
            ax = axes[i]
            
            # Set the y-limits for the plot before plotting the data
            if ylim is not None:
                lower_limit, upper_limit = ylim
                ax.set_ylim(lower_limit, upper_limit)
                
            ax.set_title(key)
            if log_scale:
                ax.set_yscale("log")
            
            # If background regions are specified, add a coloured band for each region
            if background_regions is not None:
                for reg in background_regions:
                    # Draw background coloured section over the specified x-range
                    ax.axvspan(reg['start'], reg['end'], facecolor=reg.get('color', 'lightgrey'),
                            alpha=reg.get('alpha', 0.3), zorder=0)
            
            # Plot expected values (empty markers)
            ax.scatter(np.arange(len(self.out_targetlist)), self.out_targetlist[f"{key}_profile"], 
                    label="Expected", edgecolor="k", facecolors="none", marker=marker, s=20, alpha=alpha, zorder=5)

            # Calculate sigma difference
            sigma_diff = (self.out_targetlist[key] - self.out_targetlist[f"{key}_profile"]) / self.out_targetlist[f"{key}_sigma"]
            quant_sigma = np.clip(np.floor(np.abs(sigma_diff)), *normalize_range)
            num_over3 = np.sum(np.abs(sigma_diff) > 3)
            
            # Plot error bars
            ax.errorbar(np.arange(len(self.out_targetlist)), self.out_targetlist[key],
                        yerr=self.out_targetlist[f"{key}_sigma"], fmt="none", alpha=alpha-0.2, zorder=3, color="grey")
            
            # Scatter plot with sigma-based colour mapping
            sc = ax.scatter(np.arange(len(self.out_targetlist)), self.out_targetlist[key], c=quant_sigma, 
                            cmap=cmap, norm=norm, s=20, alpha=alpha, zorder=6)
            
            # Annotation for 3σ outliers with a more opaque background (alpha set to 1.0)
            if add_annotations:
                # Get the y-axis limits
                y_min, y_max = ax.get_ylim()
                
                # If the scale is logarithmic, adjust the y_min and y_max using logarithms
                if ax.get_yscale() == 'log':
                    log_y_min = np.log10(y_min)
                    log_y_max = np.log10(y_max)
                else:
                    log_y_min = y_min
                    log_y_max = y_max
                
                # Find the points above the median of the y-range (taking the logarithmic scale into account)
                if ax.get_yscale() == 'log':
                    median_y = 10 ** ((log_y_max + log_y_min) / 2)
                else:
                    median_y = (y_max + y_min) / 2
            
                upper_half = np.where((self.out_targetlist[key]) > median_y)[0]
    
                # If more than half of the data points are in the upper half of the plot, place the annotation at the top
                if len(upper_half) > len(self.out_targetlist[key]) * 0.5:
                    annotation_position = y_min  # Place at the top
                else:
                    annotation_position = y_max  # Place at the bottom
                ax.annotate(f'{num_over3} over {len(self.out_targetlist)} points > 3σ', 
                            xy=(0.05, annotation_position),  # Position relative to the axes
                            # xycoords='axes fraction',  # Use axes fraction for positioning
                            ha='left', va='top', 
                            fontsize=10,
                            bbox=dict(facecolor='white', alpha=1.0), zorder=10)
            

            
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.set_xlabel("Planet Index")
            
            # Check if ylim is provided and annotate out-of-bound values with arrows
            if ylim is not None:
                lower_limit, upper_limit = ylim
                for xi, yi, actual, quant_color in zip(np.arange(len(self.out_targetlist)), self.out_targetlist[key], 
                                                        self.out_targetlist[f"{key}_profile"], quant_sigma):
                    # Check if the value is an expected value (i.e., _profile)
                    if actual < lower_limit:
                        # print("lower limit", actual, lower_limit)
                        arrow_color = "k"  # For expected values, the arrow is grey
                        ax.annotate('', xy=(xi, lower_limit), xytext=(xi, 2*lower_limit),
                                    arrowprops=dict(facecolor='none', edgecolor=arrow_color, arrowstyle='-|>', lw=1.2), zorder=7)
                    if yi < lower_limit:
                        # print("lower limit", yi, lower_limit)
                        arrow_color = cmap(norm(quant_color))  # Colour based on the scatter colour
                        ax.annotate('', xy=(xi, lower_limit), xytext=(xi, 2*lower_limit),
                                        arrowprops=dict(facecolor=arrow_color, edgecolor='none', arrowstyle='-|>', lw=1.2), zorder=7)
                    if actual > upper_limit:
                        # print("upper limit", actual, upper_limit)
                        arrow_color = "k"  # For expected values, the arrow is grey
                        ax.annotate('', xy=(xi, 0.8*upper_limit), xytext=(xi, 0.9*upper_limit),
                                        arrowprops=dict(facecolor='none', edgecolor=arrow_color, arrowstyle='<|-', lw=1.2), zorder=7)
                    if yi > upper_limit:
                        # print("upper limit", yi, upper_limit)
                        arrow_color = cmap(norm(quant_color))  # Colour based on the scatter colour
                        ax.annotate('', xy=(xi, 0.8*upper_limit), xytext=(xi, 0.9*upper_limit),
                                    arrowprops=dict(facecolor=arrow_color, edgecolor='none', arrowstyle='<|-', lw=1.2), zorder=7)            
            # # Update the legend and colorbar settings here as per the previous code
            # axes[0].set_ylabel(y_label)

        # Global colourbar for sigma difference
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # Position: [left, bottom, width, height]
        cbar = fig.colorbar(sc, cax=cbar_ax, ticks=range(6))
        cbar.set_label("Quantized Sigma Difference")

        # Create global legend including the expected marker and the background regions (if any)
        legend_handles =  region_handles
        fig.legend(handles=legend_handles, loc='upper left', ncol=2)

        plt.suptitle(title)
        plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust layout to fit the colourbar
        plt.show()
        return fig


    def plot_sigma_vs_input(self, keys, x_key, title="Sigma Distance vs Input",
                            y_label="Sigma Distance", yscale="linear", xscale="linear",
                            marker="o", alpha=0.7, fig_size=(15, 5), ylim=None,
                            background_regions=None, add_annotations=True):
        """
        Plot sigma distances as a function of a given input parameter, colouring by data regions.
        Annotates points outside y-limits with directional arrows.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib import cm

        num_plots = len(keys)
        fig, axes = plt.subplots(1, num_plots, figsize=fig_size, sharey=True)

        if num_plots == 1:
            axes = [axes]

        x_vals = self.out_targetlist[x_key]
        colours = ['lightgrey'] * len(x_vals)

        for j, region in enumerate(background_regions):
            colour = region.get("color", "lightgrey")
            for idx in range(region["start"], region["end"] + 1):
                if 0 <= idx < len(x_vals):
                    colours[idx] = colour

        for idx, key in enumerate(keys):
            ax = axes[idx]
            ax.set_title(key)

            sigma_diff = (self.out_targetlist[key] - self.out_targetlist[f"{key}_profile"]) / self.out_targetlist[f"{key}_sigma"]
            abs_sigma = np.abs(sigma_diff)
            num_over3 = np.sum(abs_sigma > 3)

            if ylim:
                ax.set_ylim(*ylim)
                y_min, y_max = ylim
            else:
                y_min, y_max = ax.get_ylim()

            sc = ax.scatter(x_vals, abs_sigma, c=colours, marker=marker, alpha=alpha, s=10, zorder=3)

            if add_annotations:
                y_median = (y_max + y_min) / 2
                place_at_top = np.mean(abs_sigma > y_median) > 0.5
                annotation_y = y_max if place_at_top else y_min
                ax.annotate(f'{num_over3} / {len(x_vals)} > 3σ',
                            xy=(0.05, annotation_y),
                            ha='left', va='top' if place_at_top else 'bottom',
                            fontsize=10,
                            bbox=dict(facecolor='white', alpha=1.0),
                            zorder=10)

            # Linee di soglia fisse
            for threshold, style in zip([2, 3, 5], [":", "--", "-"]):
                ax.axhline(threshold, linestyle=style, c="r", alpha=0.5)

            # Aggiungi frecce per out-of-bound
            if ylim:
                for xi, yi, colour in zip(x_vals, abs_sigma, colours):
                    if yi < y_min:
                        ax.annotate('', xy=(xi, y_min), xytext=(xi, y_min - 0.5),
                                    arrowprops=dict(facecolor=colour, edgecolor='none', arrowstyle='-|>', lw=1.2), zorder=7)
                    elif yi > y_max:
                        ax.annotate('', xy=(xi, y_max), xytext=(xi, y_max + 0.5),
                                    arrowprops=dict(facecolor=colour, edgecolor='none', arrowstyle='<|-', lw=1.2), zorder=7)

            ax.grid(True, linestyle="--", alpha=0.5)
            ax.set_xlabel(x_key)
            ax.set_yscale(yscale)
            ax.set_xscale(xscale)
            if idx == 0:
                ax.set_ylabel(y_label)

        if background_regions:
            legend_elements = []
            for i, region in enumerate(background_regions):
                legend_elements.append(Line2D([0], [0], marker='o', color='w',
                                            label=region.get('label', f"{region['start']}-{region['end']}"),
                                            markerfacecolor=region.get('color', 'lightgrey'), markersize=10))
            fig.legend(handles=legend_elements, loc='upper right')

        plt.suptitle(title)
        plt.tight_layout(rect=[0, 0, 0.95, 0.95])
        plt.show()
        return fig


    def plot_scatter_with_histograms(
        self, x_key, y_key, selection=None,
        x_err_key=None, y_err_key=None, color_key=None, contour_only=False,
        cmap="viridis", hist_color="black", bins_num=30, bins_x=None, bins_y=None, marker="o",
        ax_scatter=None, ax_hist_x=None, ax_hist_y=None, ax_legend=None, fig=None,
        label=None, alpha=0.6,
        err_color="black", elinewidth=1.5, capsize=3, capthick=1, errorevery=1, err_alpha=0.5, linestyle="-",
        global_norm=None
    ):
        """
        Plot a scatter plot with marginal histograms and optional error bars and colour encoding.

        This function visualises the relationship between two variables with optional error bars and colour-coded
        scatter points. Marginal histograms for both x and y variables are also plotted. Supports overlaying onto 
        existing axes or creating a new combined figure.

        Parameters
        ----------
        x_key : str
            Column name to use for the x-axis values.
        y_key : str
            Column name to use for the y-axis values.
        selection : array-like, optional
            A boolean or integer index array for selecting a subset of the data.
        x_err_key : str, optional
            Column name for the x-axis uncertainties.
        y_err_key : str, optional
            Column name for the y-axis uncertainties.
        color_key : str, optional
            Column name used for colour-coding points.
        contour_only : bool, optional
            If True, plot only the outline of points (useful for overlays).
        cmap : str, optional
            Name of the matplotlib colormap.
        hist_color : str, optional
            Colour for the histogram bars.
        bins_num : int, optional
            Number of bins to use for the histograms.
        bins_x : array-like, optional
            Custom bin edges for x-axis histogram.
        bins_y : array-like, optional
            Custom bin edges for y-axis histogram.
        marker : str, optional
            Marker style for the scatter plot.
        ax_scatter, ax_hist_x, ax_hist_y, ax_legend : matplotlib.axes.Axes, optional
            Axes objects for reusing an existing figure layout.
        fig : matplotlib.figure.Figure, optional
            The figure object to update (used with reusable axes).
        label : str, optional
            Label for the plotted dataset.
        alpha : float, optional
            Transparency of points and histograms.
        err_color : str, optional
            Colour for error bars.
        elinewidth, capsize, capthick : float, optional
            Styling options for the error bars.
        errorevery : int, optional
            Plot error bars for every Nth point only.
        err_alpha : float, optional
            Transparency for error bars.
        linestyle : str, optional
            Linestyle for error bars.
        global_norm : matplotlib.colors.Normalize, optional
            Normalisation for consistent colour mapping.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure containing the plot.
        ax_scatter, ax_hist_x, ax_hist_y, ax_legend : matplotlib.axes.Axes
            The axes used for scatter and histogram components.
        global_norm : matplotlib.colors.Normalize
            The normalisation object used for colour scaling.

        Notes
        -----
        - Uses logarithmic scaling for both axes.
        - Colour bar is shown only when `color_key` is provided.
        - Can be extended to overlay multiple datasets by reusing returned axes.
        """
 
        if selection is not None:
            data_table = self.out_targetlist[selection]
        else:
            data_table = self.out_targetlist
            
        x_data = np.array(data_table[x_key])
        y_data = np.array(data_table[y_key])
        x_err = np.array(data_table[x_err_key]) if x_err_key else None
        y_err = np.array(data_table[y_err_key]) if y_err_key else None
        color_data = np.array(data_table[color_key]) if color_key else None

        if global_norm is None and color_key is not None:
            global_norm = plt.Normalize(vmin=np.nanmin(color_data), vmax=np.nanmax(color_data))

        cmap_instance = plt.cm.get_cmap(cmap)
        scatter_color = cmap_instance(0.5) if color_key is None else None
        
        new_plot = False
        if ax_scatter is None or ax_hist_x is None or ax_hist_y is None:
            new_plot = True
            fig = plt.figure(figsize=(12, 10))
            gs = fig.add_gridspec(4, 5, hspace=0, wspace=0.1, width_ratios=[1, 1, 1, 1, 0.3])

            ax_scatter = fig.add_subplot(gs[1:4, 0:3])
            ax_hist_x = fig.add_subplot(gs[0, 0:3], sharex=ax_scatter)
            ax_hist_y = fig.add_subplot(gs[1:4, 3], sharey=ax_scatter)
            ax_legend = fig.add_subplot(gs[0, 3:5])
            ax_legend.axis("off")
            if color_key is not None:
                ax_colorbar = fig.add_subplot(gs[1:4, 4])

        if contour_only:
            scatter = ax_scatter.scatter(
                x_data, y_data, edgecolor=color_data if color_key else scatter_color, cmap=cmap, norm=global_norm,
                alpha=alpha, facecolors="None", label=label, marker=marker, zorder=100
            )
        else:
            scatter = ax_scatter.scatter(
                x_data, y_data, c=color_data if color_key else scatter_color, cmap=cmap, norm=global_norm,
                alpha=alpha, edgecolors="None", label=label, marker=marker, zorder=100
            )

        if x_err is not None or y_err is not None:
            for i in range(0, len(x_data), errorevery):
                ax_scatter.errorbar(
                    x_data[i], y_data[i],
                    xerr=x_err[i] if x_err is not None else None,
                    yerr=y_err[i] if y_err is not None else None,
                    alpha=err_alpha, fmt="o", color=cmap_instance(global_norm(color_data[i])) if color_data is not None else scatter_color,
                    ecolor=err_color, elinewidth=elinewidth, capsize=capsize, capthick=capthick, linestyle=linestyle, zorder=1
                )

        if bins_x is None:
            bins_x = np.logspace(np.log10(np.nanmin(x_data)*0.95), np.log10(np.nanmax(x_data)*1.05), bins_num)
        if bins_y is None:
            bins_y = np.logspace(np.log10(np.nanmin(y_data)*0.95), np.log10(np.nanmax(y_data)*1.05), bins_num)
        
        hist_color_final = hist_color if color_key else scatter_color
        
        ax_hist_x.hist(x_data, bins=bins_x, edgecolor="black", alpha=0.7, label=label, color=hist_color_final)
        ax_hist_y.hist(y_data, bins=bins_y, orientation="horizontal", edgecolor="black", alpha=0.7, label=label, color=hist_color_final)
        
        ax_hist_x.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax_hist_y.xaxis.set_major_locator(MaxNLocator(integer=True))

        if new_plot:
            ax_scatter.set_xlabel(x_key, fontsize=14)
            ax_scatter.set_ylabel(y_key, fontsize=14)
            ax_scatter.set_title(f"Scatter Plot of {x_key} vs {y_key}", fontsize=16)
            ax_scatter.set_xscale("log")
            ax_scatter.set_yscale("log")
            plt.setp(ax_hist_x.get_xticklabels(), visible=False)
            plt.setp(ax_hist_y.get_yticklabels(), visible=False)
            ax_scatter.grid(True, linestyle="--", alpha=0.6)
            if color_key is not None:
                cbar = plt.colorbar(scatter, cax=ax_colorbar)
                cbar.set_label(color_key, fontsize=14)
                
            ax_scatter.grid(True, linestyle="--", alpha=0.5)

        
            
        if not new_plot:
            # Get legend entries from the scatter and histogram axes
            handles_scatter, labels_scatter = ax_scatter.get_legend_handles_labels()
            handles_hist_x, labels_hist_x = ax_hist_x.get_legend_handles_labels()
            
            # Merge legend entries
            handles = handles_scatter + handles_hist_x
            labels = labels_scatter + labels_hist_x
            
            if len(labels) > 1:
                # Split the merged legend into two halves
                split_idx = len(labels) // 2
                handles_bottom = handles[:split_idx]
                labels_bottom = labels[:split_idx]
                handles_top = handles[split_idx:]
                labels_top = labels[split_idx:]
                
                # Place the first half inside the main scatter plot at the lower left 
                # (using bbox_to_anchor to precisely locate the legend)
                leg1 = ax_scatter.legend(handles_bottom, labels_bottom, loc='lower left',
                                        bbox_to_anchor=(0.0, 0.0), fontsize=14, framealpha=0.5)
                
                # Place the second half in the dedicated ax_legend subplot at the upper right
                ax_legend.legend(handles_top, labels_top, loc='upper right', fontsize=14, frameon=False)

        return fig, ax_scatter, ax_hist_x, ax_hist_y, ax_legend, global_norm
    
    
    def plot_scatter_with_histograms_compared_with_expectation(
        self, selection=None,
        color_key=None, expect_color="k",
        cmap="viridis", hist_color="black", bins_num=30, bins_x=None, bins_y=None, marker="o",
        ax_scatter=None, ax_hist_x=None, ax_hist_y=None, ax_legend=None, fig=None,
        label=None, alpha=0.6,
        err_color="black", elinewidth=1.5, capsize=3, capthick=1, errorevery=1, err_alpha=0.5, linestyle="-",
        global_norm=None
    ):
        """
        Plot a scatter plot of O/H vs C/O with marginal histograms and arrows to expected values.

        This method plots retrieved elemental ratios (`O/H` vs `C/O`) against their expected values 
        (`OHratio`, `COratio`) with arrows pointing from the observed to the expected point. 
        Includes histograms on both axes and optional colour-coding of the scatter points.

        Parameters
        ----------
        selection : array-like, optional
            Boolean or integer index array to select a subset of the data.
        color_key : str, optional
            Column name used to colour-code the data points.
        expect_color : str, optional
            Colour used to plot expected values and connecting arrows.
        cmap : str, optional
            Name of the colormap to use.
        hist_color : str, optional
            Colour used for histograms of the observed data.
        bins_num : int, optional
            Number of bins to use for the histograms.
        bins_x : array-like, optional
            Custom bin edges for x-axis.
        bins_y : array-like, optional
            Custom bin edges for y-axis.
        marker : str, optional
            Marker style for data points.
        ax_scatter, ax_hist_x, ax_hist_y, ax_legend : matplotlib.axes.Axes, optional
            Existing axes to overlay additional data.
        fig : matplotlib.figure.Figure, optional
            Existing figure to update.
        label : str, optional
            Label for the dataset.
        alpha : float, optional
            Transparency of plotted elements.
        err_color : str, optional
            Colour used for the error bars.
        elinewidth, capsize, capthick : float, optional
            Error bar styling options.
        errorevery : int, optional
            Plot error bars only for every Nth point.
        err_alpha : float, optional
            Transparency of error bars.
        linestyle : str, optional
            Line style for error bars.
        global_norm : matplotlib.colors.Normalize, optional
            Normalisation object for consistent colour scaling.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The complete figure with scatter and histograms.
        ax_scatter, ax_hist_x, ax_hist_y, ax_legend : matplotlib.axes.Axes
            Axes used for plot components.
        global_norm : matplotlib.colors.Normalize
            The normalisation object used for colour scaling.

        Notes
        -----
        - Arrows represent the deviation from observed to expected values.
        - Observed and expected histograms are overlaid for comparison.
        - Useful for visually assessing bias or accuracy in retrievals.
        """

        if selection is not None:
            data_table = self.out_targetlist[selection]
        else:
            data_table = self.out_targetlist
        

        x_data = np.array(data_table["O/H"])
        y_data = np.array(data_table["C/O"])
        x_exp = np.array(data_table["OHratio"])
        y_exp = np.array(data_table["COratio"])
        x_err = np.array(data_table["O/H_sigma"]) 
        y_err = np.array(data_table["C/O_sigma"]) 
        color_data = np.array(data_table[color_key]) if color_key else None

        if global_norm is None and color_key is not None:
            global_norm = plt.Normalize(vmin=np.nanmin(color_data), vmax=np.nanmax(color_data))

        cmap_instance = plt.cm.get_cmap(cmap)
        scatter_color = cmap_instance(0.5) if color_key is None else None
        
        new_plot = False
        if ax_scatter is None or ax_hist_x is None or ax_hist_y is None:
            new_plot = True
            fig = plt.figure(figsize=(12, 10))
            gs = fig.add_gridspec(4, 5, hspace=0, wspace=0.1, width_ratios=[1, 1, 1, 1, 0.3])

            ax_scatter = fig.add_subplot(gs[1:4, 0:3])
            ax_hist_x = fig.add_subplot(gs[0, 0:3], sharex=ax_scatter)
            ax_hist_y = fig.add_subplot(gs[1:4, 3], sharey=ax_scatter)
            ax_legend = fig.add_subplot(gs[0, 3:5])
            ax_legend.axis("off")
            if color_key is not None:
                ax_colorbar = fig.add_subplot(gs[1:4, 4])
        
        exp_label = label + ", input"
        exp_scatter = ax_scatter.scatter(
                x_exp, y_exp, edgecolor=expect_color,
                alpha=alpha, facecolors="None", label=exp_label, marker="s", zorder=100
            )
        data_scatter = ax_scatter.scatter(
            x_data, y_data, c=color_data if color_key else scatter_color, cmap=cmap, norm=global_norm,
            alpha=alpha, edgecolors="None", label=label, marker=marker, zorder=100
        )

        ax_scatter.quiver(
            x_data, y_data,  
            x_exp-x_data, y_exp-y_data,  
            color=expect_color,  
            angles="xy", scale_units="xy", scale=1, width=0.002
        )
        if x_err is not None or y_err is not None:
            for i in range(0, len(x_data), errorevery):
                ax_scatter.errorbar(
                    x_data[i], y_data[i],
                    xerr=x_err[i] if x_err is not None else None,
                    yerr=y_err[i] if y_err is not None else None,
                    alpha=err_alpha, fmt="o", color=cmap_instance(global_norm(color_data[i])) if color_data is not None else scatter_color,
                    ecolor=err_color, elinewidth=elinewidth, capsize=capsize, capthick=capthick, linestyle=linestyle, zorder=1
                )

        if bins_x is None:
            x_min = np.nanmin([np.nanmin(x_data), np.nanmin(x_exp)])
            x_max = np.nanmax([np.nanmax(x_data), np.nanmax(x_exp)])
            x_min *= 0.95  
            x_max *= 1.05
            bins_x = np.logspace(np.log10(x_min), np.log10(x_max), bins_num)

        if bins_y is None:
            y_min = np.nanmin([np.nanmin(y_data), np.nanmin(y_exp)])
            y_max = np.nanmax([np.nanmax(y_data), np.nanmax(y_exp)])
            y_min *= 0.95  
            y_max *= 1.05
            bins_y = np.logspace(np.log10(y_min), np.log10(y_max), bins_num)
        
        hist_color_final = hist_color if color_key else scatter_color
        
        ax_hist_x.hist(x_data, bins=bins_x, edgecolor="black", alpha=0.7, label=label, color=hist_color_final)
        ax_hist_y.hist(y_data, bins=bins_y, orientation="horizontal", edgecolor="black", alpha=0.7, label=label, color=hist_color_final)

        ax_hist_x.hist(x_exp, bins=bins_x, edgecolor="black", alpha=0.7, label=exp_label, color=expect_color)
        ax_hist_y.hist(y_exp, bins=bins_y, orientation="horizontal", edgecolor="black", alpha=0.7, label=exp_label, color=expect_color)

        
        ax_hist_x.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax_hist_y.xaxis.set_major_locator(MaxNLocator(integer=True))

        if new_plot:
            ax_scatter.set_xlabel("O/H", fontsize=14)
            ax_scatter.set_ylabel("C/O", fontsize=14)
            ax_scatter.set_title(f"Scatter Plot of O/H vs C/O", fontsize=16)
            ax_scatter.set_xscale("log")
            ax_scatter.set_yscale("log")
            plt.setp(ax_hist_x.get_xticklabels(), visible=False)
            plt.setp(ax_hist_y.get_yticklabels(), visible=False)
            ax_scatter.grid(True, linestyle="--", alpha=0.6)
            if color_key is not None:
                cbar = plt.colorbar(data_scatter, cax=ax_colorbar)
                cbar.set_label(color_key, fontsize=12)
                
            ax_scatter.grid(True, linestyle="--", alpha=0.5)

        
        
        if not new_plot:
            handles_scatter, labels_scatter = ax_scatter.get_legend_handles_labels()
            handles_hist_x, labels_hist_x = ax_hist_x.get_legend_handles_labels()
            
            # Merge legend entries
            handles = handles_scatter + handles_hist_x
            labels = labels_scatter + labels_hist_x
        
            if len(labels) > 1:  
                ax_legend.legend(handles, labels, loc="upper right", frameon=False)
    
        return fig, ax_scatter, ax_hist_x, ax_hist_y, ax_legend, global_norm
    
    
    
    def compute_differences(self, parameters=["O/H", "C/O", "He/H"]):
        """
        Create a new summary table with one row per planet base name and sigma-distance
        values between migration/redox combinations. Shared metadata columns are preserved.

        Returns
        -------
        astropy.table.Table
            A table with columns:
            - all metadata columns common to each planet
            - one column per (parameter, comparison combination), e.g. "O/H_00_vs_11"
        """
        from itertools import combinations

        suffixes = ["Mig", "Fidu", "MigR", "FiduR"]
        base_names = list(set([
            name.replace(suffix, "") for name in self.out_targetlist["Planet Name"]
            for suffix in suffixes if name.endswith(suffix)
        ] + [
            name for name in self.out_targetlist["Planet Name"] if not any(name.endswith(suffix) for suffix in suffixes)
        ]))
        base_names.sort()

        shared_columns = [
            'Star Name', 'Star Mass [Ms]', 'Spectral Type', 'Star Temperature [K]',
            'Star Radius [Rs]', 'Star Distance [pc]', 'Star K Mag', 'Star V Mag',
            'Star Metallicity', 'Star Age [Gyr]', 'Star RA', 'Star Dec', 'Planet Name',
            'Planet Period [days]', 'Planet Temperature [K]', 'Planet Semi-major Axis [m]',
            'Planet Radius [Re]', 'Planet Albedo', 'Planet Mass [Me]', 'Molecular Weight',
            'Transit Duration [s]', 'Impact Parameter', 'Heat Redistribution Factor',
            'Inclination', 'Eccentricity'
        ]

        all_combinations = list(combinations([(0,0), (0,1), (1,0), (1,1)], 2))

        # Prepare table
        new_tab = Table()
        string_cols = ['Star Name', 'Spectral Type', 'Planet Name']
        for col in shared_columns:
            dtype = 'U50' if col in string_cols else float  # usa stringa per nomi, float altrimenti
            new_tab[col] = np.array([], dtype=dtype)


        for (m0, r0), (m1, r1) in all_combinations:
            tag = f"{m0}{r0}_vs_{m1}{r1}"
            for param in parameters:
                new_tab[f"{param}_{tag}"] = []

        # Populate table
        for name in tqdm(base_names):
            mask_name = [name in p for p in self.out_targetlist["Planet Name"]]
            sub = self.out_targetlist[mask_name]

            # Find one version to use for metadata — just pick the first match
            for_metadata = sub[0]
            row = [for_metadata[col] for col in shared_columns]

            # Now calculate all distances
            for (m0, r0), (m1, r1) in all_combinations:
                sel0 = ((sub["Migration Efficiency [Fiducial = 0, Efficient = 1]"] == m0) &
                        (sub["Redox [Reduced = 0, Oxidized = 1]"] == r0))
                sel1 = ((sub["Migration Efficiency [Fiducial = 0, Efficient = 1]"] == m1) &
                        (sub["Redox [Reduced = 0, Oxidized = 1]"] == r1))

                if np.sum(sel0) == 1 and np.sum(sel1) == 1:
                    for param in parameters:
                        val0 = sub[sel0][param][0]
                        val1 = sub[sel1][param][0]
                        sigma0 = sub[sel0][f"{param}_sigma"][0]
                        sigma1 = sub[sel1][f"{param}_sigma"][0]

                        if np.isfinite(val0) and np.isfinite(val1) and np.isfinite(sigma0) and np.isfinite(sigma1):
                            diff = np.abs(val0 - val1)
                            sig_dist = diff / np.sqrt(sigma0**2 + sigma1**2)
                            row.append(sig_dist)
                        else:
                            row.append(np.nan)
                else:
                    row.extend([np.nan] * len(parameters))

            new_tab.add_row(row)

        return new_tab

    @staticmethod
    def plot_distance_panels(tbl, param_base=["O/H", "C/O", "He/H"], combo="00_vs_11", ylim=10, title=None):
        """
        Plot scatter + side histogram panels for distance-in-sigma values computed between models.

        Parameters
        ----------
        tbl : astropy.table.Table
            Table with one row per planet, containing e.g. "O/H_00_vs_11", etc.
        param_base : list of str
            List of parameters to plot (e.g. ["O/H", "C/O"]).
        combo : str
            Migration/Redox comparison tag to extract (e.g. "00_vs_11").
        ylim : float
            Maximum y-axis value; higher values are shown as arrows.
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        import numpy as np

        n_panels = len(param_base)
        fig, axs = plt.subplots(1, n_panels + 1, figsize=(10 * (n_panels + 1), 8), sharey=True)
        if n_panels == 1:
            axs = [axs]

        color_map = {'M': '#e41a1c', 'G': '#377eb8'}
        bins = np.linspace(0, ylim, 21)  # 20 bins
        bin_centers = 0.5 * (bins[:-1] + bins[1:])

        def get_colour(spec_type):
            return color_map.get(spec_type, 'grey')

        def get_percentages(data):
            total = len(data)
            return {
                "<3σ": 100 * np.sum(data < 3) / total,
                "3–5σ": 100 * np.sum((data >= 3) & (data < 5)) / total,
                ">5σ": 100 * np.sum(data >= 5) / total,
            }

        def get_bin_colour(center):
            if center < 3:
                return 'red'
            elif center < 5:
                return 'yellow'
            else:
                return 'green'

        for ax, base_param in zip(axs, param_base):
            colname = f"{base_param}_{combo}"
            if colname not in tbl.colnames:
                raise ValueError(f"Column '{colname}' not found in table.")

            data = np.array(tbl[colname], dtype=float)
            spectral_types = np.array(tbl["Spectral Type"])
            planet_names = np.array(tbl["Planet Name"])
            planet_names = [name.replace("Mig", "").replace("Fidu", "") for name in planet_names]
            planet_names = [name.replace("R", "") for name in planet_names]
            colours = [get_colour(s) for s in spectral_types]

            divider = make_axes_locatable(ax)
            ax_hist = divider.append_axes("right", size="20%", pad=0.05, sharey=ax)

            x = np.arange(len(data))
            shown_data = np.clip(data, None, ylim)

            for i, (xi, yi, actual, col) in enumerate(zip(x, shown_data, data, colours)):
                if actual > ylim:
                    ax.annotate('', xy=(xi, ylim - 0.2), xytext=(xi, ylim - 0.8),
                                arrowprops=dict(facecolor=col, edgecolor='none', arrowstyle='-|>', lw=1.2))
                else:
                    ax.scatter(xi, yi, color=col, edgecolor='k', alpha=0.8, zorder=3)

            ax.set_title(base_param, fontsize=14)
            ax.set_ylabel("Distance in sigma", fontsize=12)
            ax.set_ylim(0, ylim)
            ax.set_xticks(x)
            ax.set_xticklabels(planet_names, rotation=90, fontsize=10)
            ax.tick_params(axis='y', labelsize=12)

            ax.axhline(3, color="k", linestyle="--")
            ax.axhline(5, color="k", linestyle=":")
            ax.axhspan(0, 3, facecolor='red', alpha=0.2)
            ax.axhspan(3, 5, facecolor='yellow', alpha=0.2)
            ax.axhspan(5, ylim, facecolor='green', alpha=0.1)

            hist_vals, _ = np.histogram(np.clip(data, None, ylim), bins=bins)
            for i, val in enumerate(hist_vals):
                color = get_bin_colour(bin_centers[i])
                ax_hist.barh(bin_centers[i], val, height=bins[1] - bins[0],
                            color=color, edgecolor='black')

            ax_hist.axhline(3, color="k", linestyle="--")
            ax_hist.axhline(5, color="k", linestyle=":")

            percentages = get_percentages(data)
            summary_text = (
                f"< 3σ: {percentages['<3σ']:.1f}%\n"
                f"3–5σ: {percentages['3–5σ']:.1f}%\n"
                f"> 5σ: {percentages['>5σ']:.1f}%"
            )
            ax_hist.annotate(
                summary_text,
                xy=(0.95, 0.05),
                xycoords='axes fraction',
                ha='right', va='bottom',
                fontsize=12,
                bbox=dict(facecolor='white', edgecolor='black', alpha=0.7)
            )


            ax_hist.set_xlabel("Count", fontsize=10)
            ax_hist.tick_params(axis='y', labelleft=False, labelright=True)
            ax_hist.tick_params(axis='x', labelsize=8)
            ax_hist.grid(True, axis='x', linestyle='--', alpha=0.5)

        # === Compute total distance in quadrature (euclidean norm) ===
        total_ax = axs[-1]
        colnames = [f"{param}_{combo}" for param in param_base]
        for col in colnames:
            if col not in tbl.colnames:
                raise ValueError(f"Missing column '{col}' in table.")

        values = np.array([np.array(tbl[col], dtype=float) for col in colnames])
        total_distances = np.sqrt(np.sum(values**2, axis=0))

        spectral_types = np.array(tbl["Spectral Type"])
        planet_names = np.array(tbl["Planet Name"])
        planet_names = [name.replace("Mig", "").replace("Fidu", "").replace("R", "") for name in planet_names]
        colours = [get_colour(s) for s in spectral_types]

        divider = make_axes_locatable(total_ax)
        total_hist_ax = divider.append_axes("right", size="20%", pad=0.05, sharey=total_ax)

        x = np.arange(len(total_distances))
        shown_data = np.clip(total_distances, None, ylim)

        for i, (xi, yi, actual, col) in enumerate(zip(x, shown_data, total_distances, colours)):
            if actual > ylim:
                total_ax.annotate('', xy=(xi, ylim - 0.2), xytext=(xi, ylim - 0.8),
                                arrowprops=dict(facecolor=col, edgecolor='none', arrowstyle='-|>', lw=1.2))
            else:
                total_ax.scatter(xi, yi, color=col, edgecolor='k', alpha=0.8, zorder=3)

        total_ax.set_title("Total", fontsize=14)
        total_ax.set_ylabel("Distance in sigma", fontsize=12)
        total_ax.set_ylim(0, ylim)
        total_ax.set_xticks(x)
        total_ax.set_xticklabels(planet_names, rotation=90, fontsize=10)
        total_ax.tick_params(axis='y', labelsize=12)

        total_ax.axhline(3, color="k", linestyle="--")
        total_ax.axhline(5, color="k", linestyle=":")
        total_ax.axhspan(0, 3, facecolor='red', alpha=0.2)
        total_ax.axhspan(3, 5, facecolor='yellow', alpha=0.2)
        total_ax.axhspan(5, ylim, facecolor='green', alpha=0.1)

        hist_vals, _ = np.histogram(np.clip(total_distances, None, ylim), bins=bins)
        for i, val in enumerate(hist_vals):
            color = get_bin_colour(bin_centers[i])
            total_hist_ax.barh(bin_centers[i], val, height=bins[1] - bins[0],
                            color=color, edgecolor='black')

        total_hist_ax.axhline(3, color="k", linestyle="--")
        total_hist_ax.axhline(5, color="k", linestyle=":")

        percentages = get_percentages(total_distances)
        summary_text = (
            f"< 3σ: {percentages['<3σ']:.1f}%\n"
            f"3–5σ: {percentages['3–5σ']:.1f}%\n"
            f"> 5σ: {percentages['>5σ']:.1f}%"
        )
        total_hist_ax.annotate(
            summary_text,
            xy=(0.95, 0.05),
            xycoords='axes fraction',
            ha='right', va='bottom',
            fontsize=12,
            bbox=dict(facecolor='white', edgecolor='black', alpha=0.7)
        )

        total_hist_ax.set_xlabel("Count", fontsize=10)
        total_hist_ax.tick_params(axis='y', labelleft=False, labelright=True)
        total_hist_ax.tick_params(axis='x', labelsize=8)
        total_hist_ax.grid(True, axis='x', linestyle='--', alpha=0.5)

    # === Legend for spectral types ===
        handles = [plt.Line2D([0], [0], marker='o', color='w', label=stype,
                            markerfacecolor=col, markersize=12, markeredgecolor='k')
                for stype, col in color_map.items()]
        axs[-1].legend(handles=handles, title="Spectral Type", fontsize=10, title_fontsize=14, loc='upper right')

        if title:
            fig.suptitle(title, fontsize=16)
        else:
            fig.suptitle(f"Distance in Sigma Between Realisations: {combo}", fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        plt.show()
        return fig


    @staticmethod
    def calculate_scale_height(abundances_dict, T_list, M_list, R_list, abundance_scale="linear"):
        """
        Calculate the scale height for an array of exoplanets given the logarithmic abundances of molecules,
        temperatures, planet masses, and radii.

        Parameters:
        -----------
        abundances_dict : dict
            Dictionary where keys are species names and values are lists containing the log10 abundances
            for that species across the different planets.
            Example: {'CH4': [-2, -1.8], 'CO': [-3, -2.5], 'CO2': [-1.5, -2], 'H2O': [-2.5, -2.2]}
            
        T_list : list of floats
            List of temperatures of the atmospheres in Kelvin (K) for each planet.
            
        M_list : list of floats
            List of masses of the planets in kilograms (kg).
            
        R_list : list of floats
            List of radii of the planets in meters (m).
        abundance_scale : str
            Scale of the abundances. Options are "log" or "linear". Default is "linear".

        Returns:
        --------
        scale_heights : list of floats
            List of scale heights for each planet in meters (m).
        """
        
        import astropy.constants as c
        import astropy.units as u
        # Physical constants
        k_B = c.k_B  # Boltzmann constant in J/K
        G = c.G  # Gravitational constant in m^3 kg^-1 s^-2
        n_A = c.N_A  # Avogadro's number
        
        molecules = list(abundances_dict.keys())
        molecules.append('H2')
        
        # Molar masses of species in kg/mol (converted to kg/molecule)
        molar_masses = {
            'CH4': 16.04e-3 * u.kg / u.mol / n_A,  # in kg/molecule
            'CO': 28.01e-3 * u.kg / u.mol / n_A,   # in kg/molecule
            'CO2': 44.01e-3 * u.kg / u.mol / n_A,  # in kg/molecule
            'H2O': 18.015e-3 * u.kg / u.mol / n_A, # in kg/molecule
            'H2': 2.016e-3 * u.kg / u.mol / n_A,   # in kg/molecule
        }

        # List to store the scale heights for each planet
        scale_heights = []

        # Loop through each planet
        for abundances, T, M, R in zip(zip(*abundances_dict.values()), T_list, M_list, R_list):
            T = T.to(u.K)
            M = M.to(u.kg)
            R = R.to(u.m)

            # Convert log mass fractions to linear mass fractions if necessary
            if abundance_scale == "log":
                mass_fractions = [10**abundance for abundance in abundances]
            elif abundance_scale == "linear":
                mass_fractions = list(abundances)
            else:
                raise ValueError("abundance_scale must be either 'log' or 'linear'")

            # Normalize to get mass fractions (should sum to 1)
            total_mass_fraction = sum(mass_fractions)

            # If the total mass fraction is less than 1, fill the missing fraction with H2
            missing_fraction = 1 - total_mass_fraction
            mass_fractions.append(missing_fraction)  # Add the missing fraction to H2

                        # Normalize again to ensure the mass fractions sum to 1
            total_mass_fraction = sum(mass_fractions)
            mass_fractions = [mf / total_mass_fraction for mf in mass_fractions]
            
            # Calculate the mean molecular mass of the atmosphere (in kg)
            m_avg = sum(mass_fractions[i] * molar_masses[molecules[i]] for i in range(len(molecules)))

            # Calculate the gravitational acceleration of the planet (in m/s^2)
            g = (G * M) / (R**2)

            # Calculate the scale height for this planet (in meters)
            H = ((k_B * T) / (m_avg * g)).decompose()

            # Append the result to the list of scale heights
            scale_heights.append(H)

        scale_heights_array = u.Quantity(scale_heights, u.km)  # Convert to kilometers

        return scale_heights_array


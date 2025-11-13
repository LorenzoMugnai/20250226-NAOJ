from astropy.io import ascii
from astropy.table import Table
import astropy.units as u
import h5py
import os
import numpy as np
from tqdm import tqdm
from mol_to_ratios import calculate_ratios_with_uncertainties
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator

# Update global font settings for all plots
plt.rc('font', size=14)          # controls default text sizes
plt.rc('axes', titlesize=16)     # fontsize of the axes title
plt.rc('axes', labelsize=14)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
plt.rc('legend', fontsize=12)    # fontsize of the legend
plt.rc('figure', titlesize=18)   # fontsize of the figure title

class Inspector:

    def __init__(self, input_table, retrieval_folder, forwards_folder):
        """
        Inspector(input_table, retrieval_folder, forwards_folder)

        A class for aggregating, processing, and visualising exoplanet atmospheric retrieval results, forward models, and derived elemental ratios.

        The `Inspector` class facilitates the comparison between retrieved atmospheric parameters and expected forward model values. It supports loading and managing retrieval outputs from HDF5 files, computing molecular and elemental ratios (with uncertainties), integrating additional observational data (e.g., number of Ariel observations), and generating summary statistics and plots. It also supports identifying missing data and visualising discrepancies between predicted and retrieved parameters.

        Parameters
        ----------
        input_table : str
            Path to the input CSV or ASCII table listing planet metadata and identifiers.
        retrieval_folder : str
            Path to the folder containing retrieval output files (one per planet, in HDF5 format).
        forwards_folder : str
            Path to the folder containing forward model outputs and Ariel observation files.

        Attributes
        ----------
        targetlist : astropy.table.Table
            Table containing the initial list of planetary targets.
        retrieval_folder : str
            Directory where the retrieval result files are stored.
        forwards_folder : str
            Directory where forward model data is stored.
        out_targetlist : astropy.table.Table
            Output table that aggregates retrieved parameters, computed ratios, and observational metadata for valid planets.

        Notes
        -----
        - Retrievals are expected in a standard HDF5 format following the `Output/Solutions/solution0/fit_params` convention.
        - Computation of elemental ratios requires specific keys to be present (e.g., log-scaled abundances and uncertainties).
        - This class supports extensive plotting for analysis and validation of atmospheric parameters across a sample of planets.
        """

        self.targetlist = ascii.read(input_table)
        self.retrieval_folder = retrieval_folder
        self.forwards_folder = forwards_folder
        
        self.out_targetlist = self._prepare_out_table()
    

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
    
    def load_retrievals(self):
        """
        Load atmospheric retrieval results from HDF5 files and populate the output table.

        This method scans the retrieval folder for available HDF5 files corresponding to the planets
        listed in the input table. It dynamically detects and adds new columns found in the retrieval files,
        such as fit parameters and their uncertainties. Only planets with valid files are processed and 
        included in the output table.

        For each valid planet:
        - The method extracts the metadata from the input table.
        - It reads the retrieved atmospheric parameters from the HDF5 file (under `solution0/fit_params`).
        - It appends the extracted data as a new row in `self.out_targetlist`.

        Notes
        -----
        - Additional parameters discovered in the retrieval files are added as new columns in the output table.
        - If any expected value is missing, the method fills it with `np.nan` (for numeric types) or empty strings.

        Raises
        ------
        None

        Outputs
        -------
        Updates `self.out_targetlist` with retrieved values and metadata for all planets with valid retrieval files.
        Also prints a summary of how many valid planets were found.
        """
        print("loading retrieval data")


        # Dictionary to track additional columns dynamically discovered in HDF5 files
        additional_columns = {}

        # First pass: Identify available data and new columns
        valid_planets = []  # Store planets that have an HDF5 file

        for i, planet in tqdm(enumerate(self.targetlist["Planet Name"]), total=len(self.targetlist)):
            path = os.path.join(self.retrieval_folder, planet)
            file = os.path.join(path, f"{planet}_retrieval.hdf5")

            if not os.path.exists(file):
                print("-> File not found:", planet)
                continue  # Skip planets without an associated file

            valid_planets.append(i)  # Store index of valid planets
            # print("Valid planet:", planet)

            # Open the HDF5 file to discover new columns
            with h5py.File(file, 'r') as f:
                solution_path = "Output/Solutions/solution0/fit_params"
                if solution_path in f:
                    additional_columns["H2/He"] = f["ModelParameters/Chemistry/ratio"][()].dtype
                    for key in f[solution_path].keys():
                        additional_columns[key] = f[f"{solution_path}/{key}/value"][()].dtype
                        additional_columns[key+"_sigma_p"] = f[f"{solution_path}/{key}/sigma_p"][()].dtype
                        additional_columns[key+"_sigma_m"] = f[f"{solution_path}/{key}/sigma_m"][()].dtype
                        
        print(f"found {len(valid_planets)} valid planets")

        # Initialize `out_targetlist` with only valid planets (ones that have a file)
        self.out_targetlist = Table(
            names=self.targetlist.colnames, 
            dtype=[self.targetlist[col].dtype for col in self.targetlist.colnames]
        )

        # Ensure all additional columns exist in `out_targetlist`
        for col_name, col_dtype in additional_columns.items():
            if col_name not in self.out_targetlist.colnames:
                if np.issubdtype(col_dtype, np.number):
                    self.out_targetlist[col_name] = np.full(0, np.nan, dtype=col_dtype)
                else:
                    self.out_targetlist[col_name] = np.full(0, "", dtype="U20")  # Default string length of 20

        # Second pass: Extract data for valid planets and add them to `out_targetlist`
        for i in tqdm(valid_planets, total=len(valid_planets)):
            planet = self.targetlist["Planet Name"][i]
            path = os.path.join(self.retrieval_folder, planet)
            file = os.path.join(path, f"{planet}_retrieval.hdf5")

            # Create a dictionary with the planet's existing metadata
            planet_data = {col: self.targetlist[i][col] for col in self.targetlist.colnames}

            # Open the HDF5 file
            with h5py.File(file, 'r') as f:
                solution_path = "Output/Solutions/solution0/fit_params"
                if solution_path in f:
                    planet_data["H2/He"] = f["ModelParameters/Chemistry/ratio"][()]
                    for key in f[solution_path].keys():
                        planet_data[key] = f[f"{solution_path}/{key}/value"][()]
                        planet_data[key+"_sigma_p"] = f[f"{solution_path}/{key}/sigma_p"][()]
                        planet_data[key+"_sigma_m"] = f[f"{solution_path}/{key}/sigma_m"][()]

            # Ensure missing columns are filled with NaN or empty strings
            for col in self.out_targetlist.colnames:
                if col not in planet_data:
                    if np.issubdtype(self.out_targetlist[col].dtype, np.integer):
                        planet_data[col] = -1  # Assign a default integer value instead of NaN
                    elif np.issubdtype(self.out_targetlist[col].dtype, np.number):
                        planet_data[col] = np.nan  # Assign NaN for missing numerical values
                    else:
                        planet_data[col] = ""  # Assign empty string for missing text values

            # Add the row to `out_targetlist`
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

        
    def load_ariel_data(self,):
        """
        Load Ariel observation data (number of observations and total observing time) for each planet.

        This method reads the number of Ariel observations (`instrument_nobs`) from forward model HDF5 files 
        for each planet in the output table (`self.out_targetlist`). It also computes the total observing 
        time in hours using the planet's transit duration.

        For each planet:
        - Checks if the corresponding forward model file exists.
        - Loads the number of observations from the HDF5 file.
        - Computes observing time as: `Transit Duration [s] * nobs`, converted to hours.
        - Adds these values to the columns `nobs` and `obs_time` in the output table.

        Notes
        -----
        - Planets without a corresponding HDF5 file are skipped with a warning message.
        - If the `nobs` or `obs_time` columns are not present, they are created and initialised with `NaN`.

        Raises
        ------
        None

        Outputs
        -------
        Updates the columns `nobs` and `obs_time` in `self.out_targetlist` for planets with available data.
        """
        print("loading ariel data")

        # Ensure the required columns exist in out_targetlist
        for col in ["nobs", "obs_time"]:
            if col not in self.out_targetlist.colnames:
                self.out_targetlist[col] = np.nan  # Initialize missing columns

        for i, planet_data in tqdm(enumerate(self.out_targetlist), total=len(self.out_targetlist)):
            planet_dict = {col: planet_data[col] for col in self.out_targetlist.colnames}

            fname = os.path.join(self.forwards_folder, f"{planet_dict['Planet Name']}/{planet_dict['Planet Name']}.hdf5")
            if not os.path.isfile(fname):
                print("File not found:", planet_dict['Planet Name'])
                continue
            
            with h5py.File(fname, 'r') as f:
                ariel_path = f["Output/Spectra"]
                planet_dict["nobs"] = ariel_path["instrument_nobs"][()]
                planet_dict["obs_time"] = planet_dict['Transit Duration [s]'] * planet_dict["nobs"] * u.s.to(u.hr)

            # Ensure the row update follows the correct column order
            self.out_targetlist[i] = [planet_dict[col] for col in self.out_targetlist.colnames]
            
    def laod_profiles(self, comp_folder, pt_folder):
        """
        Load chemical composition and temperature profiles from external CSV files for each planet.

        This method searches for and reads two types of CSV files per planet:
        - Composition profiles (containing molecular abundances such as H₂, CO, H₂O, etc.)
        - Pressure-temperature (PT) profiles (used to compute a median temperature)

        It computes the median value for each species and for the temperature, and stores them in 
        dedicated columns (e.g., `H2_profile`, `CO_profile`, `T_profile`) in the output table.

        Parameters
        ----------
        comp_folder : str
            Path to the directory containing composition profile CSV files.
        pt_folder : str
            Path to the directory containing pressure-temperature profile CSV files.

        Notes
        -----
        - Profile filenames are matched using partial string matching based on the planet name and
        the presence of keywords such as "Comp" and "PT".
        - If no matching file is found, the corresponding planet is skipped.
        - Profile values are computed as the median of the data columns in the input CSVs.
        - The following columns are added to `self.out_targetlist` if not already present:
        `H2_profile`, `He_profile`, `H2O_profile`, `CO_profile`, `CO2_profile`, `CH4_profile`, `T_profile`.

        Raises
        ------
        None

        Outputs
        -------
        Updates the output table (`self.out_targetlist`) with median profile values for each molecule and temperature.
        """
        
        print("loading profiles")
        
        for col in ["H2_profile", "He_profile", "H2O_profile", "CO_profile", "CO2_profile", "CH4_profile", "T_profile"]:
            if col not in self.out_targetlist.colnames:
                self.out_targetlist[col] = np.nan  # In

        def find_file(path, keyword, selection):
            file_list = os.listdir(path)
            # print(file_list)
            # print(planet_data['Planet Name'].split("_")[0])

            matching_files = [file for file in file_list if keyword in file and selection in file and file.endswith(".csv")]
            # print(matching_files)
            return matching_files

        for i, planet_data in tqdm(enumerate(self.out_targetlist), total=len(self.out_targetlist)):

            planet_dict = {col: planet_data[col] for col in self.out_targetlist.colnames}

            matching_files =find_file(comp_folder, planet_data['Planet Name'].split("_")[0], selection="Comp")
            if matching_files:
                fname = os.path.join(comp_folder, matching_files[0])
            else:
                continue
                
            data = ascii.read(fname, format='no_header', delimiter=',', comment="#")

            column_map = ["H2", "H2O", "CO", "CO2", "CH4", "He"]
            data.rename_columns(data.colnames, column_map)
            
            for column in column_map:
                planet_dict[f"{column}_profile"] = np.median(data[column])    
            
            matching_files =find_file(pt_folder, planet_data['Planet Name'].split("_")[0], selection="PT")
            if matching_files:
                fname = os.path.join(pt_folder, matching_files[0])
            else:
                continue
            data = ascii.read(fname, format='no_header', delimiter=',', comment="#")
            planet_dict[f"T_profile"] = np.median(data["col2"])

            # Ensure the row update follows the correct column order
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
            
    def plot_nobs(self):
        """
        Plot a histogram of the number of Ariel observations per planet.

        This method generates a histogram showing the distribution of the `nobs` values
        (number of observations) across all planets in the output table. A vertical red line 
        is drawn at the threshold of 20 observations to help visualise completeness. It also 
        annotates the number of planets below and above this threshold, and highlights 
        those with `nobs >= 100`.

        Notes
        -----
        - Assumes that the column `nobs` is present and populated in `self.out_targetlist`.
        - Uses dynamic binning to account for outliers.
        - Adds axis labels, legend, and annotations to improve interpretability.

        Raises
        ------
        KeyError
            If the `nobs` column is missing from the output table.
        
        Outputs
        -------
        Displays a matplotlib figure with the histogram.
        """

        # Extract data for the histogram
        nobs_data = np.array(self.out_targetlist["nobs"])

        # Create the figure and axis
        plt.figure(figsize=(10, 6))

        # Add a vertical line at nobs = 20
        plt.axvline(x=20, color="red", linestyle="--", linewidth=2, label="Threshold: nobs = 20", zorder=10)

        # Count elements before and after nobs = 20
        below_threshold = np.sum(nobs_data < 20)
        above_threshold = np.sum(nobs_data >= 20)

        # Annotate the counts
        plt.annotate(f'< 20: {below_threshold}\n≥ 20: {above_threshold}', 
                     xy=(0.85, 0.9),xycoords='axes fraction',
                    ha='left', va='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

        
        bins = np.concatenate([np.linspace(0, 100, 31), [np.max(nobs_data)]])
        counts, bin_edges, _ = plt.hist(nobs_data, bins=bins, edgecolor="black", alpha=0.7)
        plt.annotate(f"{np.sum(nobs_data >= 100)}", xy=(105, max(counts)*0.1), fontsize=12, color="black")


        # Labels and title
        plt.xlabel("Number of Observations (nobs)", fontsize=14)
        plt.ylabel("Frequency", fontsize=14)
        plt.title("Distribution of Observations (nobs)", fontsize=16)

        plt.xticks(list(range(0, 110, 10)) + [110], labels=list(range(0, 110, 10)) + ["100+"])

        # Improve readability
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        plt.legend()
        plt.xlim(0, 110)

        # Show the plot
        plt.show()

    def plot_obs_time(self, selections=None, labels=None, colours=None, show_cumulative=True):
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

        Notes
        -----
        - Each selection will have its own histogram.
        - The cumulative observing time is plotted per bin on a log-scale Y-axis.
        - A red annotation shows how many planets exceed 200 hours for each selection.
        """

        if selections is None:
            selections = [None]

        if labels is None:
            labels = [f"Selection {i+1}" for i in range(len(selections))]

        if colours is None:
            base_colours = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple', 'tab:cyan']
            colours = base_colours[:len(selections)]

        # Setup figure and axes
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = ax1.twinx() if show_cumulative else None

        bins = np.concatenate([np.linspace(0, 200, 31), [np.max(np.array(self.out_targetlist["obs_time"]))]])

        for sel, label, colour in zip(selections, labels, colours):
            obs_time_data = np.array(self.out_targetlist["obs_time"] if sel is None else self.out_targetlist["obs_time"][sel])

            # Histogram
            counts, _, _ = ax1.hist(obs_time_data, bins=bins, edgecolor="black", alpha=0.3, label=label, color=colour)
            ax1.annotate(f"{np.sum(obs_time_data >= 200)} > 200 h", xy=(201, max(counts)*0.1), fontsize=11, color=colour)

            # Cumulative
            if show_cumulative and ax2:
                bin_indices = np.digitize(obs_time_data, bins, right=False)
                cumulative_hours = [np.sum(obs_time_data[bin_indices <= i]) for i in range(1, len(bins))]
                ax2.plot(bins[:-1], cumulative_hours, linestyle="--", marker="o", label=f"Cumulative {label}", color=colour)

        # Labels and appearance
        ax1.set_xlabel("Observing Time (hours)", fontsize=14)
        ax1.set_ylabel("Frequency", fontsize=14)
        if ax2:
            ax2.set_ylabel("Cumulative Observing Time (hours)", fontsize=14, color="darkred")
            ax2.set_yscale("log")
            ax2.tick_params(axis="y", labelcolor="darkred")

        ax1.set_xlim(0, 220)
        ax1.grid(axis="y", linestyle="--", alpha=0.5)
        plt.title("Histogram of Observing Time with Cumulative Distribution", fontsize=16)

        # Combined legend
        lines_labels = [ax.get_legend_handles_labels() for ax in [ax1, ax2] if ax]
        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        fig.legend(lines, labels, loc="upper left", bbox_to_anchor=(0.15, 0.87), fontsize=12)

        plt.tight_layout()
        plt.show()


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
                            background_regions=None):
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
            
            # If background regions are specified, add a coloured band for each region
            if background_regions is not None:
                for reg in background_regions:
                    # Draw background coloured section over the specified x-range
                    ax.axvspan(reg['start'], reg['end'], facecolor=reg.get('color', 'lightgrey'),
                            alpha=reg.get('alpha', 0.3), zorder=0)
            
            # Plot expected values (empty markers)
            ax.scatter(np.arange(len(self.out_targetlist)), self.out_targetlist[f"{key}_profile"], 
                    label="Expected", edgecolor="k", facecolors="none", marker=marker, alpha=alpha, zorder=3)
            
            # Calculate sigma difference
            sigma_diff = (self.out_targetlist[key] - self.out_targetlist[f"{key}_profile"]) / self.out_targetlist[f"{key}_sigma"]
            quant_sigma = np.clip(np.floor(np.abs(sigma_diff)), *normalize_range)
            num_over3 = np.sum(np.abs(sigma_diff) > 3)
            
            # Plot error bars
            ax.errorbar(np.arange(len(self.out_targetlist)), self.out_targetlist[key],
                        yerr=self.out_targetlist[f"{key}_sigma"], fmt="none", alpha=alpha, zorder=5, color="grey")
            
            # Scatter plot with sigma-based colour mapping
            sc = ax.scatter(np.arange(len(self.out_targetlist)), self.out_targetlist[key], c=quant_sigma, 
                            cmap=cmap, norm=norm, s=50, alpha=alpha, zorder=6)
            
            # Annotation for 3σ outliers with a more opaque background (alpha set to 1.0)
            if add_annotations:
                ax.annotate(f'{num_over3} over {len(self.out_targetlist)} points > 3σ', xy=(0.55, 0.1),
                            xycoords='axes fraction', ha='left', va='top', fontsize=10,
                            bbox=dict(facecolor='white', alpha=1.0), zorder=10)
            
            ax.set_title(key)
            if log_scale:
                ax.set_yscale("log")
            
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.set_xlabel("Planet Index")
        
        # Set common y-axis label
        axes[0].set_ylabel(y_label)
        
        
        # Global colourbar for sigma difference
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # Position: [left, bottom, width, height]
        cbar = fig.colorbar(sc, cax=cbar_ax, ticks=range(6))
        cbar.set_label("Quantized Sigma Difference")
        
        # Create global legend including the expected marker and the background regions (if any)
        legend_handles = [expected_handle] + region_handles
        # Place the legend in the upper left corner so it does not overlap the colourbar
        fig.legend(handles=legend_handles, loc='upper left')
        
        plt.suptitle(title)
        plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust layout to fit the colourbar
        plt.show()


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
            fig = plt.figure(figsize=(10, 8))
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
            fig = plt.figure(figsize=(10, 8))
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
    
    
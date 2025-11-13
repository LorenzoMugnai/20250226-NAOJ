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

class Inspector:
    def __init__(self, input_table, retrieval_folder, forwards_folder):
        self.targetlist = ascii.read(input_table)
        self.retrieval_folder = retrieval_folder
        self.forwards_folder = forwards_folder
        
        self.out_targetlist = self._prepare_out_table()
    

    def _prepare_out_table(self):
        return Table(names=self.targetlist.colnames, dtype=[self.targetlist[col].dtype for col in self.targetlist.colnames])
    
    def save_table(self, out_tab_fname):
        ascii.write(self.out_targetlist, out_tab_fname, format='csv', overwrite=True)   
        print(f"Table saved to {out_tab_fname}") 
    
    def load_retrievals(self):
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
        idx = [i for i, name in enumerate(self.targetlist["Planet Name"]) if name not in self.out_targetlist["Planet Name"]]
        print(f"Missing planets: {len(idx)}")
        print(self.targetlist["Planet Name"][idx])

        
    def load_ariel_data(self,):
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
            
    def laod_profiles(self, comp_folders, pt_folders):
        print("loading profiles")
        
        for col in ["H2_profile", "He_profile", "H2O_profile", "CO_profile", "CO2_profile", "CH4_profile", "T_profile"]:
            if col not in self.out_targetlist.colnames:
                self.out_targetlist[col] = np.nan  # In

        def find_file(path, keyword):
            file_list = os.listdir(path)
            # print(file_list)
            # print(planet_data['Planet Name'].split("_")[0])

            matching_files = [file for file in file_list if keyword in file]
            # print(matching_files)
            return matching_files

        for i, planet_data in tqdm(enumerate(self.out_targetlist), total=len(self.out_targetlist)):

            planet_dict = {col: planet_data[col] for col in self.out_targetlist.colnames}

            matching_files =find_file(comp_folders[0], planet_data['Planet Name'].split("_")[0])
            if matching_files:
                fname = os.path.join(comp_folders[0], matching_files[0])
            else:
                matching_files =find_file(comp_folders[1], planet_data['Planet Name'].split("_")[0])
                if matching_files:
                    fname = os.path.join(comp_folders[1], matching_files[0])
                else:
                    continue
                
            data = ascii.read(fname, format='no_header', delimiter=',', comment="#")

            column_map = ["H2", "H2O", "CO", "CO2", "CH4", "He"]
            data.rename_columns(data.colnames, column_map)
            for column in column_map:
                planet_dict[f"{column}_profile"] = np.median(data[column])    
            
            matching_files =find_file(pt_folders[0], planet_data['Planet Name'].split("_")[0])
            if matching_files:
                fname = os.path.join(pt_folders[0], matching_files[0])
            else:
                matching_files =find_file(pt_folders[1], planet_data['Planet Name'].split("_")[0])
                if matching_files:
                    fname = os.path.join(pt_folders[1], matching_files[0])
                else:
                    continue
            data = ascii.read(fname, format='no_header', delimiter=',', comment="#")
            planet_dict[f"T_profile"] = np.median(data["col2"])

            # Ensure the row update follows the correct column order
            self.out_targetlist[i] = [planet_dict[col] for col in self.out_targetlist.colnames]

    def compute_elemental_ratios(self):
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

    def plot_obs_time(self):

        # Extract data for the histogram
        obs_time_data = np.array(self.out_targetlist["obs_time"])  # Replace with correct column name if different


        # Create figure and axis
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Plot histogram
        bins = np.concatenate([np.linspace(0, 200, 31), [np.max(obs_time_data)]])
        counts, bin_edges, _ = plt.hist(obs_time_data, bins=bins, edgecolor="black", alpha=0.7)
        plt.annotate(f"{np.sum(obs_time_data >= 200)}", xy=(205, max(counts)*0.1), fontsize=12, color="black")

        # Compute cumulative distribution
        cumulative_hours = np.cumsum(counts * np.diff(bin_edges))  # Multiply bin count by bin width to get total hours

        # Create second Y-axis for cumulative curve
        ax2 = ax1.twinx()
        ax2.plot(bin_edges[:-1], cumulative_hours, color="red", linestyle="--", marker="o", label="Cumulative Observing Time (hrs)")

        # Labels and title
        ax1.set_xlabel("Observing Time (hours)", fontsize=14)
        ax1.set_ylabel("Frequency", fontsize=14, color="blue")
        ax2.set_ylabel("Total Cumulative Observing Time (hours)", fontsize=14, color="red")
        plt.title("Histogram of Observing Time with Cumulative Distribution", fontsize=16)

        # Improve readability
        ax1.tick_params(axis="y", labelcolor="blue")
        ax2.tick_params(axis="y", labelcolor="red")
        ax1.grid(axis="y", linestyle="--", alpha=0.6)
        ax1.set_xlim(0, 220)
        ax2.set_yscale("log")

        # Legend
        fig.legend(loc="upper left", bbox_to_anchor=(0.15, 0.85))

        # Show the plot
        plt.show()

    def plot_comparison(self, x_col, y_col, y_err_plus_col, y_err_minus_col, conversion_factor=None, title=None):
        """
        Plot a comparison between two columns (x and y) with error bars and a quantised colour bar.
        
        Parameters:
        - data: dictionary-like object (e.g. pandas DataFrame) containing the columns.
        - x_col: string, name of the x-axis column.
        - y_col: string, name of the y-axis column.
        - y_err_plus_col: string, name of the column with the positive error for y.
        - y_err_minus_col: string, name of the column with the negative error for y.
        - conversion_factor: optional; if provided, x values will be multiplied by this factor.
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
        sc = ax.scatter(x_values, y_values, c=quant_sigma, cmap=cmap, s=50)
        
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
                            add_annotations=True, fig_size=(15, 5), shared_y=False, shared_x=False):
        """
        Generalized function to create comparison plots with multiple subplots.

        Parameters:
        - data_table: dict or dataframe containing the dataset
        - keys: list of strings specifying the elements, ratios, or molecules to compare
        - y_label: label for the y-axis
        - title: overall title for the figure
        - color_map: colormap for scatter points
        - normalize_range: tuple (min, max) for color normalization
        - marker: scatter point marker shape
        - alpha: transparency of markers
        - log_scale: whether to use a log scale for the y-axis
        - add_annotations: whether to annotate number of outliers (> 3σ)
        - fig_size: figure size in inches
        - shared_y: whether to share the y-axis across subplots
        - shared_x: whether to share the x-axis across subplots
        """
        
        num_plots = len(keys)
        fig, axes = plt.subplots(1, num_plots, figsize=fig_size, sharey=shared_y, sharex=shared_x)
        
        if num_plots == 1:
            axes = [axes]  # Ensure axes is always iterable
        
        cmap = plt.get_cmap(color_map, 6)
        norm = mcolors.Normalize(vmin=normalize_range[0], vmax=normalize_range[1])
        
        for i, key in enumerate(keys):
            ax = axes[i]
            
            # Plot expected values (empty markers)
            ax.scatter(np.arange(len(self.out_targetlist)), self.out_targetlist[f"{key}_profile"], 
                    label="Expected", edgecolor="k", facecolors="none", marker=marker, alpha=alpha)
            
            # Calculate sigma difference
            sigma_diff = (self.out_targetlist[key] - self.out_targetlist[f"{key}_profile"]) / self.out_targetlist[f"{key}_sigma"]
            quant_sigma = np.clip(np.floor(np.abs(sigma_diff)), *normalize_range)
            num_over3 = np.sum(np.abs(sigma_diff) > 3)
            
            # Error bars
            ax.errorbar(np.arange(len(self.out_targetlist)), self.out_targetlist[key],
                        yerr=self.out_targetlist[f"{key}_sigma"], fmt="none", alpha=0.7, zorder=1, color="grey")
            
            # Scatter plot with color mapping
            sc = ax.scatter(np.arange(len(self.out_targetlist)), self.out_targetlist[key], c=quant_sigma, 
                            cmap=cmap, norm=norm, s=50, alpha=1, zorder=2)
            
            # Annotations for 3σ outliers
            if add_annotations:
                ax.annotate(f'{num_over3} over {len(self.out_targetlist)} points > 3σ', xy=(0.55, 0.1), xycoords='axes fraction',
                            ha='left', va='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
            
            ax.set_title(key)  # Title for each subplot
            if log_scale:
                ax.set_yscale("log")
            
            ax.grid(True, linestyle="--", alpha=0.5)
        
        # Common y-axis label
        axes[0].set_ylabel(y_label)
        
        # Global colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # Position: [left, bottom, width, height]
        cbar = fig.colorbar(sc, cax=cbar_ax, ticks=range(6))
        cbar.set_label("Quantized Sigma Difference")
        
        # Set overall title and layout
        plt.suptitle(title)
        plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust layout to fit colorbar
        plt.show()

    def plot_scatter_with_histograms(
        self, x_key, y_key, selection=None,
        x_err_key=None, y_err_key=None, color_key=None, contour_only=False,
        cmap="viridis", hist_color="black", bins_num=30,  bins_x=None, bins_y=None, marker="o",
        ax_scatter=None, ax_hist_x=None, ax_hist_y=None, ax_legend=None, fig=None,
        label=None, alpha=0.6, 
        err_color="black", elinewidth=1.5, capsize=3, capthick=1, errorevery=1, err_alpha=0.5, linestyle="-",
        global_norm=None
    ):
        """
        Creates or updates a scatter plot with histograms projected on the x and y axes.
        
        If axes are provided, it overlays new data on the existing plots.
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
    
    
    def plot_scatter_with_histograms_compared_with_expectation(
        self, selection=None,
        color_key=None, expect_color="k",
        cmap="viridis", hist_color="black", bins_num=30,  bins_x=None, bins_y=None, marker="o",
        ax_scatter=None, ax_hist_x=None, ax_hist_y=None, ax_legend=None, fig=None,
        label=None, alpha=0.6, 
        err_color="black", elinewidth=1.5, capsize=3, capthick=1, errorevery=1, err_alpha=0.5, linestyle="-",
        global_norm=None
    ):
        """
        Creates or updates a scatter plot with histograms projected on the x and y axes.
        
        If axes are provided, it overlays new data on the existing plots.
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
                alpha=alpha, facecolors="None", label=exp_label, marker=marker, zorder=100
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
    
    
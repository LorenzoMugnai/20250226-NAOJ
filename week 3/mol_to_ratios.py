import numpy as np

def calculate_ratios_with_uncertainties(ratio_H2_He, H2O, CO, CO2, CH4,
                                        sigma_H2O=None, sigma_CO=None, 
                                        sigma_CO2=None, sigma_CH4=None):
    """
    Calculates the O/H, C/O, He/H ratios using the H2/He ratio instead of separate H2 and He values.
    
    Parameters:
        ratio_H2_He : float
            The precise ratio between H2 and He.
        H2O, CO, CO2, CH4 : float or array-like
            Molar fractions of each molecule (linear scale).
        sigma_H2O, sigma_CO, sigma_CO2, sigma_CH4 : float or array-like (optional)
            Uncertainties in the molar fractions. If None, uncertainties are not computed.
    
    Returns:
        dict with keys "O/H", "C/O", "He/H" and their uncertainties (if provided).
    """
    #ratio_H2_He is provided as reverse
    ratio_H2_He = 1/ratio_H2_He
    
    # Convert inputs to NumPy arrays
    H2O, CO, CO2, CH4 = map(np.asarray, (H2O, CO, CO2, CH4))
    
    # Compute total known molecular fraction
    X_molecules = H2O + CO + CO2 + CH4

    # Compute He and H2 abundances using the H2/He ratio
    He = (1 - X_molecules) / (1 + ratio_H2_He)
    H2 = ratio_H2_He * He

    # Ensure no division by zero
    total = H2 + H2O + CO + CO2 + CH4 + He
    total = np.where(total == 0, 1, total)

    # Normalize molar fractions
    H2, H2O, CO, CO2, CH4, He = H2 / total, H2O / total, CO / total, CO2 / total, CH4 / total, He / total

    # Compute elemental abundances
    H = 2 * H2 + 2 * H2O + 4 * CH4
    O = H2O + CO + 2 * CO2
    C = CO + CO2 + CH4

    # Compute ratios
    O_H = np.divide(O, H, out=np.zeros_like(O), where=H!=0)
    C_O = np.divide(C, O, out=np.zeros_like(C), where=O!=0)
    He_H = np.divide(He, H, out=np.zeros_like(He), where=H!=0)

    # Create results dictionary
    results = {"O/H": O_H, "C/O": C_O, "He/H": He_H, "H2": H2, "He": He, "H": H, "O": O, "C": C}

    # If no uncertainties are provided, return only ratios
    if all(sigma is None for sigma in [sigma_H2O, sigma_CO, sigma_CO2, sigma_CH4]):
        return results

    # Convert uncertainties to NumPy arrays (default to zero if not provided)
    sigma_H2O = np.asarray(sigma_H2O) if sigma_H2O is not None else np.zeros_like(H2O)
    sigma_CO = np.asarray(sigma_CO) if sigma_CO is not None else np.zeros_like(CO)
    sigma_CO2 = np.asarray(sigma_CO2) if sigma_CO2 is not None else np.zeros_like(CO2)
    sigma_CH4 = np.asarray(sigma_CH4) if sigma_CH4 is not None else np.zeros_like(CH4)

    # Compute uncertainties in molecular fraction sum
    sigma_X_molecules = np.sqrt(sigma_H2O**2 + sigma_CO**2 + sigma_CO2**2 + sigma_CH4**2)

    # Compute uncertainties in He and H2 (ratio_H2_He has no uncertainty)
    sigma_He = sigma_X_molecules / (1 + ratio_H2_He)
    sigma_H2 = ratio_H2_He * sigma_He

    # Compute uncertainties in elemental abundances
    sigma_H = np.sqrt((2 * sigma_H2) ** 2 + (2 * sigma_H2O) ** 2 + (4 * sigma_CH4) ** 2)
    sigma_O = np.sqrt(sigma_H2O ** 2 + sigma_CO ** 2 + (2 * sigma_CO2) ** 2)
    sigma_C = np.sqrt(sigma_CO ** 2 + sigma_CO2 ** 2 + sigma_CH4 ** 2)

    # Compute uncertainties in the ratios using relative error propagation
    sigma_O_H = np.abs(O_H) * np.sqrt(np.divide(sigma_O ** 2, O ** 2, where=O!=0) + np.divide(sigma_H ** 2, H ** 2, where=H!=0))
    sigma_C_O = np.abs(C_O) * np.sqrt(np.divide(sigma_C ** 2, C ** 2, where=C!=0) + np.divide(sigma_O ** 2, O ** 2, where=O!=0))
    sigma_He_H = np.abs(He_H) * np.sqrt(np.divide(sigma_He ** 2, He ** 2, where=He!=0) + np.divide(sigma_H ** 2, H ** 2, where=H!=0))

    # Add uncertainties to results dictionary
    results["O/H"] = (O_H, sigma_O_H)
    results["C/O"] = (C_O, sigma_C_O)
    results["He/H"] = (He_H, sigma_He_H)
    results["H2"] = (H2, sigma_H2)
    results["He"] = (He, sigma_He)
    results["H"] = (H, sigma_H)
    results["O"] = (O, sigma_O)
    results["C"] = (C, sigma_C)

    return results

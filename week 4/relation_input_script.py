def PlanetChemistry(pp, target_no=0, df=None):

    filename = df['Chemistry filename'][target_no].replace(".csv", ".dat")
    print("RELATION SCRIPT - replaced original chem filename:", pp._raw_config["Chemistry"]["filename"], "with:", filename)
    pp._raw_config["Chemistry"]["filename"] = filename

    return pp

def PlanetTemperature(pp, target_no=0, df=None):

    
    filename = df['Pressure-Temperature filename'][target_no].replace(".csv", ".dat")
    print("RELATION SCRIPT - replaced original TP profile:", pp._raw_config["Temperature"]["filename"], " with:", filename)
    pp._raw_config["Temperature"]["filename"] = filename
    return pp




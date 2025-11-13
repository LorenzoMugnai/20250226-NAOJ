from taurex.data.profiles.temperature import TemperatureFile
def PlanetChemistry(model, target_no=0, df=None):

    filename = df['Chemistry filename'][target_no]
    print("RELATION SCRIPT - replaced original chem filename:", model.chemistry._filename, "with:", filename)
    print(target_no)
    model.chemistry._filename = filename

    return model

def PlanetTemperature(model, target_no=0, df=None):

    filename = df['Temperature filename'][target_no]
    print("RELATION SCRIPT - replaced original TP profile with:", filename)
    print(target_no)
    model._temperature_profile = TemperatureFile(filename=filename, temp_col=1, press_col=0, temp_units='K', press_units='Pa')

    return model




from enum import Enum

class Category(Enum):
    CLOUDLESS = 0
    HIGH_LEVEL_CLOUD = 1
    LOW_SIGNAL = 2

class Channel(Enum):
    MAIN     = 1
    PARALLEL = 2

class Measure(Enum):
    PRESSURE = 1
    TEMPERATURE = 2
    REL_HUMIDITY = 3
    ABS_HUMIDITY = 4
    WIND_DIRECTION = 5
    WIND_SPEED = 6

def measure_to_graph(m: Measure):
    if m == Measure.PRESSURE:
        return "Pressure", "hPa"
    elif m == Measure.TEMPERATURE:
        return "Temperature", "$C^{\circ}$"
    elif m == Measure.REL_HUMIDITY:
        return "Relative Humidity", "%"
    elif m == Measure.ABS_HUMIDITY:
        return "Absolute Humidity", "kg/kg"
    elif m == Measure.WIND_DIRECTION:
        return "Wind Direction", "$^{\circ}$"
    elif m == Measure.WIND_SPEED:
        return "Wind Speed", "m/s"
    else:
        raise Exception(m)

def collection_to_label(path: str):
    # Я помню про существование match
    if path == "M2T3NPRAD:tavg3_3d_rad_Np/DTDTSWR":
        return "Radiation Diagnostics / Air temperature tendency due to shortwave [K/s]"
    if path == "M2T3NPRAD:tavg3_3d_rad_Np/DTDTLWR":
        return "Radiation Diagnostics / Air temperature tendency due to longwave [K/s]"
    if path == "M2T3NPRAD:tavg3_3d_rad_Np/CLOUD":
        return "Radiation Diagnostics / Cloud fraction for radiation [1]"

    if path == "M2T3NVASM:tavg3_3d_asm_Nv/QI":
        return "Assimilated Meteorological Fields / Mass fraction of cloud ice water [kg/kg]"
    if path == "M2T3NVASM:tavg3_3d_asm_Nv/U":
        return "Assimilated Meteorological Fields / Eastward wind [m/s]"
    if path == "M2T3NVASM:tavg3_3d_asm_Nv/QV":
        return "Assimilated Meteorological Fields / Specific humidity [kg/kg]"
    if path == "M2T3NVASM:tavg3_3d_asm_Nv/T":
        return "Assimilated Meteorological Fields / Air temperature [K]"
    if path == "M2T3NVASM:tavg3_3d_asm_Nv/RH":
        return "Assimilated Meteorological Fields / Relative humidity after moist [1]"
    if path == "M2T3NVASM:tavg3_3d_asm_Nv/CLOUD":
        return "Assimilated Meteorological Fields / Cloud fraction for radiation [1]"
    if path == "M2T3NVASM:tavg3_3d_asm_Nv/V":
        return "Assimilated Meteorological Fields / Northward wind [m/s]"
    if path == "M2T3NVASM:tavg3_3d_asm_Nv/QL":
        return "Assimilated Meteorological Fields / Mass fraction of cloud liquid water [kg/kg]"

    if path == "M2T3NPTDT:tavg3_3d_tdt_Np/DTDTANA":
        return "Temperature Tendencies / Total temperature analysis tendency [K/s]"

    if path == "M2T3NPCLD:tavg3_3d_cld_Np/QI":
        return "Cloud Diagnostics / Mass fraction of cloud ice water [kg/kg]"
    if path == "M2T3NPCLD:tavg3_3d_cld_Np/RH":
        return "Cloud Diagnostics / Relative humidity after moist [1]"
    if path == "M2T3NPCLD:tavg3_3d_cld_Np/TAUCLI":
        return "Cloud Diagnostics / In cloud optical thickness for ice clouds [1]"
    if path == "M2T3NPCLD:tavg3_3d_cld_Np/INCLOUDQI":
        return "Cloud Diagnostics / In cloud cloud ice for radiation [kg/kg]"
    if path == "M2T3NPCLD:tavg3_3d_cld_Np/CLOUD":
        return "Cloud Diagnostics / Cloud fraction for radiation [1]"
    if path == "M2T3NPCLD:tavg3_3d_cld_Np/QL":
        return "Cloud Diagnostics / Mass fraction of cloud liquid water [kg/kg]"

    if path == "M2T3NPUDT:tavg3_3d_udt_Np/DVDTANA":
        return "Wind Tendencies / Total eastward wind analysis tendency [m/s]"
    if path == "M2T3NPUDT:tavg3_3d_udt_Np/DUDTANA":
        return "Wind Tendencies / Total northward wind analysis tendency [m/s]"

    if path == "M2T3NPQDT:tavg3_3d_qdt_Np/DQVDTANA":
        return "Moist Tendencies / Total specific humidity analysis tendency [kg / (ks*s)]"

    assert False, "Unknown path!"
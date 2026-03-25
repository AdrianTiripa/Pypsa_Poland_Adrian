import pandas as pd

PROVINCE_ORDER = [
    "DS", "KP", "LD", "LU", "LB", "MA", "MZ", "OP",
    "PK", "PD", "PM", "SL", "SK", "WN", "WP", "ZP",
]

year = 1940

old_heat_path = r"C:\Users\adria\MODEL_PyPSA\Core\Profiles_1940_2025\HeatDemand_1940_2025_working_data.csv"
new_heat_path = r"C:\Users\adria\MODEL_PyPSA\Core\Profiles_new\Core\HeatDemand\Qishare_1940_2050.csv"

old_cop_path = r"C:\Users\adria\MODEL_PyPSA\Core\Profiles_1940_2025\COP_1940_2025_working_data.csv"
new_cop_path = r"C:\Users\adria\MODEL_PyPSA\Core\Profiles_new\Core\COP\COPiavg3_1940_2050.csv"

def drop_feb29_if_needed(df, year):
    is_leap = (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0))
    if not is_leap:
        return df.copy()
    start = 59 * 24
    end = 60 * 24
    return pd.concat([df.iloc[:start], df.iloc[end:]], axis=0)

# old heat
old_heat = pd.read_csv(old_heat_path, index_col=0)
old_heat.index = pd.to_datetime(old_heat.index, errors="coerce")
old_heat = old_heat[old_heat.index.year == year].copy()
old_heat = old_heat[[c for c in old_heat.columns if c in PROVINCE_ORDER]]
if len(old_heat) == 8784:
    old_heat = drop_feb29_if_needed(old_heat, year)

# new heat (GW -> MW)
new_heat = pd.read_csv(new_heat_path, header=None)
new_heat.columns = PROVINCE_ORDER
if len(new_heat) == 8784:
    new_heat = drop_feb29_if_needed(new_heat, year)
new_heat = new_heat * 1000.0

# old COP
old_cop = pd.read_csv(old_cop_path, index_col=0)
old_cop.index = pd.to_datetime(old_cop.index, errors="coerce")
old_cop = old_cop[old_cop.index.year == year].copy()
old_cop = old_cop[[c for c in old_cop.columns if c in PROVINCE_ORDER]]
if len(old_cop) == 8784:
    old_cop = drop_feb29_if_needed(old_cop, year)

# new COP
new_cop = pd.read_csv(new_cop_path, header=None)
new_cop.columns = PROVINCE_ORDER
if len(new_cop) == 8784:
    new_cop = drop_feb29_if_needed(new_cop, year)

old_elec_for_heat = old_heat / old_cop
new_elec_for_heat = new_heat / new_cop

print("Old annual electricity for heat [MWh]:", round(old_elec_for_heat.sum().sum(), 0))
print("New annual electricity for heat [MWh]:", round(new_elec_for_heat.sum().sum(), 0))
print("Ratio new/old:", round(new_elec_for_heat.sum().sum() / old_elec_for_heat.sum().sum(), 3))
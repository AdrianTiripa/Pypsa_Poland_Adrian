from pathlib import Path
import pandas as pd

base = Path(r"C:\Users\adria\MODEL_PyPSA\Core\data")   # change if needed

buses_path = base / "buses.csv"
links_path = base / "links.csv"
loads_p_set_path = base / "loads-p_set.csv"

buses = pd.read_csv(buses_path)
links = pd.read_csv(links_path)

# --------------------------------------------------
# 1) Add missing hydrogen buses so imported links are valid
# --------------------------------------------------
elec_buses = buses.copy()
hydrogen_rows = []

for _, row in elec_buses.iterrows():
    bus = str(row["name"])
    if bus.endswith("_heat") or bus.endswith("_hydrogen") or bus.endswith("_high_temp_heat") or bus.endswith("_transport"):
        continue

    h2_bus = f"{bus}_hydrogen"
    if h2_bus in set(buses["name"]):
        continue

    new_row = row.copy()
    new_row["name"] = h2_bus
    new_row["carrier"] = "hydrogen"
    hydrogen_rows.append(new_row)

if hydrogen_rows:
    buses = pd.concat([buses, pd.DataFrame(hydrogen_rows)], ignore_index=True)

# --------------------------------------------------
# 2) Clean imported hydrogen links
# --------------------------------------------------
hyd_mask = links["name"].astype(str).str.endswith("hydrogen")

if "carrier" in links.columns:
    links.loc[hyd_mask, "carrier"] = links.loc[hyd_mask, "carrier"].fillna("hydrogen")

if "efficiency" in links.columns:
    links.loc[hyd_mask, "efficiency"] = links.loc[hyd_mask, "efficiency"].fillna(1.0)

# optional: if you want them clearly bidirectional transport links
if "p_min_pu" in links.columns:
    links.loc[hyd_mask, "p_min_pu"] = links.loc[hyd_mask, "p_min_pu"].fillna(-1.0)

buses.to_csv(buses_path, index=False)
links.to_csv(links_path, index=False)

# --------------------------------------------------
# 3) Reformat imported load timestamps to ISO strings
# --------------------------------------------------
loads = pd.read_csv(loads_p_set_path)

first_col = loads.columns[0]
dt = pd.to_datetime(loads[first_col], errors="coerce")

if dt.notna().all():
    loads[first_col] = dt.dt.strftime("%Y-%m-%d %H:%M:%S")
    loads.to_csv(loads_p_set_path, index=False)

print("Patched buses.csv, links.csv, and loads-p_set.csv")
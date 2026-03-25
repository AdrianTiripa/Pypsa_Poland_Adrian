import pandas as pd
from pathlib import Path

run_dir = Path(r"C:\Users\adria\MODEL_PyPSA\Core\Old_runs_up_until_03.21.2026\run_2020_20260301_004844_6hr_Optimal_333s")

loads = pd.read_csv(run_dir / "loads.csv")
p_set = pd.read_csv(run_dir / "loads-p_set.csv", index_col=0)

print("Number of loads:", len(loads))
print("\nSample load names:")
print(loads["name"].head(20).tolist())

def classify(name):
    s = str(name)
    if s.endswith("_heat"):
        return "heat"
    if s.endswith("_high_temp_heat"):
        return "high_temp_heat"
    if "_hydrogen" in s:
        return "hydrogen"
    if "transport" in s.lower():
        return "transport"
    return "electricity_or_other"

loads["class"] = loads["name"].apply(classify)

print("\nLoad counts by class:")
print(loads["class"].value_counts())

common = [c for c in p_set.columns if c in loads["name"].values]
meta = loads.set_index("name").loc[common].copy()
meta["class"] = meta.index.to_series().apply(classify)

annual = p_set[common].sum(axis=0)

summary = annual.groupby(meta["class"]).sum().sort_values(ascending=False)
print("\nAnnual summed load by class (raw MW-snapshot sum):")
print(summary)
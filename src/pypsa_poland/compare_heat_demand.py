import pandas as pd

PROVINCE_ORDER = [
    "DS", "KP", "LD", "LU", "LB", "MA", "MZ", "OP",
    "PK", "PD", "PM", "SL", "SK", "WN", "WP", "ZP",
]

# ---- paths ----
old_path = r"C:\Users\adria\MODEL_PyPSA\Core\Profiles_1940_2025\HeatDemand_1940_2025_working_data.csv"
new_path = r"C:\Users\adria\MODEL_PyPSA\Core\Profiles_new\Core\HeatDemand\Qishare_1940_2050.csv"

year = 1940
old_keep_feb29 = False
new_keep_feb29 = False   # use False if your model currently drops Feb 29
new_input_is_gw = True   # set True if the new Qishare file is in GW


def drop_feb29_if_needed(df, year):
    # assumes one full year hourly table, no datetime index needed
    is_leap = (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0))
    if not is_leap:
        return df.copy()
    start = 59 * 24
    end = 60 * 24
    return pd.concat([df.iloc[:start], df.iloc[end:]], axis=0)


# ---- old file ----
# old file likely has an index column and named columns
old_df = pd.read_csv(old_path, index_col=0)

# select only the target year if the file contains many years
# if the index is datetime-like:
try:
    old_idx = pd.to_datetime(old_df.index, errors="coerce")
    if old_idx.notna().all():
        old_df.index = old_idx
        old_df = old_df[old_df.index.year == year].copy()
except Exception:
    pass

if old_keep_feb29 is False and len(old_df) == 8784:
    old_df = drop_feb29_if_needed(old_df, year)

# keep only province columns if needed
old_cols = [c for c in old_df.columns if c in PROVINCE_ORDER]
old_df = old_df[old_cols].copy()


# ---- new file ----
# new file is a raw matrix: no header, no index
new_df = pd.read_csv(new_path, header=None)
new_df.columns = PROVINCE_ORDER

if new_keep_feb29 is False and len(new_df) == 8784:
    new_df = drop_feb29_if_needed(new_df, year)

if new_input_is_gw:
    new_df = new_df * 1000.0   # convert GW -> MW


# ---- summary function ----
def summarize(df, label):
    total_by_region_mwh = df.sum(axis=0)
    total_system_mwh = total_by_region_mwh.sum()
    peak_hour_mw = df.sum(axis=1).max()
    avg_hour_mw = df.sum(axis=1).mean()

    print(f"\n===== {label} =====")
    print(f"Rows: {len(df)}")
    print(f"Total annual heat demand [MWh]: {total_system_mwh:,.0f}")
    print(f"Peak hourly total heat demand [MW]: {peak_hour_mw:,.0f}")
    print(f"Average hourly total heat demand [MW]: {avg_hour_mw:,.0f}")
    print("\nTotal annual heat demand by region [MWh]:")
    print(total_by_region_mwh.round(0).sort_values(ascending=False))


summarize(old_df, "OLD")
summarize(new_df, "NEW")

# Optional direct comparison
comparison = pd.DataFrame({
    "old_MWh": old_df.sum(axis=0),
    "new_MWh": new_df.sum(axis=0),
})
comparison["ratio_new_to_old"] = comparison["new_MWh"] / comparison["old_MWh"]
print("\n===== REGION COMPARISON =====")
print(comparison.round(3))
print("\nSystem-level ratio new/old:",
      round(new_df.sum().sum() / old_df.sum().sum(), 3))
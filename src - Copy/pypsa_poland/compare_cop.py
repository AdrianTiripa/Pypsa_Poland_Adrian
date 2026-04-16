import pandas as pd

PROVINCE_ORDER = [
    "DS", "KP", "LD", "LU", "LB", "MA", "MZ", "OP",
    "PK", "PD", "PM", "SL", "SK", "WN", "WP", "ZP",
]

old_path = r"C:\Users\adria\MODEL_PyPSA\Core\Profiles_1940_2025\COP_1940_2025_working_data.csv"
new_path = r"C:\Users\adria\MODEL_PyPSA\Core\Profiles_new\Core\COP\COPiavg3_1940_2050.csv"

year = 1940

old_df = pd.read_csv(old_path, index_col=0)
old_idx = pd.to_datetime(old_df.index, errors="coerce")
old_df.index = old_idx
old_df = old_df[old_df.index.year == year].copy()

old_cols = [c for c in old_df.columns if c in PROVINCE_ORDER]
old_df = old_df[old_cols]

new_df = pd.read_csv(new_path, header=None)
new_df.columns = PROVINCE_ORDER

print("Old mean COP by region:")
print(old_df.mean().round(3))
print("\nNew mean COP by region:")
print(new_df.mean().round(3))

print("\nSystem average old COP:", round(old_df.mean().mean(), 3))
print("System average new COP:", round(new_df.mean().mean(), 3))
print("Ratio new/old:", round(new_df.mean().mean() / old_df.mean().mean(), 3))
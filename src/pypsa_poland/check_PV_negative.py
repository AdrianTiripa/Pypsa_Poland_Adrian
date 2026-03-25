import pandas as pd

path = r"C:\Users\adria\MODEL_PyPSA\Core\Profiles_new\Core\TprovCF_PV_1940_2025.csv"  # adjust if needed
df = pd.read_csv(path, index_col=0)
df.index = pd.to_datetime(df.index, errors="coerce")

pv2016 = df[df.index.year == 1946].copy()

print("Global min in 2016:", pv2016.min().min())
print("Most negative entries by column:")
print(pv2016.min().sort_values().head(10))

bad = pv2016[pv2016.min(axis=1) < 0]
print(bad.head(20))
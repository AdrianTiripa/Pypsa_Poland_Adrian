from pathlib import Path
import geopandas as gpd
import matplotlib.pyplot as plt

PATH = Path(r"C:\Users\adria\MODEL_PyPSA\Core\pypsa-poland_ADRIAN\pl.json")
OUT  = Path(r"C:\Users\adria\MODEL_PyPSA\Core\figures_maps\poland_regions_only.png")

NAME_TO_CODE = {
    "lower silesian": "DS", "dolnoslaskie": "DS",
    "kuyavian-pomeranian": "KP", "kujawsko-pomorskie": "KP",
    "lodz": "LD", "lodzkie": "LD",
    "lublin": "LU", "lubelskie": "LU",
    "lubusz": "LB", "lubuskie": "LB",
    "lesser poland": "MA", "malopolskie": "MA",
    "masovian": "MZ", "mazowieckie": "MZ",
    "opole": "OP", "opolskie": "OP",
    "subcarpathian": "PK", "podkarpackie": "PK",
    "podlaskie": "PD", "podlachian": "PD",
    "pomeranian": "PM", "pomorskie": "PM",
    "silesian": "SL", "slaskie": "SL",
    "swietokrzyskie": "SK",
    "warmian-masurian": "WN", "warminsko-mazurskie": "WN",
    "greater poland": "WP", "wielkopolskie": "WP",
    "west pomeranian": "ZP", "zachodniopomorskie": "ZP",
}

def normalize_name(s: str) -> str:
    s = str(s).strip().lower()
    for a, b in [("ł","l"),("ś","s"),("ą","a"),("ę","e"),
                 ("ń","n"),("ó","o"),("ż","z"),("ź","z"),("ć","c")]:
        s = s.replace(a, b)
    return s

gdf = gpd.read_file(PATH)
gdf["code"] = gdf["name"].map(lambda s: NAME_TO_CODE.get(normalize_name(s)))
missing = gdf.loc[gdf["code"].isna(), "name"].tolist()
if missing:
    raise ValueError(f"Unmapped names: {missing}")

OUT.parent.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(9, 9))
gdf.plot(ax=ax, color="#f2f2f2", edgecolor="black", linewidth=0.8)

for _, row in gdf.iterrows():
    pt = row.geometry.representative_point()
    ax.text(pt.x, pt.y, row["code"],
            ha="center", va="center",
            fontsize=16, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.12",
                      fc="white", ec="none", alpha=0.65))

ax.set_axis_off()
plt.tight_layout()
plt.savefig(OUT, dpi=300, bbox_inches="tight")
print(f"Saved {OUT}")
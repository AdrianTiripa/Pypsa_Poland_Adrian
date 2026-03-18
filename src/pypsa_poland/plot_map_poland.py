# src/pypsa_poland/plot_map_poland.py
# Run (PowerShell):
# & C:\Users\adria\anaconda3\envs\pypsa-legacy\python.exe C:\Users\adria\MODEL_PyPSA\Core\pypsa-poland_ADRIAN\src\pypsa_poland\plot_map_poland.py --run_dir C:\Users\adria\MODEL_PyPSA\Core\runs\run_2020_20260306_010448

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from matplotlib.lines import Line2D
from matplotlib import patheffects

# Optional basemap. Script still works without it.
try:
    import geopandas as gpd
    HAS_GPD = True
except Exception:
    HAS_GPD = False


def read_csv(run_dir: Path, name: str, index_col: int | None = None) -> pd.DataFrame:
    p = run_dir / name
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}")
    return pd.read_csv(p, index_col=index_col)


def get_bus_xy(buses: pd.DataFrame) -> pd.DataFrame:
    b = buses.copy()
    if "name" in b.columns:
        b = b.set_index("name")

    for cand in [("x", "y"), ("lon", "lat"), ("longitude", "latitude")]:
        if cand[0] in b.columns and cand[1] in b.columns:
            out = b[[cand[0], cand[1]]].rename(columns={cand[0]: "x", cand[1]: "y"})
            out["x"] = pd.to_numeric(out["x"], errors="coerce")
            out["y"] = pd.to_numeric(out["y"], errors="coerce")
            out = out.dropna()
            return out

    raise ValueError("buses.csv has no coordinates. Need columns x/y or lon/lat.")


def choose_capacity_column(df: pd.DataFrame) -> str:
    for c in ["p_nom_opt", "p_nom"]:
        if c in df.columns:
            return c
    raise ValueError("No p_nom_opt or p_nom column found.")


def poland_region_buses(bus_index: pd.Index) -> pd.Index:
    idx = pd.Index([str(b) for b in bus_index])
    m = idx.str.match(r"^PL\s+[A-Z]{2}$")
    return idx[m]


def safe_norm(v: pd.Series) -> pd.Series:
    v = v.fillna(0.0)
    s = float(v.sum())
    if s <= 0:
        return v * 0.0
    return v / s


def draw_pie(ax, x: float, y: float, fracs: list[float], colors: list, radius: float, zorder: int = 3) -> None:
    start = 0.0
    for f, c in zip(fracs, colors):
        if f <= 0:
            continue
        theta1 = start * 360.0
        theta2 = (start + f) * 360.0
        ax.add_patch(
            Wedge(
                (x, y),
                radius,
                theta1,
                theta2,
                facecolor=c,
                edgecolor="white",
                linewidth=0.4,
                zorder=zorder,
            )
        )
        start += f


def infer_link_type(row: pd.Series) -> str:
    """
    Robust link 'type' inference for legend/coloring.
    Priority:
      1) carrier column if present
      2) name heuristics
      3) bus suffix heuristics (_hydrogen/_heat/_transport)
      4) default 'Link'
    """
    # 1) carrier if present and non-empty
    if "carrier" in row.index:
        c = str(row.get("carrier", "")).strip()
        if c and c.lower() != "nan":
            return c

    name = str(row.name) if row.name is not None else ""
    nlow = name.lower()

    # 2) name heuristics
    if "dc" in nlow:
        return "DC"
    if "ac" in nlow:
        return "AC"
    if "hydrogen" in nlow or "h2" in nlow:
        return "hydrogen"
    if "heat" in nlow:
        return "heat"
    if "transport" in nlow:
        return "transport"

    # 3) bus suffix heuristics
    b0 = str(row.get("bus0", ""))
    b1 = str(row.get("bus1", ""))
    both = (b0 + " " + b1).lower()
    if "_hydrogen" in both:
        return "hydrogen"
    if "_heat" in both:
        return "heat"
    if "_transport" in both:
        return "transport"

    return "Link"


def plot_voivodeship_boundaries(ax, path: str) -> None:
    """
    Plot Poland voivodeship boundaries (admin-1) as a background layer.

    Works with GeoJSON or Shapefile or anything geopandas can read.
    Assumes data are in EPSG:4326 or reprojects to EPSG:4326.
    """
    if not HAS_GPD:
        raise RuntimeError("geopandas not available, cannot plot voivodeships")

    gdf = gpd.read_file(path)

    # If the file contains more than Poland, filter if possible.
    for col in ["NAME_0", "CNTR_NAME", "country", "ADMIN"]:
        if col in gdf.columns:
            gdf = gdf[gdf[col].astype(str).str.contains("Poland", case=False, na=False)]
            break

    # Ensure lon/lat CRS
    if gdf.crs is None:
        # If your file has unknown CRS, you MUST set it correctly here.
        # Most admin datasets ship with EPSG:4326 already.
        gdf = gdf.set_crs("EPSG:4326")
    else:
        gdf = gdf.to_crs("EPSG:4326")

    gdf.boundary.plot(ax=ax, color="#88aacc", linewidth=1.0, alpha=0.9, zorder=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, type=str)
    ap.add_argument("--out", default=None, type=str)
    ap.add_argument("--top_carriers", default=6, type=int)
    ap.add_argument("--plot_links", default="YES", type=str, help="YES/NO")

    # NEW: voivodeship boundaries layer
    ap.add_argument(
        "--voivodeships",
        default=None,
        type=str,
        help="Path to Poland voivodeship boundaries (geojson/shp). Plotted under pies/links.",
    )

    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir not found: {run_dir}")

    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else (fig_dir / "poland_map.png")

    buses = read_csv(run_dir, "buses.csv")
    gens = read_csv(run_dir, "generators.csv")
    links = read_csv(run_dir, "links.csv")

    xy = get_bus_xy(buses)
    region_bus = poland_region_buses(xy.index)
    if len(region_bus) == 0:
        raise ValueError("No regional buses matched '^PL <2 letters>$' (expected like 'PL DS').")
    xy = xy.loc[region_bus].copy()

    # ---- capacity pies ----
    cap_col = choose_capacity_column(gens)
    if "name" in gens.columns:
        gens = gens.set_index("name")
    gens = gens[gens["bus"].astype(str).isin(region_bus)].copy()
    gens[cap_col] = pd.to_numeric(gens[cap_col], errors="coerce").fillna(0.0)

    cap_bus_car = gens.groupby(["bus", "carrier"])[cap_col].sum().unstack(fill_value=0.0)
    total_by_car = cap_bus_car.sum(axis=0).sort_values(ascending=False)
    top_carriers = list(total_by_car.head(args.top_carriers).index)

    cmap = plt.get_cmap("tab20")
    carrier_list_for_colors = top_carriers + ["other"]
    carrier_color = {c: cmap(i % 20) for i, c in enumerate(carrier_list_for_colors)}

    pie_data = {}
    for b in region_bus:
        row = cap_bus_car.loc[b] if b in cap_bus_car.index else pd.Series(dtype=float)
        row = row.reindex(total_by_car.index, fill_value=0.0)

        top = row.reindex(top_carriers, fill_value=0.0)
        other = float(row.drop(index=top_carriers, errors="ignore").sum())
        vec = top.copy()
        vec["other"] = other

        fracs = safe_norm(vec).tolist()
        cols = [carrier_color[c] for c in vec.index]
        pie_data[b] = (fracs, cols)

    # ---- link filtering ----
    if "name" in links.columns:
        links = links.set_index("name")

    for col in ["bus0", "bus1"]:
        if col not in links.columns:
            raise ValueError(f"links.csv missing column '{col}'")

    links["bus0"] = links["bus0"].astype(str)
    links["bus1"] = links["bus1"].astype(str)

    mask_rr = links["bus0"].isin(region_bus) & links["bus1"].isin(region_bus)
    links_rr = links.loc[mask_rr].copy()

    link_cap_col = "p_nom_opt" if "p_nom_opt" in links_rr.columns else ("p_nom" if "p_nom" in links_rr.columns else None)
    if link_cap_col is None:
        raise ValueError("links.csv has no p_nom or p_nom_opt columns.")
    links_rr[link_cap_col] = pd.to_numeric(links_rr[link_cap_col], errors="coerce").fillna(0.0)

    # infer types
    links_rr["link_type"] = links_rr.apply(infer_link_type, axis=1)

    # type colors
    type_list = sorted(links_rr["link_type"].unique().tolist())
    type_cmap = plt.get_cmap("tab10")
    type_color = {t: type_cmap(i % 10) for i, t in enumerate(type_list)}

    # width scale
    vmax = float(links_rr[link_cap_col].max()) if len(links_rr) else 1.0
    vmax = max(vmax, 1.0)

    def width(p_nom: float) -> float:
        return 0.4 + 4.5 * (p_nom / vmax) ** 0.6

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(10, 10))

    # Optional: country outline from Natural Earth (not voivodeships)
    if HAS_GPD:
        try:
            world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
            pol = world[world["name"] == "Poland"]
            if len(pol) == 1:
                pol.plot(ax=ax, color="white", edgecolor="black", linewidth=1.0, alpha=0.35, zorder=0)
        except Exception:
            pass

    # NEW: voivodeship boundaries in same CRS/axes as buses
    if args.voivodeships:
        try:
            plot_voivodeship_boundaries(ax, args.voivodeships)
        except Exception as e:
            print(f"Warning: failed to plot voivodeships: {e}")

    # plot links colored by type
    if args.plot_links.strip().upper() != "NO":
        for _, row in links_rr.iterrows():
            b0 = row["bus0"]
            b1 = row["bus1"]
            if b0 not in xy.index or b1 not in xy.index:
                continue
            x0, y0 = float(xy.loc[b0, "x"]), float(xy.loc[b0, "y"])
            x1, y1 = float(xy.loc[b1, "x"]), float(xy.loc[b1, "y"])
            lt = row["link_type"]
            ax.plot(
                [x0, x1],
                [y0, y1],
                linewidth=width(float(row[link_cap_col])),
                alpha=0.45,
                color=type_color.get(lt, (0, 0, 0, 0.4)),
                zorder=2,
            )

    # pie radius from extent
    xmin, xmax = float(xy["x"].min()), float(xy["x"].max())
    ymin, ymax = float(xy["y"].min()), float(xy["y"].max())
    span = max(xmax - xmin, ymax - ymin)
    pie_radius = 0.02 * span

    # pies + labels
    for b in region_bus:
        x, y = float(xy.loc[b, "x"]), float(xy.loc[b, "y"])
        fracs, cols = pie_data[b]
        draw_pie(ax, x, y, fracs, cols, radius=pie_radius, zorder=3)

        # white stroke behind text for readability
        ax.text(
            x,
            y - 1.35 * pie_radius,
            b.replace("PL ", ""),
            ha="center",
            va="top",
            fontsize=9,
            zorder=4,
            path_effects=[patheffects.withStroke(linewidth=2.2, foreground="white", alpha=0.85)],
        )

    # legends
    carrier_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=carrier_color[c], markersize=10, label=c)
        for c in carrier_list_for_colors
    ]
    link_handles = [Line2D([0, 1], [0, 0], color=type_color[t], linewidth=3, label=t) for t in type_list]

    leg1 = ax.legend(handles=carrier_handles, title="Generator carrier (capacity share)", loc="lower left", frameon=True)
    ax.add_artist(leg1)
    ax.legend(handles=link_handles, title=f"Link type (color)\nwidth ~ {link_cap_col}", loc="lower right", frameon=True)

    ax.set_title("Poland regional capacity mix (pies) + inter-regional links")
    ax.set_xlabel("Longitude / x")
    ax.set_ylabel("Latitude / y")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.15)

    plt.tight_layout()
    plt.savefig(out_path, dpi=250)
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from PIL import Image

# & C:/Users/adria/anaconda3/envs/pypsa-env/python.exe c:/Users/adria/MODEL_PyPSA/Core/pypsa-poland_ADRIAN/src/pypsa_poland/plot_heat_pump_gif.py --runs_root "C:\Users\adria\MODEL_PyPSA\Core\runs" --inputs_summary "C:\Users\adria\MODEL_PyPSA\Core\runs\weather_year_comparison\weather_year_inputs_summary.csv" --map_path "C:\Users\adria\MODEL_PyPSA\Core\pypsa-poland_ADRIAN\pl.json" --out_dir "C:\Users\adria\MODEL_PyPSA\Core\pypsa-poland_ADRIAN\figures_maps\heat_pump_gif"

# ============================================================
# Province-name mapping for pl.json
# ============================================================

NAME_TO_CODE = {
    "lower silesian": "DS",
    "dolnoslaskie": "DS",
    "dolnośląskie": "DS",

    "kuyavian-pomeranian": "KP",
    "kujawsko-pomorskie": "KP",

    "lodz": "LD",
    "lodzkie": "LD",
    "łódzkie": "LD",

    "lublin": "LU",
    "lubelskie": "LU",

    "lubusz": "LB",
    "lubuskie": "LB",

    "lesser poland": "MA",
    "malopolskie": "MA",
    "małopolskie": "MA",

    "masovian": "MZ",
    "mazowieckie": "MZ",

    "opole": "OP",
    "opolskie": "OP",

    "subcarpathian": "PK",
    "podkarpackie": "PK",

    "podlaskie": "PD",
    "podlachian": "PD",

    "pomeranian": "PM",
    "pomorskie": "PM",

    "silesian": "SL",
    "slaskie": "SL",
    "śląskie": "SL",

    "swietokrzyskie": "SK",
    "świętokrzyskie": "SK",

    "warmian-masurian": "WN",
    "warminsko-mazurskie": "WN",
    "warmińsko-mazurskie": "WN",

    "greater poland": "WP",
    "wielkopolskie": "WP",

    "west pomeranian": "ZP",
    "zachodniopomorskie": "ZP",
}


# ============================================================
# Basic helpers
# ============================================================

def normalize_name(s: str) -> str:
    s = str(s).strip().lower()
    s = s.replace("ł", "l").replace("ś", "s").replace("ą", "a").replace("ę", "e")
    s = s.replace("ń", "n").replace("ó", "o").replace("ż", "z").replace("ź", "z").replace("ć", "c")
    return s


def extract_province_code(text: str | None) -> str | None:
    if text is None:
        return None
    m = re.search(r"\bPL\s+([A-Z]{2})\b", str(text))
    if m:
        return m.group(1)
    return None


def choose_capacity_column(df: pd.DataFrame, candidates=("p_nom_opt", "p_nom")) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"No capacity column found in {list(candidates)}")


def read_csv_if_exists(run_dir: Path, filename: str, index_col: int | None = None) -> pd.DataFrame | None:
    path = run_dir / filename
    if not path.exists():
        return None
    return pd.read_csv(path, index_col=index_col)


def add_code_column_from_map_name(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    if "name" not in gdf.columns:
        raise ValueError("Map file must contain a 'name' property column.")
    gdf["code"] = gdf["name"].map(lambda s: NAME_TO_CODE.get(normalize_name(s)))
    if gdf["code"].isna().any():
        missing = sorted(set(gdf.loc[gdf["code"].isna(), "name"].astype(str).tolist()))
        raise ValueError(
            "Unmapped province names in map file: "
            + ", ".join(missing)
            + "\nAdd them to NAME_TO_CODE."
        )
    return gdf


def load_map(map_path: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(map_path)
    gdf = add_code_column_from_map_name(gdf)
    gdf["rep_point"] = gdf.representative_point()
    return gdf


def merge_map_values(map_gdf: gpd.GeoDataFrame, df: pd.DataFrame) -> gpd.GeoDataFrame:
    out = map_gdf.merge(df, on="code", how="left")
    out["value"] = pd.to_numeric(out["value"], errors="coerce").fillna(0.0)
    return out


# ============================================================
# Heat pump selectors / data extraction
# ============================================================

def heat_pump_link_mask(links: pd.DataFrame) -> pd.Series:
    idx = links.index.astype(str).str.lower()
    carrier = links["carrier"].astype(str).str.lower() if "carrier" in links.columns else pd.Series("", index=links.index)
    bus0 = links["bus0"].astype(str).str.lower() if "bus0" in links.columns else pd.Series("", index=links.index)
    bus1 = links["bus1"].astype(str).str.lower() if "bus1" in links.columns else pd.Series("", index=links.index)

    pattern = r"heat[\s\-_]?pump|hp\b"

    return (
        carrier.str.contains(pattern, regex=True, na=False)
        | idx.str.contains(pattern, regex=True, na=False)
        | bus0.str.contains("_heat", na=False)
        | bus1.str.contains("_heat", na=False)
    )


def load_core_tables(run_dir: Path):
    generators = read_csv_if_exists(run_dir, "generators.csv")
    links = read_csv_if_exists(run_dir, "links.csv")
    storage_units = read_csv_if_exists(run_dir, "storage_units.csv")
    buses = read_csv_if_exists(run_dir, "buses.csv")

    if generators is not None and "name" in generators.columns:
        generators = generators.set_index("name")
    if links is not None and "name" in links.columns:
        links = links.set_index("name")
    if storage_units is not None and "name" in storage_units.columns:
        storage_units = storage_units.set_index("name")
    if buses is not None and "name" in buses.columns:
        buses = buses.set_index("name")

    return generators, links, storage_units, buses


def get_component_province(row: pd.Series, bus_cols=("bus", "bus0", "bus1")) -> str | None:
    for c in bus_cols:
        if c in row.index:
            code = extract_province_code(row[c])
            if code:
                return code
    return None


def compute_heat_pump_capacity(run_dir: Path) -> pd.DataFrame:
    _, links, _, _ = load_core_tables(run_dir)
    if links is None or links.empty:
        return pd.DataFrame(columns=["code", "value"])

    mask = heat_pump_link_mask(links)
    hp = links.loc[mask].copy()

    if hp.empty:
        print(f"[WARN] No heat-pump links found in {run_dir}")
        if "carrier" in links.columns:
            print("Available carriers:")
            print(sorted(links["carrier"].astype(str).dropna().unique().tolist())[:50])
        return pd.DataFrame(columns=["code", "value"])

    cap_col = choose_capacity_column(hp)
    hp["value"] = pd.to_numeric(hp[cap_col], errors="coerce").fillna(0.0)
    hp["code"] = hp.apply(get_component_province, axis=1, bus_cols=("bus0", "bus1"))

    out = hp.groupby("code", dropna=True)["value"].sum().reset_index()
    return out


# ============================================================
# Ordering years
# ============================================================

def detect_run_dirs(runs_root: Path) -> list[Path]:
    run_dirs = []
    for p in runs_root.rglob("*"):
        if p.is_dir() and (p / "links.csv").exists():
            run_dirs.append(p)
    return sorted(run_dirs)


def match_run_dir_for_year(run_dirs: list[Path], year: int) -> Path | None:
    year_pat = re.compile(rf"(?<!\d){year}(?!\d)")
    candidates = []

    for p in run_dirs:
        rel = str(p)
        if year_pat.search(p.name) or year_pat.search(rel):
            candidates.append(p)

    if not candidates:
        return None

    candidates = sorted(candidates, key=lambda x: (len(str(x)), str(x)))
    return candidates[0]


def load_year_order_from_inputs_summary(inputs_summary: Path) -> list[int]:
    df = pd.read_csv(inputs_summary)

    if "year" not in df.columns:
        raise ValueError("inputs_summary must contain a 'year' column.")
    if "elec_for_heat_annual_mwh" not in df.columns:
        raise ValueError("inputs_summary must contain 'elec_for_heat_annual_mwh'.")

    tmp = df[["year", "elec_for_heat_annual_mwh"]].copy()
    tmp["year"] = pd.to_numeric(tmp["year"], errors="coerce")
    tmp["elec_for_heat_annual_mwh"] = pd.to_numeric(tmp["elec_for_heat_annual_mwh"], errors="coerce")
    tmp = tmp.dropna(subset=["year", "elec_for_heat_annual_mwh"]).copy()
    tmp["year"] = tmp["year"].astype(int)

    # low electricity-for-heat = easier heating year
    tmp = tmp.sort_values("elec_for_heat_annual_mwh", ascending=True)

    return tmp["year"].tolist()


def build_ordered_runs(runs_root: Path, inputs_summary: Path | None, years: list[int] | None) -> list[tuple[int, Path]]:
    run_dirs = detect_run_dirs(runs_root)
    if not run_dirs:
        raise FileNotFoundError(
            f"No run directories found under {runs_root} "
            f"(expected subfolders containing links.csv)."
        )

    if years is None:
        if inputs_summary is None:
            raise ValueError("Provide either --inputs_summary or --years.")
        years = load_year_order_from_inputs_summary(inputs_summary)

    ordered_runs = []
    missing = []

    for y in years:
        run_dir = match_run_dir_for_year(run_dirs, int(y))
        if run_dir is None:
            missing.append(y)
        else:
            ordered_runs.append((int(y), run_dir))

    if missing:
        print("[WARN] These years were not matched to a run directory and will be skipped:")
        print(missing)

    if not ordered_runs:
        raise ValueError("No years could be matched to run directories.")

    return ordered_runs


# ============================================================
# Plotting / GIF helpers
# ============================================================

def annotate_map_values(ax, gdf: gpd.GeoDataFrame, fontsize=7):
    for _, row in gdf.iterrows():
        pt = row["rep_point"]
        label = f"{row['code']}\n{row['value_gw']:.1f}"
        ax.text(
            pt.x,
            pt.y,
            label,
            fontsize=fontsize,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.65),
        )


def draw_heat_pump_frame(
    map_gdf: gpd.GeoDataFrame,
    hp_df: pd.DataFrame,
    year: int,
    frame_idx: int,
    n_frames: int,
    vmax_gw: float,
    out_path: Path,
):
    plot_gdf = merge_map_values(map_gdf, hp_df)
    plot_gdf["value_gw"] = pd.to_numeric(plot_gdf["value"], errors="coerce").fillna(0.0) / 1000.0

    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    norm = Normalize(vmin=0.0, vmax=max(vmax_gw, 1e-9))

    plot_gdf.plot(
        column="value_gw",
        cmap="YlOrRd",
        linewidth=0.7,
        edgecolor="black",
        ax=ax,
        legend=False,
        norm=norm,
        missing_kwds={"color": "lightgrey"},
    )

    annotate_map_values(ax, plot_gdf, fontsize=7)

    ax.set_title(
        "Heat-pump capacity by province\n"
        f"Weather year {year}   •   frame {frame_idx}/{n_frames}   •   low to high electricity-for-heat",
        fontsize=13
    )
    ax.set_axis_off()

    sm = ScalarMappable(norm=norm, cmap="YlOrRd")
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.040, pad=0.02)
    cbar.set_label("Installed heat-pump capacity [GW]")

    plt.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def make_gif(frame_paths: list[Path], gif_path: Path, duration_ms: int = 800):
    images = [Image.open(fp).convert("P", palette=Image.ADAPTIVE) for fp in frame_paths]
    if not images:
        raise ValueError("No frames were created.")

    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
    )


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser(
        description="Create a GIF of heat-pump capacity by province ordered by annual electricity needed for heat."
    )
    ap.add_argument("--runs_root", required=True, type=str, help="Root folder containing weather-year run directories.")
    ap.add_argument("--map_path", default="pl.json", type=str, help="Path to Poland GeoJSON.")
    ap.add_argument("--out_dir", default="figures_maps/heat_pump_gif", type=str, help="Output folder.")

    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--inputs_summary",
        type=str,
        help="weather_year_inputs_summary.csv used to order years by elec_for_heat_annual_mwh (low to high).",
    )
    group.add_argument(
        "--years",
        nargs="+",
        type=int,
        help="Explicit year order to use in the GIF.",
    )

    ap.add_argument("--gif_name", default="heat_pump_by_province_low_to_high_elec_for_heat.gif", type=str)
    ap.add_argument("--frame_duration_ms", default=800, type=int, help="Duration per frame in milliseconds.")
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    map_path = Path(args.map_path)
    out_dir = Path(args.out_dir)
    frames_dir = out_dir / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    map_gdf = load_map(map_path)

    ordered_runs = build_ordered_runs(
        runs_root=runs_root,
        inputs_summary=Path(args.inputs_summary) if args.inputs_summary else None,
        years=args.years,
    )

    ordered_years = [y for y, _ in ordered_runs]
    print("[INFO] Ordered years used in the GIF:")
    print(ordered_years)

    # First pass: compute all heat-pump capacities so that the color scale is fixed across the whole GIF
    hp_data_by_year: dict[int, pd.DataFrame] = {}
    vmax_gw = 0.0

    for year, run_dir in ordered_runs:
        hp_df = compute_heat_pump_capacity(run_dir)
        hp_data_by_year[year] = hp_df

        if not hp_df.empty:
            local_max_gw = pd.to_numeric(hp_df["value"], errors="coerce").fillna(0.0).max() / 1000.0
            vmax_gw = max(vmax_gw, float(local_max_gw))

    vmax_gw = max(vmax_gw, 1e-9)

    # Second pass: draw frames
    frame_paths = []
    n_frames = len(ordered_runs)

    for i, (year, run_dir) in enumerate(ordered_runs, start=1):
        hp_df = hp_data_by_year[year]
        frame_path = frames_dir / f"{i:03d}_{year}.png"

        draw_heat_pump_frame(
            map_gdf=map_gdf,
            hp_df=hp_df,
            year=year,
            frame_idx=i,
            n_frames=n_frames,
            vmax_gw=vmax_gw,
            out_path=frame_path,
        )

        frame_paths.append(frame_path)
        print(f"[INFO] Saved frame for {year}: {frame_path}")

    gif_path = out_dir / args.gif_name
    make_gif(frame_paths, gif_path, duration_ms=args.frame_duration_ms)

    print(f"\n[INFO] GIF saved to: {gif_path.resolve()}")


if __name__ == "__main__":
    main()
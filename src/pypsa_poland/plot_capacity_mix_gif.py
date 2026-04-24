from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import imageio.v2 as imageio


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

CAPACITY_MIX_COLORS = {
    "PV": "#f1c40f",
    "Onshore wind": "#2e86de",
    "Offshore wind": "#1b4f72",
    "Natural gas": "#7f8c8d",
    "Nuclear": "#8e44ad",
}

DEFAULT_MIX_ORDER = ["PV", "Onshore wind", "Offshore wind", "Natural gas", "Nuclear"]


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


def read_ts(run_dir: Path, stem: str) -> pd.DataFrame:
    path = run_dir / f"{stem}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, index_col=0)
    parsed = pd.to_datetime(df.index, errors="coerce")
    if not parsed.isna().all():
        good = ~parsed.isna()
        df = df.loc[good].copy()
        df.index = parsed[good]
    return df


def try_read_ts(run_dir: Path, stems: list[str]) -> tuple[str | None, pd.DataFrame | None]:
    for stem in stems:
        try:
            return stem, read_ts(run_dir, stem)
        except FileNotFoundError:
            continue
    return None, None


def infer_step_hours(fallback_df: pd.DataFrame | None = None) -> float:
    if fallback_df is not None and isinstance(fallback_df.index, pd.DatetimeIndex) and len(fallback_df.index) > 1:
        delta = (fallback_df.index[1] - fallback_df.index[0]).total_seconds() / 3600.0
        if delta > 0:
            return float(delta)
    return 1.0


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


# ============================================================
# Carrier classification
# ============================================================

def classify_generator_carrier(text: str) -> str | None:
    s = str(text).lower()

    if "offshore" in s or "off wind" in s or "offwind" in s:
        return "Offshore wind"
    if "onshore" in s or ("wind" in s and "offshore" not in s and "offwind" not in s):
        return "Onshore wind"
    if "solar" in s or "pv" in s:
        return "PV"
    if "nuclear" in s:
        return "Nuclear"
    if "natural gas" in s or s.strip() == "gas" or "ccgt" in s or "ocgt" in s:
        return "Natural gas"

    return None


def is_vres_carrier(label: str | None) -> bool:
    return label in {"PV", "Onshore wind", "Offshore wind"}


# ============================================================
# Data loading
# ============================================================

def load_core_tables(run_dir: Path):
    generators = read_csv_if_exists(run_dir, "generators.csv")
    buses = read_csv_if_exists(run_dir, "buses.csv")

    if generators is not None and "name" in generators.columns:
        generators = generators.set_index("name")
    if buses is not None and "name" in buses.columns:
        buses = buses.set_index("name")

    return generators, buses


def get_component_province(row: pd.Series, bus_cols=("bus", "bus0", "bus1")) -> str | None:
    for c in bus_cols:
        if c in row.index:
            code = extract_province_code(row[c])
            if code:
                return code
    return None


def compute_capacity_mix(run_dir: Path) -> pd.DataFrame:
    generators, _ = load_core_tables(run_dir)
    if generators is None or generators.empty:
        return pd.DataFrame(columns=["code"] + DEFAULT_MIX_ORDER)

    cap_col = choose_capacity_column(generators)
    g = generators.copy()
    g["value"] = pd.to_numeric(g[cap_col], errors="coerce").fillna(0.0)

    carrier_source = g["carrier"].astype(str) if "carrier" in g.columns else pd.Series(g.index.astype(str), index=g.index)
    g["mix_carrier"] = carrier_source.map(classify_generator_carrier)

    g = g[g["mix_carrier"].notna()].copy()
    g["code"] = g.apply(get_component_province, axis=1, bus_cols=("bus",))
    g = g[g["code"].notna()].copy()

    out = g.pivot_table(
        index="code",
        columns="mix_carrier",
        values="value",
        aggfunc="sum",
        fill_value=0.0
    ).reset_index()

    for c in DEFAULT_MIX_ORDER:
        if c not in out.columns:
            out[c] = 0.0

    out = out[["code"] + DEFAULT_MIX_ORDER]
    return out


def compute_combined_vres_cf(run_dir: Path) -> float:
    """
    Combined VRES CF = total annual VRES generation / (installed VRES capacity * total hours)

    Tries generators-p first.
    If unavailable, falls back to generators-p_max_pu as an approximation.
    """
    generators, _ = load_core_tables(run_dir)
    if generators is None or generators.empty:
        return np.nan

    cap_col = choose_capacity_column(generators)
    g = generators.copy()
    g["cap"] = pd.to_numeric(g[cap_col], errors="coerce").fillna(0.0)

    carrier_source = g["carrier"].astype(str) if "carrier" in g.columns else pd.Series(g.index.astype(str), index=g.index)
    g["mix_carrier"] = carrier_source.map(classify_generator_carrier)

    vres = g[g["mix_carrier"].map(is_vres_carrier)].copy()
    if vres.empty:
        return np.nan

    total_cap = float(vres["cap"].sum())
    if total_cap <= 0:
        return np.nan

    # First preference: realised generation
    _, gen_ts = try_read_ts(run_dir, ["generators-p", "generators_t-p"])
    if gen_ts is not None and not gen_ts.empty:
        common = [c for c in gen_ts.columns if c in vres.index]
        if common:
            step_hours = infer_step_hours(gen_ts)
            total_gen = float(gen_ts[common].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum().sum() * step_hours)
            total_hours = float(len(gen_ts.index) * step_hours)
            if total_hours > 0:
                return total_gen / (total_cap * total_hours)

    # Fallback: availability profile
    _, pmax_pu = try_read_ts(run_dir, ["generators-p_max_pu", "generators_t-p_max_pu"])
    if pmax_pu is not None and not pmax_pu.empty:
        common = [c for c in pmax_pu.columns if c in vres.index]
        if common:
            av = pmax_pu[common].apply(pd.to_numeric, errors="coerce").fillna(0.0).mean(axis=0)
            caps = vres.loc[common, "cap"]
            if float(caps.sum()) > 0:
                return float((av * caps).sum() / caps.sum())

    return np.nan


# ============================================================
# Run discovery and ordering
# ============================================================

def infer_year_from_path(path: Path) -> int | None:
    matches = re.findall(r"(19\d{2}|20\d{2})", str(path))
    if not matches:
        return None
    return int(matches[-1])


def discover_run_dirs(runs_root: Path) -> dict[int, Path]:
    """
    Recursively find run directories containing generators.csv.
    If multiple candidates for the same year exist, keep the shortest path.
    """
    candidates = [p.parent for p in runs_root.rglob("generators.csv")]
    if not candidates:
        raise FileNotFoundError(
            f"No run directories found under {runs_root} "
            f"(expected folders somewhere below it containing generators.csv)."
        )

    by_year: dict[int, list[Path]] = {}
    for run_dir in candidates:
        year = infer_year_from_path(run_dir)
        if year is None:
            continue
        by_year.setdefault(year, []).append(run_dir)

    if not by_year:
        raise FileNotFoundError("Could not infer any weather years from discovered run directories.")

    chosen: dict[int, Path] = {}
    for year, dirs in by_year.items():
        dirs_sorted = sorted(dirs, key=lambda p: (len(str(p)), str(p)))
        chosen[year] = dirs_sorted[0]

    return chosen


def build_ordered_runs(runs_root: Path, years: list[int] | None = None) -> list[dict]:
    discovered = discover_run_dirs(runs_root)

    if years:
        missing = [y for y in years if y not in discovered]
        if missing:
            raise FileNotFoundError(
                "Could not find run directories for these years under "
                f"{runs_root}: {missing}"
            )
        selected = {y: discovered[y] for y in years}
    else:
        selected = discovered

    rows = []
    for year, run_dir in selected.items():
        vres_cf = compute_combined_vres_cf(run_dir)
        rows.append(
            {
                "year": year,
                "run_dir": run_dir,
                "vres_cf": vres_cf,
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values(["vres_cf", "year"], ascending=[True, True]).reset_index(drop=True)

    ordered = df.to_dict(orient="records")

    print("\n[INFO] Years ordered by increasing combined VRES CF:")
    for row in ordered:
        print(f"  {row['year']}: VRES CF = {row['vres_cf']:.4f} | {row['run_dir']}")

    return ordered


# ============================================================
# Plotting
# ============================================================

def merge_with_map(map_gdf: gpd.GeoDataFrame, mix_df: pd.DataFrame) -> gpd.GeoDataFrame:
    out = map_gdf[["code", "geometry", "rep_point"]].merge(mix_df, on="code", how="left")
    for c in DEFAULT_MIX_ORDER:
        if c not in out.columns:
            out[c] = 0.0
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    out["total"] = out[DEFAULT_MIX_ORDER].sum(axis=1)
    return out


def data_to_axes_fraction(ax, x, y):
    disp = ax.transData.transform((x, y))
    frac = ax.transAxes.inverted().transform(disp)
    return frac[0], frac[1]


def draw_pie_at_point(
    ax,
    x,
    y,
    values,
    colors,
    size_frac=0.08,
    startangle=90,
):
    x_frac, y_frac = data_to_axes_fraction(ax, x, y)

    x0 = x_frac - size_frac / 2
    y0 = y_frac - size_frac / 2

    pie_ax = ax.inset_axes([x0, y0, size_frac, size_frac], transform=ax.transAxes)
    pie_ax.pie(
        values,
        colors=colors,
        startangle=startangle,
        counterclock=False,
        wedgeprops=dict(edgecolor="white", linewidth=0.5),
    )
    pie_ax.set_aspect("equal")
    pie_ax.axis("off")


def plot_capacity_mix_frame(
    map_gdf: gpd.GeoDataFrame,
    mix_df: pd.DataFrame,
    year: int,
    vres_cf: float,
    out_path: Path,
    max_total: float,
    min_pie_size: float = 0.050,
    max_pie_size: float = 0.090,
):
    gdf = merge_with_map(map_gdf, mix_df)

    fig, ax = plt.subplots(figsize=(14, 12), constrained_layout=True)
    gdf.plot(ax=ax, color="#f2f2f2", edgecolor="black", linewidth=0.9)

    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    yspan = y1 - y0

    for _, row in gdf.iterrows():
        x = row["rep_point"].x
        y = row["rep_point"].y

        vals = np.array([float(row[c]) for c in DEFAULT_MIX_ORDER], dtype=float)
        total = float(vals.sum())

        if total > 0:
            frac = np.sqrt(total / max(max_total, 1e-9))
            pie_size = min_pie_size + (max_pie_size - min_pie_size) * frac
            draw_pie_at_point(
                ax=ax,
                x=x,
                y=y + 0.010 * yspan,
                values=vals,
                colors=[CAPACITY_MIX_COLORS[c] for c in DEFAULT_MIX_ORDER],
                size_frac=pie_size,
            )

        ax.text(
            x,
            y - 0.040 * yspan,
            f"{row['code']}\n{total / 1000:.1f} GW",
            ha="center",
            va="top",
            fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.78),
        )

    ax.set_title(
        f"Generation capacity mix by province\n"
        f"Weather year {year}   |   Combined VRES CF = {vres_cf:.3f}",
        fontsize=18,
        pad=18,
    )
    ax.set_axis_off()

    legend_handles = [
        Patch(facecolor=CAPACITY_MIX_COLORS[c], edgecolor="black", label=c)
        for c in DEFAULT_MIX_ORDER
    ]
    ax.legend(
        handles=legend_handles,
        title="Carrier",
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=5,
        frameon=True,
        fontsize=10,
        title_fontsize=11,
    )

    ax.text(
        0.01,
        0.01,
        "Years are ordered by increasing combined VRES capacity factor.",
        transform=ax.transAxes,
        fontsize=10,
        color="dimgray",
        ha="left",
        va="bottom",
    )

    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_gif(frame_paths: list[Path], gif_path: Path, duration: float):
    images = [imageio.imread(p) for p in frame_paths]
    imageio.mimsave(gif_path, images, duration=duration, loop=0)


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser(
        description="GIF of provincial generation capacity mixes ordered by increasing VRES CF."
    )
    ap.add_argument("--runs_root", required=True, type=str, help="Root folder containing run directories.")
    ap.add_argument("--map_path", required=True, type=str, help="Path to Poland GeoJSON / pl.json.")
    ap.add_argument("--out_dir", default="figures_maps/capacity_mix_gif", type=str, help="Output folder.")
    ap.add_argument("--years", nargs="*", type=int, default=None, help="Optional subset of years to include.")
    ap.add_argument("--duration", type=float, default=1.0, help="GIF frame duration in seconds.")
    ap.add_argument("--gif_name", type=str, default="capacity_mix_by_vres_cf.gif", help="Output GIF filename.")
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    map_path = Path(args.map_path)
    out_dir = Path(args.out_dir)
    frames_dir = out_dir / "frames"

    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    map_gdf = load_map(map_path)
    ordered_runs = build_ordered_runs(runs_root=runs_root, years=args.years)

    # Precompute all mixes and global max total for consistent pie scaling
    prepared = []
    global_max_total = 0.0

    for item in ordered_runs:
        year = item["year"]
        run_dir = item["run_dir"]
        vres_cf = item["vres_cf"]

        mix_df = compute_capacity_mix(run_dir)
        max_total_here = float(mix_df[DEFAULT_MIX_ORDER].sum(axis=1).max()) if not mix_df.empty else 0.0
        global_max_total = max(global_max_total, max_total_here)

        prepared.append(
            {
                "year": year,
                "run_dir": run_dir,
                "vres_cf": vres_cf,
                "mix_df": mix_df,
            }
        )

    frame_paths = []

    for i, item in enumerate(prepared, start=1):
        year = item["year"]
        vres_cf = item["vres_cf"]
        mix_df = item["mix_df"]

        frame_path = frames_dir / f"{i:03d}_{year}.png"
        plot_capacity_mix_frame(
            map_gdf=map_gdf,
            mix_df=mix_df,
            year=year,
            vres_cf=vres_cf,
            out_path=frame_path,
            max_total=global_max_total,
            min_pie_size=0.050,
            max_pie_size=0.090,
        )
        frame_paths.append(frame_path)
        print(f"[INFO] Saved frame: {frame_path}")

    gif_path = out_dir / args.gif_name
    make_gif(frame_paths=frame_paths, gif_path=gif_path, duration=args.duration)

    print(f"\nSaved GIF to: {gif_path.resolve()}")
    print(f"Saved frames to: {frames_dir.resolve()}")


if __name__ == "__main__":
    main()
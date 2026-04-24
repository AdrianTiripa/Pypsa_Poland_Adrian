#src/pypsa_poland/plot_map_poland.py
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from shapely.geometry import LineString


# & C:/Users/adria/anaconda3/envs/pypsa-legacy/python.exe c:/Users/adria/MODEL_PyPSA/Core/pypsa-poland_ADRIAN/src/pypsa_poland/plot_map_poland.py `--easy_run C:/Users/adria/MODEL_PyPSA/Core/runs/run_2020_20260417_132752_3hr_Optimal_974s --hard_run C:/Users/adria/MODEL_PyPSA/Core/runs/run_1987_20260417_050458_3hr_Optimal_1090s --easy_label "Least stressful (2020)" --hard_label "Most stressful (1987)" --map_path C:/Users/adria/MODEL_PyPSA/Core/pypsa-poland_ADRIAN/src/pypsa_poland/pl.json --out_dir C:/Users/adria/MODEL_PyPSA/Core/figures_maps

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


def infer_step_hours(run_dir: Path, fallback_df: pd.DataFrame | None = None) -> float:
    meta_path = run_dir / "run_metadata.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            for key in ["stepsize", "step", "step_hours"]:
                if key in meta:
                    return float(meta[key])
        except Exception:
            pass

    if fallback_df is not None and isinstance(fallback_df.index, pd.DatetimeIndex) and len(fallback_df.index) > 1:
        delta = (fallback_df.index[1] - fallback_df.index[0]).total_seconds() / 3600.0
        if delta > 0:
            return float(delta)

    return 1.0


def sanitize_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def region_bus_mask(series: pd.Series) -> pd.Series:
    return series.astype(str).str.match(r"^PL\s+[A-Z]{2}$")


def get_bus_points(buses: pd.DataFrame, map_points: dict[str, tuple[float, float]]) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}

    if buses is not None and not buses.empty and {"x", "y"}.issubset(buses.columns):
        for bus_name, row in buses.iterrows():
            x = row.get("x")
            y = row.get("y")
            if pd.notna(x) and pd.notna(y):
                out[str(bus_name)] = (float(x), float(y))

    for code, (x, y) in map_points.items():
        out.setdefault(f"PL {code}", (x, y))

    return out


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
# Carrier / asset selectors
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


def hydrogen_link_mask(links: pd.DataFrame) -> pd.Series:
    idx = links.index.astype(str).str.lower()
    carrier = links["carrier"].astype(str).str.lower() if "carrier" in links.columns else pd.Series("", index=links.index)
    b0 = links["bus0"].astype(str).str.lower() if "bus0" in links.columns else pd.Series("", index=links.index)
    b1 = links["bus1"].astype(str).str.lower() if "bus1" in links.columns else pd.Series("", index=links.index)

    return (
        carrier.str.contains("hydrogen|h2", na=False)
        | idx.str.contains("hydrogen|h2", na=False)
        | b0.str.contains("_hydrogen|_h2", na=False)
        | b1.str.contains("_hydrogen|_h2", na=False)
    )


def hydrogen_storage_mask(storage_units: pd.DataFrame) -> pd.Series:
    idx = storage_units.index.astype(str).str.lower()
    carrier = storage_units["carrier"].astype(str).str.lower() if "carrier" in storage_units.columns else pd.Series("", index=storage_units.index)
    bus = storage_units["bus"].astype(str).str.lower() if "bus" in storage_units.columns else pd.Series("", index=storage_units.index)

    return (
        carrier.str.contains("hydrogen|h2", na=False)
        | idx.str.contains("hydrogen|h2", na=False)
        | bus.str.contains("_hydrogen|_h2", na=False)
    )


def electric_interregional_link_mask(links: pd.DataFrame) -> pd.Series:
    if "bus0" not in links.columns or "bus1" not in links.columns:
        return pd.Series(False, index=links.index)

    b0 = links["bus0"].astype(str)
    b1 = links["bus1"].astype(str)
    carrier = links["carrier"].astype(str).str.lower() if "carrier" in links.columns else pd.Series("", index=links.index)
    idx = links.index.astype(str).str.lower()

    rr = region_bus_mask(b0) & region_bus_mask(b1)

    excluded = (
        carrier.str.contains("hydrogen|h2|heat|transport", na=False)
        | idx.str.contains("hydrogen|h2|heat|transport", na=False)
        | b0.str.lower().str.contains("_hydrogen|_heat|_transport", na=False)
        | b1.str.lower().str.contains("_hydrogen|_heat|_transport", na=False)
    )

    return rr & (~excluded)


# ============================================================
# Data extraction
# ============================================================

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
    print(f"[INFO] Heat-pump capacity by province for {run_dir.name}:")
    print(out.sort_values("code"))
    return out


def compute_capacity_mix(run_dir: Path) -> pd.DataFrame:
    generators, _, _, _ = load_core_tables(run_dir)
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

    out = g.pivot_table(index="code", columns="mix_carrier", values="value", aggfunc="sum", fill_value=0.0).reset_index()
    for c in DEFAULT_MIX_ORDER:
        if c not in out.columns:
            out[c] = 0.0
    out = out[["code"] + DEFAULT_MIX_ORDER]
    return out


def compute_net_imports_and_utilization(run_dir: Path, map_points: dict[str, tuple[float, float]], map_gdf: gpd.GeoDataFrame):
    _, links, _, buses = load_core_tables(run_dir)
    _, links_p0 = try_read_ts(run_dir, ["links-p0"])

    if links is None or links.empty or links_p0 is None or links_p0.empty:
        return pd.DataFrame(columns=["code", "value"]), gpd.GeoDataFrame(columns=["value", "geometry"], geometry="geometry")

    mask = electric_interregional_link_mask(links)
    link_meta = links.loc[mask].copy()
    if link_meta.empty:
        return pd.DataFrame(columns=["code", "value"]), gpd.GeoDataFrame(columns=["value", "geometry"], geometry="geometry")

    common = [c for c in links_p0.columns if c in link_meta.index]
    if not common:
        return pd.DataFrame(columns=["code", "value"]), gpd.GeoDataFrame(columns=["value", "geometry"], geometry="geometry")

    link_meta = link_meta.loc[common].copy()
    flows = sanitize_numeric(links_p0[common]).fillna(0.0)

    step_hours = infer_step_hours(run_dir, flows)

    province_balance: dict[str, float] = {}
    for link_name in common:
        row = link_meta.loc[link_name]
        c0 = extract_province_code(row["bus0"])
        c1 = extract_province_code(row["bus1"])
        if not c0 or not c1:
            continue

        series = flows[link_name]
        energy = float(series.sum() * step_hours)

        province_balance[c0] = province_balance.get(c0, 0.0) - energy
        province_balance[c1] = province_balance.get(c1, 0.0) + energy

    province_df = pd.DataFrame({"code": list(province_balance.keys()), "value": list(province_balance.values())})

    cap_col = choose_capacity_column(link_meta)
    link_meta["cap"] = pd.to_numeric(link_meta[cap_col], errors="coerce").replace(0, np.nan)

    buses = buses if buses is not None else pd.DataFrame()
    bus_points = get_bus_points(buses, map_points)
    map_rep_points = {row["code"]: (row["rep_point"].x, row["rep_point"].y) for _, row in map_gdf.iterrows()}

    line_rows = []
    for link_name in common:
        row = link_meta.loc[link_name]
        b0 = str(row["bus0"])
        b1 = str(row["bus1"])

        p0 = bus_points.get(b0)
        p1 = bus_points.get(b1)

        if p0 is None:
            c0 = extract_province_code(b0)
            p0 = map_rep_points.get(c0)
        if p1 is None:
            c1 = extract_province_code(b1)
            p1 = map_rep_points.get(c1)

        if p0 is None or p1 is None:
            continue

        flow_mean_abs = float(flows[link_name].abs().mean())
        cap = row["cap"]
        util = np.nan if pd.isna(cap) or cap == 0 else float(flow_mean_abs / cap)

        line_rows.append({"link": link_name, "value": util, "geometry": LineString([p0, p1])})

    line_gdf = gpd.GeoDataFrame(line_rows, geometry="geometry", crs=map_gdf.crs)
    return province_df, line_gdf


def compute_h2_storage_and_flow(run_dir: Path, map_points: dict[str, tuple[float, float]], map_gdf: gpd.GeoDataFrame):
    _, links, storage_units, buses = load_core_tables(run_dir)
    _, links_p0 = try_read_ts(run_dir, ["links-p0"])

    storage_df = pd.DataFrame(columns=["code", "value"])
    if storage_units is not None and not storage_units.empty:
        mask = hydrogen_storage_mask(storage_units)
        hs = storage_units.loc[mask].copy()
        if not hs.empty:
            cap_col = choose_capacity_column(hs)
            hs["p"] = pd.to_numeric(hs[cap_col], errors="coerce").fillna(0.0)
            hs["max_hours"] = pd.to_numeric(hs.get("max_hours", 1.0), errors="coerce").fillna(1.0)
            hs["value"] = hs["p"] * hs["max_hours"]
            hs["code"] = hs.apply(get_component_province, axis=1, bus_cols=("bus",))
            storage_df = hs.groupby("code", dropna=True)["value"].sum().reset_index()

    line_gdf = gpd.GeoDataFrame(columns=["value", "geometry"], geometry="geometry")
    if links is not None and not links.empty and links_p0 is not None and not links_p0.empty:
        mask = hydrogen_link_mask(links)
        hlinks = links.loc[mask].copy()

        common = [c for c in links_p0.columns if c in hlinks.index]
        if common:
            hlinks = hlinks.loc[common].copy()
            flows = sanitize_numeric(links_p0[common]).fillna(0.0)

            buses = buses if buses is not None else pd.DataFrame()
            bus_points = get_bus_points(buses, map_points)
            map_rep_points = {row["code"]: (row["rep_point"].x, row["rep_point"].y) for _, row in map_gdf.iterrows()}

            line_rows = []
            for link_name in common:
                row = hlinks.loc[link_name]
                if "bus0" not in row.index or "bus1" not in row.index:
                    continue

                b0 = str(row["bus0"])
                b1 = str(row["bus1"])

                p0 = bus_points.get(b0)
                p1 = bus_points.get(b1)

                if p0 is None:
                    c0 = extract_province_code(b0)
                    p0 = map_rep_points.get(c0)
                if p1 is None:
                    c1 = extract_province_code(b1)
                    p1 = map_rep_points.get(c1)

                if p0 is None or p1 is None:
                    continue

                flow_mean_abs = float(flows[link_name].abs().mean())
                line_rows.append({"link": link_name, "value": flow_mean_abs, "geometry": LineString([p0, p1])})

            line_gdf = gpd.GeoDataFrame(line_rows, geometry="geometry", crs=map_gdf.crs)

    return storage_df, line_gdf


# ============================================================
# Power-flow direction & volume (NEW)
# ============================================================
#
# This block produces the figure pair the advisor asked about:
# "a map showing average power flow (volume) and dominant power flow direction.
#  We are expecting a reverse of the current flows from S-N to N-S."
#
# Method (per interregional electric link):
#   1. Sum the signed link flow over the year, scaled by snapshot weight, to
#      get the NET energy transferred (TWh) and its sign in the model's
#      convention (bus0 -> bus1 positive).
#   2. Translate that into an absolute geographic direction by checking the
#      latitudes of the two endpoint buses. "Northward" means net flow from
#      lower-latitude bus to higher-latitude bus, regardless of how the model
#      named the link or which way bus0/bus1 happens to point. This is what
#      lets us cleanly answer "did the flow reverse from S->N to N->S?".
#   3. Also record gross |flow| (TWh) so the reader can see how heavily a
#      corridor is used in addition to its dominant direction.
#
# The figure draws each corridor as an arrow (FancyArrow):
#   - Arrow direction: the dominant net direction (low-lat -> high-lat if
#     flow_north > 0, otherwise high-lat -> low-lat).
#   - Arrow width: scaled by net |TWh| so heavy corridors stand out.
#   - Arrow color: red = northward (from south), blue = southward (from north).
#   - Optional grey "ghost" line: total gross |flow|, drawn under the arrow
#     so corridors with high churn but small net flow are still visible.

def compute_directional_power_flow(
    run_dir: Path,
    map_points: dict[str, tuple[float, float]],
    map_gdf: gpd.GeoDataFrame,
    drop_minor_share: float = 0.02,
) -> pd.DataFrame:
    """
    Compute, per interregional electric link, the net annual energy and its
    dominant geographic direction.

    Returns a DataFrame with columns:
        link            link name
        bus_low_lat     bus at the southern endpoint (lower y)
        bus_high_lat    bus at the northern endpoint (higher y)
        x_low, y_low    coords of the southern endpoint
        x_high, y_high  coords of the northern endpoint
        net_twh_north   net annual energy in TWh, positive = northward
        gross_twh       gross |flow| in TWh (sum of |hourly flow| × step)
        share_of_max    net |TWh| as a fraction of the corridor with the
                        largest net |TWh| in this run; useful for filtering
                        out negligible corridors.

    `drop_minor_share` removes corridors whose net flow is below this fraction
    of the largest net flow in the run, to keep the figure readable. Set to 0
    to keep every corridor.
    """
    _, links, _, buses = load_core_tables(run_dir)
    _, links_p0 = try_read_ts(run_dir, ["links-p0"])

    if (links is None or links.empty
        or links_p0 is None or links_p0.empty):
        return pd.DataFrame()

    mask = electric_interregional_link_mask(links)
    link_meta = links.loc[mask].copy()
    if link_meta.empty:
        return pd.DataFrame()

    common = [c for c in links_p0.columns if c in link_meta.index]
    if not common:
        return pd.DataFrame()

    link_meta = link_meta.loc[common].copy()
    flows = sanitize_numeric(links_p0[common]).fillna(0.0)
    step_hours = infer_step_hours(run_dir, flows)

    bus_points = get_bus_points(buses if buses is not None else pd.DataFrame(),
                                map_points)
    map_rep_points = {row["code"]: (row["rep_point"].x, row["rep_point"].y)
                      for _, row in map_gdf.iterrows()}

    rows = []
    for link_name in common:
        row = link_meta.loc[link_name]
        b0 = str(row["bus0"])
        b1 = str(row["bus1"])

        p0 = bus_points.get(b0)
        p1 = bus_points.get(b1)
        if p0 is None:
            p0 = map_rep_points.get(extract_province_code(b0))
        if p1 is None:
            p1 = map_rep_points.get(extract_province_code(b1))
        if p0 is None or p1 is None:
            continue

        # Annual energies (MWh -> TWh). Net is signed in the bus0 -> bus1
        # convention; gross is direction-blind.
        series = flows[link_name]
        net_mwh_b0_to_b1 = float(series.sum() * step_hours)
        gross_mwh = float(series.abs().sum() * step_hours)

        # Re-orient so direction is geographic, not model-internal.
        # If bus1 is north of bus0 (higher y), then bus0->bus1 == northward,
        # so net_north = +net_mwh. Otherwise flip the sign.
        if p1[1] >= p0[1]:
            bus_low, bus_high = b0, b1
            p_low, p_high = p0, p1
            net_north_mwh = net_mwh_b0_to_b1
        else:
            bus_low, bus_high = b1, b0
            p_low, p_high = p1, p0
            net_north_mwh = -net_mwh_b0_to_b1

        rows.append({
            "link":          link_name,
            "bus_low_lat":   bus_low,
            "bus_high_lat":  bus_high,
            "x_low":         p_low[0],
            "y_low":         p_low[1],
            "x_high":        p_high[0],
            "y_high":        p_high[1],
            "net_twh_north": net_north_mwh / 1e6,
            "gross_twh":     gross_mwh / 1e6,
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    max_abs = float(df["net_twh_north"].abs().max() or 1e-9)
    df["share_of_max"] = df["net_twh_north"].abs() / max_abs

    if drop_minor_share > 0:
        df = df.loc[df["share_of_max"] >= drop_minor_share].copy()

    return df


def _draw_directional_flow_panel(
    ax,
    map_gdf: gpd.GeoDataFrame,
    flow_df: pd.DataFrame,
    title: str,
    max_abs_twh: float,
    show_gross_ghost: bool = True,
):
    """
    Single-panel renderer used by directional_flow_pair.

    Each interregional electric corridor is drawn as one of three things:
      1. A grey 'ghost' line behind the arrow showing total gross |flow| (the
         busy-ness of the corridor regardless of direction).
      2. A red FancyArrow for net northward flow (drawn south->north).
      3. A blue FancyArrow for net southward flow (drawn north->south).
    Arrow width and head size scale with net |TWh|, normalised against the
    largest corridor across BOTH panels (max_abs_twh) so the easy and hard
    panels are directly comparable.
    """
    map_gdf.plot(ax=ax, color="#f2f2f2", edgecolor="black", linewidth=0.8)
    annotate_codes_only(ax, map_gdf)

    if flow_df is None or flow_df.empty:
        ax.set_title(title)
        ax.set_axis_off()
        return

    # Reserve a sensible visual scale: thinnest arrow = 0.7, thickest = 6.
    max_abs_twh = max(float(max_abs_twh), 1e-9)
    max_gross = float(flow_df["gross_twh"].max() or 1e-9)

    # Sort so that the heaviest arrows are drawn last (on top), and the
    # ghost lines underneath them.
    flow_df = flow_df.sort_values("net_twh_north", key=lambda s: s.abs())

    if show_gross_ghost:
        for _, r in flow_df.iterrows():
            lw = 0.5 + 3.0 * (r["gross_twh"] / max_gross)
            ax.plot(
                [r["x_low"], r["x_high"]],
                [r["y_low"], r["y_high"]],
                color="#bdbdbd", lw=lw, alpha=0.55, zorder=1,
                solid_capstyle="round",
            )

    for _, r in flow_df.iterrows():
        net = float(r["net_twh_north"])
        if net == 0.0:
            continue

        if net > 0:
            # Northward: arrow goes from south (low y) to north (high y).
            x0, y0 = r["x_low"], r["y_low"]
            x1, y1 = r["x_high"], r["y_high"]
            color = "#c0392b"           # red
        else:
            # Southward: arrow goes from north to south.
            x0, y0 = r["x_high"], r["y_high"]
            x1, y1 = r["x_low"], r["y_low"]
            color = "#1f4e79"           # blue

        share = abs(net) / max_abs_twh
        lw = 1.4 + 4.6 * share

        # Trim the arrow slightly so the head doesn't sit on top of the
        # province label box at the destination.
        dx, dy = x1 - x0, y1 - y0
        length = (dx * dx + dy * dy) ** 0.5
        if length > 0:
            shrink = 0.08
            x0 += dx * shrink
            y0 += dy * shrink
            x1 -= dx * shrink
            y1 -= dy * shrink

        ax.annotate(
            "",
            xy=(x1, y1), xycoords="data",
            xytext=(x0, y0), textcoords="data",
            arrowprops=dict(
                arrowstyle="-|>",
                color=color,
                lw=lw,
                mutation_scale=10 + 18 * share,
                shrinkA=0, shrinkB=0,
                alpha=0.92,
            ),
            zorder=3,
        )

        # Numeric label near the midpoint of the arrow.
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        ax.text(
            mx, my, f"{abs(net):.1f}",
            fontsize=6.5, color=color, fontweight="bold",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.10",
                      fc="white", ec="none", alpha=0.85),
            zorder=4,
        )

    ax.set_title(title)
    ax.set_axis_off()


def directional_flow_pair(
    map_gdf: gpd.GeoDataFrame,
    easy_flow_df: pd.DataFrame,
    hard_flow_df: pd.DataFrame,
    easy_title: str,
    hard_title: str,
    main_title: str,
    out_path: Path,
    show_gross_ghost: bool = True,
):
    """
    Two-panel comparison: net annual electric flow direction and volume by
    corridor, easy vs hard year.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 8.4), constrained_layout=True)

    # Use a SHARED scale across both panels so arrow widths are comparable.
    max_abs_twh = max(
        float(easy_flow_df["net_twh_north"].abs().max())
            if easy_flow_df is not None and not easy_flow_df.empty else 0.0,
        float(hard_flow_df["net_twh_north"].abs().max())
            if hard_flow_df is not None and not hard_flow_df.empty else 0.0,
        1e-9,
    )

    _draw_directional_flow_panel(
        axes[0], map_gdf, easy_flow_df, easy_title, max_abs_twh,
        show_gross_ghost=show_gross_ghost,
    )
    _draw_directional_flow_panel(
        axes[1], map_gdf, hard_flow_df, hard_title, max_abs_twh,
        show_gross_ghost=show_gross_ghost,
    )

    legend_handles = [
        Line2D([0], [0], color="#c0392b", lw=3.5,
               label="Net annual flow northward (S → N)"),
        Line2D([0], [0], color="#1f4e79", lw=3.5,
               label="Net annual flow southward (N → S)"),
    ]
    if show_gross_ghost:
        legend_handles.append(
            Line2D([0], [0], color="#bdbdbd", lw=3.5,
                   label="Gross |flow| (corridor utilisation)")
        )
    legend_handles.extend([
        Line2D([0], [0], color="black", lw=1.4,
               label=f"thin arrow ≈ small net flow  "
                     f"(scale: {max_abs_twh:.1f} TWh = thickest)"),
    ])
    axes[1].legend(handles=legend_handles, loc="lower left",
                   fontsize=7.5, frameon=True)

    fig.suptitle(main_title, fontsize=14)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Plotting helpers
# ============================================================

def merge_map_values(map_gdf: gpd.GeoDataFrame, df: pd.DataFrame) -> gpd.GeoDataFrame:
    out = map_gdf.merge(df, on="code", how="left")
    out["value"] = out["value"].fillna(0.0)
    return out


def annotate_map_values(ax, gdf: gpd.GeoDataFrame, formatter, fontsize=7):
    for _, row in gdf.iterrows():
        pt = row["rep_point"]
        label = f"{row['code']}\n{formatter(row['value'])}"
        ax.text(
            pt.x,
            pt.y,
            label,
            fontsize=fontsize,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.65),
        )


def annotate_codes_only(ax, gdf: gpd.GeoDataFrame, fontsize=7):
    for _, row in gdf.iterrows():
        pt = row["rep_point"]
        ax.text(
            pt.x,
            pt.y,
            f"{row['code']}",
            fontsize=fontsize,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.10", fc="white", ec="none", alpha=0.65),
        )


def plot_line_overlay(ax, line_gdf: gpd.GeoDataFrame, vmax: float, color="black"):
    if line_gdf is None or line_gdf.empty:
        return
    vmax = max(float(vmax), 1e-9)
    widths = 0.3 + 2.7 * (line_gdf["value"].fillna(0.0) / vmax)
    line_gdf.plot(ax=ax, linewidth=widths, color=color, alpha=0.8)

def choropleth_pair(
    easy_map: gpd.GeoDataFrame,
    hard_map: gpd.GeoDataFrame,
    easy_title: str,
    hard_title: str,
    main_title: str,
    out_path: Path,
    cmap: str,
    cbar_label: str,
    formatter,
    diverging: bool = False,
    easy_lines: gpd.GeoDataFrame | None = None,
    hard_lines: gpd.GeoDataFrame | None = None,
    line_label: str | None = None,
):
    fig, axes = plt.subplots(1, 2, figsize=(15, 8), constrained_layout=True)

    easy_vals = pd.to_numeric(easy_map["value"], errors="coerce").fillna(0.0)
    hard_vals = pd.to_numeric(hard_map["value"], errors="coerce").fillna(0.0)

    if diverging:
        max_abs = max(easy_vals.abs().max(), hard_vals.abs().max(), 1e-9)
        norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)
    else:
        vmax = max(easy_vals.max(), hard_vals.max(), 1e-9)
        norm = Normalize(vmin=0.0, vmax=vmax)

    easy_line_vals = (
        pd.to_numeric(easy_lines["value"], errors="coerce").fillna(0.0)
        if easy_lines is not None and not easy_lines.empty
        else pd.Series([0.0])
    )
    hard_line_vals = (
        pd.to_numeric(hard_lines["value"], errors="coerce").fillna(0.0)
        if hard_lines is not None and not hard_lines.empty
        else pd.Series([0.0])
    )
    vmax_lines = max(easy_line_vals.max(), hard_line_vals.max(), 1e-9)

    for ax, gdf, title, lines in [
        (axes[0], easy_map, easy_title, easy_lines),
        (axes[1], hard_map, hard_title, hard_lines),
    ]:
        gdf.plot(
            column="value",
            cmap=cmap,
            linewidth=0.7,
            edgecolor="black",
            ax=ax,
            legend=False,
            norm=norm,
            missing_kwds={"color": "lightgrey"},
        )
        if lines is not None and not lines.empty:
            plot_line_overlay(ax, lines, vmax_lines, color="black")
        annotate_map_values(ax, gdf, formatter)
        ax.set_title(title)
        ax.set_axis_off()

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.035, pad=0.02)
    cbar.set_label(cbar_label)

    if line_label:
        legend_lines = [
            Line2D([0], [0], color="black", lw=0.6, label=f"lower {line_label}"),
            Line2D([0], [0], color="black", lw=2.8, label=f"higher {line_label}"),
        ]
        axes[1].legend(handles=legend_lines, loc="lower left", fontsize=8, frameon=True)

    fig.suptitle(main_title, fontsize=14)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def hydrogen_flow_pair(
    map_gdf: gpd.GeoDataFrame,
    easy_lines: gpd.GeoDataFrame | None,
    hard_lines: gpd.GeoDataFrame | None,
    easy_title: str,
    hard_title: str,
    main_title: str,
    out_path: Path,
):
    fig, axes = plt.subplots(1, 2, figsize=(15, 8), constrained_layout=True)

    vmax_lines = max(
        float(easy_lines["value"].max()) if easy_lines is not None and not easy_lines.empty else 0.0,
        float(hard_lines["value"].max()) if hard_lines is not None and not hard_lines.empty else 0.0,
        1e-9,
    )

    for ax, title, lines in [
        (axes[0], easy_title, easy_lines),
        (axes[1], hard_title, hard_lines),
    ]:
        map_gdf.plot(ax=ax, color="#f2f2f2", edgecolor="black", linewidth=0.8)
        if lines is not None and not lines.empty:
            plot_line_overlay(ax, lines, vmax_lines, color="black")
        annotate_codes_only(ax, map_gdf)
        ax.set_title(title)
        ax.set_axis_off()

    legend_lines = [
        Line2D([0], [0], color="black", lw=0.6, label="lower mean absolute H$_2$ flow"),
        Line2D([0], [0], color="black", lw=2.8, label="higher mean absolute H$_2$ flow"),
    ]
    axes[1].legend(handles=legend_lines, loc="lower left", fontsize=8, frameon=True)

    fig.suptitle(main_title, fontsize=14)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def draw_capacity_mix_panel(ax, map_gdf: gpd.GeoDataFrame, mix_df: pd.DataFrame, title: str, max_total: float):
    gdf = map_gdf.copy()
    gdf.plot(ax=ax, color="#f2f2f2", edgecolor="black", linewidth=0.8)

    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    xspan = x1 - x0
    yspan = y1 - y0

    bar_width = xspan * 0.012
    max_bar_height = yspan * 0.08
    scale = max_bar_height / max(max_total, 1e-9)

    merged = map_gdf[["code", "rep_point"]].merge(mix_df, on="code", how="left").fillna(0.0)
    for _, row in merged.iterrows():
        x = row["rep_point"].x
        y = row["rep_point"].y

        bottom = y + 0.005 * yspan
        total = 0.0

        for carrier in DEFAULT_MIX_ORDER:
            val = float(row.get(carrier, 0.0))
            if val <= 0:
                continue
            h = val * scale
            rect = Rectangle(
                (x - bar_width / 2, bottom),
                bar_width,
                h,
                facecolor=CAPACITY_MIX_COLORS[carrier],
                edgecolor="black",
                linewidth=0.3,
                alpha=0.95,
            )
            ax.add_patch(rect)
            bottom += h
            total += val

        ax.text(
            x,
            y - 0.02 * yspan,
            f"{row['code']}\n{total / 1000:.1f} GW",
            ha="center",
            va="top",
            fontsize=7,
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.70),
        )

    ax.set_title(title)
    ax.set_axis_off()


def capacity_mix_pair(
    easy_mix: pd.DataFrame,
    hard_mix: pd.DataFrame,
    map_gdf: gpd.GeoDataFrame,
    out_path: Path,
    easy_title: str,
    hard_title: str,
    main_title: str,
):
    easy_total = float(easy_mix[DEFAULT_MIX_ORDER].sum(axis=1).max()) if not easy_mix.empty else 0.0
    hard_total = float(hard_mix[DEFAULT_MIX_ORDER].sum(axis=1).max()) if not hard_mix.empty else 0.0
    max_total = max(easy_total, hard_total, 1e-9)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), constrained_layout=True)
    draw_capacity_mix_panel(axes[0], map_gdf, easy_mix, easy_title, max_total)
    draw_capacity_mix_panel(axes[1], map_gdf, hard_mix, hard_title, max_total)

    legend_handles = [
        Rectangle((0, 0), 1, 1, facecolor=CAPACITY_MIX_COLORS[c], edgecolor="black", label=c)
        for c in DEFAULT_MIX_ORDER
    ]
    axes[1].legend(handles=legend_handles, loc="lower left", fontsize=8, frameon=True, title="Carrier")
    fig.suptitle(main_title, fontsize=14)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Main driver
# ============================================================

def main():
    ap = argparse.ArgumentParser(description="Poland provincial comparison maps for easy vs hard weather years.")
    ap.add_argument("--easy_run", required=True, type=str, help="Path to easy-year run directory.")
    ap.add_argument("--hard_run", required=True, type=str, help="Path to hard-year run directory.")
    ap.add_argument("--easy_label", default="Easy year", type=str)
    ap.add_argument("--hard_label", default="Hard year", type=str)
    ap.add_argument("--map_path", default="pl.json", type=str, help="Path to Poland GeoJSON.")
    ap.add_argument("--out_dir", default="figures_maps", type=str, help="Output folder.")
    args = ap.parse_args()

    easy_run = Path(args.easy_run)
    hard_run = Path(args.hard_run)
    map_path = Path(args.map_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    map_gdf = load_map(map_path)
    map_points = {row["code"]: (row["rep_point"].x, row["rep_point"].y) for _, row in map_gdf.iterrows()}

    # 1. Heat-pump capacity
    hp_easy = compute_heat_pump_capacity(easy_run)
    hp_hard = compute_heat_pump_capacity(hard_run)

    hp_easy_map = merge_map_values(map_gdf, hp_easy)
    hp_hard_map = merge_map_values(map_gdf, hp_hard)

    choropleth_pair(
        hp_easy_map,
        hp_hard_map,
        easy_title=f"{args.easy_label}",
        hard_title=f"{args.hard_label}",
        main_title="Heat-pump capacity by province",
        out_path=out_dir / "01_heat_pump_capacity_pair.png",
        cmap="YlOrRd",
        cbar_label="Installed heat-pump capacity [GW]",
        formatter=lambda x: f"{x / 1000:.1f}",
    )
    plt.close("all")

    # 2. Capacity mix
    mix_easy = compute_capacity_mix(easy_run)
    mix_hard = compute_capacity_mix(hard_run)

    capacity_mix_pair(
        mix_easy,
        mix_hard,
        map_gdf,
        out_path=out_dir / "02_capacity_mix_pair.png",
        easy_title=args.easy_label,
        hard_title=args.hard_label,
        main_title="Generation capacity mix by province",
    )
    plt.close("all")

    # 3. Net imports + transmission utilization
    imports_easy, lines_easy = compute_net_imports_and_utilization(easy_run, map_points, map_gdf)
    imports_hard, lines_hard = compute_net_imports_and_utilization(hard_run, map_points, map_gdf)

    imports_easy_map = merge_map_values(map_gdf, imports_easy)
    imports_hard_map = merge_map_values(map_gdf, imports_hard)

    choropleth_pair(
        imports_easy_map,
        imports_hard_map,
        easy_title=args.easy_label,
        hard_title=args.hard_label,
        main_title="Net imports by province with transmission-utilization overlay",
        out_path=out_dir / "03_net_imports_transmission_pair.png",
        cmap="RdBu_r",
        cbar_label="Net imports [TWh over model year; positive = importer]",
        formatter=lambda x: f"{x / 1e6:.1f}",
        diverging=True,
        easy_lines=lines_easy,
        hard_lines=lines_hard,
        line_label="utilization",
    )
    plt.close("all")

    # 4. Hydrogen flows only
    _, h2_lines_easy = compute_h2_storage_and_flow(easy_run, map_points, map_gdf)
    _, h2_lines_hard = compute_h2_storage_and_flow(hard_run, map_points, map_gdf)

    hydrogen_flow_pair(
        map_gdf=map_gdf,
        easy_lines=h2_lines_easy,
        hard_lines=h2_lines_hard,
        easy_title=args.easy_label,
        hard_title=args.hard_label,
        main_title="Hydrogen-flow network overlay",
        out_path=out_dir / "04_h2_flow_pair.png",
    )
    plt.close("all")

    # 5. Directional electric power flow (NEW)
    #
    # Net annual energy per interregional electric corridor, oriented in
    # absolute geographic terms (south->north or north->south) rather than
    # in the model's bus0->bus1 convention. Lets the reader directly read
    # off whether the system has reversed the historical S->N pattern to
    # an N->S pattern, as expected from offshore wind in the north.
    flow_easy = compute_directional_power_flow(easy_run, map_points, map_gdf,
                                               drop_minor_share=0.02)
    flow_hard = compute_directional_power_flow(hard_run, map_points, map_gdf,
                                               drop_minor_share=0.02)

    # Save the underlying data for the appendix / for verifying which
    # corridors flipped between easy and hard years.
    if not flow_easy.empty:
        flow_easy.to_csv(out_dir / "05_directional_flow_easy.csv", index=False)
    if not flow_hard.empty:
        flow_hard.to_csv(out_dir / "05_directional_flow_hard.csv", index=False)

    directional_flow_pair(
        map_gdf=map_gdf,
        easy_flow_df=flow_easy,
        hard_flow_df=flow_hard,
        easy_title=args.easy_label,
        hard_title=args.hard_label,
        main_title="Directional electric flow: net volume and dominant direction by corridor",
        out_path=out_dir / "05_power_flow_direction_pair.png",
    )
    plt.close("all")

    print(f"Saved figures to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
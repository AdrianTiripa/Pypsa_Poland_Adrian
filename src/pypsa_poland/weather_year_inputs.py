# weather_year_inputs.py
#
# Input-side meteorological characterisation for pypsa-poland weather-year runs.
#
# This is the first script in the weather-robustness workflow. It computes the
# main weather descriptors that explain why one year is relatively easy or
# difficult for the system before any optimisation is run. Its output CSV is
# passed to weather_year_compare.py to annotate optimisation results with the
# underlying meteorological drivers.
#
# Main outputs (written to --out_dir):
#   - weather_year_inputs_summary.csv — one row per year with all metrics.
#   - Figures: CF heatmaps, seasonal wind/PV bars, heat-demand bars,
#     COP heatmaps, electricity-for-heat lines, wind-drought bars,
#     stress-score rankings.
#
# Metrics computed per year:
#   - Annual and seasonal PV / onshore / offshore capacity factors.
#   - Combined VRES CF (installed-capacity-weighted).
#   - Annual, winter, summer, and peak heat demand.
#   - Annual, winter, and summer COP; cold-stress hours (COP < 2).
#   - Annual and peak electricity-for-heat.
#   - Wind drought duration; dark doldrum hours.
#   - Composite stress score (energy + peak + persistence sub-scores).
#
# Usage:
#   python weather_year_inputs.py \
#       --profiles_root <path> --cf_folder <path> \
#       --system_year 2050 --scenario Core
#   python weather_year_inputs.py ... --years 1980 1990 2000 2010 2020

from __future__ import annotations

import argparse
import calendar
from pathlib import Path

import numpy as np
import pandas as pd

from plot_style import (
    apply_style,
    CARRIER_COLORS, BLUE, RED, AMBER, GREY,
    bar, line, scatter_annotated, heatmap, ranking_table,
)

apply_style()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REGION_ORDER = [
    "DS", "KP", "LD", "LU", "LB", "MA", "MZ", "OP",
    "PK", "PD", "PM", "SL", "SK", "WN", "WP", "ZP",
]

SEASONS = {"DJF": [12, 1, 2], "MAM": [3, 4, 5], "JJA": [6, 7, 8], "SON": [9, 10, 11]}

INSTALLED_MW = {"pv": 69_844, "onshore": 36_400, "offshore": 45_343}
NZP = "Poland 2050 Net-Zero Pathways scenario"


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _drop_feb29(df: pd.DataFrame, year: int) -> pd.DataFrame:
    if not calendar.isleap(year):
        return df.copy()
    s, e = 59 * 24, 60 * 24
    return pd.concat([df.iloc[:s], df.iloc[e:]], axis=0, ignore_index=True)


def _load_cf_year(path: Path, year: int) -> pd.DataFrame:
    """
    Load one year of CF data from a large multi-year datetime-indexed CSV.
    """
    idx_col = pd.read_csv(path, index_col=0, usecols=[0])
    raw_idx = idx_col.index.astype(str)

    years_extracted = raw_idx.str.extract(r"(\b\d{4}\b)", expand=False)
    row_mask = years_extracted == str(year)

    if not row_mask.any():
        raise ValueError(f"No rows for year={year} in {path.name}")

    keep_positions = set(i for i, m in enumerate(row_mask) if m)
    skip = [i + 1 for i in range(len(raw_idx)) if i not in keep_positions]

    df = pd.read_csv(path, index_col=0, skiprows=skip)
    df.index = pd.to_datetime(df.index, errors="coerce")
    if df.index.isna().any():
        df = df.loc[~df.index.isna()].copy()

    if df.empty:
        raise ValueError(f"No rows for year={year} in {path.name}")

    if len(df) == 8784:
        df = df.loc[~((df.index.month == 2) & (df.index.day == 29))].copy()

    if len(df) != 8760:
        raise ValueError(
            f"Expected 8760 rows for year={year} in {path.name}, got {len(df)}"
        )
    return df.apply(pd.to_numeric, errors="coerce")


def _find_dynamic_profile(
    profiles_root: Path, scenario: str, kind: str,
    prefix: str, meteo_year: int, system_year: int,
) -> Path:
    base = profiles_root / scenario / kind
    if not base.exists():
        raise FileNotFoundError(f"Dynamic profile folder not found: {base}")
    exact = base / f"{prefix}_{meteo_year}_{system_year}.csv"
    if exact.exists():
        return exact
    candidates = [
        p for p in base.glob(f"*_{meteo_year}_{system_year}.csv")
        if not p.name.startswith("Neli2_sum_")
    ]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise ValueError(f"Ambiguous match for {kind} year={meteo_year}: "
                         + ", ".join(p.name for p in candidates))
    raise FileNotFoundError(
        f"No {kind} file in {base} for meteo_year={meteo_year}, system_year={system_year}"
    )


def _last_24_rows_all_zero(df: pd.DataFrame, tol: float = 1e-12) -> bool:
    tail = df.iloc[-24:].apply(pd.to_numeric, errors="coerce")
    vals = tail.stack().dropna()
    return (not vals.empty) and (vals.abs() <= tol).all()


def _load_dynamic_profile(
    profiles_root: Path, scenario: str, kind: str,
    prefix: str, meteo_year: int, system_year: int,
) -> pd.DataFrame:
    """
    Load a per-year dynamic profile and return an 8760-row DataFrame.
    """
    path = _find_dynamic_profile(
        profiles_root, scenario, kind, prefix, meteo_year, system_year
    )
    df = pd.read_csv(path, header=None)
    if df.shape[1] != 16:
        raise ValueError(f"{path.name}: expected 16 columns, got {df.shape[1]}")
    df.columns = REGION_ORDER
    df = df.apply(pd.to_numeric, errors="coerce")

    if len(df) == 8760:
        return df

    if len(df) == 8784:
        if _last_24_rows_all_zero(df):
            return df.iloc[:-24].reset_index(drop=True)
        return _drop_feb29(df, meteo_year)

    raise ValueError(
        f"{path.name}: unexpected row count {len(df)} "
        f"(expected 8760 or 8784) for year={meteo_year}"
    )


def _discover_years(cf_path: Path) -> list[int]:
    df = pd.read_csv(cf_path, index_col=0, usecols=[0])
    return sorted(pd.to_datetime(df.index, errors="coerce").dropna().year.unique().tolist())


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def _season_means(s8760: pd.Series) -> dict[str, float]:
    ref = pd.date_range("2001-01-01", periods=8760, freq="h")
    s = pd.Series(s8760.values, index=ref)
    return {k: float(s[s.index.month.isin(v)].mean()) for k, v in SEASONS.items()}


def _longest_below(series: pd.Series, threshold: float) -> int:
    max_run = current = 0
    for v in (series < threshold).astype(int):
        current = (current + 1) if v else 0
        max_run = max(max_run, current)
    return max_run


def _minmax_higher_worse(series: pd.Series, invert: bool = False) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if invert:
        s = -s
    rng = s.max() - s.min()
    if not np.isfinite(rng) or rng <= 0:
        return pd.Series(0.0, index=s.index)
    return (s - s.min()) / rng


_WIN = slice(0, 1416)     # Jan + Feb
_SUM = slice(3624, 5112)  # Jun + Jul + Aug


def characterize_year(
    year: int,
    cf_pv_path: Path, cf_onshore_path: Path, cf_offshore_path: Path,
    profiles_root: Path, scenario: str, system_year: int,
) -> dict:
    row: dict = {"year": year}

    pv_df  = _load_cf_year(cf_pv_path, year)
    on_df  = _load_cf_year(cf_onshore_path, year)
    off_df = _load_cf_year(cf_offshore_path, year)

    pv_sys  = pv_df.mean(axis=1)
    on_sys  = on_df.mean(axis=1)
    off_sys = off_df.mean(axis=1)

    row["cf_pv_annual_mean"] = float(pv_sys.mean())
    row["cf_pv_summer_mean"] = float(pv_sys.iloc[_SUM].mean())
    row["cf_pv_winter_mean"] = float(pv_sys.iloc[_WIN].mean())
    row["cf_onshore_annual_mean"] = float(on_sys.mean())
    row["cf_offshore_annual_mean"] = float(off_sys.mean())

    for season, v in _season_means(on_sys).items():
        row[f"cf_onshore_{season}"] = v
    for season, v in _season_means(off_sys).items():
        row[f"cf_offshore_{season}"] = v

    total_mw = sum(INSTALLED_MW.values())
    vres_sys = (
        pv_sys  * INSTALLED_MW["pv"] +
        on_sys  * INSTALLED_MW["onshore"] +
        off_sys * INSTALLED_MW["offshore"]
    ) / total_mw
    row["cf_vres_combined_annual"] = float(vres_sys.mean())
    row["cf_vres_combined_winter"] = float(vres_sys.iloc[_WIN].mean())

    wind_sys = (
        on_sys  * INSTALLED_MW["onshore"] +
        off_sys * INSTALLED_MW["offshore"]
    ) / (INSTALLED_MW["onshore"] + INSTALLED_MW["offshore"])
    row["wind_drought_max_hours"] = int(_longest_below(wind_sys, 0.10))
    row["dark_doldrums_hours"] = int(((pv_sys < 0.05) & (wind_sys < 0.05)).sum())

    heat_df = _load_dynamic_profile(
        profiles_root, scenario, "HeatDemand", "Qishare", year, system_year
    ) * 1000.0   # GW → MW

    cop_df = _load_dynamic_profile(
        profiles_root, scenario, "COP", "COPiavg3", year, system_year
    )

    heat_sys = heat_df.sum(axis=1)
    row["heat_demand_annual_mwh"] = float(heat_sys.sum())
    row["heat_demand_peak_mw"] = float(heat_sys.max())
    row["heat_demand_winter_mean"] = float(heat_sys.iloc[_WIN].mean())
    row["heat_demand_summer_mean"] = float(heat_sys.iloc[_SUM].mean())

    row["cop_annual_mean"] = float(cop_df.mean().mean())
    row["cop_winter_mean"] = float(cop_df.iloc[_WIN].mean().mean())
    row["cop_summer_mean"] = float(cop_df.iloc[_SUM].mean().mean())
    row["cop_min_hourly"] = float(cop_df.min().min())
    row["cold_stress_hours_cop_lt2"] = int((cop_df.mean(axis=1) < 2.0).sum())

    efh = heat_df.div(cop_df.replace(0, np.nan)).fillna(0.0)
    efh_sys = efh.sum(axis=1)
    row["elec_for_heat_annual_mwh"] = float(efh_sys.sum())
    row["elec_for_heat_peak_mw"] = float(efh_sys.max())
    row["elec_for_heat_winter_mean"] = float(efh_sys.iloc[_WIN].mean())

    return row


def _add_stress_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["stress_vres_low"] = _minmax_higher_worse(df["cf_vres_combined_annual"], invert=True)
    df["stress_elec_for_heat"] = _minmax_higher_worse(df["elec_for_heat_annual_mwh"])
    df["stress_heat_peak"] = _minmax_higher_worse(df["heat_demand_peak_mw"])
    df["stress_elec_for_heat_peak"] = _minmax_higher_worse(df["elec_for_heat_peak_mw"])
    df["stress_wind_drought"] = _minmax_higher_worse(df["wind_drought_max_hours"])
    df["stress_doldrums"] = _minmax_higher_worse(df["dark_doldrums_hours"])
    df["stress_cold_cop"] = _minmax_higher_worse(df["cold_stress_hours_cop_lt2"])

    df["stress_energy_block"] = (
        df["stress_vres_low"] + df["stress_elec_for_heat"]
    ) / 2.0

    df["stress_peak_block"] = (
        df["stress_heat_peak"] + df["stress_elec_for_heat_peak"]
    ) / 2.0

    df["stress_persistence_block"] = (
        df["stress_wind_drought"] + df["stress_doldrums"] + df["stress_cold_cop"]
    ) / 3.0

    df["stress_score"] = (
        df["stress_energy_block"] +
        df["stress_peak_block"] +
        df["stress_persistence_block"]
    ) / 3.0

    block_cols = ["stress_energy_block", "stress_peak_block", "stress_persistence_block"]
    block_names = {
        "stress_energy_block": "energy",
        "stress_peak_block": "peak",
        "stress_persistence_block": "persistence",
    }
    driver_cols = {
        "stress_vres_low": "low_vres",
        "stress_elec_for_heat": "annual_electricity_for_heat",
        "stress_heat_peak": "peak_heat",
        "stress_elec_for_heat_peak": "peak_electricity_for_heat",
        "stress_wind_drought": "wind_drought",
        "stress_doldrums": "dark_doldrums",
        "stress_cold_cop": "cold_cop_hours",
    }

    df["primary_stress_block"] = df[block_cols].idxmax(axis=1).map(block_names)
    df["secondary_stress_block"] = (
        df[block_cols].apply(lambda r: r.sort_values(ascending=False).index[1], axis=1).map(block_names)
    )
    df["primary_stress_driver"] = df[list(driver_cols)].idxmax(axis=1).map(driver_cols)

    return df


# ---------------------------------------------------------------------------
# Plot generation
# ---------------------------------------------------------------------------

def make_plots(df: pd.DataFrame, out_dir: Path) -> None:
    """
    Generate all input-characterisation figures and save them to out_dir.

    Covers: VRES capacity-factor time series and heatmaps, seasonal breakdowns,
    heat demand bars, COP heatmaps, electricity-for-heat, wind-drought duration,
    and stress-score ranking table.
    """
    d = df.set_index("year").sort_index()

    line(
        d[["cf_pv_annual_mean", "cf_onshore_annual_mean",
           "cf_offshore_annual_mean", "cf_vres_combined_annual"]].rename(columns={
               "cf_pv_annual_mean": "PV ground",
               "cf_onshore_annual_mean": "wind",
               "cf_offshore_annual_mean": "wind offshore",
               "cf_vres_combined_annual": "Combined VRES (capacity-weighted)",
           }),
        out_dir / "cf_annual_by_technology.png",
        "Annual mean capacity factor by technology",
        "Capacity factor", "p.u.",
        subtitle=NZP,
    )

    bar(d["wind_drought_max_hours"],
        out_dir / "wind_drought_max_hours.png",
        "Longest wind drought by weather year",
        "Duration", "hours",
        subtitle="Consecutive hours with combined wind CF < 10%",
        color=CARRIER_COLORS["wind offshore"],
        value_labels=False)

    bar(d["dark_doldrums_hours"],
        out_dir / "dark_doldrums_hours.png",
        "Dark doldrums hours by weather year",
        "Hours", "hours",
        subtitle="Wind and solar both below 5% simultaneously",
        color=GREY,
        value_labels=False)

    line(d[["heat_demand_annual_mwh"]] / 1e6,
         out_dir / "heat_demand_annual_twh.png",
         "Annual heat demand by weather year",
         "Heat demand", "TWh",
         subtitle="Sum across all 16 Polish regions — " + NZP,
         markers=True,
         color_map={"heat_demand_annual_mwh": CARRIER_COLORS["heat"]})

    line(d[["heat_demand_peak_mw"]] / 1e3,
         out_dir / "heat_demand_peak_gw.png",
         "Peak hourly heat demand by weather year",
         "Peak demand", "GW",
         subtitle="System-wide hourly peak across all regions",
         markers=True,
         color_map={"heat_demand_peak_mw": CARRIER_COLORS["heat"]})

    line(
        d[["cop_annual_mean", "cop_winter_mean", "cop_summer_mean"]].rename(columns={
            "cop_annual_mean": "Annual mean",
            "cop_winter_mean": "Winter (Jan–Feb)",
            "cop_summer_mean": "Summer (Jun–Aug)",
        }),
        out_dir / "cop_by_season.png",
        "Heat pump COP by weather year",
        "COP", "",
        subtitle="System average across all regions",
        color_map={
            "Annual mean": BLUE,
            "Winter (Jan–Feb)": CARRIER_COLORS["wind offshore"],
            "Summer (Jun–Aug)": AMBER,
        },
    )

    bar(d["cold_stress_hours_cop_lt2"],
        out_dir / "cold_stress_hours_cop_lt2.png",
        "Cold-stress hours by weather year",
        "Hours", "hours",
        subtitle="Hours when system-mean COP falls below 2.0",
        color=CARRIER_COLORS["wind offshore"],
        value_labels=False)

    line(d[["elec_for_heat_annual_mwh"]] / 1e6,
         out_dir / "elec_for_heat_annual_twh.png",
         "Annual electricity needed for heat",
         "Electricity for heat", "TWh",
         subtitle="Heat demand divided by COP — core energy-stress indicator",
         markers=True,
         color_map={"elec_for_heat_annual_mwh": CARRIER_COLORS["heat_pump"]})

    line(d[["elec_for_heat_peak_mw"]] / 1e3,
         out_dir / "elec_for_heat_peak_gw.png",
         "Peak electricity needed for heat by weather year",
         "Peak demand", "GW",
         subtitle="System-wide hourly peak of heating electricity need",
         markers=True,
         color_map={"elec_for_heat_peak_mw": CARRIER_COLORS["heat_pump"]})

    scatter_annotated(
        d["cf_vres_combined_annual"],
        d["elec_for_heat_annual_mwh"] / 1e6,
        out_dir / "stress_scatter_vres_vs_efh.png",
        xlabel="Combined VRES capacity factor",
        ylabel="Electricity for heat",
        title="Weather-year stress: VRES availability vs heating electricity demand",
        subtitle="Upper-left = most stressful for the system (low wind/solar + high electricity needed for heat)",
        xunit="p.u.", yunit="TWh",
        trend=True,
    )

    seas_cols = {f"cf_onshore_{s}": s for s in SEASONS}
    seas_data = {v: d[k] for k, v in seas_cols.items() if k in d.columns}
    if seas_data:
        heatmap(
            pd.DataFrame(seas_data, index=d.index),
            out_dir / "cf_onshore_seasonal_heatmap.png",
            "Onshore wind capacity factor by season and weather year",
            cbar_label="CF (p.u.)",
            subtitle="Poland — Provincial_Onshore_CF_1940_2025",
            cmap="RdYlGn",
            row_label="Weather year", col_label="Season",
        )

    metric_cols = [
        "cf_vres_combined_annual",
        "elec_for_heat_annual_mwh",
        "heat_demand_peak_mw",
        "elec_for_heat_peak_mw",
        "wind_drought_max_hours",
        "dark_doldrums_hours",
        "cold_stress_hours_cop_lt2",
        "cop_winter_mean",
    ]
    metric_labels = {
        "cf_vres_combined_annual": "VRES CF",
        "elec_for_heat_annual_mwh": "Elec-for-heat",
        "heat_demand_peak_mw": "Peak heat",
        "elec_for_heat_peak_mw": "Peak elec-for-heat",
        "wind_drought_max_hours": "Wind drought",
        "dark_doldrums_hours": "Doldrums",
        "cold_stress_hours_cop_lt2": "Cold COP<2",
        "cop_winter_mean": "Winter COP",
    }
    heatmap_df = d[metric_cols].copy().rename(columns=metric_labels)
    for col in ["Elec-for-heat", "Peak heat", "Peak elec-for-heat", "Wind drought", "Doldrums", "Cold COP<2"]:
        if col in heatmap_df.columns:
            s = pd.to_numeric(heatmap_df[col], errors="coerce")
            rng = s.max() - s.min()
            heatmap_df[col] = (s - s.min()) / rng if rng > 0 else 0.0
    if "VRES CF" in heatmap_df.columns:
        s = pd.to_numeric(heatmap_df["VRES CF"], errors="coerce")
        rng = s.max() - s.min()
        heatmap_df["VRES CF"] = 1 - ((s - s.min()) / rng if rng > 0 else 0.0)
    if "Winter COP" in heatmap_df.columns:
        s = pd.to_numeric(heatmap_df["Winter COP"], errors="coerce")
        rng = s.max() - s.min()
        heatmap_df["Winter COP"] = 1 - ((s - s.min()) / rng if rng > 0 else 0.0)

    heatmap(
        heatmap_df,
        out_dir / "stress_anomaly_heatmap.png",
        "Weather-year stress heatmap",
        cbar_label="Normalised stress (0–1)",
        subtitle="Higher values indicate more demanding conditions along each metric",
        cmap="RdYlGn_r",
        row_label="Weather year",
        col_label="Metric",
    )

    rank_cols = {
        "stress_score": "Overall stress",
        "primary_stress_block": "Primary block",
        "primary_stress_driver": "Primary driver",
        "cf_vres_combined_annual": "VRES CF",
        "heat_demand_annual_mwh": "Heat demand (TWh)",
        "elec_for_heat_annual_mwh": "Elec-for-heat (TWh)",
        "wind_drought_max_hours": "Wind drought (h)",
        "dark_doldrums_hours": "Doldrums (h)",
    }
    rank_df = d[[c for c in rank_cols if c in d.columns]].rename(columns=rank_cols).copy()
    for col in ["Heat demand (TWh)", "Elec-for-heat (TWh)"]:
        if col in rank_df.columns:
            rank_df[col] = (rank_df[col] / 1e6).round(1)
    rank_df["Overall stress"] = rank_df["Overall stress"].round(3)
    rank_df = rank_df.sort_values("Overall stress", ascending=False).head(20)
    ranking_table(
        rank_df,
        out_dir / "stress_ranking_table.png",
        "Weather years ranked by system stress (top 20)",
        subtitle="Use the overall score for ranking, then interpret years through their stress block and driver",
        highlight_top=3, highlight_bottom=3,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Characterise all available meteorological years from input profiles and
    write a summary CSV plus figures to the output directory.
    """
    ap = argparse.ArgumentParser(
        description="Characterise meteorological years from input profiles."
    )
    ap.add_argument("--profiles_root", required=True)
    ap.add_argument("--cf_folder", required=True)
    ap.add_argument("--scenario", default="Core")
    ap.add_argument("--system_year", type=int, default=2050)
    ap.add_argument("--years", nargs="+", type=int, default=None)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    profiles_root = Path(args.profiles_root)
    cf_folder = Path(args.cf_folder)
    out_dir = Path(args.out_dir) if args.out_dir else Path("weather_year_inputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    cf_pv_path = cf_folder / "TprovCF_PV_1940_2025.csv"
    cf_onshore_path = cf_folder / "Provincial_Onshore_CF_1940_2025.csv"
    cf_offshore_path = cf_folder / "Offshore_CF_hourly_1940_2025.csv"

    for p in [cf_pv_path, cf_onshore_path, cf_offshore_path]:
        if not p.exists():
            raise FileNotFoundError(f"CF file not found: {p}")

    years = args.years or _discover_years(cf_pv_path)
    print(f"Processing {len(years)} year(s): {years[0]}–{years[-1]}")

    rows, failures = [], []
    for i, year in enumerate(years, 1):
        print(f"[{i}/{len(years)}] Year {year} ...", end=" ", flush=True)
        try:
            rows.append(characterize_year(
                year=year,
                cf_pv_path=cf_pv_path,
                cf_onshore_path=cf_onshore_path,
                cf_offshore_path=cf_offshore_path,
                profiles_root=profiles_root,
                scenario=args.scenario,
                system_year=args.system_year,
            ))
            print("OK")
        except Exception as e:
            failures.append({"year": year, "error": str(e)})
            print(f"FAILED — {e}")

    if not rows:
        print("No years processed successfully.")
        return

    df = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)
    df = _add_stress_metrics(df)
    df.to_csv(out_dir / "weather_year_inputs_summary.csv", index=False)

    ranked = df.sort_values("stress_score", ascending=False)[[
        "year", "stress_score", "primary_stress_block", "primary_stress_driver",
        "cf_vres_combined_annual", "heat_demand_annual_mwh", "elec_for_heat_annual_mwh",
        "wind_drought_max_hours", "dark_doldrums_hours", "cold_stress_hours_cop_lt2",
    ]]
    ranked.to_csv(out_dir / "weather_year_stress_ranking.csv", index=False)

    if failures:
        pd.DataFrame(failures).to_csv(out_dir / "failures.csv", index=False)

    print("\nTop 10 most stressful years:")
    print(ranked.head(10).to_string(index=False))

    make_plots(df, out_dir)
    print(f"\nAll outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()

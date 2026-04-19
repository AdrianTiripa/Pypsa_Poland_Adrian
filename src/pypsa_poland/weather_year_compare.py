# weather_year_compare.py
#
# Cross-year output comparison for pypsa-poland weather-year runs.
#
# Reads the structured CSV summaries produced by results_to_csv.py for each
# run folder and produces figures and tables comparing every major output
# dimension across meteorological years:
#   - System cost (objective value).
#   - Installed capacities by carrier (generators, storage, links).
#   - Annual generation mix and VRES curtailment.
#   - Demand by sector (electricity, heat, hydrogen, transport).
#   - Storage sizing (H2 cavern, battery, TES, PSH).
#   - Electrolyser, heat pump, H2 pipeline, and CHP capacities.
#   - Interregional transmission utilisation.
#   - Stress correlation plots (if --inputs_csv is supplied from weather_year_inputs.py).
#
# Usage:
#   python weather_year_compare.py --runs_root <folder>
#   python weather_year_compare.py --runs_root <folder> --inputs_csv <path> --out_dir <path>

# & C:/Users/adria/anaconda3/envs/pypsa-legacy/python.exe c:/Users/adria/MODEL_PyPSA/Core/pypsa-poland_ADRIAN/src/pypsa_poland/weather_year_compare.py 
# --runs_root C:/Users/adria/MODEL_PyPSA/Core/runs --inputs_csv C:/Users/adria/MODEL_PyPSA/Core/weather_year_inputs/weather_year_inputs_summary.csv 
# --out_dir C:/Users/adria/MODEL_PyPSA/Core/weather_year_comparison

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from plot_style import (
    apply_style,
    CARRIER_COLORS, BLUE, RED, AMBER, GREY,
    bar, stacked_bar, line, scatter_annotated, ranking_table,
    savefig,
)

apply_style()

# constants

VRES_CARRIERS = {"PV ground", "wind", "wind offshore"}

SECTOR_CLASSES = [
    "electricity_or_other", "heat", "high_temp_heat", "hydrogen", "transport"
]

NZP = "Poland 2050 Net-Zero Pathways scenario"


# ---------------------------------------------------------------------------
# I/O and weighting helpers
# ---------------------------------------------------------------------------

def is_run_dir(path: Path) -> bool:
    return path.is_dir() and all(
        (path / f).exists() for f in ["generators.csv", "buses.csv", "carriers.csv"]
    )


def find_run_dirs(runs_root: Path) -> list[Path]:
    return sorted([p for p in runs_root.iterdir() if is_run_dir(p)], key=lambda p: p.name)


def read_ts(run_dir: Path, stem: str) -> pd.DataFrame:
    path = run_dir / f"{stem}.csv"
    if not path.exists():
        raise FileNotFoundError(str(path))
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index, errors="coerce")
    if df.index.isna().any():
        df = df.loc[~df.index.isna()].copy()
    return df


def try_read_ts(run_dir: Path, stems: list[str]) -> tuple[str | None, pd.DataFrame | None]:
    for s in stems:
        try:
            return s, read_ts(run_dir, s)
        except FileNotFoundError:
            continue
    return None, None


def read_snapshot_weights(run_dir: Path, index: pd.DatetimeIndex) -> pd.Series:
    path = run_dir / "snapshot_weightings.csv"
    if not path.exists():
        return pd.Series(1.0, index=index)
    w = pd.read_csv(path, index_col=0)
    w.index = pd.to_datetime(w.index, errors="coerce")
    if w.index.isna().any():
        w = w.loc[~w.index.isna()].copy()
    for col in ["generators", "objective", "stores"]:
        if col in w.columns:
            return pd.to_numeric(w[col], errors="coerce").fillna(1.0).reindex(index).fillna(1.0)
    return pd.Series(1.0, index=index)


def wsum(df: pd.DataFrame, weights: pd.Series) -> pd.Series:
    return df.mul(weights.reindex(df.index).fillna(1.0), axis=0).sum(axis=0)


def wmean(df: pd.DataFrame, weights: pd.Series) -> pd.Series:
    w = weights.reindex(df.index).fillna(1.0)
    d = float(w.sum())
    return df.mul(w, axis=0).sum(axis=0) / d if d > 0 else pd.Series(0.0, index=df.columns)


def choose_cap(df: pd.DataFrame) -> str:
    for c in ["p_nom_opt", "p_nom"]:
        if c in df.columns:
            return c
    raise ValueError("No p_nom_opt or p_nom found.")


def sanitize(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(pd.to_numeric, errors="coerce")


def region_bus_mask(bus_series: pd.Series) -> pd.Series:
    return bus_series.astype(str).str.match(r"^PL\s+[A-Z]{2}$")


def electric_interregional_mask(links: pd.DataFrame) -> pd.Series:
    if "bus0" not in links.columns or "bus1" not in links.columns:
        return pd.Series(False, index=links.index)

    b0 = links["bus0"].astype(str)
    b1 = links["bus1"].astype(str)

    rr = region_bus_mask(b0) & region_bus_mask(b1)

    carrier = (
        links["carrier"].astype(str).str.lower()
        if "carrier" in links.columns
        else pd.Series("", index=links.index)
    )

    idx_str = pd.Series(links.index.astype(str), index=links.index).str.lower()

    excl = (
        carrier.str.contains("hydrogen|heat|transport", na=False)
        | idx_str.str.contains("hydrogen|heat|transport", na=False)
        | b0.str.lower().str.contains("_hydrogen|_heat|_transport", na=False)
        | b1.str.lower().str.contains("_hydrogen|_heat|_transport", na=False)
    )

    return pd.Series(rr, index=links.index) & ~pd.Series(excl, index=links.index)


# ---------------------------------------------------------------------------
# Run metadata helpers
# ---------------------------------------------------------------------------

def _from_name(name: str) -> dict:
    out = {"year": None, "step_hr": None, "solve_label": None, "runtime_s": None}
    for pat, key in [
        (r"run_(\d{4})",                           "year"),
        (r"_(\d+)hr_",                             "step_hr"),
        (r"_(\d+)s(?:_|$)",                        "runtime_s"),
    ]:
        m = re.search(pat, name)
        if m:
            out[key] = int(m.group(1))
    m = re.search(r"_(Optimal|Suboptimal|Infeasible|Unbounded)(?:_|$)", name, re.I)
    if m:
        out["solve_label"] = m.group(1).title()
    return out


def read_run_meta(run_dir: Path) -> dict:
    fm: dict = {}
    p = run_dir / "run_metadata.json"
    if p.exists():
        try:
            with open(p, encoding="utf-8") as f:
                fm = json.load(f)
        except Exception:
            pass
    fn = _from_name(run_dir.name)
    year  = fm.get("year",           fn["year"])
    step  = fm.get("stepsize",        fn["step_hr"])
    rt    = fm.get("elapsed_seconds", fn["runtime_s"])
    term  = fm.get("termination_condition")
    stat  = fm.get("solver_status")
    label = fn["solve_label"]
    if label is None:
        label = (term or stat or "Unknown").strip().title()
    return {"run_name": run_dir.name, "year": year, "step_hr": step,
            "solve_label": label, "runtime_s": rt}


def _get_objective(run_dir: Path) -> float | None:
    p_net = run_dir / "network.csv"
    if not p_net.exists():
        return None

    try:
        net = pd.read_csv(p_net)
        if net.empty:
            return None

        obj = pd.to_numeric(net["objective"], errors="coerce").iloc[0]
        obj_const = pd.to_numeric(net["objective_constant"], errors="coerce").iloc[0]

        if pd.notna(obj) and pd.notna(obj_const):
            return float(obj + obj_const)
        if pd.notna(obj):
            return float(obj)
    except Exception:
        pass

    return None



def _classify_load(name: str) -> str:
    s = str(name)
    if s.endswith("_high_temp_heat"): return "high_temp_heat"
    if s.endswith("_heat"):           return "heat"
    if "_hydrogen" in s:              return "hydrogen"
    if "transport" in s.lower():      return "transport"
    return "electricity_or_other"


# ---------------------------------------------------------------------------
# Per-run data extraction
# ---------------------------------------------------------------------------

def extract_run(run_dir: Path) -> dict:
    meta = read_run_meta(run_dir)
    row: dict = {**meta, "objective": _get_objective(run_dir)}

    gens    = pd.read_csv(run_dir / "generators.csv")
    links   = pd.read_csv(run_dir / "links.csv")   if (run_dir / "links.csv").exists()   else None
    storage = pd.read_csv(run_dir / "storage_units.csv") \
              if (run_dir / "storage_units.csv").exists() else None
    loads   = pd.read_csv(run_dir / "loads.csv")   if (run_dir / "loads.csv").exists()   else None

    for df in [gens, links, storage, loads]:
        if df is not None and "name" in df.columns:
            df.set_index("name", inplace=True)

    # Loads
    _, loads_p = try_read_ts(run_dir, ["loads-p_set", "loads-p"])
    if loads_p is not None and loads is not None:
        w = read_snapshot_weights(run_dir, loads_p.index)
        ts = sanitize(loads_p).fillna(0.0).sum(axis=1)
        row["peak_load_mw"]          = float(ts.max())
        row["total_annual_load_mwh"] = float((ts * w.reindex(ts.index).fillna(1.0)).sum())
        common = [c for c in loads_p.columns if c in loads.index]
        if common:
            meta_l = loads.loc[common].copy()
            meta_l["class"] = meta_l.index.to_series().apply(_classify_load)
            annual = wsum(sanitize(loads_p[common]).fillna(0.0), w)
            by_cls = annual.groupby(meta_l["class"]).sum()
            for cls in SECTOR_CLASSES:
                row[f"load_{cls}_mwh"] = float(by_cls.get(cls, 0.0))
    else:
        row["peak_load_mw"] = row["total_annual_load_mwh"] = np.nan
        for cls in SECTOR_CLASSES:
            row[f"load_{cls}_mwh"] = np.nan

    # Generation and curtailment
    _, gen_p    = try_read_ts(run_dir, ["generators-p", "generators-p_set"])
    _, gen_pmax = try_read_ts(run_dir, ["generators-p_max_pu"])

    if gens is not None:
        cap_col = choose_cap(gens)
        gens[cap_col] = pd.to_numeric(gens[cap_col], errors="coerce").fillna(0.0)

        if "carrier" in gens.columns:
            for carrier, mw in gens.groupby("carrier")[cap_col].sum().items():
                row[f"cap_gen_{carrier}_mw"] = float(mw)

        if gen_p is not None and "carrier" in gens.columns:
            w = read_snapshot_weights(run_dir, gen_p.index)
            common = [c for c in gen_p.columns if c in gens.index]
            if common:
                annual = wsum(sanitize(gen_p[common]).fillna(0.0), w)
                for carrier, mwh in annual.groupby(gens.loc[common, "carrier"]).sum().items():
                    row[f"gen_{carrier}_mwh"] = float(mwh)
                row["total_generation_mwh"] = float(annual.sum())

        if gen_p is not None and gen_pmax is not None and "carrier" in gens.columns:
            w = read_snapshot_weights(run_dir, gen_p.index)
            cv = [c for c in gen_p.columns
                  if c in gens.index and c in gen_pmax.columns
                  and str(gens.loc[c, "carrier"]) in VRES_CARRIERS]
            if cv:
                cap_v    = pd.to_numeric(gens.loc[cv, cap_col], errors="coerce").fillna(0.0)
                dispatch = sanitize(gen_p[cv]).fillna(0.0).clip(lower=0.0)
                avail    = sanitize(gen_pmax[cv]).fillna(0.0).clip(lower=0.0).multiply(cap_v, axis=1)
                curtail  = (avail - dispatch).clip(lower=0.0)
                a_mwh    = wsum(avail,   w).sum()
                c_mwh    = wsum(curtail, w).sum()
                row["curtailment_total_mwh"] = float(c_mwh)
                row["curtailment_share_pu"]  = float(c_mwh / a_mwh) if a_mwh > 0 else 0.0
                for car in VRES_CARRIERS:
                    mask = gens.loc[cv, "carrier"] == car
                    cm = float(wsum(curtail, w)[mask.index[mask]].sum())
                    am = float(wsum(avail,   w)[mask.index[mask]].sum())
                    tag = car.replace(" ", "_")
                    row[f"curtailment_{tag}_mwh"]   = cm
                    row[f"curtailment_{tag}_share"] = (cm / am) if am > 0 else 0.0

    # ---- storage ------------------------------------------------------------
    if storage is not None:
        cap_col_s = choose_cap(storage)
        storage[cap_col_s] = pd.to_numeric(storage[cap_col_s], errors="coerce").fillna(0.0)
        storage["max_hours"] = pd.to_numeric(
            storage.get("max_hours", pd.Series(0.0, index=storage.index)),
            errors="coerce"
        ).fillna(0.0)
        storage["energy_mwh"] = storage[cap_col_s] * storage["max_hours"]
        if "carrier" in storage.columns:
            for carrier, grp in storage.groupby("carrier"):
                tag = str(carrier).replace(" ", "_")
                row[f"storage_power_{tag}_mw"]  = float(grp[cap_col_s].sum())
                row[f"storage_energy_{tag}_mwh"] = float(grp["energy_mwh"].sum())

    # ---- links --------------------------------------------------------------
    if links is not None:
        cap_col_l = choose_cap(links)
        links[cap_col_l] = pd.to_numeric(links[cap_col_l], errors="coerce").fillna(0.0)

        idx_str = pd.Series(links.index.astype(str), index=links.index)

        ely_mask = idx_str.str.endswith("_electrolyzer")
        hp_mask = (
            idx_str.str.endswith("_heat_pump")
            | (
                links["carrier"].astype(str) == "heat_pump"
                if "carrier" in links.columns
                else pd.Series(False, index=links.index)
            )
        )
        h2_pipe_mask = (
            idx_str.str.contains("hydrogen", case=False, na=False)
            & ~ely_mask
            & ~idx_str.str.contains("chp", case=False, na=False)
        )
        chp_mask = idx_str.str.contains("chp_hydrogen", case=False, na=False)
        el_mask = electric_interregional_mask(links)

        row["electrolyser_total_mw"] = float(links.loc[ely_mask, cap_col_l].sum())
        row["heat_pump_total_mw"] = float(links.loc[hp_mask, cap_col_l].sum())
        row["h2_pipeline_total_mw"] = float(links.loc[h2_pipe_mask, cap_col_l].sum())
        row["chp_h2_total_mw"] = float(links.loc[chp_mask, cap_col_l].sum())
        row["elec_transmission_total_mw"] = float(links.loc[el_mask, cap_col_l].sum())

        _, links_p0 = try_read_ts(run_dir, ["links-p0"])
        if links_p0 is not None:
            el_names = [c for c in links_p0.columns if c in links.index and bool(el_mask.loc[c])]
            if el_names:
                w = read_snapshot_weights(run_dir, links_p0.index)
                cap = links.loc[el_names, cap_col_l].replace(0.0, np.nan)
                fl = sanitize(links_p0[el_names]).fillna(0.0)
                util = fl.abs().divide(cap, axis=1).replace([np.inf, -np.inf], np.nan)
                row["transmission_mean_utilisation"] = float(wmean(util.fillna(0.0), w).mean())
                row["transmission_peak_utilisation"] = float(util.max().max())

            h2_names = [c for c in links_p0.columns if c in links.index and bool(h2_pipe_mask.loc[c])]
            if h2_names:
                w = read_snapshot_weights(run_dir, links_p0.index)
                row["h2_network_abs_flow_mwh"] = float(
                    wsum(sanitize(links_p0[h2_names]).fillna(0.0).abs(), w).sum()
                )

    return row


# cross-run helpers

def _wide_by_year(records: list[dict], prefix: str, suffix: str = "") -> pd.DataFrame:
    rows = []
    for r in records:
        year = r.get("year")
        if year is None:
            continue
        entry = {"year": year}
        for k, v in r.items():
            if k.startswith(prefix) and (not suffix or k.endswith(suffix)):
                entry[k[len(prefix):(-len(suffix) if suffix else None)]] = v
        rows.append(entry)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("year").sort_index()


# comparison plots

def make_all_plots(
    records: list[dict],
    out_dir: Path,
    top_carriers: int,
    inputs_df: pd.DataFrame | None,
) -> None:
    summary = pd.DataFrame(records).sort_values("year").set_index("year")

    # 1. Objective
    if "objective" in summary.columns:
        obj = summary["objective"].dropna() / 1e9
        bar(obj, out_dir / "objective_by_year.png",
            "System cost by weather year",
            "Total system cost", "bn €",
            subtitle=NZP,
            color=BLUE,
            ref_line=float(obj.mean()),
            ref_label=f"Mean: {obj.mean():.1f} bn €")

    # 2. Load totals
    bar(summary["peak_load_mw"].dropna() / 1e3,
        out_dir / "peak_load_by_year.png",
        "Peak system load by weather year",
        "Peak load", "GW", color=BLUE, subtitle=NZP)

    bar(summary["total_annual_load_mwh"].dropna() / 1e6,
        out_dir / "total_load_by_year.png",
        "Total annual system load by weather year",
        "Total load", "TWh", color=BLUE, subtitle=NZP)

    # 3. Load by sector
    load_wide = _wide_by_year(records, "load_", "_mwh")
    if not load_wide.empty:
        stacked_bar(
            load_wide / 1e6,
            out_dir / "load_by_sector_stacked.png",
            "Annual demand by sector across weather years",
            "Demand", "TWh",
            top_n=len(SECTOR_CLASSES),
            subtitle=NZP,
            color_map={
                "electricity_or_other": CARRIER_COLORS["electricity_or_other"],
                "heat":                 CARRIER_COLORS["heat"],
                "high_temp_heat":       CARRIER_COLORS["high_temp_heat"],
                "hydrogen":             CARRIER_COLORS["hydrogen"],
                "transport":            CARRIER_COLORS["transport"],
            },
        )
        line(
            load_wide / 1e6,
            out_dir / "load_by_sector_lines.png",
            "Annual demand by sector across weather years",
            "Demand", "TWh",
            subtitle=NZP,
            color_map={
                "electricity_or_other": CARRIER_COLORS["electricity_or_other"],
                "heat":                 CARRIER_COLORS["heat"],
                "high_temp_heat":       CARRIER_COLORS["high_temp_heat"],
                "hydrogen":             CARRIER_COLORS["hydrogen"],
                "transport":            CARRIER_COLORS["transport"],
            },
        )

    # 4. Generation mix
    gen_wide = _wide_by_year(records, "gen_", "_mwh")
    if not gen_wide.empty:
        stacked_bar(
            gen_wide / 1e6,
            out_dir / "generation_mix_stacked.png",
            "Annual generation by carrier across weather years",
            "Generation", "TWh",
            top_n=top_carriers, subtitle=NZP,
        )

    # 5. Installed generation capacity
    cap_wide = _wide_by_year(records, "cap_gen_", "_mw")
    if not cap_wide.empty:
        stacked_bar(
            cap_wide / 1e3,
            out_dir / "installed_gen_capacity_stacked.png",
            "Installed generation capacity by carrier across weather years",
            "Capacity", "GW",
            top_n=top_carriers, subtitle=NZP,
        )

    # 6. VRES curtailment
    if "curtailment_total_mwh" in summary.columns:
        bar(summary["curtailment_total_mwh"].dropna() / 1e6,
            out_dir / "curtailment_total_twh.png",
            "Total VRES curtailment by weather year",
            "Curtailed energy", "TWh",
            subtitle=NZP, color=AMBER)

        bar(summary["curtailment_share_pu"].dropna() * 100,
            out_dir / "curtailment_share_pct.png",
            "VRES curtailment share by weather year",
            "Curtailment share", "%",
            subtitle=NZP, color=AMBER)

    curt_share_cols = {
        c: c.replace("curtailment_", "").replace("_share", "").replace("_", " ")
        for c in summary.columns if c.startswith("curtailment_") and c.endswith("_share")
    }
    if curt_share_cols:
        line(
            summary[list(curt_share_cols)].dropna() * 100,
            out_dir / "curtailment_share_by_carrier.png",
            "VRES curtailment share by carrier across weather years",
            "Curtailment share", "%",
            subtitle=NZP,
            cols=list(curt_share_cols),
            color_map={c: CARRIER_COLORS.get(v, BLUE) for c, v in curt_share_cols.items()},
        )

    # 7. Storage power + energy
    stor_mw  = _wide_by_year(records, "storage_power_",  "_mw")
    stor_mwh = _wide_by_year(records, "storage_energy_", "_mwh")
    if not stor_mw.empty:
        stacked_bar(stor_mw / 1e3, out_dir / "storage_power_stacked.png",
                    "Installed storage power capacity by carrier across weather years",
                    "Capacity", "GW", top_n=top_carriers, subtitle=NZP)
    if not stor_mwh.empty:
        stacked_bar(stor_mwh / 1e6, out_dir / "storage_energy_stacked.png",
                    "Installed storage energy capacity by carrier across weather years",
                    "Energy", "TWh", top_n=top_carriers, subtitle=NZP)

    # 8. Sector coupling capacities
    for col, title, color in [
        ("electrolyser_total_mw",  "Total electrolyser capacity",    CARRIER_COLORS["hydrogen"]),
        ("heat_pump_total_mw",     "Total heat pump capacity",        CARRIER_COLORS["heat_pump"]),
        ("h2_pipeline_total_mw",   "Total H₂ pipeline capacity",      CARRIER_COLORS["hydrogen storage"]),
        ("chp_h2_total_mw",        "Total H₂ CHP plant capacity",     CARRIER_COLORS["high_temp_heat"]),
    ]:
        if col in summary.columns:
            bar(summary[col].dropna() / 1e3,
                out_dir / f"{col.replace('_mw','')}_by_year.png",
                f"{title} by weather year",
                "Capacity", "GW",
                subtitle=NZP, color=color)

    # 9. Transmission
    if "elec_transmission_total_mw" in summary.columns:
        bar(summary["elec_transmission_total_mw"].dropna() / 1e3,
            out_dir / "transmission_total_gw.png",
            "Total electric interregional capacity by weather year",
            "Capacity", "GW", color=BLUE, subtitle=NZP)

    if "transmission_mean_utilisation" in summary.columns:
        bar(summary["transmission_mean_utilisation"].dropna() * 100,
            out_dir / "transmission_mean_utilisation.png",
            "Mean transmission line utilisation by weather year",
            "Utilisation", "%",
            color=BLUE, subtitle=NZP,
            ref_line=80, ref_label="80% threshold")

    # 10. H2 network throughput
    if "h2_network_abs_flow_mwh" in summary.columns:
        bar(summary["h2_network_abs_flow_mwh"].dropna() / 1e6,
            out_dir / "h2_network_throughput_twh.png",
            "Total H₂ network throughput by weather year",
            "Throughput", "TWh",
            color=CARRIER_COLORS["hydrogen"], subtitle=NZP)

    # 11. Stress correlation plots
    if inputs_df is not None:
        inp = inputs_df.set_index("year") if "year" in inputs_df.columns else inputs_df
        inp.index  = inp.index.astype(int)
        summary.index = summary.index.astype(int)

        stress_dir = out_dir / "stress_correlations"
        stress_dir.mkdir(exist_ok=True)

        pairs = [
            ("cf_vres_combined_annual",   "curtailment_share_pu",
             "Combined VRES CF (p.u.)",   "Curtailment share (p.u.)"),
            ("cf_vres_combined_annual",   "objective",
             "Combined VRES CF (p.u.)",   "System cost (bn €)"),
            ("elec_for_heat_annual_mwh",  "heat_pump_total_mw",
             "Electricity for heat (TWh)","Heat pump capacity (GW)"),
            ("elec_for_heat_annual_mwh",  "objective",
             "Electricity for heat (TWh)","System cost (bn €)"),
            ("wind_drought_max_hours",    "storage_energy_hydrogen_storage_mwh",
             "Wind drought (hours)",      "H₂ storage energy (TWh)"),
            ("wind_drought_max_hours",    "electrolyser_total_mw",
             "Wind drought (hours)",      "Electrolyser capacity (GW)"),
            ("stress_score",              "objective",
             "Composite stress score",    "System cost (bn €)"),
            ("cold_stress_hours_cop_lt2", "heat_pump_total_mw",
             "Cold-stress hours (COP<2)", "Heat pump capacity (GW)"),
        ]

        for inp_col, out_col, xlabel, ylabel in pairs:
            if inp_col not in inp.columns or out_col not in summary.columns:
                continue

            x = inp[inp_col].copy()
            y = summary[out_col].copy()

            # Scale units for readability
            if "mwh" in inp_col:
                x = x / 1e6
            if out_col == "objective":
                y = y / 1e9
            elif "_mw" in out_col and "utilisation" not in out_col:
                y = y / 1e3
            elif "energy_" in out_col and "_mwh" in out_col:
                y = y / 1e6

            fname = f"corr_{inp_col}_vs_{out_col}.png"
            scatter_annotated(
                x, y,
                stress_dir / fname,
                xlabel=xlabel, ylabel=ylabel,
                title=f"{ylabel} vs {xlabel}",
                subtitle=NZP,
                trend=True,
            )

        # Ranked table joining input stress + output metrics
        common_years = summary.index.intersection(inp.index)
        if len(common_years) >= 2:
            merge_cols_out = [c for c in
                ["solve_label", "objective", "curtailment_share_pu",
                 "electrolyser_total_mw", "heat_pump_total_mw",
                 "storage_energy_hydrogen_storage_mwh"]
                if c in summary.columns]
            merge_cols_inp = [c for c in
                ["stress_score", "cf_vres_combined_annual",
                 "elec_for_heat_annual_mwh", "wind_drought_max_hours"]
                if c in inp.columns]

            merged = (
                summary.loc[common_years, merge_cols_out]
                       .join(inp.loc[common_years, merge_cols_inp], how="inner")
                       .sort_values("stress_score", ascending=False)
            )

            for col in merged.columns:
                if col != "solve_label":
                    merged[col] = pd.to_numeric(merged[col], errors="coerce")

            # Display in human-readable units
            disp = merged.copy()

            numeric_cols = [
                "objective",
                "curtailment_share_pu",
                "electrolyser_total_mw",
                "heat_pump_total_mw",
                "storage_energy_hydrogen_storage_mwh",
                "elec_for_heat_annual_mwh",
                "stress_score",
                "cf_vres_combined_annual",
                "wind_drought_max_hours",
            ]
            for col in numeric_cols:
                if col in disp.columns:
                    disp[col] = pd.to_numeric(disp[col], errors="coerce")

            if "objective" in disp.columns:
                disp["objective"] = (disp["objective"] / 1e9).round(1)
            if "curtailment_share_pu" in disp.columns:
                disp["curtailment_share_pu"] = (disp["curtailment_share_pu"] * 100).round(1)
            for col in ["electrolyser_total_mw", "heat_pump_total_mw"]:
                if col in disp.columns:
                    disp[col] = (disp[col] / 1e3).round(1)
            if "storage_energy_hydrogen_storage_mwh" in disp.columns:
                disp["storage_energy_hydrogen_storage_mwh"] = (
                    disp["storage_energy_hydrogen_storage_mwh"] / 1e6
                ).round(1)
            if "elec_for_heat_annual_mwh" in disp.columns:
                disp["elec_for_heat_annual_mwh"] = (
                    disp["elec_for_heat_annual_mwh"] / 1e6
                ).round(1)

            disp = disp.rename(columns={
                "objective":                           "Cost (bn €)",
                "curtailment_share_pu":                "Curtailment (%)",
                "electrolyser_total_mw":               "Electrolyser (GW)",
                "heat_pump_total_mw":                  "Heat pump (GW)",
                "storage_energy_hydrogen_storage_mwh": "H₂ storage (TWh)",
                "stress_score":                        "Stress score",
                "cf_vres_combined_annual":             "VRES CF",
                "elec_for_heat_annual_mwh":            "Elec-for-heat (TWh)",
                "wind_drought_max_hours":              "Wind drought (h)",
            })

            disp.to_csv(stress_dir / "runs_ranked_by_stress.csv")
            ranking_table(
                disp.drop(columns=["solve_label"], errors="ignore"),
                stress_dir / "runs_ranked_by_stress_table.png",
                "Weather years ranked by system stress",
                subtitle="Red = most stressful · Green = least stressful",
                highlight_top=3, highlight_bottom=3,
            )

            print(f"\nTop 5 most stressful runs:")
            print(disp.head(5).to_string())


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compare PyPSA weather-year runs across all output dimensions."
    )
    ap.add_argument("--runs_root",    required=True, type=str)
    ap.add_argument("--out_dir",      default=None,  type=str)
    ap.add_argument("--inputs_csv",   default=None,  type=str,
                    help="Path to weather_year_inputs_summary.csv for correlation plots")
    ap.add_argument("--top_carriers", default=8,     type=int)
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    if not runs_root.exists():
        raise FileNotFoundError(f"runs_root not found: {runs_root}")

    out_dir = Path(args.out_dir) if args.out_dir else runs_root / "weather_year_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = find_run_dirs(runs_root)
    if not run_dirs:
        raise FileNotFoundError(f"No valid run directories found in: {runs_root}")

    print(f"Found {len(run_dirs)} run directories.")

    records, failures = [], []
    for i, run_dir in enumerate(run_dirs, 1):
        print(f"[{i}/{len(run_dirs)}] {run_dir.name}")
        try:
            records.append(extract_run(run_dir))
        except Exception as e:
            failures.append({"run_name": run_dir.name, "error": str(e)})
            print(f"  FAILED: {e}")

    if not records:
        print("No runs processed. Exiting.")
        return

    summary_df = pd.DataFrame(records).sort_values(["year", "run_name"])
    summary_df.to_csv(out_dir / "all_runs_summary.csv", index=False)
    if failures:
        pd.DataFrame(failures).to_csv(out_dir / "failures.csv", index=False)

    inputs_df = None
    if args.inputs_csv:
        p = Path(args.inputs_csv)
        if p.exists():
            inputs_df = pd.read_csv(p)
            print(f"Loaded input characterisation: {len(inputs_df)} years.")
        else:
            print(f"Warning: --inputs_csv path not found: {p}")

    make_all_plots(records, out_dir, args.top_carriers, inputs_df)

    print(f"\nAll outputs saved to: {out_dir}")
    if failures:
        print(f"{len(failures)} run(s) failed — see failures.csv")


if __name__ == "__main__":
    main()
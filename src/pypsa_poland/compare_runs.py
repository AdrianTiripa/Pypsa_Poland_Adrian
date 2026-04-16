# compare_runs.py
#
# Cross-run comparison suite for pypsa-poland weather-year runs.
#
# Reads all valid run folders found under a root directory, extracts key scalar
# summaries (total load, generation, installed capacity, objective, solve status),
# and produces a set of comparison plots and CSV exports covering:
#   - Solve status and runtime across weather years.
#   - Total and sector-split annual demand by year.
#   - Annual generation mix (stacked by carrier) by year.
#   - Installed capacity mix (stacked by carrier) by year.
#   - System cost by year with mean reference line.
#   - Scatter plot of heat demand vs system cost.
#
# Usage:
#   python compare_runs.py --runs_root <folder>
#   python compare_runs.py --runs_root <folder> --out_dir <output_path>
#   python compare_runs.py --runs_root <folder> --top_carriers 10

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Shared plot style
# ---------------------------------------------------------------------------
from plot_style import (
    apply_style,
    CARRIER_COLORS, BLUE, RED, AMBER,
    bar, stacked_bar, line, scatter_annotated,
    savefig,
)

apply_style()


# ---------------------------------------------------------------------------
# I/O and weighting helpers
# ---------------------------------------------------------------------------

def is_run_dir(path: Path) -> bool:
    required = ["generators.csv", "buses.csv", "carriers.csv"]
    return path.is_dir() and all((path / f).exists() for f in required)


def find_run_dirs(runs_root: Path) -> list[Path]:
    return sorted([p for p in runs_root.iterdir() if is_run_dir(p)],
                  key=lambda p: p.name)


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


def weighted_time_sum(df: pd.DataFrame, weights: pd.Series) -> pd.Series:
    return df.mul(weights.reindex(df.index).fillna(1.0), axis=0).sum(axis=0)


def choose_capacity_column(df: pd.DataFrame) -> str:
    for c in ["p_nom_opt", "p_nom"]:
        if c in df.columns:
            return c
    raise ValueError("No p_nom_opt or p_nom found.")


def classify_load(name: str) -> str:
    """Classify a load name into a sector for stacked demand plots."""
    s = str(name)
    if s.endswith("_high_temp_heat"):  return "high_temp_heat"
    if s.endswith("_heat"):            return "heat"
    if "_hydrogen" in s:               return "hydrogen"
    if "transport" in s.lower():       return "transport"
    return "electricity_or_other"


# ---------------------------------------------------------------------------
# Run metadata helpers
# ---------------------------------------------------------------------------

def _extract_from_name(name: str) -> dict:
    """Extract year, step size, solve label, and runtime from the run folder name."""
    out = {"year": None, "step_hr": None, "solve_label": None, "runtime_s": None}
    m = re.search(r"run_(\d{4})", name)
    if m:
        out["year"] = int(m.group(1))
    m = re.search(r"_(\d+)hr_", name)
    if m:
        out["step_hr"] = int(m.group(1))
    m = re.search(r"_(Optimal|Suboptimal|Infeasible|Unbounded)(?:_|$)", name, re.I)
    if m:
        out["solve_label"] = m.group(1).title()
    m = re.search(r"_(\d+)s(?:_|$)", name)
    if m:
        out["runtime_s"] = int(m.group(1))
    return out


def read_run_metadata(run_dir: Path) -> dict:
    """
    Read run metadata from run_metadata.json, falling back to folder-name parsing.

    Folder-name parsing is used when the JSON file is absent or incomplete,
    so the comparison suite works even on runs with partial metadata.
    """
    file_meta: dict = {}
    path = run_dir / "run_metadata.json"
    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                file_meta = json.load(f)
        except Exception:
            pass

    fn = _extract_from_name(run_dir.name)
    year      = file_meta.get("year",           fn["year"])
    step_hr   = file_meta.get("stepsize",        fn["step_hr"])
    runtime_s = file_meta.get("elapsed_seconds", fn["runtime_s"])
    term      = file_meta.get("termination_condition")
    status    = file_meta.get("solver_status")

    solve_label = fn["solve_label"]
    if solve_label is None:
        if isinstance(term, str) and term.strip():
            solve_label = term.strip().title()
        elif isinstance(status, str) and status.strip():
            solve_label = status.strip().title()
        else:
            solve_label = "Unknown"

    return {
        "run_name":    run_dir.name,
        "year":        year,
        "step_hr":     step_hr,
        "solve_label": solve_label,
        "runtime_s":   runtime_s,
    }


def get_objective(run_dir: Path) -> float | None:
    """Read the optimisation objective value from run_metadata.json or objective.txt."""
    path = run_dir / "run_metadata.json"
    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                m = json.load(f)
            for k in ["objective", "objective_value"]:
                if k in m:
                    return float(m[k])
        except Exception:
            pass
    txt = run_dir / "objective.txt"
    if txt.exists():
        try:
            raw = txt.read_text(encoding="utf-8")
            match = re.search(r"([-+]?\d+(\.\d+)?([eE][-+]?\d+)?)", raw)
            if match:
                return float(match.group(1))
        except Exception:
            pass
    return None


# ---------------------------------------------------------------------------
# Per-run summary extraction
# ---------------------------------------------------------------------------

def summarize_run(
    run_dir: Path,
) -> tuple[dict, pd.Series, pd.Series, pd.Series]:
    """
    Extract scalar and per-carrier summaries from a single run directory.

    Returns a tuple of (summary_dict, load_by_class, gen_by_carrier, cap_by_carrier).
    All energy values are in MWh, capacity values in MW.
    """
    meta = read_run_metadata(run_dir)
    summary = {
        "run_name":    meta["run_name"],
        "year":        meta["year"],
        "step_hr":     meta["step_hr"],
        "solve_label": meta["solve_label"],
        "runtime_s":   meta["runtime_s"],
        "objective":   get_objective(run_dir),
    }

    loads = pd.read_csv(run_dir / "loads.csv") if (run_dir / "loads.csv").exists() else None
    gens  = pd.read_csv(run_dir / "generators.csv")

    # loads
    _, loads_p = try_read_ts(run_dir, ["loads-p_set"])
    load_by_class   = pd.Series(dtype=float)
    total_annual    = np.nan
    peak_mw         = np.nan

    if loads is not None and loads_p is not None and "name" in loads.columns:
        weights    = read_snapshot_weights(run_dir, loads_p.index)
        total_ts   = loads_p.sum(axis=1)
        peak_mw    = float(total_ts.max())
        total_annual = float(
            (total_ts * weights.reindex(total_ts.index).fillna(1.0)).sum()
        )
        common = [c for c in loads_p.columns if c in set(loads["name"])]
        if common:
            meta_loads = loads.set_index("name").loc[common].copy()
            meta_loads["class"] = meta_loads.index.to_series().apply(classify_load)
            annual = weighted_time_sum(loads_p[common], weights)
            load_by_class = annual.groupby(meta_loads["class"]).sum().sort_values(ascending=False)

    summary["total_annual_load_mwh"] = total_annual
    summary["peak_load_mw"]          = peak_mw
    for cls in ["electricity_or_other", "heat", "high_temp_heat", "hydrogen", "transport"]:
        summary[f"load_{cls}_mwh"] = float(load_by_class.get(cls, 0.0))

    # generation
    _, gen_p = try_read_ts(run_dir, ["generators-p", "generators-p_set"])
    gen_by_carrier = pd.Series(dtype=float)

    if gen_p is not None and "name" in gens.columns and "carrier" in gens.columns:
        weights   = read_snapshot_weights(run_dir, gen_p.index)
        gens_idx  = gens.set_index("name")
        common    = [c for c in gen_p.columns if c in gens_idx.index]
        if common:
            annual = weighted_time_sum(gen_p[common], weights)
            gen_by_carrier = annual.groupby(gens_idx.loc[common, "carrier"]).sum() \
                                   .sort_values(ascending=False)

    summary["total_generation_mwh"] = float(gen_by_carrier.sum()) if len(gen_by_carrier) else np.nan

    # installed capacity
    cap_col = choose_capacity_column(gens)
    gens[cap_col] = pd.to_numeric(gens[cap_col], errors="coerce").fillna(0.0)
    cap_by_carrier = (
        gens.groupby("carrier")[cap_col].sum().sort_values(ascending=False)
        if "carrier" in gens.columns else pd.Series(dtype=float)
    )

    return summary, load_by_class, gen_by_carrier, cap_by_carrier


# ---------------------------------------------------------------------------
# Cross-run comparison plots
# ---------------------------------------------------------------------------

def _wide(records: list[dict], value_col: str, group_col: str = "carrier_or_class",
          by: str = "year") -> pd.DataFrame:
    """Pivot a list of records into a wide DataFrame indexed by `by` and columned by `group_col`."""
    df = pd.DataFrame(records)
    if df.empty or by not in df.columns:
        return pd.DataFrame()
    return (
        df.pivot_table(index=by, columns=group_col, values=value_col,
                       aggfunc="sum", fill_value=0.0)
          .sort_index()
    )


def make_comparison_plots(
    summary_df: pd.DataFrame,
    gen_df: pd.DataFrame,
    cap_df: pd.DataFrame,
    load_df: pd.DataFrame,
    out_dir: Path,
    top_carriers: int,
) -> None:
    """
    Produce all cross-run comparison figures and save them to out_dir.

    Covers: solve-status counts, runtime, total/sectoral loads, generation mix,
    capacity mix, system cost, and a heat-load vs cost scatter plot.
    """
    # solve status bar
    counts = summary_df["solve_label"].fillna("Unknown").value_counts().sort_index()
    bar(
        counts,
        out_dir / "solve_status_counts.png",
        "Solve status across weather-year runs",
        "Count", "",
        color=BLUE,
        value_labels=True,
        rotate_xticks=False,
    )

    # runtime
    d = summary_df.dropna(subset=["year", "runtime_s"]).sort_values("year")
    if not d.empty:
        line(
            pd.DataFrame({"Runtime": d["runtime_s"].values / 60}, index=d["year"].values),
            out_dir / "runtime_by_year.png",
            "Solver runtime by weather year",
            "Runtime", "minutes",
            markers=True,
            color_map={"Runtime": BLUE},
        )

    # total annual load
    d = summary_df.dropna(subset=["year", "total_annual_load_mwh"]).sort_values("year")
    if not d.empty:
        bar(
            pd.Series(d["total_annual_load_mwh"].values / 1e6, index=d["year"].astype(int)),
            out_dir / "total_annual_load_by_year.png",
            "Total annual system load by weather year",
            "Load", "TWh",
            color=BLUE,
        )

    # heat load
    d = summary_df.dropna(subset=["year", "load_heat_mwh"]).sort_values("year")
    if not d.empty:
        bar(
            pd.Series(d["load_heat_mwh"].values / 1e6, index=d["year"].astype(int)),
            out_dir / "heat_load_by_year.png",
            "Annual heat sector load by weather year",
            "Load", "TWh",
            color=CARRIER_COLORS.get("heat", RED),
        )

    # peak load
    d = summary_df.dropna(subset=["year", "peak_load_mw"]).sort_values("year")
    if not d.empty:
        bar(
            pd.Series(d["peak_load_mw"].values / 1e3, index=d["year"].astype(int)),
            out_dir / "peak_load_by_year.png",
            "Peak system load by weather year",
            "Peak load", "GW",
            color=BLUE,
        )

    # total generation
    d = summary_df.dropna(subset=["year", "total_generation_mwh"]).sort_values("year")
    if not d.empty:
        bar(
            pd.Series(d["total_generation_mwh"].values / 1e6, index=d["year"].astype(int)),
            out_dir / "total_generation_by_year.png",
            "Total annual generation by weather year",
            "Generation", "TWh",
            color=BLUE,
        )

    # objective
    d = summary_df.dropna(subset=["year", "objective"]).sort_values("year")
    if not d.empty:
        obj = pd.Series(d["objective"].values / 1e9, index=d["year"].astype(int))
        mean_obj = float(obj.mean())
        bar(
            obj,
            out_dir / "objective_by_year.png",
            "System cost by weather year",
            "Total system cost", "bn €",
            color=BLUE,
            ref_line=mean_obj,
            ref_label=f"Mean: {mean_obj:.1f} bn €",
        )

    # generation stacked by carrier
    if not gen_df.empty:
        gen_wide = _wide(gen_df.to_dict("records"), "annual_mwh")
        if not gen_wide.empty:
            stacked_bar(
                gen_wide / 1e6,
                out_dir / "generation_by_carrier_over_years.png",
                "Annual generation by carrier across weather years",
                "Generation", "TWh",
                top_n=top_carriers,
            )

    # installed capacity stacked
    if not cap_df.empty:
        cap_wide = _wide(cap_df.to_dict("records"), "capacity_mw")
        if not cap_wide.empty:
            stacked_bar(
                cap_wide / 1e3,
                out_dir / "capacity_by_carrier_over_years.png",
                "Installed generation capacity by carrier across weather years",
                "Capacity", "GW",
                top_n=top_carriers,
            )

    # load by sector stacked
    if not load_df.empty:
        load_wide = _wide(load_df.to_dict("records"), "annual_mwh")
        if not load_wide.empty:
            stacked_bar(
                load_wide / 1e6,
                out_dir / "load_by_class_over_years.png",
                "Annual demand by sector across weather years",
                "Demand", "TWh",
                top_n=10,
                color_map={
                    "electricity_or_other": CARRIER_COLORS["electricity_or_other"],
                    "heat":                 CARRIER_COLORS["heat"],
                    "high_temp_heat":       CARRIER_COLORS["high_temp_heat"],
                    "hydrogen":             CARRIER_COLORS["hydrogen"],
                    "transport":            CARRIER_COLORS["transport"],
                },
            )

    # objective vs heat load scatter (if enough points)
    if len(summary_df) >= 4:
        d = summary_df.dropna(subset=["year", "objective", "load_heat_mwh"]).sort_values("year")
        if not d.empty:
            scatter_annotated(
                pd.Series(d["load_heat_mwh"].values / 1e6, index=d["year"].astype(int)),
                pd.Series(d["objective"].values / 1e9,     index=d["year"].astype(int)),
                out_dir / "scatter_heat_load_vs_objective.png",
                xlabel="Annual heat sector load",
                ylabel="System cost",
                title="System cost vs heat demand by weather year",
                subtitle="Poland 2050 Net-Zero Pathways scenario",
                xunit="TWh",
                yunit="bn €",
                trend=True,
            )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Discover run folders, build summaries, write CSVs, and produce comparison plots."""
    ap = argparse.ArgumentParser(
        description="Cross-run weather-year comparison for pypsa-poland."
    )
    ap.add_argument("--runs_root",    required=True, type=str)
    ap.add_argument("--out_dir",      default=None,  type=str)
    ap.add_argument("--top_carriers", default=8,     type=int)
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    if not runs_root.exists():
        raise FileNotFoundError(f"runs_root not found: {runs_root}")

    out_dir = Path(args.out_dir) if args.out_dir else runs_root / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = find_run_dirs(runs_root)
    if not run_dirs:
        raise FileNotFoundError(f"No run folders found in: {runs_root}")

    print(f"Found {len(run_dirs)} run folders.")

    summary_rows, load_records, gen_records, cap_records = [], [], [], []
    failures = []

    for i, run_dir in enumerate(run_dirs, 1):
        print(f"[{i}/{len(run_dirs)}] {run_dir.name}")
        try:
            summary, load_by_class, gen_by_carrier, cap_by_carrier = summarize_run(run_dir)
            summary_rows.append(summary)
            key = {k: summary[k] for k in ("run_name", "year", "step_hr", "solve_label")}
            for cls, val in load_by_class.items():
                load_records.append({**key, "carrier_or_class": cls, "annual_mwh": float(val)})
            for car, val in gen_by_carrier.items():
                gen_records.append({**key, "carrier_or_class": car, "annual_mwh": float(val)})
            for car, val in cap_by_carrier.items():
                cap_records.append({**key, "carrier_or_class": car, "capacity_mw": float(val)})
        except Exception as e:
            failures.append((run_dir.name, str(e)))
            print(f"  Failed: {e}")

    summary_df = pd.DataFrame(summary_rows).sort_values(["year", "run_name"], na_position="last")
    gen_df     = pd.DataFrame(gen_records)
    cap_df     = pd.DataFrame(cap_records)
    load_df    = pd.DataFrame(load_records)

    summary_df.to_csv(out_dir / "all_runs_summary.csv", index=False)
    gen_df.to_csv(out_dir / "generation_by_carrier.csv", index=False)
    cap_df.to_csv(out_dir / "capacity_by_carrier.csv", index=False)
    load_df.to_csv(out_dir / "load_by_class.csv", index=False)

    if failures:
        pd.DataFrame(failures, columns=["run_name", "error"]).to_csv(
            out_dir / "failures.csv", index=False
        )

    make_comparison_plots(summary_df, gen_df, cap_df, load_df, out_dir, args.top_carriers)

    print(f"\nSaved comparison outputs to: {out_dir}")
    if failures:
        print("Failures:")
        for name, err in failures:
            print(f"  - {name}: {err}")


if __name__ == "__main__":
    main()

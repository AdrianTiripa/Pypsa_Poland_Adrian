from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Usage examples:
#
# 1) Compare all runs inside the runs folder:
# & C:\Users\adria\anaconda3\envs\pypsa-legacy\python.exe C:\Users\adria\MODEL_PyPSA\Core\pypsa-poland_ADRIAN\src\pypsa_poland\compare_runs.py --runs_root C:\Users\adria\MODEL_PyPSA\Core\runs
#
# 2) Compare all runs and write outputs to a custom folder:
# & C:\Users\adria\anaconda3\envs\pypsa-legacy\python.exe C:\Users\adria\MODEL_PyPSA\Core\pypsa-poland_ADRIAN\src\pypsa_poland\compare_runs.py --runs_root C:\Users\adria\MODEL_PyPSA\Core\runs --out_dir C:\Users\adria\MODEL_PyPSA\Core\run_comparison
#
# 3) Compare all runs and keep only top 10 carriers in carrier plots:
# & C:\Users\adria\anaconda3\envs\pypsa-legacy\python.exe C:\Users\adria\MODEL_PyPSA\Core\pypsa-poland_ADRIAN\src\pypsa_poland\compare_runs.py --runs_root C:\Users\adria\MODEL_PyPSA\Core\runs --top_carriers 10


def is_run_dir(path: Path) -> bool:
    required = ["generators.csv", "buses.csv", "carriers.csv"]
    return path.is_dir() and all((path / f).exists() for f in required)


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
            s = pd.to_numeric(w[col], errors="coerce").fillna(1.0)
            return s.reindex(index).fillna(1.0)

    return pd.Series(1.0, index=index)


def weighted_time_sum(df: pd.DataFrame, weights: pd.Series) -> pd.Series:
    w = weights.reindex(df.index).fillna(1.0)
    return df.mul(w, axis=0).sum(axis=0)


def choose_capacity_column(df: pd.DataFrame) -> str:
    for c in ["p_nom_opt", "p_nom"]:
        if c in df.columns:
            return c
    raise ValueError("No p_nom_opt or p_nom found.")


def classify_load(name: str) -> str:
    s = str(name)
    if s.endswith("_high_temp_heat"):
        return "high_temp_heat"
    if s.endswith("_heat"):
        return "heat"
    if "_hydrogen" in s:
        return "hydrogen"
    if "transport" in s.lower():
        return "transport"
    return "electricity_or_other"


def extract_from_folder_name(name: str) -> dict:
    out = {
        "year": None,
        "step_hr": None,
        "solve_label": None,
        "runtime_s": None,
    }

    m_year = re.search(r"run_(\d{4})", name)
    if m_year:
        out["year"] = int(m_year.group(1))

    m_step = re.search(r"_(\d+)hr_", name)
    if m_step:
        out["step_hr"] = int(m_step.group(1))

    m_runtime = re.search(r"_(\d+)s(?:_|$)", name)
    if m_runtime:
        out["runtime_s"] = int(m_runtime.group(1))

    m_label = re.search(r"_(Optimal|Suboptimal|Infeasible|Unbounded|Unknown)(?:_|$)", name, re.IGNORECASE)
    if m_label:
        out["solve_label"] = m_label.group(1).title()

    return out


def read_run_metadata(run_dir: Path) -> dict:
    meta = {}

    path = run_dir / "run_metadata.json"
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            meta = {}

    from_name = extract_from_folder_name(run_dir.name)

    year = meta.get("year", from_name["year"])
    step_hr = meta.get("stepsize", from_name["step_hr"])
    runtime_s = meta.get("elapsed_seconds", from_name["runtime_s"])

    term = meta.get("termination_condition")
    status = meta.get("solver_status")

    solve_label = from_name["solve_label"]
    if solve_label is None:
        if isinstance(term, str) and term.strip():
            solve_label = term.strip().title()
        elif isinstance(status, str) and status.strip():
            solve_label = status.strip().title()
        else:
            solve_label = "Unknown"

    return {
        "run_name": run_dir.name,
        "year": year,
        "step_hr": step_hr,
        "solve_label": solve_label,
        "runtime_s": runtime_s,
    }


def get_objective(run_dir: Path) -> float | None:
    # optional sources
    candidates = [
        run_dir / "run_metadata.json",
        run_dir / "objective.txt",
    ]

    meta_path = run_dir / "run_metadata.json"
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            for k in ["objective", "objective_value"]:
                if k in meta:
                    return float(meta[k])
        except Exception:
            pass

    txt_path = run_dir / "objective.txt"
    if txt_path.exists():
        try:
            txt = txt_path.read_text(encoding="utf-8")
            m = re.search(r"([-+]?\d+(\.\d+)?([eE][-+]?\d+)?)", txt)
            if m:
                return float(m.group(1))
        except Exception:
            pass

    return None


def summarize_run(run_dir: Path) -> tuple[dict, pd.Series, pd.Series, pd.Series]:
    meta = read_run_metadata(run_dir)

    summary = {
        "run_name": meta["run_name"],
        "year": meta["year"],
        "step_hr": meta["step_hr"],
        "solve_label": meta["solve_label"],
        "runtime_s": meta["runtime_s"],
        "objective": get_objective(run_dir),
    }

    loads = pd.read_csv(run_dir / "loads.csv") if (run_dir / "loads.csv").exists() else None
    gens = pd.read_csv(run_dir / "generators.csv")

    # ----- loads -----
    _, loads_p = try_read_ts(run_dir, ["loads-p_set"])
    load_by_class = pd.Series(dtype=float)
    total_annual_load = np.nan
    peak_load_mw = np.nan

    if loads is not None and loads_p is not None and "name" in loads.columns:
        weights = read_snapshot_weights(run_dir, loads_p.index)
        total_load_ts = loads_p.sum(axis=1)
        peak_load_mw = float(total_load_ts.max())
        total_annual_load = float((total_load_ts * weights.reindex(total_load_ts.index).fillna(1.0)).sum())

        common = [c for c in loads_p.columns if c in set(loads["name"])]
        if common:
            meta_loads = loads.set_index("name").loc[common].copy()
            meta_loads["class"] = meta_loads.index.to_series().apply(classify_load)
            annual_by_load = weighted_time_sum(loads_p[common], weights)
            load_by_class = annual_by_load.groupby(meta_loads["class"]).sum().sort_values(ascending=False)

    summary["total_annual_load_mwh"] = total_annual_load
    summary["peak_load_mw"] = peak_load_mw

    for cls in ["electricity_or_other", "heat", "high_temp_heat", "hydrogen", "transport"]:
        summary[f"load_{cls}_mwh"] = float(load_by_class.get(cls, 0.0))

    # ----- generation by carrier -----
    _, gen_p = try_read_ts(run_dir, ["generators-p", "generators-p_set"])
    gen_by_carrier = pd.Series(dtype=float)

    if gen_p is not None and "name" in gens.columns and "carrier" in gens.columns:
        weights = read_snapshot_weights(run_dir, gen_p.index)
        gens_idx = gens.set_index("name")
        common = [c for c in gen_p.columns if c in gens_idx.index]
        if common:
            annual_by_gen = weighted_time_sum(gen_p[common], weights)
            carrier = gens_idx.loc[common, "carrier"]
            gen_by_carrier = annual_by_gen.groupby(carrier).sum().sort_values(ascending=False)

    summary["total_generation_mwh"] = float(gen_by_carrier.sum()) if len(gen_by_carrier) else np.nan

    # ----- installed capacity by carrier -----
    cap_col = choose_capacity_column(gens)
    gens[cap_col] = pd.to_numeric(gens[cap_col], errors="coerce").fillna(0.0)
    if "carrier" in gens.columns:
        cap_by_carrier = gens.groupby("carrier")[cap_col].sum().sort_values(ascending=False)
    else:
        cap_by_carrier = pd.Series(dtype=float)

    return summary, load_by_class, gen_by_carrier, cap_by_carrier


def pivot_long_records(records: list[dict], index_cols: list[str], value_name: str) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=index_cols + ["carrier_or_class", value_name])

    return pd.DataFrame(records)


def save_status_bar(df: pd.DataFrame, out_dir: Path) -> None:
    counts = df["solve_label"].fillna("Unknown").value_counts().sort_index()
    plt.figure(figsize=(8, 4))
    plt.bar(counts.index.astype(str), counts.values)
    plt.title("Run status counts")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_dir / "status_counts.png", dpi=220)
    plt.close()


def save_runtime_plot(df: pd.DataFrame, out_dir: Path) -> None:
    d = df.dropna(subset=["year", "runtime_s"]).sort_values("year")
    if d.empty:
        return
    plt.figure(figsize=(10, 4))
    plt.plot(d["year"], d["runtime_s"], marker="o")
    plt.title("Runtime by year")
    plt.xlabel("Year")
    plt.ylabel("Seconds")
    plt.tight_layout()
    plt.savefig(out_dir / "runtime_by_year.png", dpi=220)
    plt.close()


def save_total_metric_plot(df: pd.DataFrame, out_dir: Path, ycol: str, title: str, ylabel: str, filename: str) -> None:
    d = df.dropna(subset=["year", ycol]).sort_values("year")
    if d.empty:
        return
    plt.figure(figsize=(10, 4))
    plt.plot(d["year"], d[ycol], marker="o")
    plt.title(title)
    plt.xlabel("Year")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=220)
    plt.close()


def save_stacked_area_by_year(
    wide_df: pd.DataFrame,
    out_dir: Path,
    title: str,
    ylabel: str,
    filename: str,
    top_n: int = 8,
) -> None:
    if wide_df.empty:
        return

    wide_df = wide_df.sort_index()
    totals = wide_df.sum(axis=0).sort_values(ascending=False)
    top_cols = list(totals.head(top_n).index)

    plot_df = wide_df[top_cols].copy()
    rest = wide_df.drop(columns=top_cols, errors="ignore")
    if rest.shape[1] > 0:
        plot_df["other"] = rest.sum(axis=1)

    plt.figure(figsize=(12, 5))
    plt.stackplot(plot_df.index, [plot_df[c].values for c in plot_df.columns], labels=plot_df.columns)
    plt.title(title)
    plt.xlabel("Year")
    plt.ylabel(ylabel)
    plt.legend(loc="upper left", ncols=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=220)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", required=True, type=str)
    ap.add_argument("--out_dir", default=None, type=str)
    ap.add_argument("--top_carriers", default=8, type=int)
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    if not runs_root.exists():
        raise FileNotFoundError(f"runs_root not found: {runs_root}")

    out_dir = Path(args.out_dir) if args.out_dir else (runs_root / "comparison")
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = find_run_dirs(runs_root)
    if not run_dirs:
        raise FileNotFoundError(f"No run folders found in: {runs_root}")

    print(f"Found {len(run_dirs)} run folders.")

    summary_rows = []
    load_records = []
    gen_records = []
    cap_records = []
    failures = []

    for i, run_dir in enumerate(run_dirs, start=1):
        print(f"[{i}/{len(run_dirs)}] Reading {run_dir.name}")
        try:
            summary, load_by_class, gen_by_carrier, cap_by_carrier = summarize_run(run_dir)
            summary_rows.append(summary)

            key = {
                "run_name": summary["run_name"],
                "year": summary["year"],
                "step_hr": summary["step_hr"],
                "solve_label": summary["solve_label"],
            }

            for cls, val in load_by_class.items():
                load_records.append({**key, "carrier_or_class": cls, "annual_mwh": float(val)})

            for car, val in gen_by_carrier.items():
                gen_records.append({**key, "carrier_or_class": car, "annual_mwh": float(val)})

            for car, val in cap_by_carrier.items():
                cap_records.append({**key, "carrier_or_class": car, "capacity_mw": float(val)})

        except Exception as e:
            failures.append((run_dir.name, str(e)))
            print(f"Failed on {run_dir.name}: {e}")

    summary_df = pd.DataFrame(summary_rows).sort_values(["year", "run_name"], na_position="last")
    load_df = pd.DataFrame(load_records)
    gen_df = pd.DataFrame(gen_records)
    cap_df = pd.DataFrame(cap_records)

    summary_df.to_csv(out_dir / "all_runs_summary.csv", index=False)
    load_df.to_csv(out_dir / "load_by_class.csv", index=False)
    gen_df.to_csv(out_dir / "generation_by_carrier.csv", index=False)
    cap_df.to_csv(out_dir / "capacity_by_carrier.csv", index=False)

    if failures:
        pd.DataFrame(failures, columns=["run_name", "error"]).to_csv(out_dir / "failures.csv", index=False)

    if not summary_df.empty:
        save_status_bar(summary_df, out_dir)
        save_runtime_plot(summary_df, out_dir)
        save_total_metric_plot(
            summary_df, out_dir,
            "total_annual_load_mwh",
            "Total annual load by year",
            "MWh",
            "total_annual_load_by_year.png",
        )
        save_total_metric_plot(
            summary_df, out_dir,
            "load_heat_mwh",
            "Annual heat load by year",
            "MWh",
            "heat_load_by_year.png",
        )
        save_total_metric_plot(
            summary_df, out_dir,
            "peak_load_mw",
            "Peak load by year",
            "MW",
            "peak_load_by_year.png",
        )
        save_total_metric_plot(
            summary_df, out_dir,
            "total_generation_mwh",
            "Total annual generation by year",
            "MWh",
            "total_generation_by_year.png",
        )

        if "objective" in summary_df.columns and summary_df["objective"].notna().any():
            save_total_metric_plot(
                summary_df, out_dir,
                "objective",
                "Objective by year",
                "Objective",
                "objective_by_year.png",
            )

    if not gen_df.empty:
        gen_wide = (
            gen_df.pivot_table(index="year", columns="carrier_or_class", values="annual_mwh", aggfunc="sum", fill_value=0.0)
            .sort_index()
        )
        save_stacked_area_by_year(
            gen_wide, out_dir,
            "Generation by carrier over years",
            "MWh",
            "generation_by_carrier_over_years.png",
            top_n=args.top_carriers,
        )

    if not cap_df.empty:
        cap_wide = (
            cap_df.pivot_table(index="year", columns="carrier_or_class", values="capacity_mw", aggfunc="sum", fill_value=0.0)
            .sort_index()
        )
        save_stacked_area_by_year(
            cap_wide, out_dir,
            "Installed capacity by carrier over years",
            "MW",
            "capacity_by_carrier_over_years.png",
            top_n=args.top_carriers,
        )

    if not load_df.empty:
        load_wide = (
            load_df.pivot_table(index="year", columns="carrier_or_class", values="annual_mwh", aggfunc="sum", fill_value=0.0)
            .sort_index()
        )
        save_stacked_area_by_year(
            load_wide, out_dir,
            "Load by class over years",
            "MWh",
            "load_by_class_over_years.png",
            top_n=10,
        )

    print(f"\nSaved comparison outputs to: {out_dir}")
    if failures:
        print("\nSome runs failed during summary:")
        for name, err in failures:
            print(f"- {name}: {err}")

if __name__ == "__main__":
    main()
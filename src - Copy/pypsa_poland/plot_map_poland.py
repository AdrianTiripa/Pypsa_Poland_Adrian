# src/pypsa_poland/hydrogen_plots.py

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Usage examples
#
# 1) One run:
# & C:\Users\adria\anaconda3\envs\pypsa-legacy\python.exe C:\Users\adria\MODEL_PyPSA\Core\pypsa-poland_ADRIAN\src\pypsa_poland\hydrogen_plots.py --run_dir C:\Users\adria\MODEL_PyPSA\Core\runs\run_2025_20260322_060526_3hr_Optimal_1401s
#
# 2) One run, custom output folder:
# & C:\Users\adria\anaconda3\envs\pypsa-legacy\python.exe C:\Users\adria\MODEL_PyPSA\Core\pypsa-poland_ADRIAN\src\pypsa_poland\hydrogen_plots.py --run_dir C:\Users\adria\MODEL_PyPSA\Core\runs\run_2025_20260322_060526_3hr_Optimal_1401s --out_dir C:\Users\adria\MODEL_PyPSA\Core\my_figures
#
# 3) All runs:
# & C:\Users\adria\anaconda3\envs\pypsa-legacy\python.exe C:\Users\adria\MODEL_PyPSA\Core\pypsa-poland_ADRIAN\src\pypsa_poland\hydrogen_plots.py --runs_root C:\Users\adria\MODEL_PyPSA\Core\runs
#
# 4) All runs, keep more lines in top-flow plots:
# & C:\Users\adria\anaconda3\envs\pypsa-legacy\python.exe C:\Users\adria\MODEL_PyPSA\Core\pypsa-poland_ADRIAN\src\pypsa_poland\hydrogen_plots.py --runs_root C:\Users\adria\MODEL_PyPSA\Core\runs --top_n 12
#
# Notes:
# - Use exactly one of: --run_dir or --runs_root
# - In single-run mode, figures go to <run_dir>\figures unless --out_dir is given
# - In batch mode, each run gets its own <run_dir>\figures folder


# -----------------------------
# Basic helpers
# -----------------------------

def is_run_dir(path: Path) -> bool:
    required = ["buses.csv", "links.csv", "loads.csv"]
    return path.is_dir() and all((path / f).exists() for f in required)


def find_run_dirs(runs_root: Path) -> list[Path]:
    run_dirs = [p for p in runs_root.iterdir() if is_run_dir(p)]
    return sorted(run_dirs, key=lambda p: p.name)


def read_csv_if_exists(run_dir: Path, filename: str, index_col: int | None = None) -> pd.DataFrame | None:
    path = run_dir / filename
    if not path.exists():
        return None
    return pd.read_csv(path, index_col=index_col)


def read_ts(run_dir: Path, stem: str) -> pd.DataFrame:
    path = run_dir / f"{stem}.csv"
    if not path.exists():
        raise FileNotFoundError(str(path))

    df = pd.read_csv(path, index_col=0)

    # Try datetime parse first
    parsed = pd.to_datetime(df.index, errors="coerce")
    if not parsed.isna().all():
        good = ~parsed.isna()
        df = df.loc[good].copy()
        df.index = parsed[good]
        return df

    # Otherwise keep numeric / original index
    try:
        df.index = pd.to_numeric(df.index, errors="raise")
    except Exception:
        pass

    return df


def try_read_ts(run_dir: Path, stems: list[str]) -> tuple[str | None, pd.DataFrame | None]:
    for s in stems:
        try:
            return s, read_ts(run_dir, s)
        except FileNotFoundError:
            continue
    return None, None


def choose_capacity_column(df: pd.DataFrame) -> str:
    for c in ["p_nom_opt", "p_nom"]:
        if c in df.columns:
            return c
    raise ValueError("No p_nom_opt or p_nom found.")


def region_bus_mask(bus_series: pd.Series) -> pd.Series:
    b = bus_series.astype(str)
    return b.str.match(r"^PL\s+[A-Z]{2}$")


def try_load_metadata(run_dir: Path) -> dict:
    path = run_dir / "run_metadata.json"
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def x_label_from_index(index: pd.Index) -> str:
    if isinstance(index, pd.DatetimeIndex):
        return "Time"
    return "Snapshot"


def sanitize_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def title_suffix_from_meta(meta: dict) -> str:
    if not meta:
        return ""
    year = meta.get("year")
    step = meta.get("stepsize")
    if year is None and step is None:
        return ""
    parts = []
    if year is not None:
        parts.append(f"year={year}")
    if step is not None:
        parts.append(f"step={step}h")
    return " (" + ", ".join(parts) + ")"


# -----------------------------
# Hydrogen selectors
# -----------------------------

def hydrogen_link_mask(links: pd.DataFrame) -> pd.Series:
    idx = links.index.astype(str)
    carrier = links["carrier"].astype(str).str.lower() if "carrier" in links.columns else pd.Series("", index=links.index)
    bus0 = links["bus0"].astype(str).str.lower() if "bus0" in links.columns else pd.Series("", index=links.index)
    bus1 = links["bus1"].astype(str).str.lower() if "bus1" in links.columns else pd.Series("", index=links.index)

    return (
        carrier.str.contains("hydrogen|h2", case=False, na=False)
        | idx.str.lower().str.contains("hydrogen|h2", na=False)
        | bus0.str.contains("_hydrogen|_h2", case=False, na=False)
        | bus1.str.contains("_hydrogen|_h2", case=False, na=False)
    )


def hydrogen_load_mask(loads: pd.DataFrame) -> pd.Series:
    idx = loads.index.astype(str)
    carrier = loads["carrier"].astype(str).str.lower() if "carrier" in loads.columns else pd.Series("", index=loads.index)
    bus = loads["bus"].astype(str).str.lower() if "bus" in loads.columns else pd.Series("", index=loads.index)

    return (
        carrier.str.contains("hydrogen|h2", case=False, na=False)
        | idx.str.lower().str.contains("hydrogen|h2", na=False)
        | bus.str.contains("_hydrogen|_h2", case=False, na=False)
    )


def hydrogen_storage_mask(storage_units: pd.DataFrame) -> pd.Series:
    idx = storage_units.index.astype(str)
    carrier = storage_units["carrier"].astype(str).str.lower() if "carrier" in storage_units.columns else pd.Series("", index=storage_units.index)
    bus = storage_units["bus"].astype(str).str.lower() if "bus" in storage_units.columns else pd.Series("", index=storage_units.index)

    return (
        carrier.str.contains("hydrogen|h2", case=False, na=False)
        | idx.str.lower().str.contains("hydrogen|h2", na=False)
        | bus.str.contains("_hydrogen|_h2", case=False, na=False)
    )


def electric_interregional_link_mask(links: pd.DataFrame) -> pd.Series:
    if "bus0" not in links.columns or "bus1" not in links.columns:
        return pd.Series(False, index=links.index)

    b0 = links["bus0"].astype(str)
    b1 = links["bus1"].astype(str)
    carrier = links["carrier"].astype(str).str.lower() if "carrier" in links.columns else pd.Series("", index=links.index)
    idx = links.index.astype(str).str.lower()

    rr = region_bus_mask(b0) & region_bus_mask(b1)

    # exclude hydrogen / heat / transport / other sector links
    excluded = (
        carrier.str.contains("hydrogen|h2|heat|transport", case=False, na=False)
        | idx.str.contains("hydrogen|h2|heat|transport", na=False)
        | b0.str.lower().str.contains("_hydrogen|_heat|_transport", na=False)
        | b1.str.lower().str.contains("_hydrogen|_heat|_transport", na=False)
    )

    return rr & (~excluded)


# -----------------------------
# Aggregation helpers
# -----------------------------

def top_columns_by_abs_sum(df: pd.DataFrame, n: int) -> list[str]:
    if df is None or df.empty:
        return []
    s = df.abs().sum(axis=0).sort_values(ascending=False)
    return list(s.head(n).index)


def top_columns_by_sum(df: pd.DataFrame, n: int) -> list[str]:
    if df is None or df.empty:
        return []
    s = df.sum(axis=0).sort_values(ascending=False)
    return list(s.head(n).index)


def maybe_set_index_name(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if out.index.name is None:
        out.index.name = "snapshot"
    return out


def group_sum_by_bus(meta_df: pd.DataFrame, value_df: pd.DataFrame, bus_col: str) -> pd.DataFrame:
    if value_df is None or value_df.empty or meta_df is None or meta_df.empty:
        return pd.DataFrame()

    common = [c for c in value_df.columns if c in meta_df.index]
    if not common:
        return pd.DataFrame()

    meta = meta_df.loc[common].copy()
    vals = value_df[common].copy()

    valid = meta[bus_col].notna()
    meta = meta.loc[valid]
    vals = vals[meta.index]

    buses = meta[bus_col].astype(str)
    grouped = {}
    for bus in sorted(buses.unique()):
        cols = buses.index[buses == bus].tolist()
        grouped[bus] = vals[cols].sum(axis=1)

    if not grouped:
        return pd.DataFrame(index=vals.index)

    out = pd.DataFrame(grouped, index=vals.index)
    return out


def duration_curve_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    out = {}
    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce").fillna(0.0).sort_values(ascending=False).reset_index(drop=True)
        out[c] = s.values
    return pd.DataFrame(out)


# -----------------------------
# Plot helpers
# -----------------------------

def save_simple_line_plot(
    df: pd.DataFrame,
    out_path: Path,
    title: str,
    ylabel: str,
    top_n: int | None = None,
    abs_rank: bool = False,
) -> None:
    if df is None or df.empty or df.shape[1] == 0:
        return

    plot_df = sanitize_numeric(df).fillna(0.0)

    if top_n is not None and plot_df.shape[1] > top_n:
        cols = top_columns_by_abs_sum(plot_df, top_n) if abs_rank else top_columns_by_sum(plot_df, top_n)
        plot_df = plot_df[cols]

    plt.figure(figsize=(12, 5))
    for col in plot_df.columns:
        plt.plot(plot_df.index, plot_df[col].values, label=str(col), linewidth=1.2)

    plt.title(title)
    plt.xlabel(x_label_from_index(plot_df.index))
    plt.ylabel(ylabel)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def save_stacked_area_plot(
    df: pd.DataFrame,
    out_path: Path,
    title: str,
    ylabel: str,
    top_n: int | None = None,
) -> None:
    if df is None or df.empty or df.shape[1] == 0:
        return

    plot_df = sanitize_numeric(df).fillna(0.0)

    # For stacked area, negative values can make the plot ugly/confusing.
    # Clip only for demand/storage SOC style positive quantities.
    plot_df = plot_df.clip(lower=0.0)

    if top_n is not None and plot_df.shape[1] > top_n:
        cols = top_columns_by_sum(plot_df, top_n)
        rest = plot_df.drop(columns=cols, errors="ignore")
        plot_df = plot_df[cols].copy()
        if rest.shape[1] > 0:
            plot_df["other"] = rest.sum(axis=1)

    plt.figure(figsize=(12, 5))
    plt.stackplot(plot_df.index, [plot_df[c].values for c in plot_df.columns], labels=[str(c) for c in plot_df.columns])
    plt.title(title)
    plt.xlabel(x_label_from_index(plot_df.index))
    plt.ylabel(ylabel)
    plt.legend(fontsize=8, ncol=2, loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def save_bar_plot(
    series: pd.Series,
    out_path: Path,
    title: str,
    ylabel: str,
    top_n: int | None = None,
    abs_sort: bool = False,
) -> None:
    if series is None or series.empty:
        return

    s = pd.to_numeric(series, errors="coerce").fillna(0.0)

    if abs_sort:
        s = s.loc[s.abs().sort_values(ascending=False).index]
    else:
        s = s.sort_values(ascending=False)

    if top_n is not None:
        s = s.head(top_n)

    plt.figure(figsize=(12, 5))
    x = np.arange(len(s))
    plt.bar(x, s.values)
    plt.xticks(x, [str(i) for i in s.index], rotation=90)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


# -----------------------------
# Main plotting logic
# -----------------------------

def make_hydrogen_plots_for_run(run_dir: Path, out_dir: Path, top_n: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = try_load_metadata(run_dir)
    suffix = title_suffix_from_meta(meta)

    # Static tables
    links = read_csv_if_exists(run_dir, "links.csv")
    loads = read_csv_if_exists(run_dir, "loads.csv")
    storage_units = read_csv_if_exists(run_dir, "storage_units.csv")
    buses = read_csv_if_exists(run_dir, "buses.csv")

    if links is None:
        raise FileNotFoundError(f"Missing {run_dir / 'links.csv'}")

    if "name" in links.columns:
        links = links.set_index("name")
    if loads is not None and "name" in loads.columns:
        loads = loads.set_index("name")
    if storage_units is not None and "name" in storage_units.columns:
        storage_units = storage_units.set_index("name")
    if buses is not None and "name" in buses.columns:
        buses = buses.set_index("name")

    # Time series
    _, links_p0 = try_read_ts(run_dir, ["links-p0"])
    _, links_p1 = try_read_ts(run_dir, ["links-p1"])
    _, loads_p = try_read_ts(run_dir, ["loads-p", "loads-p_set"])
    _, storage_soc = try_read_ts(run_dir, ["storage_units-state_of_charge"])
    _, storage_dispatch = try_read_ts(run_dir, ["storage_units-p_dispatch", "storage_units-p"])
    _, storage_store = try_read_ts(run_dir, ["storage_units-p_store"])

    # ---------- A. Hydrogen network static capacity ----------
    if links is not None and not links.empty:
        hmask = hydrogen_link_mask(links)
        hlinks = links.loc[hmask].copy()

        if not hlinks.empty:
            cap_col = choose_capacity_column(hlinks)
            hlinks[cap_col] = pd.to_numeric(hlinks[cap_col], errors="coerce").fillna(0.0)

            save_bar_plot(
                series=hlinks[cap_col].sort_values(ascending=False),
                out_path=out_dir / "hydrogen_network_capacities.png",
                title=f"Hydrogen network link capacities{suffix}",
                ylabel=cap_col,
                top_n=top_n,
                abs_sort=False,
            )

    # ---------- B. Hydrogen network flows over time ----------
    if links_p0 is not None and links is not None:
        common_h = [c for c in links_p0.columns if c in links.index and hydrogen_link_mask(links.loc[[c]]).iloc[0]]
        if common_h:
            hflow_p0 = sanitize_numeric(links_p0[common_h]).fillna(0.0)

            save_simple_line_plot(
                df=hflow_p0,
                out_path=out_dir / "hydrogen_network_flows_timeseries.png",
                title=f"Hydrogen network flows (p0){suffix}",
                ylabel="MW",
                top_n=top_n,
                abs_rank=True,
            )

            hflow_dc = duration_curve_df(hflow_p0.abs())
            save_simple_line_plot(
                df=hflow_dc,
                out_path=out_dir / "hydrogen_network_flows_duration_curve.png",
                title=f"Hydrogen network absolute flow duration curves{suffix}",
                ylabel="MW",
                top_n=top_n,
                abs_rank=False,
            )

            total_abs_hflow = hflow_p0.abs().sum(axis=1)
            save_simple_line_plot(
                df=pd.DataFrame({"total_abs_hydrogen_flow": total_abs_hflow}, index=hflow_p0.index),
                out_path=out_dir / "hydrogen_network_total_abs_flow.png",
                title=f"Total absolute hydrogen network flow{suffix}",
                ylabel="MW",
            )

    # ---------- C. Hydrogen demand by asset ----------
    if loads is not None and loads_p is not None:
        hmask_load = hydrogen_load_mask(loads)
        hload_names = [c for c in loads_p.columns if c in loads.index and hmask_load.loc[c]]
        if hload_names:
            hload_ts = sanitize_numeric(loads_p[hload_names]).fillna(0.0)

            save_simple_line_plot(
                df=hload_ts,
                out_path=out_dir / "hydrogen_demand_timeseries.png",
                title=f"Hydrogen demand by asset{suffix}",
                ylabel="MW",
                top_n=top_n,
                abs_rank=False,
            )

            hload_dc = duration_curve_df(hload_ts)
            save_simple_line_plot(
                df=hload_dc,
                out_path=out_dir / "hydrogen_demand_duration_curve.png",
                title=f"Hydrogen demand duration curves{suffix}",
                ylabel="MW",
                top_n=top_n,
                abs_rank=False,
            )

            total_h_demand = hload_ts.sum(axis=0).sort_values(ascending=False)
            save_bar_plot(
                series=total_h_demand,
                out_path=out_dir / "hydrogen_demand_total_by_asset.png",
                title=f"Total hydrogen demand by asset{suffix}",
                ylabel="Sum over snapshots",
                top_n=top_n,
                abs_sort=False,
            )

            hload_bus = group_sum_by_bus(loads.loc[hload_names], hload_ts, "bus")
            if not hload_bus.empty:
                save_stacked_area_plot(
                    df=hload_bus,
                    out_path=out_dir / "hydrogen_demand_by_bus_stacked.png",
                    title=f"Hydrogen demand by bus{suffix}",
                    ylabel="MW",
                    top_n=top_n,
                )

                save_bar_plot(
                    series=hload_bus.sum(axis=0).sort_values(ascending=False),
                    out_path=out_dir / "hydrogen_demand_total_by_bus.png",
                    title=f"Total hydrogen demand by bus{suffix}",
                    ylabel="Sum over snapshots",
                    top_n=top_n,
                    abs_sort=False,
                )

    # ---------- D. Hydrogen storage state of charge ----------
    hstorage_names = []
    if storage_units is not None and not storage_units.empty:
        hmask_storage = hydrogen_storage_mask(storage_units)
        hstorage_names = storage_units.index[hmask_storage].tolist()

    if storage_soc is not None and hstorage_names:
        common_soc = [c for c in storage_soc.columns if c in hstorage_names]
        if common_soc:
            hsoc = sanitize_numeric(storage_soc[common_soc]).fillna(0.0)

            save_simple_line_plot(
                df=hsoc,
                out_path=out_dir / "hydrogen_storage_soc_timeseries.png",
                title=f"Hydrogen storage state of charge{suffix}",
                ylabel="MWh",
                top_n=top_n,
                abs_rank=False,
            )

            save_bar_plot(
                series=hsoc.max(axis=0).sort_values(ascending=False),
                out_path=out_dir / "hydrogen_storage_soc_max_by_asset.png",
                title=f"Max hydrogen storage SOC by asset{suffix}",
                ylabel="MWh",
                top_n=top_n,
                abs_sort=False,
            )

            hsoc_bus = group_sum_by_bus(storage_units.loc[common_soc], hsoc, "bus")
            if not hsoc_bus.empty:
                save_stacked_area_plot(
                    df=hsoc_bus,
                    out_path=out_dir / "hydrogen_storage_soc_by_bus_stacked.png",
                    title=f"Hydrogen storage SOC by bus{suffix}",
                    ylabel="MWh",
                    top_n=top_n,
                )

    # ---------- E. Hydrogen storage charge / discharge ----------
    if hstorage_names and (storage_dispatch is not None or storage_store is not None):
        hdisp = None
        hstore = None

        if storage_dispatch is not None:
            common_disp = [c for c in storage_dispatch.columns if c in hstorage_names]
            if common_disp:
                hdisp = sanitize_numeric(storage_dispatch[common_disp]).fillna(0.0)

        if storage_store is not None:
            common_store = [c for c in storage_store.columns if c in hstorage_names]
            if common_store:
                hstore = sanitize_numeric(storage_store[common_store]).fillna(0.0)

        if hdisp is not None:
            save_simple_line_plot(
                df=hdisp,
                out_path=out_dir / "hydrogen_storage_discharge_timeseries.png",
                title=f"Hydrogen storage discharge{suffix}",
                ylabel="MW",
                top_n=top_n,
                abs_rank=False,
            )

        if hstore is not None:
            save_simple_line_plot(
                df=hstore,
                out_path=out_dir / "hydrogen_storage_charge_timeseries.png",
                title=f"Hydrogen storage charge{suffix}",
                ylabel="MW",
                top_n=top_n,
                abs_rank=False,
            )

        if hdisp is not None or hstore is not None:
            idx = hdisp.index if hdisp is not None else hstore.index
            total_discharge = hdisp.sum(axis=1) if hdisp is not None else pd.Series(0.0, index=idx)
            total_charge = hstore.sum(axis=1) if hstore is not None else pd.Series(0.0, index=idx)

            compare = pd.DataFrame(
                {
                    "discharge": total_discharge,
                    "charge": total_charge,
                    "net_discharge_minus_charge": total_discharge - total_charge,
                },
                index=idx,
            )

            save_simple_line_plot(
                df=compare,
                out_path=out_dir / "hydrogen_storage_charge_discharge_total.png",
                title=f"Hydrogen storage total charge and discharge{suffix}",
                ylabel="MW",
            )

    # ---------- F. Electricity interregional flow ----------
    if links is not None and links_p0 is not None:
        emask = electric_interregional_link_mask(links)
        elink_names = [c for c in links_p0.columns if c in links.index and emask.loc[c]]
        if elink_names:
            eflow = sanitize_numeric(links_p0[elink_names]).fillna(0.0)

            save_simple_line_plot(
                df=eflow,
                out_path=out_dir / "power_flow_interregional_timeseries.png",
                title=f"Electric interregional link flows (p0){suffix}",
                ylabel="MW",
                top_n=top_n,
                abs_rank=True,
            )

            ef_dc = duration_curve_df(eflow.abs())
            save_simple_line_plot(
                df=ef_dc,
                out_path=out_dir / "power_flow_interregional_duration_curve.png",
                title=f"Electric interregional absolute flow duration curves{suffix}",
                ylabel="MW",
                top_n=top_n,
                abs_rank=False,
            )

            total_abs_eflow = eflow.abs().sum(axis=1)
            save_simple_line_plot(
                df=pd.DataFrame({"total_abs_electric_flow": total_abs_eflow}, index=eflow.index),
                out_path=out_dir / "power_flow_interregional_total_abs.png",
                title=f"Total absolute electric interregional flow{suffix}",
                ylabel="MW",
            )

            if "bus0" in links.columns and "bus1" in links.columns:
                pair_labels = []
                for name in elink_names:
                    row = links.loc[name]
                    pair_labels.append(f"{row['bus0']} -> {row['bus1']}")
                totals = eflow.abs().sum(axis=0)
                totals.index = pair_labels

                save_bar_plot(
                    series=totals.sort_values(ascending=False),
                    out_path=out_dir / "power_flow_interregional_total_by_link.png",
                    title=f"Total absolute electric interregional flow by link{suffix}",
                    ylabel="Sum of |flow| over snapshots",
                    top_n=top_n,
                    abs_sort=False,
                )

    print(f"Saved hydrogen/power-flow figures to: {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default=None, type=str, help="Single run folder")
    ap.add_argument("--runs_root", default=None, type=str, help="Folder containing many run folders")
    ap.add_argument("--out_dir", default=None, type=str, help="Only used in single-run mode")
    ap.add_argument("--top_n", default=8, type=int, help="How many top assets/links to keep in busy plots")
    args = ap.parse_args()

    if bool(args.run_dir) == bool(args.runs_root):
        raise ValueError("Provide exactly one of --run_dir or --runs_root.")

    if args.run_dir:
        run_dir = Path(args.run_dir)
        if not run_dir.exists():
            raise FileNotFoundError(f"run_dir not found: {run_dir}")

        out_dir = Path(args.out_dir) if args.out_dir else (run_dir / "figures")
        make_hydrogen_plots_for_run(run_dir, out_dir, args.top_n)
        return

    runs_root = Path(args.runs_root)
    if not runs_root.exists():
        raise FileNotFoundError(f"runs_root not found: {runs_root}")

    run_dirs = find_run_dirs(runs_root)
    if not run_dirs:
        raise FileNotFoundError(f"No run folders found in: {runs_root}")

    print(f"Found {len(run_dirs)} run folders in {runs_root}")

    failures = []
    for i, run_dir in enumerate(run_dirs, start=1):
        print(f"[{i}/{len(run_dirs)}] Processing {run_dir.name}")
        try:
            out_dir = run_dir / "figures"
            make_hydrogen_plots_for_run(run_dir, out_dir, args.top_n)
        except Exception as e:
            failures.append((run_dir.name, str(e)))
            print(f"Failed for {run_dir.name}: {e}")

    if failures:
        print("\nFinished with some failures:")
        for name, err in failures:
            print(f"- {name}: {err}")
    else:
        print("\nFinished all runs successfully.")


if __name__ == "__main__":
    main()
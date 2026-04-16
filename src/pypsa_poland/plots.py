# src/pypsa_poland/plots.py
#
# Single-run plot suite for pypsa-poland.
#
# Reads the CSV outputs from one run directory and produces a standardised set
# of figures covering:
#   1. Installed generation capacity mix by region (stacked bar).
#   2. Annual energy generation mix by region and by carrier.
#   3. Net inter-regional imports by region.
#   4. Total system load time series and load duration curve.
#   5. Sector-coupling capacities (electrolysers, heat pumps, storage).
#   6. Transmission utilisation summary.
#   7. VRES curtailment by carrier.
#   8. Dominant inter-regional power-flow direction.
#
# Can be run in single-run mode (--run_dir) or batch mode (--runs_root), in
# which case each run folder gets its own figures/ sub-directory.
#
# Usage:
#   python plots.py --run_dir  <path>
#   python plots.py --runs_root <folder>
#   python plots.py --runs_root <folder> --top_carriers 12

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Shared plot style
# ---------------------------------------------------------------------------
from plot_style import (
    apply_style,
    CARRIER_COLORS, BLUE, RED,
    bar, stacked_bar, line, scatter_annotated,
    savefig,
)

apply_style()

# Carriers treated as variable renewables for curtailment analysis.
VRES_CARRIERS = {"PV ground", "wind", "wind offshore"}


# ---------------------------------------------------------------------------
# I/O and weighting helpers
# ---------------------------------------------------------------------------

def read_ts(run_dir: Path, stem: str) -> pd.DataFrame:
    """Read a time-series CSV and parse its index as a DatetimeIndex."""
    path = run_dir / f"{stem}.csv"
    if not path.exists():
        raise FileNotFoundError(str(path))
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index, errors="coerce")
    if df.index.isna().any():
        df = df.loc[~df.index.isna()].copy()
    return df


def try_read_ts(run_dir: Path, stems: list[str]) -> tuple[str | None, pd.DataFrame | None]:
    """Try each stem in order and return the first that loads successfully."""
    for s in stems:
        try:
            return s, read_ts(run_dir, s)
        except FileNotFoundError:
            continue
    return None, None


def read_snapshot_weights(run_dir: Path, index: pd.DatetimeIndex) -> pd.Series:
    """
    Return snapshot weightings aligned to `index`.

    Falls back to uniform weights of 1.0 if snapshot_weightings.csv is absent,
    which is correct for hourly (stepsize=1) runs.
    """
    path = run_dir / "snapshot_weightings.csv"
    if not path.exists():
        return pd.Series(1.0, index=index)
    w = pd.read_csv(path, index_col=0)
    w.index = pd.to_datetime(w.index, errors="coerce")
    if w.index.isna().any():
        w = w.loc[~w.index.isna()].copy()
    # Use the first recognised weighting column.
    for col in ["generators", "objective", "stores"]:
        if col in w.columns:
            s = pd.to_numeric(w[col], errors="coerce").fillna(1.0)
            return s.reindex(index).fillna(1.0)
    return pd.Series(1.0, index=index)


def weighted_time_sum(df: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """Compute a snapshot-weighted column sum (e.g. hourly MW → annual MWh)."""
    w = weights.reindex(df.index).fillna(1.0)
    return df.mul(w, axis=0).sum(axis=0)


def weighted_time_mean(df: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """Compute a snapshot-weighted column mean (e.g. weighted-average capacity factor)."""
    w = weights.reindex(df.index).fillna(1.0)
    denom = float(w.sum())
    if denom <= 0:
        return pd.Series(0.0, index=df.columns)
    return df.mul(w, axis=0).sum(axis=0) / denom


def choose_capacity_column(df: pd.DataFrame) -> str:
    """Return 'p_nom_opt' if available (post-solve result), otherwise 'p_nom'."""
    for c in ["p_nom_opt", "p_nom"]:
        if c in df.columns:
            return c
    raise ValueError("No p_nom_opt or p_nom found.")


def region_bus_mask(bus_series: pd.Series) -> pd.Series:
    """Boolean mask selecting buses that match the 'PL XX' primary-region pattern."""
    return bus_series.astype(str).str.match(r"^PL\s+[A-Z]{2}$")


def sanitize_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce all columns to numeric, converting non-parseable values to NaN."""
    return df.apply(pd.to_numeric, errors="coerce")


def set_index_if_name(df: pd.DataFrame | None) -> pd.DataFrame | None:
    """Set 'name' as the index if present, otherwise return df unchanged."""
    if df is None:
        return None
    if "name" in df.columns:
        return df.set_index("name")
    return df


def is_run_dir(path: Path) -> bool:
    """Return True if path contains the minimum set of PyPSA output CSVs."""
    required = ["generators.csv", "buses.csv", "carriers.csv"]
    return path.is_dir() and all((path / f).exists() for f in required)


def find_run_dirs(runs_root: Path) -> list[Path]:
    """Return all valid run directories under runs_root, sorted by name."""
    return sorted([p for p in runs_root.iterdir() if is_run_dir(p)],
                  key=lambda p: p.name)


def electric_interregional_link_mask(links: pd.DataFrame) -> pd.Series:
    """
    Boolean mask selecting links that connect two primary region buses via
    electricity (excludes hydrogen, heat, and transport links).
    """
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
    idx = links.index.astype(str).str.lower()
    excluded = (
        carrier.str.contains("hydrogen|heat|transport", case=False, na=False)
        | idx.str.contains("hydrogen|heat|transport", na=False)
        | b0.str.lower().str.contains("_hydrogen|_heat|_transport", na=False)
        | b1.str.lower().str.contains("_hydrogen|_heat|_transport", na=False)
    )
    return rr & (~excluded)


def save_summary_csv(df: pd.DataFrame | pd.Series, out_path: Path) -> None:
    """Save a DataFrame or Series to CSV at out_path."""
    if isinstance(df, pd.Series):
        df.to_csv(out_path, header=True)
    else:
        df.to_csv(out_path, index=True)


# ---------------------------------------------------------------------------
# Per-run plot builder
# ---------------------------------------------------------------------------

def make_plots_for_run(run_dir: Path, out_dir: Path, top_carriers: int) -> None:
    """
    Generate all standard figures for a single run directory.

    Figures are saved as PNG files in out_dir. A CSV summary is also saved
    alongside the transmission utilisation figure for tabular inspection.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    gens    = pd.read_csv(run_dir / "generators.csv")
    links   = pd.read_csv(run_dir / "links.csv")   if (run_dir / "links.csv").exists()   else None
    loads   = pd.read_csv(run_dir / "loads.csv")   if (run_dir / "loads.csv").exists()   else None
    storage = pd.read_csv(run_dir / "storage_units.csv") \
              if (run_dir / "storage_units.csv").exists() else None

    if "bus" not in gens.columns or "carrier" not in gens.columns:
        raise ValueError("generators.csv must include 'bus' and 'carrier' columns.")

    gens_idx    = set_index_if_name(gens)
    links_idx   = set_index_if_name(links)
    storage_idx = set_index_if_name(storage)

    cap_col_gen = choose_capacity_column(gens)
    gens[cap_col_gen] = pd.to_numeric(gens[cap_col_gen], errors="coerce").fillna(0.0)

    # Derive a subtitle from the weather year embedded in the folder name.
    year     = run_dir.name.split("_")[1] if run_dir.name.startswith("run_") else ""
    subtitle = f"Weather year {year}" if year else None

    _, gen_p        = try_read_ts(run_dir, ["generators-p", "generators-p_set"])
    _, links_p0     = try_read_ts(run_dir, ["links-p0"])
    _, loads_p      = try_read_ts(run_dir, ["loads-p_set", "loads-p"])
    _, gen_p_max_pu = try_read_ts(run_dir, ["generators-p_max_pu"])

    # 1. Capacity mix by region (stacked bar)
    gens_reg = gens[region_bus_mask(gens["bus"])].copy()
    cap = gens_reg.groupby(["bus", "carrier"])[cap_col_gen].sum().unstack(fill_value=0.0)
    cap.index = cap.index.str.replace("PL ", "", regex=False)
    cap = cap / 1e3   # MW → GW

    stacked_bar(
        cap,
        out_dir / "capacity_mix_by_region_stacked.png",
        "Installed generation capacity by region",
        "Capacity", "GW",
        top_n=top_carriers,
        subtitle=subtitle,
        rotate_xticks=False,
    )

    # 2. Energy mix by region and by carrier
    if gen_p is not None and gens_idx is not None:
        weights = read_snapshot_weights(run_dir, gen_p.index)
        common  = [c for c in gen_p.columns if c in gens_idx.index]
        if common:
            meta = gens_idx.loc[common, ["bus", "carrier"]].copy()
            meta = meta[region_bus_mask(meta["bus"])]
            if not meta.empty:
                E = weighted_time_sum(gen_p[meta.index], weights)
                tmp = pd.DataFrame(
                    {"bus": meta["bus"], "carrier": meta["carrier"], "E": E.values},
                    index=meta.index,
                )
                e_bus_car = tmp.groupby(["bus", "carrier"])["E"].sum().unstack(fill_value=0.0)
                e_bus_car.index = e_bus_car.index.str.replace("PL ", "", regex=False)
                e_bus_car = e_bus_car / 1e6   # MWh → TWh
                stacked_bar(
                    e_bus_car,
                    out_dir / "energy_mix_by_region_stacked.png",
                    "Annual generation mix by region",
                    "Generation", "TWh",
                    top_n=top_carriers,
                    subtitle=subtitle,
                    rotate_xticks=False,
                )

            energy_per_gen = weighted_time_sum(gen_p[common], weights)
            energy_by_carrier = (
                energy_per_gen.groupby(gens_idx.loc[common, "carrier"]).sum()
                              .sort_values(ascending=False) / 1e6
            )
            bar(
                energy_by_carrier,
                out_dir / "energy_by_carrier.png",
                "Annual energy generation by carrier",
                "Generation", "TWh",
                subtitle=subtitle,
            )

    # 3. Net imports by region
    if links_p0 is not None and links_idx is not None:
        weights = read_snapshot_weights(run_dir, links_p0.index)
        common  = [c for c in links_p0.columns if c in links_idx.index]
        if common:
            meta = links_idx.loc[common, ["bus0", "bus1"]].copy()
            meta["bus0"] = meta["bus0"].astype(str)
            meta["bus1"] = meta["bus1"].astype(str)
            rr = meta[region_bus_mask(meta["bus0"]) & region_bus_mask(meta["bus1"])]
            if not rr.empty:
                annual = weighted_time_sum(links_p0[rr.index], weights)
                net: dict[str, float] = {}
                for lname, row in rr.iterrows():
                    b0, b1 = row["bus0"], row["bus1"]
                    v = float(annual.loc[lname])
                    # Positive signed flow means bus0 → bus1, so bus0 exports.
                    net[b0] = net.get(b0, 0.0) - v
                    net[b1] = net.get(b1, 0.0) + v
                net_s = (
                    pd.Series(net)
                      .rename(index=lambda x: x.replace("PL ", ""))
                      .sort_index() / 1e6
                )
                bar(
                    net_s,
                    out_dir / "net_imports_by_region.png",
                    "Net inter-regional imports by region",
                    "Net import", "TWh",
                    subtitle="Positive = net importer",
                    color=BLUE,
                    value_labels=True,
                    rotate_xticks=False,
                )

    # 4. Total load time series and load duration curve
    if loads_p is not None:
        total_load = sanitize_numeric(loads_p).sum(axis=1) / 1e3   # MW → GW
        line(
            pd.DataFrame({"Total load": total_load}, index=total_load.index),
            out_dir / "total_load.png",
            "Total system load",
            "Load", "GW",
            subtitle=subtitle,
            markers=False,
            color_map={"Total load": BLUE},
        )

        # Load duration curve: hours sorted from highest to lowest,
        # x-axis expressed as a percentage of total hours.
        ldc = total_load.sort_values(ascending=False).reset_index(drop=True)
        ldc.index = ldc.index / len(ldc) * 100
        line(
            pd.DataFrame({"Load duration curve": ldc}),
            out_dir / "load_duration_curve.png",
            "Load duration curve",
            "Load", "GW",
            subtitle=subtitle,
            markers=False,
            color_map={"Load duration curve": BLUE},
        )

    # 5. Sector-coupling capacities (electrolysers, heat pumps, storage)
    installed_rows = []

    if links_idx is not None and not links_idx.empty:
        cap_col_link = choose_capacity_column(links_idx)
        links_idx[cap_col_link] = pd.to_numeric(
            links_idx[cap_col_link], errors="coerce"
        ).fillna(0.0)

        ely_mask = links_idx.index.astype(str).str.endswith("_electrolyzer")
        hp_mask  = (
            links_idx.index.astype(str).str.endswith("_heat_pump")
            | (links_idx["carrier"].astype(str) == "heat_pump")
        )

        if ely_mask.any():
            ely_caps = links_idx.loc[ely_mask, cap_col_link].sort_values(ascending=False)
            ely_caps.index = (
                ely_caps.index.str.replace("PL ", "").str.replace("_electrolyzer", "")
            )
            installed_rows.append(
                pd.DataFrame(
                    {"category": "electrolyser", "capacity_mw": ely_caps.values},
                    index=ely_caps.index,
                )
            )
            bar(
                ely_caps / 1e3,
                out_dir / "installed_electrolyser_capacity_by_asset.png",
                "Installed electrolyser capacity by region",
                "Capacity", "GW",
                subtitle=subtitle,
                color=CARRIER_COLORS.get("hydrogen", BLUE),
            )

        if hp_mask.any():
            hp_caps = links_idx.loc[hp_mask, cap_col_link].sort_values(ascending=False)
            hp_caps.index = (
                hp_caps.index.str.replace("PL ", "").str.replace("_heat_pump", "")
            )
            installed_rows.append(
                pd.DataFrame(
                    {"category": "heat_pump", "capacity_mw": hp_caps.values},
                    index=hp_caps.index,
                )
            )
            bar(
                hp_caps / 1e3,
                out_dir / "installed_heat_pump_capacity_by_asset.png",
                "Installed heat pump capacity by region",
                "Capacity", "GW",
                subtitle=subtitle,
                color=CARRIER_COLORS.get("heat_pump", RED),
            )

    if storage_idx is not None and not storage_idx.empty:
        cap_col_s = choose_capacity_column(storage_idx)
        storage_idx[cap_col_s] = pd.to_numeric(
            storage_idx[cap_col_s], errors="coerce"
        ).fillna(0.0)

        by_carrier = (
            storage_idx.groupby("carrier")[cap_col_s].sum()
                       .sort_values(ascending=False) / 1e3
        )
        bar(
            by_carrier,
            out_dir / "installed_storage_capacity_by_carrier.png",
            "Installed storage power capacity by carrier",
            "Capacity", "GW",
            subtitle=subtitle,
        )

        save_summary_csv(
            storage_idx[["carrier", "bus", cap_col_s]].rename(
                columns={cap_col_s: "capacity_mw"}
            ),
            out_dir / "installed_storage_capacity_detail.csv",
        )

    # 6. Transmission utilisation
    if links_idx is not None and links_p0 is not None and not links_idx.empty:
        emask          = electric_interregional_link_mask(links_idx)
        electric_links = links_idx.loc[emask].copy()

        if not electric_links.empty:
            cap_col_link = choose_capacity_column(electric_links)
            electric_links[cap_col_link] = pd.to_numeric(
                electric_links[cap_col_link], errors="coerce"
            ).fillna(0.0)

            common         = [c for c in links_p0.columns if c in electric_links.index]
            electric_links = electric_links.loc[common]

            if common:
                flow    = sanitize_numeric(links_p0[common]).fillna(0.0)
                weights = read_snapshot_weights(run_dir, flow.index)
                # Replace zero capacity with NaN to avoid division-by-zero.
                cap_s   = electric_links[cap_col_link].replace(0.0, np.nan)
                util    = flow.abs().divide(cap_s, axis=1).replace([np.inf, -np.inf], np.nan)

                util_mean   = weighted_time_mean(util.fillna(0.0), weights)
                util_peak   = util.max(axis=0).fillna(0.0)
                hours_ge_90 = util.ge(0.9).mul(weights, axis=0).sum(axis=0).fillna(0.0)

                util_summary = pd.DataFrame({
                    "bus0":                electric_links["bus0"].astype(str),
                    "bus1":                electric_links["bus1"].astype(str),
                    "capacity_mw":         electric_links[cap_col_link],
                    "mean_utilisation_pu": util_mean.reindex(electric_links.index).fillna(0.0),
                    "peak_utilisation_pu": util_peak.reindex(electric_links.index).fillna(0.0),
                    "hours_ge_90pct":      hours_ge_90.reindex(electric_links.index).fillna(0.0),
                }, index=electric_links.index).sort_values("peak_utilisation_pu", ascending=False)

                util_summary.to_csv(out_dir / "transmission_line_utilisation_summary.csv")

                # Build short "A → B" labels for the bar chart x-axis.
                mean_s = util_summary["mean_utilisation_pu"].copy()
                mean_s.index = [
                    f"{util_summary.loc[i,'bus0'].replace('PL ','')} → "
                    f"{util_summary.loc[i,'bus1'].replace('PL ','')}"
                    for i in mean_s.index
                ]
                bar(
                    mean_s * 100,
                    out_dir / "transmission_line_mean_utilisation.png",
                    "Transmission line mean utilisation",
                    "Utilisation", "%",
                    subtitle=subtitle,
                    color=BLUE,
                    ref_line=80,
                    ref_label="80% threshold",
                )

    # 7. VRES curtailment
    if gen_p is not None and gen_p_max_pu is not None and gens_idx is not None:
        common = [
            c for c in gen_p.columns
            if c in gens_idx.index and c in gen_p_max_pu.columns
        ]
        if common:
            vres_meta = gens_idx.loc[common].copy()
            vres_meta = vres_meta[vres_meta["carrier"].astype(str).isin(VRES_CARRIERS)]

            if not vres_meta.empty:
                cv       = vres_meta.index.tolist()
                weights  = read_snapshot_weights(run_dir, gen_p.index)
                dispatch = sanitize_numeric(gen_p[cv]).fillna(0.0).clip(lower=0.0)
                pmaxpu   = sanitize_numeric(gen_p_max_pu[cv]).fillna(0.0).clip(lower=0.0)
                cap_v    = pd.to_numeric(
                    gens_idx.loc[cv, choose_capacity_column(gens_idx)], errors="coerce"
                ).fillna(0.0)

                available   = pmaxpu.multiply(cap_v, axis=1)
                curtailment = (available - dispatch).clip(lower=0.0)

                avail_mwh   = weighted_time_sum(available,   weights)
                curtail_mwh = weighted_time_sum(curtailment, weights)
                # Share of available resource that was curtailed, per generator.
                share = (curtail_mwh / avail_mwh.replace(0.0, np.nan)).fillna(0.0)

                curtail_by_carrier = (
                    curtail_mwh.groupby(gens_idx.loc[cv, "carrier"]).sum()
                               .sort_values(ascending=False) / 1e6
                )
                share_by_carrier = (
                    share.groupby(gens_idx.loc[cv, "carrier"]).mean()
                         .sort_values(ascending=False) * 100
                )

                bar(
                    curtail_by_carrier,
                    out_dir / "vres_curtailment_by_carrier.png",
                    "VRES curtailment by carrier",
                    "Curtailed energy", "TWh",
                    subtitle=subtitle,
                )
                bar(
                    share_by_carrier,
                    out_dir / "vres_curtailment_share_by_carrier.png",
                    "VRES curtailment share by carrier",
                    "Curtailment share", "%",
                    subtitle=subtitle,
                )

    # 8. Dominant power flow direction
    if links_idx is not None and links_p0 is not None and not links_idx.empty:
        emask          = electric_interregional_link_mask(links_idx)
        electric_links = links_idx.loc[emask].copy()
        if not electric_links.empty:
            common = [c for c in links_p0.columns if c in electric_links.index]
            if common:
                flow    = sanitize_numeric(links_p0[common]).fillna(0.0)
                weights = read_snapshot_weights(run_dir, flow.index)
                signed  = weighted_time_sum(flow,       weights)
                abs_mwh = weighted_time_sum(flow.abs(), weights)

                dom_rows = []
                for name in common:
                    row    = electric_links.loc[name]
                    b0, b1 = str(row["bus0"]), str(row["bus1"])
                    sv     = float(signed.loc[name])
                    av     = float(abs_mwh.loc[name])
                    # Dominant direction is the sign that carries more net energy.
                    if sv >= 0:
                        label = f"{b0.replace('PL ','')} → {b1.replace('PL ','')}"
                        dom   = sv
                    else:
                        label = f"{b1.replace('PL ','')} → {b0.replace('PL ','')}"
                        dom   = -sv
                    dom_rows.append({"label": label, "dominant_mwh": dom, "abs_mwh": av})

                dom_df = (
                    pd.DataFrame(dom_rows)
                      .set_index("label")
                      .sort_values("dominant_mwh", ascending=False)
                )
                bar(
                    dom_df["dominant_mwh"] / 1e6,
                    out_dir / "dominant_power_flow_volume_by_link.png",
                    "Dominant inter-regional power-flow volume",
                    "Energy", "TWh",
                    subtitle=subtitle,
                    color=BLUE,
                )

    print(f"Saved figures to: {out_dir}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Produce all standard single-run figures for one or a batch of run folders."""
    ap = argparse.ArgumentParser(
        description="Single-run plot suite for pypsa-poland."
    )
    ap.add_argument("--run_dir",      default=None, type=str,
                    help="Path to a single run folder.")
    ap.add_argument("--runs_root",    default=None, type=str,
                    help="Root folder containing multiple run sub-folders.")
    ap.add_argument("--out_dir",      default=None, type=str,
                    help="Output folder (single-run mode only).")
    ap.add_argument("--top_carriers", default=8,    type=int,
                    help="Number of carriers to show in stacked bar charts.")
    args = ap.parse_args()

    if bool(args.run_dir) == bool(args.runs_root):
        raise ValueError("Provide exactly one of --run_dir or --runs_root.")

    if args.run_dir:
        run_dir = Path(args.run_dir)
        if not run_dir.exists():
            raise FileNotFoundError(f"run_dir not found: {run_dir}")
        out_dir = Path(args.out_dir) if args.out_dir else run_dir / "figures"
        make_plots_for_run(run_dir, out_dir, args.top_carriers)
        return

    runs_root = Path(args.runs_root)
    if not runs_root.exists():
        raise FileNotFoundError(f"runs_root not found: {runs_root}")

    run_dirs = find_run_dirs(runs_root)
    if not run_dirs:
        raise FileNotFoundError(f"No run folders found in: {runs_root}")

    print(f"Found {len(run_dirs)} run folders.")

    failures = []
    for i, run_dir in enumerate(run_dirs, 1):
        print(f"[{i}/{len(run_dirs)}] {run_dir.name}")
        try:
            make_plots_for_run(run_dir, run_dir / "figures", args.top_carriers)
        except Exception as e:
            failures.append((run_dir.name, str(e)))
            print(f"  Failed: {e}")

    if failures:
        print("\nFinished with failures:")
        for name, err in failures:
            print(f"  - {name}: {err}")
    else:
        print("\nAll runs completed successfully.")


if __name__ == "__main__":
    main()

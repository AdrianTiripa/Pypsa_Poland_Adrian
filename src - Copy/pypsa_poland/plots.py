# src/pypsa_poland/plots.py

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Usage examples (same style as before):
#
# 1) One run:
# & C:\Users\adria\anaconda3\envs\pypsa-legacy\python.exe C:\Users\adria\MODEL_PyPSA\Core\pypsa-poland_ADRIAN\src\pypsa_poland\plots.py --run_dir C:\Users\adria\MODEL_PyPSA\Core\runs\run_2020_20260321_231455_3hr_Optimal_1577s
#
# 2) One run with custom output folder:
# & C:\Users\adria\anaconda3\envs\pypsa-legacy\python.exe C:\Users\adria\MODEL_PyPSA\Core\pypsa-poland_ADRIAN\src\pypsa_poland\plots.py --run_dir C:\Users\adria\MODEL_PyPSA\Core\runs\run_2020_20260321_231455_3hr_Optimal_1577s --out_dir C:\Users\adria\MODEL_PyPSA\Core\my_figures
#
# 3) All runs inside the runs folder:
# & C:\Users\adria\anaconda3\envs\pypsa-legacy\python.exe C:\Users\adria\MODEL_PyPSA\Core\pypsa-poland_ADRIAN\src\pypsa_poland\plots.py --runs_root C:\Users\adria\MODEL_PyPSA\Core\runs
#
# 4) All runs, with top 12 carriers in stacked plots:
# & C:\Users\adria\anaconda3\envs\pypsa-legacy\python.exe C:\Users\adria\MODEL_PyPSA\Core\pypsa-poland_ADRIAN\src\pypsa_poland\plots.py --runs_root C:\Users\adria\MODEL_PyPSA\Core\runs --top_carriers 12
#
# Notes:
# - Use exactly one of: --run_dir or --runs_root
# - In single-run mode, figures go to <run_dir>\figures unless --out_dir is given
# - In batch mode, each run gets its own <run_dir>\figures folder


VRES_CARRIERS = {"PV ground", "wind", "wind offshore"}


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


def weighted_time_mean(df: pd.DataFrame, weights: pd.Series) -> pd.Series:
    w = weights.reindex(df.index).fillna(1.0)
    denom = float(w.sum())
    if denom <= 0:
        return pd.Series(0.0, index=df.columns)
    return df.mul(w, axis=0).sum(axis=0) / denom


def choose_capacity_column(df: pd.DataFrame) -> str:
    for c in ["p_nom_opt", "p_nom"]:
        if c in df.columns:
            return c
    raise ValueError("No p_nom_opt or p_nom found.")


def region_bus_mask(bus_series: pd.Series) -> pd.Series:
    b = bus_series.astype(str)
    return b.str.match(r"^PL\s+[A-Z]{2}$")


def sanitize_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def set_index_if_name(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None:
        return None
    if "name" in df.columns:
        return df.set_index("name")
    return df


def is_run_dir(path: Path) -> bool:
    required = ["generators.csv", "buses.csv", "carriers.csv"]
    return path.is_dir() and all((path / f).exists() for f in required)


def find_run_dirs(runs_root: Path) -> list[Path]:
    run_dirs = [p for p in runs_root.iterdir() if is_run_dir(p)]
    return sorted(run_dirs, key=lambda p: p.name)


def electric_interregional_link_mask(links: pd.DataFrame) -> pd.Series:
    if "bus0" not in links.columns or "bus1" not in links.columns:
        return pd.Series(False, index=links.index)

    b0 = links["bus0"].astype(str)
    b1 = links["bus1"].astype(str)
    rr = region_bus_mask(b0) & region_bus_mask(b1)

    carrier = links["carrier"].astype(str).str.lower() if "carrier" in links.columns else pd.Series("", index=links.index)
    idx = links.index.astype(str).str.lower()

    excluded = (
        carrier.str.contains("hydrogen|heat|transport", case=False, na=False)
        | idx.str.contains("hydrogen|heat|transport", na=False)
        | b0.str.lower().str.contains("_hydrogen|_heat|_transport", na=False)
        | b1.str.lower().str.contains("_hydrogen|_heat|_transport", na=False)
    )

    return rr & (~excluded)


def save_bar_plot(
    series: pd.Series,
    out_path: Path,
    title: str,
    ylabel: str,
    *,
    top_n: int | None = None,
    sort_by_abs: bool = False,
) -> None:
    if series is None or series.empty:
        return

    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return

    if sort_by_abs:
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


def save_line_plot(
    df: pd.DataFrame,
    out_path: Path,
    title: str,
    ylabel: str,
    *,
    top_n: int | None = None,
    abs_rank: bool = False,
) -> None:
    if df is None or df.empty:
        return

    plot_df = sanitize_numeric(df).copy()
    plot_df = plot_df.dropna(axis=1, how="all")
    if plot_df.empty:
        return

    if top_n is not None and plot_df.shape[1] > top_n:
        if abs_rank:
            rank = plot_df.abs().sum(axis=0).sort_values(ascending=False)
        else:
            rank = plot_df.sum(axis=0).sort_values(ascending=False)
        plot_df = plot_df[rank.head(top_n).index]

    plt.figure(figsize=(12, 5))
    for col in plot_df.columns:
        plt.plot(plot_df.index, plot_df[col], label=str(col))
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("Time")
    if plot_df.shape[1] <= 12:
        plt.legend(fontsize=8, ncols=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def save_stacked_by_region(df: pd.DataFrame, out_path: Path, title: str, ylabel: str, top_n: int) -> None:
    if df is None or df.empty:
        return

    data = df.copy()
    totals = data.sum(axis=0)
    data = data.loc[:, totals[totals > 1e-6].index]
    if data.empty:
        return

    top = data.sum(axis=0).sort_values(ascending=False).head(top_n).index
    plot_df = data[top].copy()
    rest = data.drop(columns=top, errors="ignore")
    if rest.shape[1] > 0:
        plot_df["other"] = rest.sum(axis=1)

    plot_df = plot_df.sort_index()

    plt.figure(figsize=(12, 6))
    bottom = np.zeros(len(plot_df))
    x = np.arange(len(plot_df.index))
    for col in plot_df.columns:
        y = plot_df[col].values
        plt.bar(x, y, bottom=bottom, label=col)
        bottom += y

    plt.xticks(x, [str(i).replace("PL ", "") for i in plot_df.index], rotation=0)
    plt.title(title)
    plt.xlabel("Region")
    plt.ylabel(ylabel)
    plt.legend(ncols=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def save_summary_csv(df: pd.DataFrame | pd.Series, out_path: Path) -> None:
    if isinstance(df, pd.Series):
        df.to_csv(out_path, header=True)
    else:
        df.to_csv(out_path, index=True)


def make_plots_for_run(run_dir: Path, out_dir: Path, top_carriers: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    gens = pd.read_csv(run_dir / "generators.csv")
    links = pd.read_csv(run_dir / "links.csv") if (run_dir / "links.csv").exists() else None
    loads = pd.read_csv(run_dir / "loads.csv") if (run_dir / "loads.csv").exists() else None
    storage_units = pd.read_csv(run_dir / "storage_units.csv") if (run_dir / "storage_units.csv").exists() else None

    if "bus" not in gens.columns or "carrier" not in gens.columns:
        raise ValueError("generators.csv must include 'bus' and 'carrier' columns.")

    gens_idx = set_index_if_name(gens)
    links_idx = set_index_if_name(links)
    loads_idx = set_index_if_name(loads)
    storage_idx = set_index_if_name(storage_units)

    cap_col_gen = choose_capacity_column(gens)
    gens[cap_col_gen] = pd.to_numeric(gens[cap_col_gen], errors="coerce").fillna(0.0)

    # ---------- Existing core plots ----------
    gens_reg = gens[region_bus_mask(gens["bus"])].copy()
    cap = gens_reg.groupby(["bus", "carrier"])[cap_col_gen].sum().unstack(fill_value=0.0)
    save_stacked_by_region(
        cap,
        out_dir / "capacity_mix_by_region_stacked.png",
        "Installed capacity mix by region (stacked)",
        "MW",
        top_carriers,
    )

    _, gen_p = try_read_ts(run_dir, ["generators-p", "generators-p_set"])
    _, links_p0 = try_read_ts(run_dir, ["links-p0"])
    _, loads_p = try_read_ts(run_dir, ["loads-p_set", "loads-p"])
    _, gen_p_max_pu = try_read_ts(run_dir, ["generators-p_max_pu"])

    if gen_p is not None and gens_idx is not None:
        weights_gen = read_snapshot_weights(run_dir, gen_p.index)
        common = [c for c in gen_p.columns if c in gens_idx.index]
        if common:
            meta = gens_idx.loc[common, ["bus", "carrier"]].copy()
            meta = meta[region_bus_mask(meta["bus"])]
            if not meta.empty:
                E = weighted_time_sum(gen_p[meta.index], weights_gen)
                tmp = pd.DataFrame({"bus": meta["bus"], "carrier": meta["carrier"], "E": E.values}, index=meta.index)
                e_bus_car = tmp.groupby(["bus", "carrier"])["E"].sum().unstack(fill_value=0.0)
                save_stacked_by_region(
                    e_bus_car,
                    out_dir / "energy_mix_by_region_stacked.png",
                    "Energy generation mix by region (stacked)",
                    "MWh (snapshot-weighted)",
                    top_carriers,
                )

            energy_per_gen = weighted_time_sum(gen_p[common], weights_gen)
            energy_by_carrier = energy_per_gen.groupby(gens_idx.loc[common, "carrier"]).sum().sort_values(ascending=False)
            save_bar_plot(
                energy_by_carrier,
                out_dir / "energy_by_carrier.png",
                "Annual energy by carrier",
                "MWh (snapshot-weighted)",
            )

    if links_p0 is not None and links_idx is not None:
        weights_links = read_snapshot_weights(run_dir, links_p0.index)
        common = [c for c in links_p0.columns if c in links_idx.index]
        if common:
            meta = links_idx.loc[common, ["bus0", "bus1"]].copy()
            meta["bus0"] = meta["bus0"].astype(str)
            meta["bus1"] = meta["bus1"].astype(str)
            rr = meta[region_bus_mask(meta["bus0"]) & region_bus_mask(meta["bus1"])]

            if not rr.empty:
                annual = weighted_time_sum(links_p0[rr.index], weights_links)
                net = {b: 0.0 for b in sorted(pd.concat([rr["bus0"], rr["bus1"]]).unique())}
                for link_name, row in rr.iterrows():
                    b0, b1 = row["bus0"], row["bus1"]
                    v = float(annual.loc[link_name])
                    net[b0] -= v
                    net[b1] += v

                net_s = pd.Series(net).sort_index()
                save_bar_plot(
                    net_s,
                    out_dir / "net_imports_by_region.png",
                    "Net inter-regional imports by region (positive = net importer)",
                    "MWh (snapshot-weighted)",
                )

                total_abs_flow = sanitize_numeric(links_p0[rr.index]).abs().sum(axis=1)
                save_line_plot(
                    pd.DataFrame({"total_abs_interregional_flow": total_abs_flow}, index=total_abs_flow.index),
                    out_dir / "total_interregional_flow.png",
                    "Total absolute inter-regional link flow (sum |p0|)",
                    "MW",
                )

    if loads_p is not None:
        total_load = sanitize_numeric(loads_p).sum(axis=1)
        save_line_plot(
            pd.DataFrame({"total_load": total_load}, index=total_load.index),
            out_dir / "total_load.png",
            "Total load (system)",
            "MW",
        )

        ldc = total_load.sort_values(ascending=False).reset_index(drop=True)
        save_line_plot(
            pd.DataFrame({"load_duration_curve": ldc}),
            out_dir / "load_duration_curve.png",
            "Load duration curve",
            "MW",
        )

    # ---------- New 1: Installed capacities in electrolysers / heat pumps / storage ----------
    installed_rows = []

    if links_idx is not None and not links_idx.empty:
        cap_col_link = choose_capacity_column(links_idx)
        links_idx[cap_col_link] = pd.to_numeric(links_idx[cap_col_link], errors="coerce").fillna(0.0)

        electrolyser_mask = links_idx.index.astype(str).str.endswith("_electrolyzer")
        heat_pump_mask = (
            links_idx.index.astype(str).str.endswith("_heat_pump")
            | (links_idx["carrier"].astype(str) == "heat_pump")
        )

        if electrolyser_mask.any():
            ely_caps = links_idx.loc[electrolyser_mask, cap_col_link].sort_values(ascending=False)
            installed_rows.append(pd.DataFrame({"category": "electrolyser", "name": ely_caps.index, "capacity_mw": ely_caps.values}))
            save_bar_plot(
                ely_caps,
                out_dir / "installed_electrolyser_capacity_by_asset.png",
                "Installed electrolyser capacity by asset",
                "MW",
                top_n=top_carriers,
            )

        if heat_pump_mask.any():
            hp_caps = links_idx.loc[heat_pump_mask, cap_col_link].sort_values(ascending=False)
            installed_rows.append(pd.DataFrame({"category": "heat_pump", "name": hp_caps.index, "capacity_mw": hp_caps.values}))
            save_bar_plot(
                hp_caps,
                out_dir / "installed_heat_pump_capacity_by_asset.png",
                "Installed heat-pump capacity by asset",
                "MW",
                top_n=top_carriers,
            )

    if storage_idx is not None and not storage_idx.empty:
        cap_col_storage = choose_capacity_column(storage_idx)
        storage_idx[cap_col_storage] = pd.to_numeric(storage_idx[cap_col_storage], errors="coerce").fillna(0.0)

        storage_by_carrier = storage_idx.groupby("carrier")[cap_col_storage].sum().sort_values(ascending=False)
        save_bar_plot(
            storage_by_carrier,
            out_dir / "installed_storage_capacity_by_carrier.png",
            "Installed storage power capacity by carrier",
            "MW",
        )

        storage_by_asset = storage_idx[cap_col_storage].sort_values(ascending=False)
        save_bar_plot(
            storage_by_asset,
            out_dir / "installed_storage_capacity_by_asset.png",
            "Installed storage power capacity by asset",
            "MW",
            top_n=top_carriers,
        )

        installed_rows.append(
            pd.DataFrame(
                {
                    "category": "storage_unit",
                    "name": storage_idx.index,
                    "carrier": storage_idx["carrier"].astype(str).values,
                    "capacity_mw": storage_idx[cap_col_storage].values,
                }
            )
        )

        save_summary_csv(
            storage_idx[["carrier", "bus", cap_col_storage]].rename(columns={cap_col_storage: "capacity_mw"}),
            out_dir / "installed_storage_capacity_detail.csv",
        )

    if installed_rows:
        installed_detail = pd.concat(installed_rows, ignore_index=True, sort=False)
        installed_detail.to_csv(out_dir / "installed_sector_coupling_capacity_detail.csv", index=False)

    # ---------- New 2: Transmission line utilization ----------
    if links_idx is not None and links_p0 is not None and not links_idx.empty:
        emask = electric_interregional_link_mask(links_idx)
        electric_links = links_idx.loc[emask].copy()

        if not electric_links.empty:
            cap_col_link = choose_capacity_column(electric_links)
            electric_links[cap_col_link] = pd.to_numeric(electric_links[cap_col_link], errors="coerce").fillna(0.0)

            common = [c for c in links_p0.columns if c in electric_links.index]
            electric_links = electric_links.loc[common]
            if common:
                flow = sanitize_numeric(links_p0[common]).fillna(0.0)
                weights = read_snapshot_weights(run_dir, flow.index)

                cap_series = electric_links[cap_col_link].replace(0.0, np.nan)
                util = flow.abs().divide(cap_series, axis=1).replace([np.inf, -np.inf], np.nan)

                util_mean = weighted_time_mean(util.fillna(0.0), weights)
                util_peak = util.max(axis=0).fillna(0.0)
                hours_ge_90 = util.ge(0.9).mul(weights, axis=0).sum(axis=0).fillna(0.0)
                hours_ge_99 = util.ge(0.99).mul(weights, axis=0).sum(axis=0).fillna(0.0)
                headroom_min = (1.0 - util_peak).clip(lower=0.0)

                util_summary = pd.DataFrame(
                    {
                        "bus0": electric_links["bus0"].astype(str),
                        "bus1": electric_links["bus1"].astype(str),
                        "carrier": electric_links["carrier"].astype(str) if "carrier" in electric_links.columns else "",
                        "capacity_mw": electric_links[cap_col_link],
                        "mean_utilization_pu": util_mean.reindex(electric_links.index).fillna(0.0),
                        "peak_utilization_pu": util_peak.reindex(electric_links.index).fillna(0.0),
                        "hours_ge_90pct": hours_ge_90.reindex(electric_links.index).fillna(0.0),
                        "hours_ge_99pct": hours_ge_99.reindex(electric_links.index).fillna(0.0),
                        "min_headroom_to_peak_pu": headroom_min.reindex(electric_links.index).fillna(0.0),
                    },
                    index=electric_links.index,
                ).sort_values("peak_utilization_pu", ascending=False)

                util_summary.to_csv(out_dir / "transmission_line_utilization_summary.csv")
                save_bar_plot(
                    util_summary["mean_utilization_pu"],
                    out_dir / "transmission_line_mean_utilization.png",
                    "Transmission line mean utilization",
                    "p.u.",
                    top_n=top_carriers,
                )
                save_bar_plot(
                    util_summary["peak_utilization_pu"],
                    out_dir / "transmission_line_peak_utilization.png",
                    "Transmission line peak utilization",
                    "p.u.",
                    top_n=top_carriers,
                )
                save_bar_plot(
                    util_summary["hours_ge_90pct"],
                    out_dir / "transmission_line_hours_ge_90pct.png",
                    "Transmission line hours at or above 90% utilization",
                    "Weighted hours",
                    top_n=top_carriers,
                )

    # ---------- New 3: VRES curtailment ----------
    if gen_p is not None and gen_p_max_pu is not None and gens_idx is not None:
        common = [c for c in gen_p.columns if c in gens_idx.index and c in gen_p_max_pu.columns]
        if common:
            vres_meta = gens_idx.loc[common].copy()
            vres_meta = vres_meta[vres_meta["carrier"].astype(str).isin(VRES_CARRIERS)]

            if not vres_meta.empty:
                common_vres = vres_meta.index.tolist()
                dispatch = sanitize_numeric(gen_p[common_vres]).fillna(0.0).clip(lower=0.0)
                pmaxpu = sanitize_numeric(gen_p_max_pu[common_vres]).fillna(0.0).clip(lower=0.0)
                cap = pd.to_numeric(gens_idx.loc[common_vres, choose_capacity_column(gens_idx)], errors="coerce").fillna(0.0)

                available = pmaxpu.multiply(cap, axis=1)
                curtailment = (available - dispatch).clip(lower=0.0)

                weights = read_snapshot_weights(run_dir, dispatch.index)
                available_mwh = weighted_time_sum(available, weights)
                dispatch_mwh = weighted_time_sum(dispatch, weights)
                curtailed_mwh = weighted_time_sum(curtailment, weights)

                share = curtailed_mwh / available_mwh.replace(0.0, np.nan)
                share = share.fillna(0.0)

                curtail_detail = pd.DataFrame(
                    {
                        "carrier": gens_idx.loc[common_vres, "carrier"].astype(str),
                        "bus": gens_idx.loc[common_vres, "bus"].astype(str),
                        "installed_capacity_mw": cap,
                        "available_mwh": available_mwh,
                        "dispatched_mwh": dispatch_mwh,
                        "curtailed_mwh": curtailed_mwh,
                        "curtailment_share": share,
                    },
                    index=common_vres,
                ).sort_values("curtailed_mwh", ascending=False)

                curtail_detail.to_csv(out_dir / "vres_curtailment_detail.csv")

                curtail_by_carrier = curtail_detail.groupby("carrier")["curtailed_mwh"].sum().sort_values(ascending=False)
                avail_by_carrier = curtail_detail.groupby("carrier")["available_mwh"].sum()
                share_by_carrier = (curtail_by_carrier / avail_by_carrier.replace(0.0, np.nan)).fillna(0.0).sort_values(ascending=False)

                save_bar_plot(
                    curtail_by_carrier,
                    out_dir / "vres_curtailment_by_carrier.png",
                    "VRES curtailment by carrier",
                    "MWh (snapshot-weighted)",
                )
                save_bar_plot(
                    share_by_carrier,
                    out_dir / "vres_curtailment_share_by_carrier.png",
                    "VRES curtailment share by carrier",
                    "p.u.",
                )
                save_bar_plot(
                    curtail_detail["curtailed_mwh"],
                    out_dir / "vres_curtailment_top_assets.png",
                    "VRES curtailment by asset",
                    "MWh (snapshot-weighted)",
                    top_n=top_carriers,
                )

    # ---------- New 4: Dominant power-flow direction and volume ----------
    if links_idx is not None and links_p0 is not None and not links_idx.empty:
        emask = electric_interregional_link_mask(links_idx)
        electric_links = links_idx.loc[emask].copy()

        if not electric_links.empty:
            common = [c for c in links_p0.columns if c in electric_links.index]
            if common:
                flow = sanitize_numeric(links_p0[common]).fillna(0.0)
                weights = read_snapshot_weights(run_dir, flow.index)
                signed_mwh = weighted_time_sum(flow, weights)
                abs_mwh = weighted_time_sum(flow.abs(), weights)

                dominant_rows = []
                for name in common:
                    row = electric_links.loc[name]
                    b0 = str(row["bus0"])
                    b1 = str(row["bus1"])
                    signed_val = float(signed_mwh.loc[name])
                    abs_val = float(abs_mwh.loc[name])

                    if signed_val >= 0:
                        dominant_from = b0
                        dominant_to = b1
                        dominant_mwh = signed_val
                    else:
                        dominant_from = b1
                        dominant_to = b0
                        dominant_mwh = -signed_val

                    dominant_rows.append(
                        {
                            "link_name": name,
                            "bus0": b0,
                            "bus1": b1,
                            "dominant_from": dominant_from,
                            "dominant_to": dominant_to,
                            "dominant_direction": f"{dominant_from} -> {dominant_to}",
                            "dominant_signed_mwh": dominant_mwh,
                            "total_absolute_mwh": abs_val,
                            "dominant_share_of_absolute": (dominant_mwh / abs_val) if abs_val > 0 else 0.0,
                        }
                    )

                dominant_df = pd.DataFrame(dominant_rows).set_index("link_name").sort_values(
                    "dominant_signed_mwh", ascending=False
                )
                dominant_df.to_csv(out_dir / "dominant_power_flow_direction.csv")

                labels = dominant_df["dominant_direction"].copy()
                labels.index = dominant_df.index
                dominant_series = pd.Series(dominant_df["dominant_signed_mwh"].values, index=labels.values)
                share_series = pd.Series(dominant_df["dominant_share_of_absolute"].values, index=labels.values)

                save_bar_plot(
                    dominant_series,
                    out_dir / "dominant_power_flow_volume_by_link.png",
                    "Dominant interregional power-flow volume by link direction",
                    "MWh (snapshot-weighted)",
                    top_n=top_carriers,
                )
                save_bar_plot(
                    share_series,
                    out_dir / "dominant_power_flow_share_by_link.png",
                    "Dominant-direction share of absolute flow by link",
                    "p.u.",
                    top_n=top_carriers,
                )

    print(f"Saved figures to: {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default=None, type=str, help="Single run folder")
    ap.add_argument("--runs_root", default=None, type=str, help="Folder containing many run folders")
    ap.add_argument("--out_dir", default=None, type=str, help="Only used in single-run mode")
    ap.add_argument("--top_carriers", default=8, type=int)
    args = ap.parse_args()

    if bool(args.run_dir) == bool(args.runs_root):
        raise ValueError("Provide exactly one of --run_dir or --runs_root.")

    if args.run_dir:
        run_dir = Path(args.run_dir)
        if not run_dir.exists():
            raise FileNotFoundError(f"run_dir not found: {run_dir}")

        out_dir = Path(args.out_dir) if args.out_dir else (run_dir / "figures")
        make_plots_for_run(run_dir, out_dir, args.top_carriers)
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
            make_plots_for_run(run_dir, out_dir, args.top_carriers)
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
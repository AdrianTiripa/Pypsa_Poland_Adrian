from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# python -m pypsa_poland.results_to_csv --run_dir 

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


def region_bus_mask(bus_series: pd.Series) -> pd.Series:
    b = bus_series.astype(str)
    return b.str.match(r"^PL\s+[A-Z]{2}$")


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


def ensure_out_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)


def save_csv(df: pd.DataFrame | pd.Series, out_dir: Path, name: str) -> None:
    path = out_dir / name
    if isinstance(df, pd.Series):
        df.to_frame(name="value").to_csv(path)
    else:
        df.to_csv(path, index=True)


def export_run_csvs(run_dir: Path, out_dir: Path) -> None:
    ensure_out_dir(out_dir)

    gens = pd.read_csv(run_dir / "generators.csv")
    links = pd.read_csv(run_dir / "links.csv") if (run_dir / "links.csv").exists() else None
    loads = pd.read_csv(run_dir / "loads.csv") if (run_dir / "loads.csv").exists() else None
    storage_units = pd.read_csv(run_dir / "storage_units.csv") if (run_dir / "storage_units.csv").exists() else None

    gens_idx = set_index_if_name(gens)
    links_idx = set_index_if_name(links)
    loads_idx = set_index_if_name(loads)
    storage_idx = set_index_if_name(storage_units)

    cap_col_gen = choose_capacity_column(gens)
    gens[cap_col_gen] = pd.to_numeric(gens[cap_col_gen], errors="coerce").fillna(0.0)

    _, gen_p = try_read_ts(run_dir, ["generators-p", "generators-p_set"])
    _, gen_p_max_pu = try_read_ts(run_dir, ["generators-p_max_pu"])
    _, links_p0 = try_read_ts(run_dir, ["links-p0"])
    _, loads_p = try_read_ts(run_dir, ["loads-p_set", "loads-p"])
    _, stores_p = try_read_ts(run_dir, ["storage_units-p"])
    _, stores_soc = try_read_ts(run_dir, ["storage_units-state_of_charge"])

    # ------------------------------------------------------------------
    # 01, 02: installed generation capacity
    # ------------------------------------------------------------------
    gens_reg = gens[region_bus_mask(gens["bus"])].copy()

    cap_by_region_carrier = (
        gens_reg.groupby(["bus", "carrier"])[cap_col_gen]
        .sum()
        .reset_index()
        .rename(columns={"bus": "region_bus", "carrier": "carrier", cap_col_gen: "capacity_mw"})
        .sort_values(["region_bus", "carrier"])
    )
    save_csv(cap_by_region_carrier, out_dir, "01_capacity_by_region_and_carrier.csv")

    cap_by_carrier_system = (
        gens.groupby("carrier")[cap_col_gen]
        .sum()
        .reset_index()
        .rename(columns={cap_col_gen: "capacity_mw"})
        .sort_values("capacity_mw", ascending=False)
    )
    save_csv(cap_by_carrier_system, out_dir, "02_capacity_by_carrier_system.csv")

    # ------------------------------------------------------------------
    # 03, 04, 05: generation summaries
    # ------------------------------------------------------------------
    if gen_p is not None and gens_idx is not None:
        weights = read_snapshot_weights(run_dir, gen_p.index)
        common = [c for c in gen_p.columns if c in gens_idx.index]

        if common:
            gen_p_num = sanitize_numeric(gen_p[common]).fillna(0.0)
            energy_per_gen = weighted_time_sum(gen_p_num, weights)

            meta = gens_idx.loc[common, ["bus", "carrier"]].copy()
            meta["energy_mwh"] = energy_per_gen.values

            gen_region_carrier = (
                meta[region_bus_mask(meta["bus"])]
                .groupby(["bus", "carrier"])["energy_mwh"]
                .sum()
                .reset_index()
                .rename(columns={"bus": "region_bus"})
                .sort_values(["region_bus", "carrier"])
            )
            save_csv(gen_region_carrier, out_dir, "03_generation_by_region_and_carrier_mwh.csv")

            gen_carrier_system = (
                meta.groupby("carrier")["energy_mwh"]
                .sum()
                .reset_index()
                .sort_values("energy_mwh", ascending=False)
            )
            save_csv(gen_carrier_system, out_dir, "04_generation_by_carrier_system_mwh.csv")

            cap_series = pd.to_numeric(gens_idx.loc[common, choose_capacity_column(gens_idx)], errors="coerce").fillna(0.0)
            weighted_mean_dispatch = weighted_time_mean(gen_p_num, weights)
            cf = weighted_mean_dispatch / cap_series.replace(0.0, np.nan)
            cf = cf.fillna(0.0)

            cf_detail = pd.DataFrame(
                {
                    "bus": gens_idx.loc[common, "bus"].astype(str),
                    "carrier": gens_idx.loc[common, "carrier"].astype(str),
                    "installed_capacity_mw": cap_series,
                    "annual_generation_mwh": energy_per_gen,
                    "capacity_factor": cf,
                },
                index=common,
            ).sort_values("annual_generation_mwh", ascending=False)

            save_csv(cf_detail, out_dir, "05_generator_capacity_factor_summary.csv")

    # ------------------------------------------------------------------
    # 06, 07: load summaries
    # ------------------------------------------------------------------
    if loads_p is not None:
        weights = read_snapshot_weights(run_dir, loads_p.index)
        loads_num = sanitize_numeric(loads_p).fillna(0.0)

        load_energy = weighted_time_sum(loads_num, weights)
        load_detail = pd.DataFrame(
            {
                "load_name": load_energy.index.astype(str),
                "annual_load_mwh": load_energy.values,
            }
        )

        def infer_region(load_name: str) -> str | None:
            s = str(load_name)
            if s.startswith("PL ") and len(s) >= 5:
                return s[:5]
            return None

        load_detail["region_bus"] = load_detail["load_name"].map(infer_region)

        load_by_region = (
            load_detail.dropna(subset=["region_bus"])
            .groupby("region_bus")["annual_load_mwh"]
            .sum()
            .reset_index()
            .sort_values("region_bus")
        )
        save_csv(load_by_region, out_dir, "06_load_by_region_mwh.csv")

        total_load_system = pd.DataFrame(
            {"metric": ["annual_load_mwh"], "value": [float(load_energy.sum())]}
        )
        save_csv(total_load_system, out_dir, "07_total_load_system_mwh.csv")

    # ------------------------------------------------------------------
    # 08, 09, 10: transmission summaries
    # ------------------------------------------------------------------
    if links_idx is not None and links_p0 is not None and not links_idx.empty:
        emask = electric_interregional_link_mask(links_idx)
        electric_links = links_idx.loc[emask].copy()

        if not electric_links.empty:
            common = [c for c in links_p0.columns if c in electric_links.index]
            if common:
                electric_links = electric_links.loc[common]
                flow = sanitize_numeric(links_p0[common]).fillna(0.0)
                weights = read_snapshot_weights(run_dir, flow.index)

                annual_signed = weighted_time_sum(flow, weights)
                annual_abs = weighted_time_sum(flow.abs(), weights)

                flow_summary_rows = []
                net = {}

                for name in common:
                    row = electric_links.loc[name]
                    b0 = str(row["bus0"])
                    b1 = str(row["bus1"])
                    signed_mwh = float(annual_signed.loc[name])
                    abs_mwh = float(annual_abs.loc[name])

                    flow_summary_rows.append(
                        {
                            "link_name": name,
                            "bus0": b0,
                            "bus1": b1,
                            "signed_flow_mwh_from_bus0_to_bus1": signed_mwh,
                            "absolute_flow_mwh": abs_mwh,
                        }
                    )

                    net[b0] = net.get(b0, 0.0) - signed_mwh
                    net[b1] = net.get(b1, 0.0) + signed_mwh

                flow_summary = pd.DataFrame(flow_summary_rows).sort_values("absolute_flow_mwh", ascending=False)
                save_csv(flow_summary, out_dir, "08_interregional_transmission_flows_summary.csv")

                net_imports = pd.DataFrame(
                    [{"region_bus": k, "net_import_mwh": v} for k, v in sorted(net.items())]
                ).sort_values("region_bus")
                save_csv(net_imports, out_dir, "09_net_imports_by_region_mwh.csv")

                cap_col_link = choose_capacity_column(electric_links)
                electric_links[cap_col_link] = pd.to_numeric(electric_links[cap_col_link], errors="coerce").fillna(0.0)

                cap_series = electric_links[cap_col_link].replace(0.0, np.nan)
                util = flow.abs().divide(cap_series, axis=1).replace([np.inf, -np.inf], np.nan)

                util_mean = weighted_time_mean(util.fillna(0.0), weights)
                util_peak = util.max(axis=0).fillna(0.0)
                hours_ge_90 = util.ge(0.9).mul(weights, axis=0).sum(axis=0).fillna(0.0)
                hours_ge_99 = util.ge(0.99).mul(weights, axis=0).sum(axis=0).fillna(0.0)

                util_summary = pd.DataFrame(
                    {
                        "link_name": electric_links.index.astype(str),
                        "bus0": electric_links["bus0"].astype(str).values,
                        "bus1": electric_links["bus1"].astype(str).values,
                        "carrier": electric_links["carrier"].astype(str).values if "carrier" in electric_links.columns else "",
                        "capacity_mw": electric_links[cap_col_link].values,
                        "mean_utilization_pu": util_mean.reindex(electric_links.index).fillna(0.0).values,
                        "peak_utilization_pu": util_peak.reindex(electric_links.index).fillna(0.0).values,
                        "hours_ge_90pct": hours_ge_90.reindex(electric_links.index).fillna(0.0).values,
                        "hours_ge_99pct": hours_ge_99.reindex(electric_links.index).fillna(0.0).values,
                    }
                ).sort_values("peak_utilization_pu", ascending=False)

                save_csv(util_summary, out_dir, "10_transmission_utilization_summary.csv")

    # ------------------------------------------------------------------
    # 11, 12, 13: storage summaries
    # ------------------------------------------------------------------
    if storage_idx is not None and not storage_idx.empty:
        cap_col_storage = choose_capacity_column(storage_idx)
        storage_idx[cap_col_storage] = pd.to_numeric(storage_idx[cap_col_storage], errors="coerce").fillna(0.0)

        storage_by_carrier = (
            storage_idx.groupby("carrier")[cap_col_storage]
            .sum()
            .reset_index()
            .rename(columns={cap_col_storage: "capacity_mw"})
            .sort_values("capacity_mw", ascending=False)
        )
        save_csv(storage_by_carrier, out_dir, "11_storage_capacity_by_carrier.csv")

        storage_by_asset = (
            storage_idx.reset_index()[["name", "carrier", "bus", cap_col_storage]]
            .rename(columns={cap_col_storage: "capacity_mw"})
            .sort_values("capacity_mw", ascending=False)
        )
        save_csv(storage_by_asset, out_dir, "12_storage_capacity_by_asset.csv")

        if stores_p is not None:
            weights = read_snapshot_weights(run_dir, stores_p.index)
            common = [c for c in stores_p.columns if c in storage_idx.index]
            if common:
                p = sanitize_numeric(stores_p[common]).fillna(0.0)

                charge_mwh = weighted_time_sum((-p).clip(lower=0.0), weights)
                discharge_mwh = weighted_time_sum(p.clip(lower=0.0), weights)

                summary = pd.DataFrame(
                    {
                        "carrier": storage_idx.loc[common, "carrier"].astype(str),
                        "bus": storage_idx.loc[common, "bus"].astype(str),
                        "capacity_mw": pd.to_numeric(storage_idx.loc[common, cap_col_storage], errors="coerce").fillna(0.0),
                        "annual_charge_mwh": charge_mwh,
                        "annual_discharge_mwh": discharge_mwh,
                    },
                    index=common,
                )

                if stores_soc is not None:
                    soc_common = [c for c in common if c in stores_soc.columns]
                    if soc_common:
                        soc = sanitize_numeric(stores_soc[soc_common]).fillna(0.0)
                        summary.loc[soc_common, "mean_soc_mwh"] = weighted_time_mean(soc, weights).values
                        summary.loc[soc_common, "max_soc_mwh"] = soc.max(axis=0).values

                summary = summary.reset_index().rename(columns={"index": "storage_name"})
                save_csv(summary, out_dir, "13_storage_dispatch_summary.csv")

    # ------------------------------------------------------------------
    # 14: sector coupling capacities
    # ------------------------------------------------------------------
    sector_rows = []

    if links_idx is not None and not links_idx.empty:
        cap_col_link = choose_capacity_column(links_idx)
        links_idx[cap_col_link] = pd.to_numeric(links_idx[cap_col_link], errors="coerce").fillna(0.0)

        electrolyser_mask = links_idx.index.astype(str).str.endswith("_electrolyzer")
        heat_pump_mask = (
            links_idx.index.astype(str).str.endswith("_heat_pump")
            | links_idx["carrier"].astype(str).eq("heat_pump")
        )
        transport_link_mask = (
            links_idx.index.astype(str).str.endswith("_transport_link")
            | links_idx["carrier"].astype(str).eq("transport")
        )

        for label, mask in [
            ("electrolyser", electrolyser_mask),
            ("heat_pump", heat_pump_mask),
            ("transport_link", transport_link_mask),
        ]:
            if mask.any():
                tmp = links_idx.loc[mask, [cap_col_link, "bus0", "bus1", "carrier"]].copy()
                tmp["component_type"] = label
                tmp = tmp.reset_index().rename(columns={"index": "asset_name", cap_col_link: "capacity_mw"})
                sector_rows.append(tmp)

    if storage_idx is not None and not storage_idx.empty:
        cap_col_storage = choose_capacity_column(storage_idx)
        if storage_idx is not None and not storage_idx.empty:
            cap_col_storage = choose_capacity_column(storage_idx)
            tmp = storage_idx[[cap_col_storage, "bus", "carrier"]].copy()
            tmp["component_type"] = "storage_unit"
            tmp["bus0"] = tmp["bus"]
            tmp["bus1"] = ""
            tmp = tmp.reset_index()

            first_col = tmp.columns[0]
            tmp = tmp.rename(columns={first_col: "asset_name", cap_col_storage: "capacity_mw"})

            sector_rows.append(tmp[["asset_name", "capacity_mw", "bus0", "bus1", "carrier", "component_type"]])

    if sector_rows:
        sector_df = pd.concat(sector_rows, ignore_index=True, sort=False)
        save_csv(sector_df, out_dir, "14_sector_coupling_capacity_summary.csv")

    # ------------------------------------------------------------------
    # 15: hydrogen summary
    # ------------------------------------------------------------------
    hydrogen_parts = []

    if loads_idx is not None:
        h2_loads = loads_idx[loads_idx["bus"].astype(str).str.endswith("_hydrogen")]
        if not h2_loads.empty:
            tmp = h2_loads.reset_index()[["name", "bus", "carrier"]].copy()
            tmp["category"] = "hydrogen_load"
            hydrogen_parts.append(tmp)

    if links_idx is not None:
        h2_links = links_idx[
            links_idx["bus1"].astype(str).str.endswith("_hydrogen")
            | links_idx.index.astype(str).str.endswith("_electrolyzer")
            | links_idx.index.astype(str).str.endswith("hydrogen")
        ].copy()
        if not h2_links.empty:
            cap_col_link = choose_capacity_column(h2_links)
            tmp = h2_links.reset_index()[["name", "bus0", "bus1", "carrier", cap_col_link]].copy()
            tmp["category"] = "hydrogen_link"
            tmp = tmp.rename(columns={cap_col_link: "capacity_mw"})
            hydrogen_parts.append(tmp)

    if storage_idx is not None:
        h2_storage = storage_idx[storage_idx["bus"].astype(str).str.endswith("_hydrogen")].copy()
        if not h2_storage.empty:
            cap_col_storage = choose_capacity_column(h2_storage)
            tmp = h2_storage.reset_index()[["name", "bus", "carrier", cap_col_storage]].copy()
            tmp["category"] = "hydrogen_storage"
            tmp = tmp.rename(columns={cap_col_storage: "capacity_mw"})
            hydrogen_parts.append(tmp)

    if hydrogen_parts:
        hydrogen_df = pd.concat(hydrogen_parts, ignore_index=True, sort=False)
        save_csv(hydrogen_df, out_dir, "15_hydrogen_summary.csv")

    # ------------------------------------------------------------------
    # 16: heat summary
    # ------------------------------------------------------------------
    heat_parts = []

    if loads_idx is not None:
        heat_loads = loads_idx[loads_idx["bus"].astype(str).str.endswith("_heat")]
        if not heat_loads.empty:
            tmp = heat_loads.reset_index()[["name", "bus", "carrier"]].copy()
            tmp["category"] = "heat_load"
            heat_parts.append(tmp)

    if links_idx is not None:
        heat_links = links_idx[
            links_idx["bus1"].astype(str).str.endswith("_heat")
            | links_idx.index.astype(str).str.endswith("_heat_pump")
            | links_idx["carrier"].astype(str).eq("heat_pump")
        ].copy()
        if not heat_links.empty:
            cap_col_link = choose_capacity_column(heat_links)
            tmp = heat_links.reset_index()[["name", "bus0", "bus1", "carrier", cap_col_link]].copy()
            tmp["category"] = "heat_link"
            tmp = tmp.rename(columns={cap_col_link: "capacity_mw"})
            heat_parts.append(tmp)

    if storage_idx is not None:
        heat_storage = storage_idx[storage_idx["bus"].astype(str).str.endswith("_heat")].copy()
        if not heat_storage.empty:
            cap_col_storage = choose_capacity_column(heat_storage)
            tmp = heat_storage.reset_index()[["name", "bus", "carrier", cap_col_storage]].copy()
            tmp["category"] = "heat_storage"
            tmp = tmp.rename(columns={cap_col_storage: "capacity_mw"})
            heat_parts.append(tmp)

    if heat_parts:
        heat_df = pd.concat(heat_parts, ignore_index=True, sort=False)
        save_csv(heat_df, out_dir, "16_heat_summary.csv")

    # ------------------------------------------------------------------
    # 17: transport summary
    # ------------------------------------------------------------------
    transport_parts = []

    if loads_idx is not None:
        t_loads = loads_idx[loads_idx["bus"].astype(str).str.endswith("_transport")]
        if not t_loads.empty:
            tmp = t_loads.reset_index()[["name", "bus", "carrier"]].copy()
            tmp["category"] = "transport_load"
            transport_parts.append(tmp)

    if links_idx is not None:
        t_links = links_idx[
            links_idx["bus1"].astype(str).str.endswith("_transport")
            | links_idx.index.astype(str).str.endswith("_transport_link")
            | links_idx["carrier"].astype(str).eq("transport")
        ].copy()
        if not t_links.empty:
            cap_col_link = choose_capacity_column(t_links)
            tmp = t_links.reset_index()[["name", "bus0", "bus1", "carrier", cap_col_link]].copy()
            tmp["category"] = "transport_link"
            tmp = tmp.rename(columns={cap_col_link: "capacity_mw"})
            transport_parts.append(tmp)

    if transport_parts:
        transport_df = pd.concat(transport_parts, ignore_index=True, sort=False)
        save_csv(transport_df, out_dir, "17_transport_summary.csv")

    # ------------------------------------------------------------------
    # 18, 19: VRES curtailment
    # ------------------------------------------------------------------
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
                        "generator_name": common_vres,
                        "carrier": gens_idx.loc[common_vres, "carrier"].astype(str).values,
                        "bus": gens_idx.loc[common_vres, "bus"].astype(str).values,
                        "installed_capacity_mw": cap.values,
                        "available_mwh": available_mwh.values,
                        "dispatched_mwh": dispatch_mwh.values,
                        "curtailed_mwh": curtailed_mwh.values,
                        "curtailment_share": share.values,
                    }
                ).sort_values("curtailed_mwh", ascending=False)
                save_csv(curtail_detail, out_dir, "18_vres_curtailment_detail.csv")

                curtail_by_carrier = (
                    curtail_detail.groupby("carrier")[["available_mwh", "dispatched_mwh", "curtailed_mwh"]]
                    .sum()
                    .reset_index()
                )
                curtail_by_carrier["curtailment_share"] = (
                    curtail_by_carrier["curtailed_mwh"] / curtail_by_carrier["available_mwh"].replace(0.0, np.nan)
                ).fillna(0.0)
                curtail_by_carrier = curtail_by_carrier.sort_values("curtailed_mwh", ascending=False)
                save_csv(curtail_by_carrier, out_dir, "19_vres_curtailment_by_carrier.csv")

    # ------------------------------------------------------------------
    # 20: dominant power flow direction
    # ------------------------------------------------------------------
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

                dominant_df = pd.DataFrame(dominant_rows).sort_values("dominant_signed_mwh", ascending=False)
                save_csv(dominant_df, out_dir, "20_dominant_power_flow_direction.csv")

    print(f"Saved CSV summaries to: {out_dir}")


def is_run_dir(path: Path) -> bool:
    required = ["generators.csv", "buses.csv", "carriers.csv"]
    return path.is_dir() and all((path / f).exists() for f in required)


def find_run_dirs(runs_root: Path) -> list[Path]:
    run_dirs = [p for p in runs_root.iterdir() if is_run_dir(p)]
    return sorted(run_dirs, key=lambda p: p.name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default=None, type=str, help="Single run folder")
    ap.add_argument("--runs_root", default=None, type=str, help="Folder containing many run folders")
    ap.add_argument("--out_dir", default=None, type=str, help="Only used in single-run mode")
    args = ap.parse_args()

    if bool(args.run_dir) == bool(args.runs_root):
        raise ValueError("Provide exactly one of --run_dir or --runs_root.")

    if args.run_dir:
        run_dir = Path(args.run_dir)
        if not run_dir.exists():
            raise FileNotFoundError(f"run_dir not found: {run_dir}")

        out_dir = Path(args.out_dir) if args.out_dir else (run_dir / "csv_results")
        export_run_csvs(run_dir, out_dir)
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
            out_dir = run_dir / "csv_results"
            export_run_csvs(run_dir, out_dir)
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
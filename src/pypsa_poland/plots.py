# src/pypsa_poland/plots.py
# Run:
# & C:\Users\adria\anaconda3\envs\pypsa-legacy\python.exe C:\Users\adria\MODEL_PyPSA\Core\pypsa-poland_ADRIAN\src\pypsa_poland\plots.py --run_dir C:\Users\adria\MODEL_PyPSA\Core\runs\run_2020_20260306_010448_GasAdded_1hr_SubOptimal_3545s

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_ts(run_dir: Path, stem: str) -> pd.DataFrame:
    path = run_dir / f"{stem}.csv"
    if not path.exists():
        raise FileNotFoundError(str(path))
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index, errors="coerce")
    if df.index.isna().any():
        df = df.loc[~df.index.isna()]
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, type=str)
    ap.add_argument("--out_dir", default=None, type=str)
    ap.add_argument("--top_carriers", default=8, type=int)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir not found: {run_dir}")

    out_dir = Path(args.out_dir) if args.out_dir else (run_dir / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Load metadata ----------
    gens = pd.read_csv(run_dir / "generators.csv")
    links = pd.read_csv(run_dir / "links.csv") if (run_dir / "links.csv").exists() else None
    loads = pd.read_csv(run_dir / "loads.csv") if (run_dir / "loads.csv").exists() else None

   
    # ---------- Plot A: Capacity mix per region (stacked bar) ----------
    cap_col = choose_capacity_column(gens)
    gens[cap_col] = pd.to_numeric(gens[cap_col], errors="coerce").fillna(0.0)

    if "bus" not in gens.columns or "carrier" not in gens.columns:
        raise ValueError("generators.csv must include 'bus' and 'carrier' columns.")

    gens_reg = gens[region_bus_mask(gens["bus"])].copy()
    cap = gens_reg.groupby(["bus", "carrier"])[cap_col].sum().unstack(fill_value=0.0)

    # Drop carriers that are essentially zero system-wide (cleans legend)
    tot = cap.sum(axis=0)
    cap = cap.loc[:, tot[tot > 1e-6].index]

    # Keep top carriers by total capacity, rest -> other
    top = cap.sum(axis=0).sort_values(ascending=False).head(args.top_carriers).index
    cap_plot = cap[top].copy()
    rest = cap.drop(columns=top, errors="ignore")
    if rest.shape[1] > 0:
        cap_plot["other"] = rest.sum(axis=1)

    cap_plot = cap_plot.sort_index()

    plt.figure(figsize=(12, 6))
    bottom = np.zeros(len(cap_plot))
    x = np.arange(len(cap_plot.index))
    for col in cap_plot.columns:
        y = cap_plot[col].values
        plt.bar(x, y, bottom=bottom, label=col)
        bottom += y

    plt.xticks(x, [b.replace("PL ", "") for b in cap_plot.index], rotation=0)
    plt.title("Installed capacity mix by region (stacked)")
    plt.xlabel("Region")
    plt.ylabel("MW")
    plt.legend(ncols=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_dir / "capacity_mix_by_region_stacked.png", dpi=220)
    plt.close()

    # ---------- Plot A2: Energy mix by region (stacked, requires generators-p or generators-p_set) ----------
    stem_gen, gen_p = try_read_ts(run_dir, ["generators-p", "generators-p_set"])
    if gen_p is not None and "name" in gens.columns:
        gens_idx = gens.set_index("name")
        common = [c for c in gen_p.columns if c in gens_idx.index]
        if common:
            meta = gens_idx.loc[common, ["bus", "carrier"]].copy()
            meta = meta[region_bus_mask(meta["bus"])]

            if len(meta) > 0:
                gen_p2 = gen_p[meta.index]

                # annual energy per generator (MW * hours step -> MWh if hourly)
                E = gen_p2.sum(axis=0)

                # energy by (bus, carrier)
                tmp = pd.DataFrame({"bus": meta["bus"], "carrier": meta["carrier"], "E": E.values}, index=meta.index)
                E_bus_car = tmp.groupby(["bus", "carrier"])["E"].sum().unstack(fill_value=0.0)

                # drop tiny carriers for clean legend
                totE = E_bus_car.sum(axis=0)
                E_bus_car = E_bus_car.loc[:, totE[totE > 1e-6].index]

                topE = E_bus_car.sum(axis=0).sort_values(ascending=False).head(args.top_carriers).index
                E_plot = E_bus_car[topE].copy()
                restE = E_bus_car.drop(columns=topE, errors="ignore")
                if restE.shape[1] > 0:
                    E_plot["other"] = restE.sum(axis=1)

                E_plot = E_plot.sort_index()

                plt.figure(figsize=(12, 6))
                bottom = np.zeros(len(E_plot))
                x = np.arange(len(E_plot.index))
                for col in E_plot.columns:
                    y = E_plot[col].values
                    plt.bar(x, y, bottom=bottom, label=col)
                    bottom += y

                plt.xticks(x, [b.replace("PL ", "") for b in E_plot.index], rotation=0)
                plt.title("Energy generation mix by region (stacked)")
                plt.xlabel("Region")
                plt.ylabel("MW·h (relative MWh if hourly)")
                plt.legend(ncols=2, fontsize=9)
                plt.tight_layout()
                plt.savefig(out_dir / "energy_mix_by_region_stacked.png", dpi=220)
                plt.close()

    # ---------- Plot A3: Net imports by region (requires links-p0 and links.csv with bus0/bus1) ----------
    stem_l0, links_p0 = try_read_ts(run_dir, ["links-p0"])
    if links_p0 is not None and links is not None and all(c in links.columns for c in ["name", "bus0", "bus1"]):
        links_meta = links.set_index("name")
        common = [c for c in links_p0.columns if c in links_meta.index]
        if common:
            meta = links_meta.loc[common, ["bus0", "bus1"]].copy()
            meta["bus0"] = meta["bus0"].astype(str)
            meta["bus1"] = meta["bus1"].astype(str)

            rr = meta[region_bus_mask(meta["bus0"]) & region_bus_mask(meta["bus1"])]
            if len(rr) > 0:
                p0 = links_p0[rr.index]

                # p0 is flow from bus0 -> bus1 (positive means export from bus0)
                net = {b: 0.0 for b in sorted(pd.concat([rr["bus0"], rr["bus1"]]).unique())}
                annual = p0.sum(axis=0)  # MW·h if hourly

                for link_name, row in rr.iterrows():
                    b0, b1 = row["bus0"], row["bus1"]
                    v = float(annual.loc[link_name])
                    net[b0] -= v  # export reduces net imports
                    net[b1] += v  # import increases net imports

                net_s = pd.Series(net).sort_index()

                plt.figure(figsize=(12, 4))
                x = np.arange(len(net_s.index))
                plt.bar(x, net_s.values)
                plt.xticks(x, [b.replace("PL ", "") for b in net_s.index])
                plt.axhline(0, linewidth=1)
                plt.title("Net inter-regional imports by region (positive = net importer)")
                plt.xlabel("Region")
                plt.ylabel("MW·h (relative MWh if hourly)")
                plt.tight_layout()
                plt.savefig(out_dir / "net_imports_by_region.png", dpi=220)
                plt.close()

    # ---------- Plot B: Total load + Load duration curve ----------
    stem_load, loads_p = try_read_ts(run_dir, ["loads-p_set"])
    if loads_p is not None:
        total_load = loads_p.sum(axis=1)

        plt.figure(figsize=(12, 4))
        plt.plot(total_load.values)
        plt.title("Total load (system)")
        plt.xlabel("Hour")
        plt.ylabel("MW")
        plt.tight_layout()
        plt.savefig(out_dir / "total_load.png", dpi=220)
        plt.close()

        ldc = total_load.sort_values(ascending=False).values
        plt.figure(figsize=(8, 4))
        plt.plot(ldc)
        plt.title("Load duration curve")
        plt.xlabel("Hour rank")
        plt.ylabel("MW")
        plt.tight_layout()
        plt.savefig(out_dir / "load_duration_curve.png", dpi=220)
        plt.close()

    # ---------- Plot C: Energy by carrier (annual MWh) ----------
    stem_gen, gen_p = try_read_ts(run_dir, ["generators-p", "generators-p_set"])
    if gen_p is not None:
        # Ensure gens index aligns to gen_p columns
        # generators.csv usually has a 'name' column; if not, assume same order (rare)
        if "name" in gens.columns:
            gens_idx = gens.set_index("name")
            common = [c for c in gen_p.columns if c in gens_idx.index]
            if common:
                gen_p2 = gen_p[common]
                carrier = gens_idx.loc[common, "carrier"]
                gen_by_carrier = gen_p2.T.groupby(carrier).sum().T
                energy = gen_by_carrier.sum(axis=0)  # MW * hours step ~ MWh if hourly
                energy = energy.sort_values(ascending=False)

                plt.figure(figsize=(10, 5))
                plt.bar(np.arange(len(energy.index)), energy.values)
                plt.xticks(np.arange(len(energy.index)), energy.index, rotation=45, ha="right")
                plt.title("Annual energy by carrier (sum over time; units ~ MWh if hourly)")
                plt.ylabel("MW·h (relative)")
                plt.tight_layout()
                plt.savefig(out_dir / "energy_by_carrier.png", dpi=220)
                plt.close()

    # ---------- Plot D: Link capacity matrix between regions (AC/DC/hydrogen etc.) ----------
    if links is not None and all(c in links.columns for c in ["bus0", "bus1"]):
        links2 = links.copy()
        for c in ["bus0", "bus1"]:
            links2[c] = links2[c].astype(str)

        rr = links2[region_bus_mask(links2["bus0"]) & region_bus_mask(links2["bus1"])].copy()
        if len(rr) > 0:
            cap_col_l = "p_nom_opt" if "p_nom_opt" in rr.columns else ("p_nom" if "p_nom" in rr.columns else None)
            if cap_col_l is not None:
                rr[cap_col_l] = pd.to_numeric(rr[cap_col_l], errors="coerce").fillna(0.0)
                rr["link_type"] = rr["carrier"].astype(str) if "carrier" in rr.columns else "Link"

                # choose one type to show matrix for: prefer AC if present
                preferred = "AC" if "carrier" in rr.columns and (rr["carrier"] == "AC").any() else rr["link_type"].iloc[0]
                mat = rr[rr["link_type"] == preferred].pivot_table(
                    index="bus0", columns="bus1", values=cap_col_l, aggfunc="sum", fill_value=0.0
                )
                # make symmetric for visual
                buses = sorted(set(mat.index).union(set(mat.columns)))
                mat = mat.reindex(index=buses, columns=buses, fill_value=0.0)
                mat = mat + mat.T

                plt.figure(figsize=(8, 7))
                plt.imshow(mat.values, aspect="auto")
                plt.title(f"Inter-regional link capacity matrix ({preferred})")
                plt.xticks(np.arange(len(buses)), [b.replace("PL ", "") for b in buses], rotation=90)
                plt.yticks(np.arange(len(buses)), [b.replace("PL ", "") for b in buses])
                plt.colorbar(label=f"{cap_col_l} (MW)")
                plt.tight_layout()
                plt.savefig(out_dir / "link_capacity_matrix.png", dpi=220)
                plt.close()

    # ---------- Plot E (optional): Total interregional flow over time if links-p0 exists ----------
    stem_l0, links_p0 = try_read_ts(run_dir, ["links-p0"])
    if links_p0 is not None and links is not None and "name" in links.columns:
        # Keep only interregional region->region links if possible
        links_meta = links.set_index("name")
        common = [c for c in links_p0.columns if c in links_meta.index]
        if common:
            meta = links_meta.loc[common]
            mask_rr = region_bus_mask(meta["bus0"].astype(str)) & region_bus_mask(meta["bus1"].astype(str))
            rr_cols = meta[mask_rr].index.tolist()
            if rr_cols:
                total_abs_flow = links_p0[rr_cols].abs().sum(axis=1)
                plt.figure(figsize=(12, 4))
                plt.plot(total_abs_flow.values)
                plt.title("Total absolute inter-regional link flow (sum |p0|)")
                plt.xlabel("Hour")
                plt.ylabel("MW")
                plt.tight_layout()
                plt.savefig(out_dir / "total_interregional_flow.png", dpi=220)
                plt.close()

    print(f"Saved figures to: {out_dir}")


if __name__ == "__main__":
    main()
# src/pypsa_poland/plots.py

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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


def region_bus_mask(bus_series: pd.Series) -> pd.Series:
    b = bus_series.astype(str)
    return b.str.match(r"^PL\s+[A-Z]{2}$")


def is_run_dir(path: Path) -> bool:
    required = ["generators.csv", "buses.csv", "carriers.csv"]
    return path.is_dir() and all((path / f).exists() for f in required)


def find_run_dirs(runs_root: Path) -> list[Path]:
    run_dirs = [p for p in runs_root.iterdir() if is_run_dir(p)]
    return sorted(run_dirs, key=lambda p: p.name)


def make_plots_for_run(run_dir: Path, out_dir: Path, top_carriers: int) -> None:
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

    tot = cap.sum(axis=0)
    cap = cap.loc[:, tot[tot > 1e-6].index]

    top = cap.sum(axis=0).sort_values(ascending=False).head(top_carriers).index
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

    # ---------- Plot A2: Energy mix by region (stacked, weighted) ----------
    stem_gen, gen_p = try_read_ts(run_dir, ["generators-p", "generators-p_set"])
    if gen_p is not None and "name" in gens.columns:
        weights = read_snapshot_weights(run_dir, gen_p.index)
        gens_idx = gens.set_index("name")
        common = [c for c in gen_p.columns if c in gens_idx.index]
        if common:
            meta = gens_idx.loc[common, ["bus", "carrier"]].copy()
            meta = meta[region_bus_mask(meta["bus"])]

            if len(meta) > 0:
                gen_p2 = gen_p[meta.index]
                E = weighted_time_sum(gen_p2, weights)

                tmp = pd.DataFrame(
                    {"bus": meta["bus"], "carrier": meta["carrier"], "E": E.values},
                    index=meta.index,
                )
                E_bus_car = tmp.groupby(["bus", "carrier"])["E"].sum().unstack(fill_value=0.0)

                totE = E_bus_car.sum(axis=0)
                E_bus_car = E_bus_car.loc[:, totE[totE > 1e-6].index]

                topE = E_bus_car.sum(axis=0).sort_values(ascending=False).head(top_carriers).index
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
                plt.ylabel("MWh (snapshot-weighted)")
                plt.legend(ncols=2, fontsize=9)
                plt.tight_layout()
                plt.savefig(out_dir / "energy_mix_by_region_stacked.png", dpi=220)
                plt.close()

    # ---------- Plot A3: Net imports by region (weighted) ----------
    stem_l0, links_p0 = try_read_ts(run_dir, ["links-p0"])
    if links_p0 is not None and links is not None and all(c in links.columns for c in ["name", "bus0", "bus1"]):
        weights = read_snapshot_weights(run_dir, links_p0.index)
        links_meta = links.set_index("name")
        common = [c for c in links_p0.columns if c in links_meta.index]
        if common:
            meta = links_meta.loc[common, ["bus0", "bus1"]].copy()
            meta["bus0"] = meta["bus0"].astype(str)
            meta["bus1"] = meta["bus1"].astype(str)

            rr = meta[region_bus_mask(meta["bus0"]) & region_bus_mask(meta["bus1"])]
            if len(rr) > 0:
                p0 = links_p0[rr.index]

                net = {b: 0.0 for b in sorted(pd.concat([rr["bus0"], rr["bus1"]]).unique())}
                annual = weighted_time_sum(p0, weights)

                for link_name, row in rr.iterrows():
                    b0, b1 = row["bus0"], row["bus1"]
                    v = float(annual.loc[link_name])
                    net[b0] -= v
                    net[b1] += v

                net_s = pd.Series(net).sort_index()

                plt.figure(figsize=(12, 4))
                x = np.arange(len(net_s.index))
                plt.bar(x, net_s.values)
                plt.xticks(x, [b.replace("PL ", "") for b in net_s.index])
                plt.axhline(0, linewidth=1)
                plt.title("Net inter-regional imports by region (positive = net importer)")
                plt.xlabel("Region")
                plt.ylabel("MWh (snapshot-weighted)")
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
        plt.xlabel("Snapshot")
        plt.ylabel("MW")
        plt.tight_layout()
        plt.savefig(out_dir / "total_load.png", dpi=220)
        plt.close()

        ldc = total_load.sort_values(ascending=False).values
        plt.figure(figsize=(8, 4))
        plt.plot(ldc)
        plt.title("Load duration curve")
        plt.xlabel("Snapshot rank")
        plt.ylabel("MW")
        plt.tight_layout()
        plt.savefig(out_dir / "load_duration_curve.png", dpi=220)
        plt.close()

    # ---------- Plot C: Energy by carrier (annual MWh, weighted) ----------
    stem_gen, gen_p = try_read_ts(run_dir, ["generators-p", "generators-p_set"])
    if gen_p is not None and "name" in gens.columns:
        weights = read_snapshot_weights(run_dir, gen_p.index)
        gens_idx = gens.set_index("name")
        common = [c for c in gen_p.columns if c in gens_idx.index]
        if common:
            gen_p2 = gen_p[common]
            carrier = gens_idx.loc[common, "carrier"]
            energy_per_gen = weighted_time_sum(gen_p2, weights)
            energy = energy_per_gen.groupby(carrier).sum().sort_values(ascending=False)

            plt.figure(figsize=(10, 5))
            plt.bar(np.arange(len(energy.index)), energy.values)
            plt.xticks(np.arange(len(energy.index)), energy.index, rotation=45, ha="right")
            plt.title("Annual energy by carrier")
            plt.ylabel("MWh (snapshot-weighted)")
            plt.tight_layout()
            plt.savefig(out_dir / "energy_by_carrier.png", dpi=220)
            plt.close()

    # ---------- Plot D: Link capacity matrix between regions ----------
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

                preferred = "AC" if "carrier" in rr.columns and (rr["carrier"] == "AC").any() else rr["link_type"].iloc[0]
                mat = rr[rr["link_type"] == preferred].pivot_table(
                    index="bus0", columns="bus1", values=cap_col_l, aggfunc="sum", fill_value=0.0
                )

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

    # ---------- Plot E: Total interregional flow over time ----------
    stem_l0, links_p0 = try_read_ts(run_dir, ["links-p0"])
    if links_p0 is not None and links is not None and "name" in links.columns:
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
                plt.xlabel("Snapshot")
                plt.ylabel("MW")
                plt.tight_layout()
                plt.savefig(out_dir / "total_interregional_flow.png", dpi=220)
                plt.close()

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
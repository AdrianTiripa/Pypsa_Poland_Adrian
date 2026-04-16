# src/pypsa_poland/summary_stats.py
#
# Quick human-readable summary of a single pypsa-poland run.
#
# Reads the static component CSVs from a run directory (generators, links,
# storage_units) and prints a formatted summary covering:
#   1. Generator installed capacities by carrier.
#   2. Storage installed capacities (power and energy) by carrier.
#   3. Interregional electric transmission link capacities.
#   4. H₂ pipeline and electrolyser capacities.
#   5. Heat pump capacities by region.
#
# Usage:
#   python -m pypsa_poland.summary_stats --run_dir <path_to_run_folder>
#
# The output can be redirected to a text file:
#   python -m pypsa_poland.summary_stats --run_dir <path> > summary.txt

from __future__ import annotations

import argparse
from pathlib import Path

import sys
sys.stdout.reconfigure(encoding="utf-8")   # ensure Unicode output on Windows

import pandas as pd


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def read_csv(run_dir: Path, filename: str) -> pd.DataFrame | None:
    """Read a component CSV from the run directory; return None if absent."""
    path = run_dir / filename
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "name" in df.columns:
        df = df.set_index("name")
    return df


def choose_cap(df: pd.DataFrame) -> str:
    """Return 'p_nom_opt' if it exists (post-solve), otherwise 'p_nom'."""
    return "p_nom_opt" if "p_nom_opt" in df.columns else "p_nom"


def fmt(val: float, unit: str = "GW", decimals: int = 2) -> str:
    """Format a MW value to GW or MWh to GWh with a given number of decimal places."""
    divisor = 1000 if unit in ("GW", "GWh") else 1
    return f"{val / divisor:.{decimals}f} {unit}"


def print_section(title: str) -> None:
    """Print a section header with a fixed-width separator line."""
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")


# ---------------------------------------------------------------------------
# Summary logic
# ---------------------------------------------------------------------------

def run_summary(run_dir: Path) -> None:
    """Print a formatted capacity and dispatch summary for the given run folder."""
    run_dir = Path(run_dir)
    print(f"\nRun: {run_dir.name}")

    gens    = read_csv(run_dir, "generators.csv")
    links   = read_csv(run_dir, "links.csv")
    storage = read_csv(run_dir, "storage_units.csv")

    # 1. Generator installed capacities
    print_section("1. Generator installed capacities")
    if gens is not None and "carrier" in gens.columns:
        cap_col = choose_cap(gens)
        gens[cap_col] = pd.to_numeric(gens[cap_col], errors="coerce").fillna(0.0)
        by_carrier = gens.groupby("carrier")[cap_col].sum().sort_values(ascending=False)
        for carrier, mw in by_carrier.items():
            if mw > 1:
                print(f"  {carrier:<30} {fmt(mw)}")

    # 2. Storage installed capacities (MW and GWh)
    print_section("2. Storage installed capacities")
    if storage is not None and "carrier" in storage.columns:
        cap_col = choose_cap(storage)
        storage[cap_col] = pd.to_numeric(storage[cap_col], errors="coerce").fillna(0.0)
        storage["max_hours"]   = pd.to_numeric(storage.get("max_hours", 0), errors="coerce").fillna(0.0)
        storage["energy_mwh"]  = storage[cap_col] * storage["max_hours"]

        by_carrier = storage.groupby("carrier").agg(
            power_mw=(cap_col, "sum"),
            energy_mwh=("energy_mwh", "sum"),
        ).sort_values("power_mw", ascending=False)

        print(f"  {'Carrier':<30} {'Power':>10} {'Energy':>12}")
        print(f"  {'-'*54}")
        for carrier, row in by_carrier.iterrows():
            if row["power_mw"] > 1:
                print(f"  {carrier:<30} {fmt(row['power_mw']):>10} {fmt(row['energy_mwh'], 'GWh'):>12}")

    # 3. Transmission lines (electric interregional)
    print_section("3. Transmission lines (electric interregional)")
    if links is not None and "bus0" in links.columns and "bus1" in links.columns:
        cap_col = choose_cap(links)
        links[cap_col] = pd.to_numeric(links[cap_col], errors="coerce").fillna(0.0)

        b0      = links["bus0"].astype(str)
        b1      = links["bus1"].astype(str)
        carrier = (
            links["carrier"].astype(str).str.lower()
            if "carrier" in links.columns
            else pd.Series("", index=links.index)
        )
        idx = links.index.astype(str).str.lower()

        # Keep only region-to-region electric links (exclude hydrogen, heat, transport).
        elec_mask = (
            b0.str.match(r"^PL\s+[A-Z]{2}$") & b1.str.match(r"^PL\s+[A-Z]{2}$")
            & ~carrier.str.contains("hydrogen|heat|transport", na=False)
            & ~idx.str.contains("hydrogen|heat|transport", na=False)
            & ~b0.str.contains("_hydrogen|_heat|_transport", na=False)
            & ~b1.str.contains("_hydrogen|_heat|_transport", na=False)
        )
        elinks = links.loc[elec_mask].copy()

        print(f"  {'Link':<35} {'Capacity':>10}")
        print(f"  {'-'*47}")
        for name, row in elinks.sort_values(cap_col, ascending=False).iterrows():
            mw = row[cap_col]
            if mw > 1:
                label = f"{row['bus0']} -> {row['bus1']}"
                print(f"  {label:<35} {fmt(mw):>10}")

        print(f"\n  Total electric interregional capacity: {fmt(elinks[cap_col].sum())}")

    # 4. H₂ pipelines and electrolysers
    print_section("4. H₂ pipelines and electrolysers")
    if links is not None:
        cap_col  = choose_cap(links)
        ely_mask = links.index.astype(str).str.endswith("_electrolyzer")
        pipe_mask = (
            links.index.astype(str).str.contains("hydrogen", case=False, na=False)
            & ~ely_mask
            & ~links.index.astype(str).str.contains("chp", case=False, na=False)
        )

        if ely_mask.any():
            ely = links.loc[ely_mask, cap_col].sort_values(ascending=False)
            print(f"\n  Electrolysers (by region):")
            for name, mw in ely.items():
                mw = float(mw)
                if mw > 1:
                    region = str(name).replace("_electrolyzer", "").replace("PL ", "")
                    print(f"    {region:<10} {fmt(mw)}")
            print(f"    {'TOTAL':<10} {fmt(ely.sum())}")

        if pipe_mask.any():
            pipes = links.loc[pipe_mask, cap_col].sort_values(ascending=False)
            print(f"\n  H₂ pipelines (top 10 by capacity):")
            for name, mw in pipes.head(10).items():
                mw = float(mw)
                if mw > 1:
                    print(f"    {str(name):<45} {fmt(mw)}")
            print(f"    {'TOTAL (all pipes)':<45} {fmt(pipes.sum())}")

    # 5. Heat pumps
    print_section("5. Heat pumps")
    if links is not None:
        cap_col = choose_cap(links)
        hp_mask = (
            links.index.astype(str).str.endswith("_heat_pump")
            | (
                links["carrier"].astype(str).eq("heat_pump")
                if "carrier" in links.columns
                else pd.Series(False, index=links.index)
            )
        )
        if hp_mask.any():
            hp = links.loc[hp_mask, cap_col].sort_values(ascending=False)
            print(f"\n  Heat pumps (by region):")
            for name, mw in hp.items():
                mw = float(mw)
                if mw > 0.1:
                    region = str(name).replace("_heat_pump", "").replace("PL ", "")
                    print(f"    {region:<10} {fmt(mw)}")
            print(f"    {'TOTAL':<10} {fmt(hp.sum())}")

    print(f"\n{'='*55}\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Print a capacity summary for a single run folder.")
    ap.add_argument("--run_dir", required=True, type=str, help="Path to the run folder.")
    args = ap.parse_args()
    run_summary(Path(args.run_dir))


if __name__ == "__main__":
    main()

# src/pypsa_poland/orchestration.py

from __future__ import annotations

import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
import pypsa

from .components import REGISTRY

logger = logging.getLogger(__name__)


def _make_run_dir(runs_folder: Path, base_name: str = "run") -> Path:
    runs_folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = runs_folder / f"{base_name}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def downsample_snapshots(n: pypsa.Network, step: int) -> pypsa.Network:
    """
    Keep every `step`-th snapshot and scale snapshot_weightings so each kept
    snapshot represents `step` hours.

    IMPORTANT:
    Call this only AFTER importing time-dependent CSVs (import_from_csv_folder),
    otherwise PyPSA will try to index 8760-row time series with a shorter snapshots index.
    """
    step = int(step)
    if step <= 1:
        return n

    before = len(n.snapshots)
    kept = n.snapshots[::step]

    # Use set_snapshots so PyPSA reindexes all *_t tables consistently
    n.set_snapshots(kept)

    # Scale weightings so objective still represents the full year
    w = n.snapshot_weightings.loc[kept].copy()
    for col in w.columns:
        w[col] = w[col] * step
    n.snapshot_weightings = w

    logger.info("Downsampled snapshots by step=%d: %d -> %d", step, before, len(kept))
    return n


def build_network(cfg: dict) -> pypsa.Network:
    n = pypsa.Network()

    year = int(cfg["snapshots"]["year"])

    # Full calendar year, hourly, left-inclusive (8760 after dropping Feb 29 if leap)
    snapshots = pd.date_range(
        start=f"{year}-01-01 00:00",
        end=f"{year+1}-01-01 00:00",
        freq="H",
        inclusive="left",
    )

    if len(snapshots) == 8784:
        snapshots = snapshots[~((snapshots.month == 2) & (snapshots.day == 29))]

    if len(snapshots) != 8760:
        raise ValueError(f"Snapshot length {len(snapshots)} != 8760 for year={year}")

    # Set FULL hourly snapshots first (needed for CSV import of time series)
    n.set_snapshots(snapshots)

    # Import base network from CSV folder (may include 8760-row time series tables)
    input_folder = Path(cfg["paths"]["input_folder"])
    n.import_from_csv_folder(str(input_folder))

    # Ensure all carriers referenced by components exist
    for c in pd.unique(n.buses.carrier.dropna()):
        if c not in n.carriers.index:
            n.add("Carrier", c)

    for c in pd.unique(n.generators.carrier.dropna()):
        if c not in n.carriers.index:
            n.add("Carrier", c)

    for c in pd.unique(n.links.carrier.dropna()):
        if c not in n.carriers.index:
            n.add("Carrier", c)

    for c in pd.unique(n.storage_units.carrier.dropna()):
        if c not in n.carriers.index:
            n.add("Carrier", c)

    # Links that don't have bus2 shouldn't need efficiency2; set to 0 to avoid NaNs
    if "efficiency2" in n.links.columns:
        mask = n.links["bus2"].isna() & n.links["efficiency2"].isna()
        n.links.loc[mask, "efficiency2"] = 0.0

    # NOW apply config stepsize (safe because time-series import already matched 8760)
    step = int(cfg["snapshots"].get("stepsize", 1))
    n = downsample_snapshots(n, step)

    return n

def ensure_all_component_carriers(n: pypsa.Network) -> pypsa.Network:
    """
    Ensure every carrier referenced by buses, generators, links, loads, storage_units exists in n.carriers.
    This prevents PyPSA consistency warnings and downstream surprises.
    """
    def _ensure(series: pd.Series) -> None:
        for c in pd.unique(series.dropna()):
            if c not in n.carriers.index:
                n.add("Carrier", c)

    if hasattr(n, "buses") and "carrier" in n.buses.columns:
        _ensure(n.buses["carrier"])
    if hasattr(n, "generators") and "carrier" in n.generators.columns:
        _ensure(n.generators["carrier"])
    if hasattr(n, "links") and "carrier" in n.links.columns:
        _ensure(n.links["carrier"])
    if hasattr(n, "loads") and "carrier" in n.loads.columns:
        _ensure(n.loads["carrier"])
    if hasattr(n, "storage_units") and "carrier" in n.storage_units.columns:
        _ensure(n.storage_units["carrier"])

    return n

def apply_pipeline(n: pypsa.Network, cfg: dict) -> pypsa.Network:
    for step in cfg.get("pipeline", []):
        if step not in REGISTRY:
            raise KeyError(
                f"Pipeline step '{step}' not registered. Available: {list(REGISTRY)}"
            )
        logger.info("Running step: %s", step)
        n = REGISTRY[step](n, cfg)
    return n


def solve(n: pypsa.Network, cfg: dict) -> pypsa.Network:
    # Sanitize known bad static values
    if "efficiency2" in n.links.columns:
        n.links["efficiency2"] = n.links["efficiency2"].fillna(1.0)

    solver = cfg.get("solver", {})
    name = solver.get("name", "gurobi")
    options = solver.get("options", {}) or {}

    n.optimize(solver_name=name, solver_options=options)
    return n


def run_pipeline(cfg: dict) -> Path:
    run_dir = _make_run_dir(
        Path(cfg["paths"]["runs_folder"]),
        base_name=f"run_{cfg['snapshots']['year']}",
    )

    n = build_network(cfg)
    n = apply_pipeline(n, cfg)
    n = ensure_all_component_carriers(n)
    n = solve(n, cfg)

    n.export_to_csv_folder(str(run_dir))
    logger.info("Exported results to %s", run_dir)

    return run_dir
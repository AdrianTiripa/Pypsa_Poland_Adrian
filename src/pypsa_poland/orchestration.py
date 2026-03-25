from __future__ import annotations

import json
import logging
import time
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


def _safe_rename_run_dir(run_dir: Path, suffix: str) -> Path:
    target = run_dir.with_name(f"{run_dir.name}_{suffix}")
    if not target.exists():
        run_dir.rename(target)
        return target

    i = 1
    while True:
        candidate = run_dir.with_name(f"{run_dir.name}_{suffix}_{i}")
        if not candidate.exists():
            run_dir.rename(candidate)
            return candidate
        i += 1


def _write_run_metadata(
    run_dir: Path,
    cfg: dict,
    elapsed_seconds: float,
    status: str,
    termination_condition: str,
) -> None:
    meta = {
        "year": int(cfg["snapshots"]["year"]),
        "stepsize": int(cfg["snapshots"].get("stepsize", 1)),
        "keep_feb29": bool(cfg["snapshots"].get("keep_feb29", False)),
        "elapsed_seconds": elapsed_seconds,
        "elapsed_minutes": elapsed_seconds / 60.0,
        "solver_status": status,
        "termination_condition": termination_condition,
        "timestamp_finished": datetime.now().isoformat(timespec="seconds"),
    }

    with open(run_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    with open(run_dir / "runtime.txt", "w", encoding="utf-8") as f:
        f.write(f"elapsed_seconds={elapsed_seconds:.2f}\n")
        f.write(f"elapsed_minutes={elapsed_seconds/60.0:.2f}\n")
        f.write(f"solver_status={status}\n")
        f.write(f"termination_condition={termination_condition}\n")


def _label_from_termination(status: str, termination_condition: str) -> str:
    s = str(status).strip().lower()
    t = str(termination_condition).strip().lower()

    if t == "optimal":
        return "Optimal"
    if t == "suboptimal":
        return "Suboptimal"
    if t == "infeasible":
        return "Infeasible"
    if t == "unbounded":
        return "Unbounded"
    if s == "ok" and t:
        return t.replace("_", " ").title().replace(" ", "")
    if s:
        return s.replace("_", " ").title().replace(" ", "")
    return "Unknown"


def downsample_snapshots(n: pypsa.Network, step: int) -> pypsa.Network:
    """
    Keep every `step`-th snapshot and scale snapshot_weightings so each kept
    snapshot represents `step` hours.
    """
    step = int(step)
    if step <= 1:
        return n

    before = len(n.snapshots)
    kept = n.snapshots[::step]

    n.set_snapshots(kept)

    w = n.snapshot_weightings.loc[kept].copy()
    for col in w.columns:
        w[col] = w[col] * step
    n.snapshot_weightings = w

    logger.info("Downsampled snapshots by step=%d: %d -> %d", step, before, len(kept))
    return n


def _build_hourly_snapshots(cfg: dict) -> pd.DatetimeIndex:
    year = int(cfg["snapshots"]["year"])
    keep_feb29 = bool(cfg["snapshots"].get("keep_feb29", False))

    snapshots = pd.date_range(
        start=f"{year}-01-01 00:00",
        end=f"{year + 1}-01-01 00:00",
        freq="H",
        inclusive="left",
    )

    if not keep_feb29 and len(snapshots) == 8784:
        snapshots = snapshots[~((snapshots.month == 2) & (snapshots.day == 29))]

    return snapshots


def build_network(cfg: dict) -> pypsa.Network:
    n = pypsa.Network()

    snapshots = _build_hourly_snapshots(cfg)

    configured_hours = cfg["snapshots"].get("hours")
    if configured_hours is not None and int(configured_hours) != len(snapshots):
        logger.warning(
            "snapshots.hours=%s but actual snapshot length for year=%s is %d. Using actual snapshot length.",
            configured_hours,
            cfg["snapshots"]["year"],
            len(snapshots),
        )

    n.set_snapshots(snapshots)

    input_folder = Path(cfg["paths"]["input_folder"])
    n.import_from_csv_folder(str(input_folder))

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

    if "efficiency2" in n.links.columns:
        mask = n.links["bus2"].isna() & n.links["efficiency2"].isna()
        n.links.loc[mask, "efficiency2"] = 0.0

    step = int(cfg["snapshots"].get("stepsize", 1))
    n = downsample_snapshots(n, step)

    return n


def ensure_all_component_carriers(n: pypsa.Network) -> pypsa.Network:
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




def solve(n: pypsa.Network, cfg: dict) -> tuple[pypsa.Network, str, str]:
    if "efficiency2" in n.links.columns:
        n.links["efficiency2"] = n.links["efficiency2"].fillna(1.0)

    solver = cfg.get("solver", {})
    name = solver.get("name", "gurobi")
    options = solver.get("options", {}) or {}

    result = n.optimize(solver_name=name, solver_options=options)

    status = "unknown"
    termination_condition = "unknown"

    # PyPSA/Linopy typically returns a tuple like ("ok", "optimal")
    if isinstance(result, tuple) and len(result) >= 2:
        status = str(result[0])
        termination_condition = str(result[1])
    else:
        # fallback in case API differs
        status = getattr(n, "status", status)
        termination_condition = getattr(n, "termination_condition", termination_condition)

    logger.info("Solve finished with status=%s, termination_condition=%s", status, termination_condition)
    return n, status, termination_condition


def run_pipeline(cfg: dict) -> Path:
    year = int(cfg["snapshots"]["year"])
    step = int(cfg["snapshots"].get("stepsize", 1))

    run_dir = _make_run_dir(
        Path(cfg["paths"]["runs_folder"]),
        base_name=f"run_{year}",
    )

    t0 = time.perf_counter()

    n = build_network(cfg)
    n = apply_pipeline(n, cfg)
    n = ensure_all_component_carriers(n)

    #  PV FIX ===========================================================
    pv_mask = n.generators.carrier.astype(str).str.contains("PV", case=False, na=False)
    pv_cols = n.generators.index[pv_mask]

    print("Static p_min_pu (PV):")
    print(n.generators.loc[pv_mask, "p_min_pu"].unique())

    print("\nTime-dependent p_min_pu exists?")
    print(hasattr(n.generators_t, "p_min_pu"))

    if hasattr(n.generators_t, "p_min_pu"):
        common_pmin = [c for c in pv_cols if c in n.generators_t.p_min_pu.columns]
        print("PV columns found in generators_t.p_min_pu:", len(common_pmin))
        if common_pmin:
            print("Max time-dependent PV p_min_pu:")
            print(n.generators_t.p_min_pu[common_pmin].max().max())

    if hasattr(n.generators_t, "p_max_pu"):
        common_pmax = [c for c in pv_cols if c in n.generators_t.p_max_pu.columns]
        print("\nPV columns found in generators_t.p_max_pu:", len(common_pmax))
        if common_pmax:
            print("Global min PV p_max_pu:")
            print(n.generators_t.p_max_pu[common_pmax].min().min())

            bad_ts = pd.Timestamp("1946-08-31 00:00:00")
            if bad_ts in n.generators_t.p_max_pu.index:
                print("\nPV p_max_pu at bad timestamp:")
                print(n.generators_t.p_max_pu.loc[bad_ts, common_pmax].sort_values().head(20))

    # ======================================================================

    n, status, termination_condition = solve(n, cfg)

    elapsed = time.perf_counter() - t0

    n.export_to_csv_folder(str(run_dir))
    _write_run_metadata(run_dir, cfg, elapsed, status, termination_condition)

    solve_label = _label_from_termination(status, termination_condition)
    runtime_tag = f"{int(round(elapsed))}s"
    step_tag = f"{step}hr"

    run_dir = _safe_rename_run_dir(run_dir, f"{step_tag}_{solve_label}_{runtime_tag}")

    logger.info("Exported results to %s", run_dir)
    return run_dir
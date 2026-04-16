# src/pypsa_poland/orchestration.py
#
# Top-level orchestration for a single pypsa-poland optimisation run.
#
# Responsibilities:
#   - Build the PyPSA network from CSV inputs and set hourly snapshots.
#   - Reindex imported time-series DataFrames to a proper DatetimeIndex
#     (fixes a class of silent bugs described in _reindex_timeseries_to_snapshots).
#   - Apply the user-defined pipeline of component-addition steps.
#   - Solve the network with Gurobi (or any configured solver).
#   - Export results to a timestamped run directory and write run metadata.
#
# The public entry point is run_pipeline(cfg), which returns the run directory
# Path so downstream scripts can find the outputs.

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


# ---------------------------------------------------------------------------
# Run directory helpers
# ---------------------------------------------------------------------------

def _make_run_dir(runs_folder: Path, base_name: str = "run") -> Path:
    """Create a timestamped run directory inside runs_folder."""
    runs_folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = runs_folder / f"{base_name}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _safe_rename_run_dir(run_dir: Path, suffix: str) -> Path:
    """
    Rename run_dir by appending suffix. If the target name already exists,
    append a numeric counter (_1, _2, ...) to avoid collisions.
    """
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
    """Write run_metadata.json and a plain-text runtime.txt to the run directory."""
    meta = {
        "year":                  int(cfg["snapshots"]["year"]),
        "stepsize":              int(cfg["snapshots"].get("stepsize", 1)),
        "keep_feb29":            bool(cfg["snapshots"].get("keep_feb29", False)),
        "elapsed_seconds":       elapsed_seconds,
        "elapsed_minutes":       elapsed_seconds / 60.0,
        "solver_status":         status,
        "termination_condition": termination_condition,
        "timestamp_finished":    datetime.now().isoformat(timespec="seconds"),
    }

    with open(run_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Plain-text version for quick inspection without a JSON parser.
    with open(run_dir / "runtime.txt", "w", encoding="utf-8") as f:
        f.write(f"elapsed_seconds={elapsed_seconds:.2f}\n")
        f.write(f"elapsed_minutes={elapsed_seconds/60.0:.2f}\n")
        f.write(f"solver_status={status}\n")
        f.write(f"termination_condition={termination_condition}\n")


def _label_from_termination(status: str, termination_condition: str) -> str:
    """
    Derive a human-readable CamelCase label from solver status/termination strings.
    Used to suffix the run directory name for easy identification.
    """
    s = str(status).strip().lower()
    t = str(termination_condition).strip().lower()

    if t == "optimal":    return "Optimal"
    if t == "suboptimal": return "Suboptimal"
    if t == "infeasible": return "Infeasible"
    if t == "unbounded":  return "Unbounded"
    # Fall back to a cleaned-up version of whatever the solver returned.
    if s == "ok" and t:
        return t.replace("_", " ").title().replace(" ", "")
    if s:
        return s.replace("_", " ").title().replace(" ", "")
    return "Unknown"


# ---------------------------------------------------------------------------
# Snapshot utilities
# ---------------------------------------------------------------------------

def downsample_snapshots(n: pypsa.Network, step: int) -> pypsa.Network:
    """
    Keep every `step`-th snapshot and scale snapshot_weightings accordingly.

    Each retained snapshot is given a weighting of `step` hours so that
    energy sums (generation, curtailment, etc.) remain correct when computed
    as weighted sums over the reduced snapshot set.
    """
    step = int(step)
    if step <= 1:
        return n

    before = len(n.snapshots)
    kept = n.snapshots[::step]

    n.set_snapshots(kept)

    # Scale all weighting columns so each kept snapshot represents `step` hours.
    w = n.snapshot_weightings.loc[kept].copy()
    for col in w.columns:
        w[col] = w[col] * step
    n.snapshot_weightings = w

    logger.info("Downsampled snapshots by step=%d: %d -> %d", step, before, len(kept))
    return n


def _build_hourly_snapshots(cfg: dict) -> pd.DatetimeIndex:
    """
    Build a full-year hourly DatetimeIndex for the configured year.
    Feb 29 is dropped by default (keep_feb29=False) so that all years
    have the same length (8760 h), simplifying cross-year comparisons.
    """
    year = int(cfg["snapshots"]["year"])
    keep_feb29 = bool(cfg["snapshots"].get("keep_feb29", False))

    snapshots = pd.date_range(
        start=f"{year}-01-01 00:00",
        end=f"{year + 1}-01-01 00:00",
        freq="h",
        inclusive="left",
    )

    # Drop Feb 29 for non-leap treatment unless the user explicitly opts in.
    if not keep_feb29 and len(snapshots) == 8784:
        snapshots = snapshots[~((snapshots.month == 2) & (snapshots.day == 29))]

    return snapshots


# ---------------------------------------------------------------------------
# Time-series reindexing fix
# ---------------------------------------------------------------------------

def _reindex_timeseries_to_snapshots(n: pypsa.Network) -> pypsa.Network:
    """
    Re-index all imported time-series DataFrames to ``n.snapshots``.

    ROOT CAUSE OF THE BUG THIS FIXES
    ---------------------------------
    The input CSV files (generators-p_max_pu.csv, loads-p_set.csv, etc.) store
    their time axis as plain integers (0, 1, 2, ...) rather than timestamps.
    When PyPSA reads them via ``import_from_csv_folder`` it preserves whatever
    index the CSV has.  ``set_snapshots()`` updates ``n.snapshots`` to a proper
    DatetimeIndex but does NOT retroactively reindex the already-loaded
    time-series DataFrames stored in each component's ``_t`` attribute.

    Consequences
    ------------
    1. Exported results have integer indices — every output CSV comes out with
       rows labelled 0, 1, 2 ... instead of timestamps. Downstream code that
       calls pd.to_datetime() on those values gets nanosecond epoch timestamps
       (1970-01-01 00:00:00.000000001, etc.), making time-series analysis useless.

    2. supply.py CF assignments silently fail for existing columns — when
       ``add_generators`` writes ``n.generators_t.p_max_pu.loc[:, pv_cols] = pv_cf``
       where ``pv_cf`` has a DatetimeIndex and ``p_max_pu`` still has an integer
       index, pandas aligns on labels and produces NaN for every row.

    3. The H2 min-SOC linopy constraint fails — linopy attaches the network's
       snapshot coordinate (DatetimeIndex) to its variables. The slice in
       constraints.py uses a DatetimeIndex sub-slice that only works correctly
       if the network's snapshots are already a DatetimeIndex before solve time.

    Fix
    ---
    After ``import_from_csv_folder`` and ``set_snapshots`` (but before the
    pipeline runs), walk every component's ``_t`` attribute and replace any
    DataFrame whose index length matches ``len(n.snapshots)`` but whose index
    is not already a DatetimeIndex with a copy reindexed positionally to
    ``n.snapshots``. Matching by length is safe because the integer index
    0..N-1 is positional — row i corresponds to snapshot i.
    """
    snapshots = n.snapshots
    n_snaps = len(snapshots)

    # All PyPSA component types that carry time-series (_t) attributes.
    component_names = [
        "generators", "loads", "storage_units", "stores",
        "links", "lines", "transformers", "buses", "shunt_impedances",
    ]

    total_reindexed = 0
    for comp_name in component_names:
        comp_t = getattr(n, f"{comp_name}_t", None)
        if comp_t is None:
            continue

        # comp_t is a pypsa.descriptors.Dict — iterate over its DataFrames.
        for attr_name in list(comp_t.keys()):
            df = comp_t[attr_name]
            if not isinstance(df, pd.DataFrame) or df.empty:
                continue
            if len(df) != n_snaps:
                continue
            # Already a DatetimeIndex of the right length — leave it alone.
            if isinstance(df.index, pd.DatetimeIndex):
                continue
            # Integer / RangeIndex of matching length: reindex positionally.
            comp_t[attr_name] = df.set_index(snapshots)
            total_reindexed += 1
            logger.debug(
                "Reindexed %s_t.%s (%d rows) to DatetimeIndex.",
                comp_name, attr_name, n_snaps,
            )

    logger.info(
        "_reindex_timeseries_to_snapshots: reindexed %d time-series DataFrame(s) "
        "to %d-snapshot DatetimeIndex starting %s.",
        total_reindexed,
        n_snaps,
        snapshots[0],
    )
    return n


# ---------------------------------------------------------------------------
# Network construction
# ---------------------------------------------------------------------------

def build_network(cfg: dict) -> pypsa.Network:
    """
    Create a PyPSA network, set snapshots, import CSV inputs, and apply fixes.

    Steps performed:
      1. Build hourly DatetimeIndex for the configured year.
      2. Import network topology and static/time-series data from CSV folder.
      3. Register any carriers found in the imported data that are not yet
         in n.carriers (prevents PyPSA warnings about missing carriers).
      4. Patch a NaN in efficiency2 for links with no bus2 connection.
      5. Reindex all time-series DataFrames to the proper DatetimeIndex.
      6. Downsample snapshots if stepsize > 1.
    """
    n = pypsa.Network()

    snapshots = _build_hourly_snapshots(cfg)

    # Warn if the config specifies a different number of hours than we actually
    # produce — this can happen when the user forgets to account for Feb 29.
    configured_hours = cfg["snapshots"].get("hours")
    if configured_hours is not None and int(configured_hours) != len(snapshots):
        logger.warning(
            "snapshots.hours=%s but actual snapshot length for year=%s is %d. "
            "Using actual snapshot length.",
            configured_hours,
            cfg["snapshots"]["year"],
            len(snapshots),
        )

    n.set_snapshots(snapshots)

    input_folder = Path(cfg["paths"]["input_folder"])
    n.import_from_csv_folder(str(input_folder))

    # Register any carriers present in imported components that are not yet
    # in n.carriers, so PyPSA does not raise carrier-not-found warnings.
    for series in [
        n.buses.carrier,
        n.generators.carrier,
        n.links.carrier,
        n.storage_units.carrier,
    ]:
        for c in pd.unique(series.dropna()):
            if c not in n.carriers.index:
                n.add("Carrier", c)

    # Links with no bus2 connection may have a NaN efficiency2, which can
    # cause numerical issues. Set it to 0.0 for those links explicitly.
    if "efficiency2" in n.links.columns:
        mask = n.links["bus2"].isna() & n.links["efficiency2"].isna()
        n.links.loc[mask, "efficiency2"] = 0.0

    # Reindex all imported time-series to the proper DatetimeIndex BEFORE the
    # pipeline runs (so supply.py CF assignments and the H2 SOC constraint work).
    n = _reindex_timeseries_to_snapshots(n)

    step = int(cfg["snapshots"].get("stepsize", 1))
    n = downsample_snapshots(n, step)

    return n


# ---------------------------------------------------------------------------
# Pipeline application
# ---------------------------------------------------------------------------

def ensure_all_component_carriers(n: pypsa.Network) -> pypsa.Network:
    """
    Ensure every carrier referenced by any network component exists in n.carriers.

    Called after the pipeline has run to catch any carriers introduced by
    component-addition steps that did not call _ensure_carrier() themselves.
    """
    def _ensure(series: pd.Series) -> None:
        for c in pd.unique(series.dropna()):
            if c not in n.carriers.index:
                n.add("Carrier", c)

    if hasattr(n, "buses")         and "carrier" in n.buses.columns:         _ensure(n.buses["carrier"])
    if hasattr(n, "generators")    and "carrier" in n.generators.columns:    _ensure(n.generators["carrier"])
    if hasattr(n, "links")         and "carrier" in n.links.columns:          _ensure(n.links["carrier"])
    if hasattr(n, "loads")         and "carrier" in n.loads.columns:          _ensure(n.loads["carrier"])
    if hasattr(n, "storage_units") and "carrier" in n.storage_units.columns: _ensure(n.storage_units["carrier"])

    return n


def apply_pipeline(n: pypsa.Network, cfg: dict) -> pypsa.Network:
    """
    Execute each pipeline step listed in cfg['pipeline'] in order.

    Each step name must be registered in REGISTRY. Steps are called as
    REGISTRY[step](n, cfg) and must return the (possibly modified) network.
    """
    for step in cfg.get("pipeline", []):
        if step not in REGISTRY:
            raise KeyError(
                f"Pipeline step '{step}' not registered. Available: {list(REGISTRY)}"
            )
        logger.info("Running step: %s", step)
        n = REGISTRY[step](n, cfg)
    return n


# ---------------------------------------------------------------------------
# Solve
# ---------------------------------------------------------------------------

def solve(n: pypsa.Network, cfg: dict) -> tuple[pypsa.Network, str, str]:
    """
    Optimise the network and return (network, solver_status, termination_condition).

    efficiency2 NaNs on links with a real bus2 connection are filled with 1.0
    immediately before solve so that the linopy model is numerically clean.
    The extra_functionality hook injects capacity-target constraints from cfg.
    """
    # Fill any remaining efficiency2 NaNs before linopy builds the model.
    if "efficiency2" in n.links.columns:
        n.links["efficiency2"] = n.links["efficiency2"].fillna(1.0)

    solver = cfg.get("solver", {})
    name    = solver.get("name", "gurobi")
    options = solver.get("options", {}) or {}

    from .components.constraints import fix_total_capacity_by_carrier

    result = n.optimize(
        solver_name=name,
        solver_options=options,
        # Inject user-defined capacity targets as additional linopy constraints.
        extra_functionality=lambda n, snapshots: fix_total_capacity_by_carrier(
            n, snapshots, cfg
        ),
    )

    # PyPSA can return a tuple (status, termination_condition) or set attributes
    # directly on the network — handle both API styles.
    status = "unknown"
    termination_condition = "unknown"

    if isinstance(result, tuple) and len(result) >= 2:
        status               = str(result[0])
        termination_condition = str(result[1])
    else:
        status               = getattr(n, "status", status)
        termination_condition = getattr(n, "termination_condition", termination_condition)

    logger.info(
        "Solve finished with status=%s, termination_condition=%s",
        status, termination_condition,
    )
    return n, status, termination_condition


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def run_pipeline(cfg: dict) -> Path:
    """
    Run the full build → pipeline → solve → export cycle for one year.

    Returns the final run directory path (renamed with solve status and runtime).
    """
    year = int(cfg["snapshots"]["year"])
    step = int(cfg["snapshots"].get("stepsize", 1))

    # Create the run directory before timing starts so directory creation time
    # is not included in the reported solve time.
    run_dir = _make_run_dir(
        Path(cfg["paths"]["runs_folder"]),
        base_name=f"run_{year}",
    )

    t0 = time.perf_counter()

    n = build_network(cfg)
    n = apply_pipeline(n, cfg)
    n = ensure_all_component_carriers(n)

    n, status, termination_condition = solve(n, cfg)

    elapsed = time.perf_counter() - t0

    # Export full PyPSA results (components + time-series) to the run directory.
    n.export_to_csv_folder(str(run_dir))
    _write_run_metadata(run_dir, cfg, elapsed, status, termination_condition)

    # Rename with human-readable tags so the folder name summarises the run.
    solve_label  = _label_from_termination(status, termination_condition)
    runtime_tag  = f"{int(round(elapsed))}s"
    step_tag     = f"{step}hr"

    run_dir = _safe_rename_run_dir(run_dir, f"{step_tag}_{solve_label}_{runtime_tag}")

    logger.info("Exported results to %s", run_dir)
    return run_dir

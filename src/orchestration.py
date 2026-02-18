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


def build_network(cfg: dict) -> pypsa.Network:
    n = pypsa.Network()

    year = int(cfg["snapshots"]["year"])
    hours = int(cfg["snapshots"]["hours"])
    n.set_snapshots(pd.date_range(f"{year}-01-01", periods=hours, freq="H"))

    input_folder = Path(cfg["paths"]["input_folder"])
    n.import_from_csv_folder(str(input_folder))

    return n


def apply_pipeline(n: pypsa.Network, cfg: dict) -> pypsa.Network:
    for step in cfg.get("pipeline", []):
        if step not in REGISTRY:
            raise KeyError(f"Pipeline step '{step}' not registered. Available: {list(REGISTRY)}")
        logger.info("Running step: %s", step)
        n = REGISTRY[step](n, cfg)
    return n


def solve(n: pypsa.Network, cfg: dict) -> pypsa.Network:
    solver = cfg.get("solver", {})
    name = solver.get("name", "gurobi")
    options = solver.get("options", {}) or {}
    n.optimize(solver_name=name, solver_options=options)
    return n


def run_pipeline(cfg: dict) -> Path:
    run_dir = _make_run_dir(Path(cfg["paths"]["runs_folder"]), base_name="run")
    n = build_network(cfg)
    n = apply_pipeline(n, cfg)
    n = solve(n, cfg)

    # export
    n.export_to_csv_folder(str(run_dir))
    logger.info("Exported results to %s", run_dir)

    return run_dir

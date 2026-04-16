# src/pypsa_poland/config.py
from __future__ import annotations

from pathlib import Path
import yaml


def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError("Config YAML must define a mapping (dictionary) at the top level.")

    return cfg

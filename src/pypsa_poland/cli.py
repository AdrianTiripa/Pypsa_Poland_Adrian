# src/pypsa_poland/cli.py
#
# Command-line entry point for pypsa-poland.
#
# Accepts a YAML config path and an optional year range, then calls
# run_pipeline() once per year. A deep copy of the base config is made
# for each year so that individual runs cannot mutate shared state.
#
# Usage examples:
#   python -m pypsa_poland --config configs/default.yaml --year 2020
#   python -m pypsa_poland --config configs/default.yaml --years 1980-1985

from pathlib import Path
from .config import load_config
from .orchestration import run_pipeline

import argparse
import copy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml",
                   help="Path to YAML config file.")
    p.add_argument("--year", type=int,
                   help="Run a single meteorological year (overrides config).")
    p.add_argument("--years", type=str,
                   help="Run an inclusive year range, e.g. '1980-1985'.")
    args = p.parse_args()

    cfg0 = load_config(Path(args.config))

    # Determine which years to run based on CLI flags, falling back to config.
    if args.years:
        a, b = args.years.split("-", 1)
        years = range(int(a), int(b) + 1)
    elif args.year is not None:
        years = [args.year]
    else:
        years = [int(cfg0["snapshots"]["year"])]

    for y in years:
        # Deep-copy so each year's run starts from an identical, clean config.
        cfg = copy.deepcopy(cfg0)
        cfg["snapshots"]["year"] = y
        run_pipeline(cfg)


if __name__ == "__main__":
    main()

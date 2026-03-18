# src/pypsa_poland/cli.py

from pathlib import Path
from .config import load_config
# from .logging import setup_logging
from .orchestration import run_pipeline

import argparse
import copy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--year", type=int, help="Run a single year (overrides config)")
    p.add_argument("--years", type=str, help="Run a range like 1980-1985 (inclusive)")
    args = p.parse_args()

    cfg0 = load_config(Path(args.config))

    # Determine which years to run
    if args.years:
        a, b = args.years.split("-", 1)
        years = range(int(a), int(b) + 1)
    elif args.year is not None:
        years = [args.year]
    else:
        years = [int(cfg0["snapshots"]["year"])]

    for y in years:
        cfg = copy.deepcopy(cfg0)
        cfg["snapshots"]["year"] = y
        run_pipeline(cfg)


if __name__ == "__main__":
    main()
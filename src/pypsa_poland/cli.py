# src/pypsa_poland/cli.py

from pathlib import Path
from .config import load_config
from .logging import setup_logging
from .orchestration import run_pipeline


def main():
    setup_logging("INFO")
    cfg = load_config(Path("configs/default.yaml"))
    run_pipeline(cfg)


if __name__ == "__main__":
    main()

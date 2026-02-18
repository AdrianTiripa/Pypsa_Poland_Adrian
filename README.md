# pypsa-poland

A modular, YAML-driven PyPSA-based energy system model for Poland.

This repository provides a structured and extensible framework for
energy system modelling using PyPSA.\
The model is organized as a configurable pipeline of components
(generation, heat, hydrogen, transport, network, constraints), enabling
reproducible scenario analysis.

------------------------------------------------------------------------

## Key Features

-   YAML-driven configuration
-   Modular component architecture
-   Registry-based pipeline execution
-   Separation of network data and parameter tables
-   Reproducible run directories
-   Ready for CI integration

------------------------------------------------------------------------

## Repository Structure

    pypsa-poland/
    ├── configs/           # YAML configuration files
    ├── data/              # Network CSV input data (imported via PyPSA)
    ├── database/          # Time series & parameter tables (Excel)
    ├── runs/              # Model outputs (ignored by git)
    ├── src/pypsa_poland/
    │   ├── orchestration.py
    │   └── components/

------------------------------------------------------------------------

## Architecture

The model follows a modular pipeline architecture:

    CLI → config.yaml → orchestration → REGISTRY → components → solve → export

Each pipeline step is defined in `configs/default.yaml`:

``` yaml
pipeline:
  - generators
  - heat
  - hydrogen
  - cop
  - hydrogen_storage
  - heat_storage
  - high_grade_heat
  - transport
  - dc_link
  - co2_constraint
```

Each step corresponds to a function call:

``` python
REGISTRY[step](network, cfg)
```

All components follow a standardized interface:

``` python
def component_step(network: pypsa.Network, cfg: dict) -> pypsa.Network:
```

------------------------------------------------------------------------

## Installation

Create and activate your Python environment, then install in editable
mode:

``` bash
pip install -e .
```

------------------------------------------------------------------------

## Running the Model

``` bash
pypsa-poland --config configs/default.yaml
```

Model outputs are written to the `runs/` directory with timestamped
subfolders.

------------------------------------------------------------------------

## Configuration

All model settings are controlled via YAML files:

-   Paths
-   Snapshot settings
-   Solver configuration
-   Technology parameters
-   Pipeline order

This design enables reproducible scenario definition without modifying
source code.

------------------------------------------------------------------------

## Development Notes

-   No hard-coded paths
-   Idempotent component functions
-   Clear separation between orchestration and domain logic
-   Designed for research transparency and extensibility

------------------------------------------------------------------------

## License

See `LICENSE` file.

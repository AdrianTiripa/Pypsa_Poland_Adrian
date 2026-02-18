# src/pypsa_poland/components/heat.py

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pypsa

logger = logging.getLogger(__name__)


def add_heat(n: pypsa.Network, cfg: dict) -> pypsa.Network:
    data_folder = Path(cfg["paths"]["data_folder"])
    """
    Add low-temperature (non-industrial) heat demand and heat pumps.

    - Creates a "{bus}_heat" bus for each electricity bus b (if not exists).
    - Adds a heat pump Link from elec bus -> heat bus for each region in the demand table.
    - Adds a Load on each heat bus with time series from Excel.
    """
    data_folder = Path(data_folder)

    logger.info("Adding low-temp heat demand started.")
    heat_demand = pd.read_excel(data_folder / "2020_Heat_NonIndustiral.xlsx", index_col=0)
    heat_demand.index = n.snapshots

    # Heat bus per electricity bus
    for b in n.buses.index:
        heat_bus = f"{b}_heat"
        if heat_bus not in n.buses.index:
            n.add("Bus", heat_bus, carrier="heat")

    cop = 3.0  # constant COP (keep as-is)

    # Heat pumps and (ensure) heat buses for regions present in demand file
    for region in heat_demand.columns:
        elec_bus = region
        heat_bus = f"{region}_heat"

        if elec_bus not in n.buses.index:
            logger.warning("Skipping heat demand region '%s': elec bus not in network.", elec_bus)
            continue

        if heat_bus not in n.buses.index:
            n.add("Bus", heat_bus, carrier="heat")

        link_name = f"{region}_heat_pump"
        if link_name not in n.links.index:
            n.add(
                "Link",
                name=link_name,
                bus0=elec_bus,     # electricity input
                bus1=heat_bus,     # heat output
                efficiency=cop,
                carrier="heat_pump",
                capital_cost=10_000,
                marginal_cost=0.01,
                p_nom_extendable=True,
            )

    # Add heat loads with time series
    heat_demand.columns = [f"{col}_heat" for col in heat_demand.columns]

    # Ensure buses exist for each load column (they should, but keep safe)
    for bus in heat_demand.columns:
        if bus not in n.buses.index:
            n.add("Bus", bus, carrier="heat")

    # Add loads
    for load_name, bus_name in zip(heat_demand.columns, heat_demand.columns):
        if load_name not in n.loads.index:
            n.add("Load", name=load_name, bus=bus_name)

    # Set time series
    n.loads_t.p_set[heat_demand.columns] = heat_demand * 1.0

    logger.info("Adding low-temp heat demand done.")
    return n


def add_heat_storage(n: pypsa.Network, cfg: dict) -> pypsa.Network:
    """
    Add thermal storage on each *_heat bus (hot water storage).
    """
    if "hot_water" not in n.carriers.index:
        n.add("Carrier", "hot_water")

    storage_parameters = dict(
        carrier="hot_water",
        efficiency_store=0.95,
        efficiency_dispatch=0.95,
        standing_loss=0.001,
        capital_cost=30_000,
        marginal_cost=0.0,
        lifetime=20,
        max_hours=9.2,
        p_nom_extendable=True,
    )

    for bus in n.buses.index:
        if bus.endswith("_heat"):
            storage_name = f"thermal_storage_{bus}"
            if storage_name not in n.storage_units.index:
                n.add("StorageUnit", name=storage_name, bus=bus, **storage_parameters)

    return n


def add_high_grade_heat(n: pypsa.Network, cfg: dict) -> pypsa.Network:
    data_folder = Path(cfg["paths"]["data_folder"])
    """
    Add industrial (high-temp) heat demand and CHP-like hydrogen plant producing electricity + heat.
    """
    data_folder = Path(data_folder)

    # Use one consistent carrier name
    carrier = "high_temp_heat"
    if carrier not in n.carriers.index:
        n.add("Carrier", carrier)

    # Create high-temp heat buses and CHP links
    for bus in n.buses.index:
        # Only for "electricity buses" (heuristic: exclude existing heat/hydrogen buses)
        if bus.endswith("_heat") or bus.endswith("_hydrogen") or bus.endswith("_high_temp_heat"):
            continue

        high_temp_heat_bus = f"{bus}_high_temp_heat"
        if high_temp_heat_bus not in n.buses.index:
            n.add("Bus", high_temp_heat_bus, carrier=carrier)

        link_name = f"{bus}_chp_h2_plant"
        h2_bus = f"{bus}_hydrogen"
        if h2_bus in n.buses.index and link_name not in n.links.index:
            n.add(
                "Link",
                name=link_name,
                bus0=h2_bus,                # hydrogen input
                bus1=bus,                   # electricity output
                bus2=high_temp_heat_bus,    # heat output
                p_nom=0,
                efficiency=0.3,
                efficiency2=0.6,
                capital_cost=20_000.0,
                p_nom_extendable=True,
            )

    # Industrial heat demand
    heat_demand = pd.read_excel(data_folder / "2020_HeatDemandIndustry.xlsx", index_col=0)
    heat_demand.index = n.snapshots

    # Ensure buses exist for all regions in file
    for region in heat_demand.columns:
        elec_bus = region
        high_temp_heat_bus = f"{region}_high_temp_heat"

        if elec_bus not in n.buses.index:
            logger.warning("Skipping industrial heat region '%s': elec bus not in network.", elec_bus)
            continue

        if high_temp_heat_bus not in n.buses.index:
            n.add("Bus", high_temp_heat_bus, carrier=carrier)

    heat_demand.columns = [f"{col}_high_temp_heat" for col in heat_demand.columns]

    # Add loads
    for load_name, bus_name in zip(heat_demand.columns, heat_demand.columns):
        if load_name not in n.loads.index:
            n.add("Load", name=load_name, bus=bus_name)

    n.loads_t.p_set[heat_demand.columns] = heat_demand * 1.0
    return n


def add_cop(n: pypsa.Network, cfg: dict) -> pypsa.Network:
    """
    Set time-dependent COP (efficiency) for heat pump links from Excel.

    Expects Excel columns to be regions like "PL KP" and maps them to link names "<region>_heat_pump".
    """
    data_folder = Path(cfg["paths"]["data_folder"])

    cop = pd.read_excel(data_folder / "2020_COP.xlsx", index_col=0)
    cop.index = n.snapshots
    cop.columns = [f"{col}_heat_pump" for col in cop.columns]

    # Only apply to links that exist (avoid overwriting unrelated link efficiencies)
    existing = [c for c in cop.columns if c in n.links.index]
    missing = [c for c in cop.columns if c not in n.links.index]
    if missing:
        logger.warning("COP: %d columns have no matching Link; skipping those (e.g. %s).",
                       len(missing), missing[0])

    if existing:
        n.links_t.efficiency[existing] = cop[existing]
    else:
        logger.warning("COP: no columns matched existing heat pump links; nothing set.")

    return n
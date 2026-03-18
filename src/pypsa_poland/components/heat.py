from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pypsa

from .profile_io import read_profile_csv, read_excel_timeseries

logger = logging.getLogger(__name__)


def add_heat(n: pypsa.Network, cfg: dict) -> pypsa.Network:
    """
    Add low-temperature (non-industrial) heat demand and heat pumps.

    - Creates a "{bus}_heat" bus for each electricity bus b (if not exists).
    - Adds a heat pump Link from elec bus -> heat bus for each region in the demand table.
    - Adds a Load on each heat bus with time series from multi-year CSV.
    """
    logger.info("Adding low-temp heat demand started.")

    # Multi-year CSV profile, conformed to n.snapshots (stepsize-safe)
    heat_demand = read_profile_csv(cfg, "heat_demand_multi", n.snapshots)

    # Heat bus per electricity bus
    for b in n.buses.index:
        heat_bus = f"{b}_heat"
        if heat_bus not in n.buses.index:
            n.add("Bus", heat_bus, carrier="heat")

    # constant COP (time-varying COP applied later by add_cop)
    cop_default = float(cfg.get("heat", {}).get("heat_pump", {}).get("cop_default", 3.0))

    # Heat pumps and (ensure) heat buses for regions present in demand file
    for region in heat_demand.columns:
        elec_bus = f"PL {region}"
        heat_bus = f"PL {region}_heat"

        if elec_bus not in n.buses.index:
            logger.warning("Skipping heat demand region '%s': elec bus not in network.", elec_bus)
            continue

        if heat_bus not in n.buses.index:
            n.add("Bus", heat_bus, carrier="heat")

        link_name = f"PL {region}_heat_pump"
        if link_name not in n.links.index:
            n.add(
                "Link",
                name=link_name,
                bus0=elec_bus,
                bus1=heat_bus,
                efficiency=cop_default,
                carrier=cfg.get("heat", {}).get("heat_pump", {}).get("carrier", "heat_pump"),
                capital_cost=float(cfg.get("heat", {}).get("heat_pump", {}).get("capital_cost", 10_000)),
                marginal_cost=float(cfg.get("heat", {}).get("heat_pump", {}).get("marginal_cost", 0.01)),
                p_nom_extendable=bool(cfg.get("heat", {}).get("heat_pump", {}).get("p_nom_extendable", True)),
            )

    # Add heat loads with time series
    heat_demand.columns = [f"PL {col}_heat" for col in heat_demand.columns]

    # Ensure buses exist
    for bus in heat_demand.columns:
        if bus not in n.buses.index:
            n.add("Bus", bus, carrier="heat")

    # Add loads + set time series
    for load_name in heat_demand.columns:
        if load_name not in n.loads.index:
            n.add("Load", name=load_name, bus=load_name)

    n.loads_t.p_set[heat_demand.columns] = heat_demand

    logger.info("Adding low-temp heat demand done.")
    return n


def add_heat_storage(n: pypsa.Network, cfg: dict) -> pypsa.Network:
    """
    Add thermal storage on each *_heat bus (hot water storage).
    """
    carrier = cfg.get("heat", {}).get("thermal_storage", {}).get("carrier", "hot_water")
    if carrier not in n.carriers.index:
        n.add("Carrier", carrier)

    ts_cfg = cfg.get("heat", {}).get("thermal_storage", {})
    storage_parameters = dict(
        carrier=carrier,
        efficiency_store=float(ts_cfg.get("efficiency_store", 0.95)),
        efficiency_dispatch=float(ts_cfg.get("efficiency_dispatch", 0.95)),
        standing_loss=float(ts_cfg.get("standing_loss", 0.001)),
        capital_cost=float(ts_cfg.get("capital_cost", 30_000)),
        marginal_cost=float(ts_cfg.get("marginal_cost", 0.0)),
        lifetime=int(ts_cfg.get("lifetime", 20)),
        max_hours=float(ts_cfg.get("max_hours", 9.2)),
        p_nom_extendable=bool(ts_cfg.get("p_nom_extendable", True)),
    )

    for bus in n.buses.index:
        if bus.endswith("_heat"):
            storage_name = f"thermal_storage_{bus}"
            if storage_name not in n.storage_units.index:
                n.add("StorageUnit", name=storage_name, bus=bus, **storage_parameters)

    return n


def add_high_grade_heat(n: pypsa.Network, cfg: dict) -> pypsa.Network:
    """
    Add industrial (high-temp) heat demand and CHP-like hydrogen plant producing electricity + heat.
    """
    data_folder = Path(cfg["paths"]["data_folder"])

    carrier = cfg.get("heat", {}).get("high_temp_heat", {}).get("carrier", "high_temp_heat")
    if carrier not in n.carriers.index:
        n.add("Carrier", carrier)

    chp_cfg = cfg.get("heat", {}).get("high_temp_heat", {}).get("chp_h2_plant", {})

    # Create high-temp heat buses and CHP links
    for bus in n.buses.index:
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
                bus0=h2_bus,
                bus1=bus,
                bus2=high_temp_heat_bus,
                p_nom=float(chp_cfg.get("p_nom", 0)),
                efficiency=float(chp_cfg.get("efficiency_el", 0.3)),
                efficiency2=float(chp_cfg.get("efficiency_heat", 0.6)),
                capital_cost=float(chp_cfg.get("capital_cost", 20_000.0)),
                p_nom_extendable=bool(chp_cfg.get("p_nom_extendable", True)),
            )

    # Industrial heat demand (Excel), conformed to n.snapshots
    ind_fname = cfg["files"].get("heat_demand_industry", "2020_HeatDemandIndustry.xlsx")
    ind_path = data_folder / ind_fname
    heat_demand = read_excel_timeseries(ind_path, cfg, n.snapshots)

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

    # Add loads + time series
    for load_name in heat_demand.columns:
        if load_name not in n.loads.index:
            n.add("Load", name=load_name, bus=load_name)

    n.loads_t.p_set[heat_demand.columns] = heat_demand
    return n


def add_cop(n: pypsa.Network, cfg: dict) -> pypsa.Network:
    """
    Set time-dependent COP (efficiency) for heat pump links from CSV multi-year profile.

    Expects columns to be regions like "KP" and maps them to link names "PL <region>_heat_pump".
    """
    cop = read_profile_csv(cfg, "cop_multi", n.snapshots)
    cop.columns = [f"PL {col}_heat_pump" for col in cop.columns]

    existing = [c for c in cop.columns if c in n.links.index]
    missing = [c for c in cop.columns if c not in n.links.index]
    if missing:
        logger.warning(
            "COP: %d columns have no matching Link; skipping those (e.g. %s).",
            len(missing),
            missing[0],
        )

    if existing:
        n.links_t.efficiency[existing] = cop[existing]
    else:
        logger.warning("COP: no columns matched existing heat pump links; nothing set.")

    return n
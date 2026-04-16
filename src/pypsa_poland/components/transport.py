# src/pypsa_poland/components/transport.py
#
# Electric transport sector component additions for pypsa-poland.
#
# For each primary electricity region the function adds:
#   - A "<bus>_transport" bus with carrier "transport".
#   - A link from the electricity bus to the transport bus (modelling EV charging
#     infrastructure with a small efficiency loss).
#   - A load on the transport bus with the demand time series from Excel.
#   - A small EV buffer storage unit sized at 25% of mean hourly transport demand,
#     representing the managed-charging flexibility of the vehicle fleet.
#
# Carrier note
# ------------
# EV storage units use carrier="transport" (not "electricity"). The old carrier
# caused these units to appear as a mystery ~1.86 GW "electricity" storage
# category in results summaries, separate from all other storage carriers.
# Using "transport" keeps them correctly grouped with the transport sector in all
# downstream reporting and map plots.

from __future__ import annotations

import logging
from pathlib import Path

import pypsa

from .profile_io import read_excel_timeseries

logger = logging.getLogger(__name__)


def add_transport(n: pypsa.Network, cfg: dict) -> pypsa.Network:
    """
    Add electric transport demand infrastructure to the network.

    Demand time series are loaded from Excel, aligned to n.snapshots, and
    written to n.loads_t.p_set. EV buffer storage is then added on each
    transport bus with a p_nom proportional to the mean hourly load.
    """
    data_folder   = Path(cfg["paths"]["data_folder"])
    transport_cfg = cfg.get("transport", {})
    storage_cfg   = transport_cfg.get("ev_storage", {})

    logger.info("Adding transport started.")

    if "transport" not in n.carriers.index:
        n.add("Carrier", "transport")

    # ---- Create transport buses and electricity→transport links ----
    for bus in n.buses.index:
        # Skip derived buses — only add transport to primary electricity regions.
        if (
            bus.endswith("_heat")
            or bus.endswith("_hydrogen")
            or bus.endswith("_high_temp_heat")
            or bus.endswith("_transport")
        ):
            continue

        t_bus = f"{bus}_transport"
        if t_bus not in n.buses.index:
            n.add("Bus", t_bus, carrier="transport")

        link_name = f"{bus}_transport_link"
        if link_name not in n.links.index:
            n.add(
                "Link",
                name=link_name,
                bus0=bus,
                bus1=t_bus,
                p_nom=0,
                efficiency=float(transport_cfg.get("link_efficiency", 0.99)),
                capital_cost=float(transport_cfg.get("link_capital_cost", 1_000.0)),
                p_nom_extendable=True,
                carrier="transport",
            )

    # ---- Load transport demand time series ----
    t_path    = data_folder / cfg["files"].get("transport_demand", "2020_ElectricTransport.xlsx")
    transport = read_excel_timeseries(t_path, cfg, n.snapshots)
    transport.columns = [f"{col}_transport" for col in transport.columns]

    # Ensure buses and loads exist for each column in the demand file.
    for load_name in transport.columns:
        region = load_name.replace("_transport", "")
        t_bus  = f"{region}_transport"

        if region not in n.buses.index:
            logger.warning(
                "Skipping transport region '%s': electricity bus not in network.", region
            )
            continue

        if t_bus not in n.buses.index:
            n.add("Bus", t_bus, carrier="transport")

        if load_name not in n.loads.index:
            n.add("Load", name=load_name, bus=t_bus, p_set=0.0, carrier="transport")

    existing = [c for c in transport.columns if c in n.loads.index]
    if existing:
        n.loads_t.p_set[existing] = transport[existing]
    else:
        logger.warning("No transport demand columns matched loads; time series not set.")

    # ---- EV buffer storage ----
    # Sized at 25% of mean hourly demand per region to represent the managed-
    # charging flexibility of the EV fleet.
    # Carrier is "transport" (not "electricity") so these units appear in the
    # correct sector grouping in all downstream results and map plots.
    storage_parameters = dict(
        carrier="transport",
        efficiency_store=float(storage_cfg.get("efficiency_store", 0.95)),
        efficiency_dispatch=float(storage_cfg.get("efficiency_dispatch", 0.95)),
        standing_loss=float(storage_cfg.get("standing_loss", 0.001)),
        capital_cost=float(storage_cfg.get("capital_cost", 1_000)),
        marginal_cost=float(storage_cfg.get("marginal_cost", 0.0)),
        lifetime=int(storage_cfg.get("lifetime", 20)),
        max_hours=float(storage_cfg.get("max_hours", 8)),
        p_nom_extendable=False,   # sized exogenously from demand, not optimised
    )

    for bus in n.buses.index:
        if not bus.endswith("_transport"):
            continue

        storage_name = f"transport_storage_{bus}"
        if storage_name in n.storage_units.index:
            continue

        # p_nom = 25% of mean load on this transport bus (0 if no data yet).
        p_nom = 0.0
        if hasattr(n.loads_t, "p_set") and bus in n.loads_t.p_set.columns:
            p_nom = 0.25 * float(n.loads_t.p_set[bus].mean())

        n.add("StorageUnit", name=storage_name, bus=bus, p_nom=p_nom, **storage_parameters)

    logger.info("Adding transport done.")
    return n

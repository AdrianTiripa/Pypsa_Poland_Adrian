from __future__ import annotations

import logging
from pathlib import Path

import pypsa

from .profile_io import read_excel_timeseries

logger = logging.getLogger(__name__)


def add_transport(n: pypsa.Network, cfg: dict) -> pypsa.Network:
    """
    Add electric transport demand as a separate carrier/bus per region,
    linked from the electricity bus, plus a simple storage unit on each transport bus.
    """
    data_folder = Path(cfg["paths"]["data_folder"])
    logger.info("Adding transport started.")

    if "transport" not in n.carriers.index:
        n.add("Carrier", "transport")

    # Create transport buses + links from electricity to transport
    for bus in n.buses.index:
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
                efficiency=0.99,
                capital_cost=1_000.0,
                p_nom_extendable=True,
                carrier="transport",
            )

    # Load transport demand time series
    t_path = data_folder / cfg["files"].get("transport_demand", "2020_ElectricTransport.xlsx")
    transport = read_excel_timeseries(t_path, cfg, n.snapshots)
    transport.columns = [f"{col}_transport" for col in transport.columns]

    # Ensure buses + loads exist
    for load_name in transport.columns:
        region = load_name.replace("_transport", "")
        t_bus = f"{region}_transport"

        if region not in n.buses.index:
            logger.warning("Skipping transport region '%s': electricity bus not in network.", region)
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

    # Add transport storage
    storage_parameters = dict(
        carrier="electricity",
        efficiency_store=0.95,
        efficiency_dispatch=0.95,
        standing_loss=0.001,
        capital_cost=1_000,
        marginal_cost=0.0,
        lifetime=20,
        max_hours=8,
        p_nom_extendable=False,
    )

    for bus in n.buses.index:
        if not bus.endswith("_transport"):
            continue

        storage_name = f"transport_storage_{bus}"
        if storage_name in n.storage_units.index:
            continue

        p_nom = 0.0
        if hasattr(n.loads_t, "p_set") and bus in n.loads_t.p_set.columns:
            p_nom = 0.25 * float(n.loads_t.p_set[bus].mean())

        n.add("StorageUnit", name=storage_name, bus=bus, p_nom=p_nom, **storage_parameters)

    logger.info("Adding transport done.")
    return n
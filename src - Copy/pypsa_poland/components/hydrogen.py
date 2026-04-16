#src/pypsa_poland/components/hydrogen.py

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pypsa

from .profile_io import read_excel_timeseries

logger = logging.getLogger(__name__)


def add_hydrogen(n: pypsa.Network, cfg: dict) -> pypsa.Network:
    """
    Add hydrogen buses, electrolysers (elec -> H2), and hydrogen demand time series.
    """
    data_folder = Path(cfg["paths"]["data_folder"])
    logger.info("Adding hydrogen started.")

    h2_cfg = cfg.get("hydrogen", {}).get("electrolyser", {})

    if "hydrogen" not in n.carriers.index:
        n.add("Carrier", "hydrogen")

    for bus in n.buses.index:
        if bus.endswith("_heat") or bus.endswith("_hydrogen") or bus.endswith("_high_temp_heat"):
            continue

        h2_bus = f"{bus}_hydrogen"
        if h2_bus not in n.buses.index:
            n.add("Bus", h2_bus, carrier="hydrogen")

        ely_name = f"{bus}_electrolyzer"
        if ely_name not in n.links.index:
            n.add(
                "Link",
                name=ely_name,
                bus0=bus,
                bus1=h2_bus,
                efficiency=float(h2_cfg.get("efficiency", 0.67)),
                capital_cost=float(h2_cfg.get("capital_cost", 330_000)),
                p_nom_extendable=bool(h2_cfg.get("p_nom_extendable", True)),
                ramp_limit_up=float(h2_cfg.get("ramp_limit_up", 0.2)),
                ramp_limit_down=float(h2_cfg.get("ramp_limit_down", 0.2)),
                carrier=str(h2_cfg.get("carrier", "hydrogen")),
            )

        load_name = f"{bus}_hydrogen_demand"
        if load_name not in n.loads.index:
            n.add("Load", name=load_name, bus=h2_bus, p_set=0.0, carrier="hydrogen")

    # Demand time series (Excel)
    h2_path = data_folder / cfg["files"].get("hydrogen_demand", "2020_PureHydrogen.xlsx")
    h2_demand = read_excel_timeseries(h2_path, cfg, n.snapshots)
    h2_demand.columns = [f"{col}_hydrogen_demand" for col in h2_demand.columns]

    for load_name in h2_demand.columns:
        region = load_name.replace("_hydrogen_demand", "")
        h2_bus = f"{region}_hydrogen"
        if h2_bus not in n.buses.index:
            logger.warning("Skipping H2 demand '%s': bus '%s' missing.", load_name, h2_bus)
            continue
        if load_name not in n.loads.index:
            n.add("Load", name=load_name, bus=h2_bus, p_set=0.0, carrier="hydrogen")

    existing = [c for c in h2_demand.columns if c in n.loads.index]
    if existing:
        n.loads_t.p_set[existing] = h2_demand[existing]
    else:
        logger.warning("No hydrogen demand columns matched existing loads; time series not set.")

    logger.info("Adding hydrogen done.")
    return n


def _allowed_cavern_buses_from_caps(hs_cfg: dict) -> set[str]:
    """
    Derive allowed cavern buses from YAML cap entries like:
    Hydrogen_Storage_PL DS_hydrogen
    Hydrogen_Storage_PL KP_hydrogen
    """
    caps = hs_cfg.get("caps", {}) or {}
    allowed = set()

    prefix = "Hydrogen_Storage_"
    other_prefix = "Hydrogen_Storage_other_"

    for su_name in caps:
        su_name = str(su_name)

        if su_name.startswith(other_prefix):
            continue

        if su_name.startswith(prefix):
            bus = su_name[len(prefix):]
            if bus.endswith("_hydrogen"):
                allowed.add(bus)

    return allowed


def add_hydrogen_storage(n: pypsa.Network, cfg: dict) -> pypsa.Network:
    """
    Add hydrogen storage units on *_hydrogen buses.

    Cavern storage is allowed only on buses explicitly listed in
    cfg['hydrogen_storage']['caps'] through names like:
        Hydrogen_Storage_PL DS_hydrogen

    "Other" hydrogen storage can still exist on every hydrogen bus.

    This also removes imported cavern storage units that appear on disallowed buses,
    so a messy storage_units.csv cannot silently reintroduce bad cavern locations.
    """
    hs_cfg = cfg.get("hydrogen_storage", {})

    cavern_cfg = hs_cfg.get("cavern", {})
    other_cfg = hs_cfg.get("other", {})

    carrier_cavern = str(cavern_cfg.get("carrier", "hydrogen storage"))
    carrier_other = str(other_cfg.get("carrier", "hydrogen storage other"))

    if carrier_cavern not in n.carriers.index:
        n.add("Carrier", carrier_cavern)
    if carrier_other not in n.carriers.index:
        n.add("Carrier", carrier_other)

    params_cavern = dict(
        carrier=carrier_cavern,
        p_nom_extendable=bool(cavern_cfg.get("p_nom_extendable", True)),
        max_hours=float(cavern_cfg.get("max_hours", 106.0)),
        efficiency_store=float(cavern_cfg.get("efficiency_store", 0.99)),
        efficiency_dispatch=float(cavern_cfg.get("efficiency_dispatch", 0.995)),
        capital_cost=float(cavern_cfg.get("capital_cost", 60_000)),
        marginal_cost=float(cavern_cfg.get("marginal_cost", 0.0)),
        standing_loss=float(cavern_cfg.get("standing_loss", 0.001)),
        cyclic_state_of_charge=bool(cavern_cfg.get("cyclic_state_of_charge", False)),
    )
    params_other = dict(
        carrier=carrier_other,
        p_nom_extendable=bool(other_cfg.get("p_nom_extendable", True)),
        max_hours=float(other_cfg.get("max_hours", 24.0)),
        efficiency_store=float(other_cfg.get("efficiency_store", 0.8)),
        efficiency_dispatch=float(other_cfg.get("efficiency_dispatch", 0.95)),
        capital_cost=float(other_cfg.get("capital_cost", 90_000)),
        marginal_cost=float(other_cfg.get("marginal_cost", 0.0)),
        standing_loss=float(other_cfg.get("standing_loss", 0.0001)),
        cyclic_state_of_charge=bool(other_cfg.get("cyclic_state_of_charge", False)),
    )

    hydrogen_buses = [bus for bus in n.buses.index if str(bus).endswith("_hydrogen")]
    allowed_cavern_buses = _allowed_cavern_buses_from_caps(hs_cfg)

    if allowed_cavern_buses:
        missing_allowed = sorted(b for b in allowed_cavern_buses if b not in n.buses.index)
        if missing_allowed:
            logger.warning(
                "Some YAML cavern locations are not present as buses and will be skipped: %s",
                missing_allowed,
            )
    else:
        logger.warning(
            "No cavern locations found in hydrogen_storage.caps. "
            "No cavern storage units will be added."
        )

    # Remove imported / pre-existing cavern storage on disallowed buses
    to_remove = []
    for su_name in list(n.storage_units.index):
        su_bus = str(n.storage_units.loc[su_name, "bus"]) if "bus" in n.storage_units.columns else ""
        su_carrier = str(n.storage_units.loc[su_name, "carrier"]) if "carrier" in n.storage_units.columns else ""

        is_cavern = (
            su_carrier == carrier_cavern
            or (
                str(su_name).startswith("Hydrogen_Storage_")
                and not str(su_name).startswith("Hydrogen_Storage_other_")
            )
        )

        if is_cavern and su_bus.endswith("_hydrogen") and su_bus not in allowed_cavern_buses:
            to_remove.append(su_name)

    if to_remove:
        logger.info("Removing cavern hydrogen storage on disallowed buses: %s", to_remove)
        n.remove("StorageUnit", to_remove)

    # Add cavern only on allowed buses
    for bus in sorted(allowed_cavern_buses):
        if bus not in n.buses.index:
            continue

        name1 = f"Hydrogen_Storage_{bus}"
        if name1 not in n.storage_units.index:
            n.add("StorageUnit", name=name1, bus=bus, **params_cavern)

    # Add "other" storage on every hydrogen bus
    for bus in hydrogen_buses:
        name2 = f"Hydrogen_Storage_other_{bus}"
        if name2 not in n.storage_units.index:
            n.add("StorageUnit", name=name2, bus=bus, **params_other)

    # Apply YAML caps if provided
    caps = hs_cfg.get("caps", {}) or {}
    for su_name, vals in caps.items():
        if su_name not in n.storage_units.index:
            logger.warning("H2 storage cap provided for missing StorageUnit '%s' (skipping).", su_name)
            continue
        for k, v in (vals or {}).items():
            n.storage_units.loc[su_name, k] = float(v)

    # Price anchor from YAML if present
    anchor = hs_cfg.get("price_anchor", {}) or {}
    anchor_bus = anchor.get("bus", None)
    anchor_name = anchor.get("load_name", "H2_price_anchor")
    if anchor_bus and anchor_bus in n.buses.index and anchor_name not in n.loads.index:
        n.add("Load", name=anchor_name, bus=anchor_bus, carrier="hydrogen", p_set=0.0)
        if hasattr(n.loads_t, "p_set"):
            n.loads_t.p_set[anchor_name] = 0.0
        logger.info("Added hydrogen price anchor load '%s' at bus '%s'.", anchor_name, anchor_bus)

    return n


def add_electrolyser_ramp_constraints(
    n: pypsa.Network, snapshots, ramp_rel_per_hour: float = 0.1
) -> None:
    """
    Adds ramp constraints to the optimization model for electrolyser links.

    IMPORTANT: Call only AFTER the optimization model exists (e.g. after n.optimize.create_model()).
    """
    m = n.model
    p = m.variables["Link-p"]
    p_nom = m.variables["Link-p_nom"]

    ely = [name for name in n.links.index if name.endswith("electrolyzer")]
    if not ely:
        return

    times = list(snapshots)
    for i in range(1, len(times)):
        t_prev, t_curr = times[i - 1], times[i]
        dt_hours = (pd.Timestamp(t_curr) - pd.Timestamp(t_prev)).total_seconds() / 3600.0
        rhs = ramp_rel_per_hour * dt_hours * p_nom.loc[ely]

        m.add_constraints(
            p.loc[t_curr, ely] - p.loc[t_prev, ely] <= rhs,
            name=f"ely_ramp_up__{t_curr}",
        )
        m.add_constraints(
            p.loc[t_prev, ely] - p.loc[t_curr, ely] <= rhs,
            name=f"ely_ramp_dn__{t_curr}",
        )
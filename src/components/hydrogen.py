# src/pypsa_poland/components/hydrogen.py

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pypsa

logger = logging.getLogger(__name__)


def add_hydrogen(n: pypsa.Network, data_folder: str | Path) -> pypsa.Network:
    """
    Add hydrogen buses, electrolysers (elec -> H2), and hydrogen demand time series.
    """
    data_folder = Path(data_folder)
    logger.info("Adding hydrogen started.")

    if "hydrogen" not in n.carriers.index:
        n.add("Carrier", "hydrogen")

    # Add hydrogen buses + electrolysers (skip non-electricity buses)
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
                efficiency=0.67,
                capital_cost=330_000,
                p_nom_extendable=True,
                ramp_limit_up=0.2,
                ramp_limit_down=0.2,
                carrier="hydrogen",
            )

        # Create a demand load placeholder (time series filled below if data exists)
        load_name = f"{bus}_hydrogen_demand"
        if load_name not in n.loads.index:
            n.add("Load", name=load_name, bus=h2_bus, p_set=0.0, carrier="hydrogen")

    # Load hydrogen demand time series
    # NOTE: your old code used data_folder + "2020_PureHydrogen.xlsx" (missing slash)
    h2_demand = pd.read_excel(data_folder / "2020_PureHydrogen.xlsx", index_col=0)
    h2_demand.index = n.snapshots

    # Expect columns to be regions like "PL KP", turn them into load names "<region>_hydrogen_demand"
    h2_demand.columns = [f"{col}_hydrogen_demand" for col in h2_demand.columns]

    # Ensure those loads exist and point to the right bus
    for load_name in h2_demand.columns:
        region = load_name.replace("_hydrogen_demand", "")
        h2_bus = f"{region}_hydrogen"
        if h2_bus not in n.buses.index:
            logger.warning("Skipping H2 demand '%s': bus '%s' missing.", load_name, h2_bus)
            continue
        if load_name not in n.loads.index:
            n.add("Load", name=load_name, bus=h2_bus, p_set=0.0, carrier="hydrogen")

    # Assign time series (only for loads that exist)
    existing = [c for c in h2_demand.columns if c in n.loads.index]
    if existing:
        n.loads_t.p_set[existing] = h2_demand[existing]
    else:
        logger.warning("No hydrogen demand columns matched existing loads; time series not set.")

    logger.info("Adding hydrogen done.")
    return n


def add_hydrogen_storage(n: pypsa.Network, cfg: dict | None = None) -> pypsa.Network:
    """
    Add hydrogen storage units on each *_hydrogen bus.
    Keeps your two storage types + your p_nom_max/p_nom_min assignments (guarded).
    """
    if "hydrogen storage" not in n.carriers.index:
        n.add("Carrier", "hydrogen storage")
    if "hydrogen storage other" not in n.carriers.index:
        n.add("Carrier", "hydrogen storage other")

    params_cavern = dict(
        carrier="hydrogen storage",
        p_nom_extendable=True,
        max_hours=106.0,
        efficiency_store=0.990,
        efficiency_dispatch=0.995,
        capital_cost=60_000,
        marginal_cost=0.0,
        standing_loss=0.001,
        cyclic_state_of_charge=False,
    )
    params_other = dict(
        carrier="hydrogen storage other",
        p_nom_extendable=True,
        max_hours=24.0,
        efficiency_store=0.8,
        efficiency_dispatch=0.95,
        capital_cost=90_000,
        marginal_cost=0.0,
        standing_loss=0.0001,
        cyclic_state_of_charge=False,
    )

    for bus in n.buses.index:
        if not bus.endswith("_hydrogen"):
            continue

        name1 = f"Hydrogen_Storage_{bus}"
        if name1 not in n.storage_units.index:
            n.add("StorageUnit", name=name1, bus=bus, **params_cavern)

        name2 = f"Hydrogen_Storage_other_{bus}"
        if name2 not in n.storage_units.index:
            n.add("StorageUnit", name=name2, bus=bus, **params_other)

    # Your manual caps (guard to avoid KeyErrors)
    caps = {
        "Hydrogen_Storage_PL KP_hydrogen": dict(p_nom_max=(1_755_000) / 72.0, p_nom_min=2000.0),
        "Hydrogen_Storage_PL DS_hydrogen": dict(p_nom_max=(3_900_000) / 72.0),
        "Hydrogen_Storage_PL PK_hydrogen": dict(p_nom_max=(2_888_000) / 72.0),
        "Hydrogen_Storage_PL MA_hydrogen": dict(p_nom_max=(270_000) / 72.0),
        "Hydrogen_Storage_PL LU_hydrogen": dict(p_nom_max=0.0),
        "Hydrogen_Storage_PL LB_hydrogen": dict(p_nom_max=0.0),
        "Hydrogen_Storage_PL LD_hydrogen": dict(p_nom_max=0.0),
        "Hydrogen_Storage_PL MZ_hydrogen": dict(p_nom_max=0.0),
        "Hydrogen_Storage_PL OP_hydrogen": dict(p_nom_max=0.0),
        "Hydrogen_Storage_PL ZP_hydrogen": dict(p_nom_max=0.0),
        "Hydrogen_Storage_PL PD_hydrogen": dict(p_nom_max=0.0),
        "Hydrogen_Storage_PL PM_hydrogen": dict(p_nom_max=0.0),
        "Hydrogen_Storage_PL SL_hydrogen": dict(p_nom_max=0.0),
        "Hydrogen_Storage_PL SK_hydrogen": dict(p_nom_max=0.0),
        "Hydrogen_Storage_PL WN_hydrogen": dict(p_nom_max=0.0),
        "Hydrogen_Storage_PL WP_hydrogen": dict(p_nom_max=0.0),
    }
    for su_name, vals in caps.items():
        if su_name in n.storage_units.index:
            for k, v in vals.items():
                n.storage_units.loc[su_name, k] = v

    # Hydrogen price anchor (keep behavior, but no prints)
    anchor_bus = "PL SK_hydrogen"
    anchor_name = "H2_price_anchor"
    if anchor_bus in n.buses.index and anchor_name not in n.loads.index:
        n.add("Load", name=anchor_name, bus=anchor_bus, carrier="hydrogen", p_set=0.0)
        # p_set is already scalar; p_set time series optional. Keep safe:
        if hasattr(n.loads_t, "p_set"):
            n.loads_t.p_set[anchor_name] = 0.0
        logger.info("Added hydrogen price anchor load at bus '%s'.", anchor_bus)

    return n


def add_electrolyser_ramp_constraints(n: pypsa.Network, snapshots, ramp_rel_per_hour: float = 0.1) -> None:
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

        m.add_constraints(p.loc[t_curr, ely] - p.loc[t_prev, ely] <= rhs, name=f"ely_ramp_up__{t_curr}")
        m.add_constraints(p.loc[t_prev, ely] - p.loc[t_curr, ely] <= rhs, name=f"ely_ramp_dn__{t_curr}")

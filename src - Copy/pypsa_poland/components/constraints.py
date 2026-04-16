# src/pypsa_poland/components/constraints.py

from __future__ import annotations

import logging
import pypsa
import xarray as xr

logger = logging.getLogger(__name__)


def add_co2_constraint(n: pypsa.Network, cfg: dict) -> pypsa.Network:
    co2_limit = float(cfg.get("constraints", {}).get("co2_limit_tonnes", 0.005 * 475e6))

    name = "CO2_limit"
    if name not in n.global_constraints.index:
        n.add(
            "GlobalConstraint",
            name,
            carrier_attribute="co2_emissions",
            type="primary_energy",
            sense="<=",
            constant=co2_limit,
        )
    else:
        logger.info("GlobalConstraint '%s' already exists; skipping.", name)

    return n


def fix_total_capacity_by_carrier(n: pypsa.Network, snapshots, cfg: dict) -> None:
    """
    Enforce capacity mix (ported from your old script) using extra_functionality.

    IMPORTANT: Compatible with older linopy where m.variables is not dict-like.
    """
    m = n.model

    # ---- Required model variables (no .get!) ----
    if "Generator-p_nom" not in m.variables:
        raise KeyError("Model variable 'Generator-p_nom' not found. Is the model created correctly?")
    p_nom_gen = m.variables["Generator-p_nom"]

    cap_cfg = cfg.get("capacity_targets", {})

    solar_total = float(cap_cfg.get("solar_pv_ground_total_mw", 69_844))
    wind_onshore_total = float(cap_cfg.get("wind_onshore_total_mw", 36_400))
    wind_offshore_total = float(cap_cfg.get("wind_offshore_total_mw", 45_358))
    gas_total = float(cap_cfg.get("natural_gas_total_mw", 20_200))
    nuclear_total = float(cap_cfg.get("nuclear_total_mw", 19_843))

    storage_min_cfg = cap_cfg.get("storage_unit_p_nom_min_mw", {})
    h2_storage_min = float(storage_min_cfg.get("hydrogen_storage", 3_700))
    h2_storage_other_min = float(storage_min_cfg.get("hydrogen_storage_other", 22_400))
    battery_min = float(storage_min_cfg.get("battery", 19_469))
    flow_min = float(storage_min_cfg.get("flow", 156))
    psh_min = float(storage_min_cfg.get("PSH", 9_070))
    heat_storage_min = float(storage_min_cfg.get("hot_water", 3_300))

    link_caps_cfg = cap_cfg.get("link_p_nom_caps_mw", {})
    h2_links_cap = float(link_caps_cfg.get("links_ending_hydrogen_max", 67_000))
    ely_links_cap = float(link_caps_cfg.get("links_ending_electrolyzer_max", 60_000))

    # ---- 1) Solar PV ground total minimum ----
    solar_gens = n.generators.index[n.generators.carrier == "PV ground"]
    if len(solar_gens) > 0:
        expr = p_nom_gen.loc[solar_gens].sum()
        m.add_constraints(expr >= solar_total, name="total_solar_capacity_min")
    else:
        logger.warning("No generators with carrier 'PV ground'; solar constraint skipped.")

    # ---- 2) Wind onshore total minimum ----
    onshore_gens = n.generators.index[n.generators.carrier == "wind"]
    if len(onshore_gens) > 0:
        expr = p_nom_gen.loc[onshore_gens].sum()
        m.add_constraints(expr >= wind_onshore_total, name="total_onshore_capacity_min")
    else:
        logger.warning("No generators with carrier 'wind'; wind constraint skipped.")

    # ---- 3) Wind offshore total minimum ----
    offshore_gens = n.generators.index[n.generators.carrier == "wind offshore"]
    if len(offshore_gens) > 0:
        expr = p_nom_gen.loc[offshore_gens].sum()
        m.add_constraints(expr >= wind_offshore_total, name="total_offshore_capacity_min")
    else:
        logger.warning("No generators with carrier 'wind offshore'; offshore wind constraint skipped.")

    # ---- 4) Gas total minimum ----
    gas_gens = n.generators.index[n.generators.carrier == "Natural gas"]
    if len(gas_gens) > 0:
        expr = p_nom_gen.loc[gas_gens].sum()
        m.add_constraints(expr >= gas_total, name="total_gas_capacity_min")
    else:
        logger.warning("No generators with carrier 'Natural gas'; gas constraint skipped.")

    # ---- 5) Nuclear total minimum ----
    nuclear_gens = n.generators.index[n.generators.carrier == "nuclear"]
    if len(nuclear_gens) > 0:
        expr = p_nom_gen.loc[nuclear_gens].sum()
        m.add_constraints(expr >= nuclear_total, name="total_nuclear_capacity_min")
    else:
        logger.warning("No generators with carrier 'nuclear'; nuclear constraint skipped.")

    # ---- StorageUnit p_nom mins ----
    if "StorageUnit-p_nom" not in m.variables:
        logger.warning("Model variable 'StorageUnit-p_nom' not found; storage constraints skipped.")
        p_nom_su = None
    else:
        p_nom_su = m.variables["StorageUnit-p_nom"]

        def _min_storage_by_carrier(carrier: str, min_val: float, cname: str) -> None:
            su = n.storage_units.index[n.storage_units.carrier == carrier]
            if len(su) == 0:
                logger.warning("No storage_units with carrier '%s'; %s skipped.", carrier, cname)
                return
            expr = p_nom_su.loc[su].sum()
            m.add_constraints(expr >= min_val, name=cname)

        _min_storage_by_carrier("hydrogen storage", h2_storage_min, "min_hydrogen_storage_power")
        _min_storage_by_carrier("hydrogen storage other", h2_storage_other_min, "min_hydrogen_storage_other_power")
        _min_storage_by_carrier("battery", battery_min, "min_battery_storage_power")
        _min_storage_by_carrier("flow", flow_min, "min_flow_storage_power")
        _min_storage_by_carrier("PSH", psh_min, "min_PSH_storage_power")
        _min_storage_by_carrier("hot_water", heat_storage_min, "min_hot_water_storage_power")

    # ---- Link p_nom caps ----
    if "Link-p_nom" not in m.variables:
        logger.warning("Model variable 'Link-p_nom' not found; link cap constraints skipped.")
    else:
        p_nom_link = m.variables["Link-p_nom"]

        hydrogen_links = [name for name in n.links.index if str(name).endswith("hydrogen")]
        if hydrogen_links:
            expr = p_nom_link.loc[hydrogen_links].sum()
            m.add_constraints(expr <= h2_links_cap, name="cap_links_ending_hydrogen")
        else:
            logger.warning("No links ending with 'hydrogen'; hydrogen link cap skipped.")

        electrolyzer_links = [name for name in n.links.index if str(name).endswith("electrolyzer")]
        if electrolyzer_links:
            expr = p_nom_link.loc[electrolyzer_links].sum()
            m.add_constraints(expr <= ely_links_cap, name="cap_links_ending_electrolyzer")
        else:
            logger.warning("No links ending with 'electrolyzer'; electrolyzer link cap skipped.")

    # ---- Optional: min SOC constraint (guarded) ----
    soc_cfg = cfg.get("operational_constraints", {}).get("hydrogen_min_soc", {})
    if soc_cfg.get("enabled", True) and (p_nom_su is not None):
        if "StorageUnit-state_of_charge" not in m.variables:
            logger.warning("No 'StorageUnit-state_of_charge' var; min SOC skipped.")
            return

        try:
            soc = m.variables["StorageUnit-state_of_charge"]
            min_soc_frac = float(soc_cfg.get("min_soc_fraction", 0.3))
            enforce_from = int(soc_cfg.get("enforce_from_snapshot_index", 360))
            enforce_snapshots = n.snapshots[enforce_from:]

            units = soc_cfg.get("units", ["Hydrogen_Storage_PL KP_hydrogen"])
            units = [u for u in units if u in n.storage_units.index]
            if not units:
                return

            soc_sel = soc.loc[enforce_snapshots, units]
            p_nom_sel = p_nom_su.loc[units]

            max_hours_da = xr.DataArray(
                n.storage_units.loc[units, "max_hours"],
                coords={"StorageUnit": units},
                dims=["StorageUnit"],
            )
            rhs = min_soc_frac * max_hours_da * p_nom_sel
            rhs = rhs.expand_dims(snapshot=enforce_snapshots)

            m.add_constraints(soc_sel >= rhs, name="min_soc_30pct_hydrogen_after_startup")

        except Exception as e:
            logger.warning("H2 min SOC constraint failed/skipped: %s", e)
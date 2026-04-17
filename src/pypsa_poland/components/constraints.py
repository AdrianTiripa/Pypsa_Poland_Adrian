# src/pypsa_poland/components/constraints.py
#
# Custom linopy constraints for the pypsa-poland optimisation model.
#
# Contains two public functions:
#   - add_co2_constraint: adds a global CO2 cap as a PyPSA GlobalConstraint.
#   - fix_total_capacity_by_carrier: injected via extra_functionality at solve
#     time to enforce capacity targets and storage sizing floors from the config.
#
# A private helper _add_h2_min_soc_constraint enforces a minimum state-of-charge
# floor on H2 cavern storage units to prevent the model from over-drawing
# caverns and producing unrealistically optimistic results.

from __future__ import annotations

import logging
import traceback

import pypsa
import xarray as xr

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CO2 global constraint
# ---------------------------------------------------------------------------

def add_co2_constraint(n: pypsa.Network, cfg: dict) -> pypsa.Network:
    """
    Add a system-wide CO2 emission cap as a PyPSA GlobalConstraint.

    The limit is read from cfg['constraints']['co2_limit_tonnes']. If the
    constraint already exists (e.g. imported from the CSV folder), it is left
    unchanged to avoid duplicate constraint names in the linopy model.
    """
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


# ---------------------------------------------------------------------------
# Capacity-target constraints (injected via extra_functionality)
# ---------------------------------------------------------------------------

def fix_total_capacity_by_carrier(n: pypsa.Network, snapshots, cfg: dict) -> None:
    """
    Enforce technology capacity targets by adding constraints to the linopy model.

    Called via n.optimize(extra_functionality=...) so that n.model is already
    built when this function runs. Compatible with older linopy versions where
    m.variables is not dict-like.

    Targets (all in MW) are read from cfg['capacity_targets'] with built-in
    fallback defaults that reflect the NZP 2050 scenario.
    """
    m = n.model

    # Generator p_nom is required — raise immediately if absent rather than
    # silently skipping all generation constraints.
    if "Generator-p_nom" not in m.variables:
        raise KeyError("Model variable 'Generator-p_nom' not found. Is the model created correctly?")
    p_nom_gen = m.variables["Generator-p_nom"]

    cap_cfg = cfg.get("capacity_targets", {})

    # --- Read generation capacity targets from config with NZP 2050 defaults ---
    solar_total         = float(cap_cfg.get("solar_pv_ground_total_mw",  69_844))
    wind_onshore_total  = float(cap_cfg.get("wind_onshore_total_mw",     36_400))
    wind_offshore_total = float(cap_cfg.get("wind_offshore_total_mw",    45_358))
    gas_total           = float(cap_cfg.get("natural_gas_total_mw",      20_200))
    nuclear_total       = float(cap_cfg.get("nuclear_total_mw",          19_843))

    # --- Read storage power floor targets ---
    storage_min_cfg   = cap_cfg.get("storage_unit_p_nom_min_mw", {})
    h2_storage_min        = float(storage_min_cfg.get("hydrogen_storage",        3_700))
    h2_storage_other_min  = float(storage_min_cfg.get("hydrogen_storage_other", 22_400))
    battery_min           = float(storage_min_cfg.get("battery",                19_469))
    flow_min              = float(storage_min_cfg.get("flow",                      156))
    psh_min               = float(storage_min_cfg.get("PSH",                    9_070))
    heat_storage_min      = float(storage_min_cfg.get("hot_water",              3_300))

    # --- Read link capacity caps ---
    link_caps_cfg  = cap_cfg.get("link_p_nom_caps_mw", {})
    h2_links_cap   = float(link_caps_cfg.get("links_ending_hydrogen_max",    67_000))
    ely_links_cap  = float(link_caps_cfg.get("links_ending_electrolyzer_max", 60_000))

    # ---- 1) Solar PV ground — total minimum ----
    solar_gens = n.generators.index[n.generators.carrier == "PV ground"]
    if len(solar_gens) > 0:
        expr = p_nom_gen.loc[solar_gens].sum()
        m.add_constraints(expr >= solar_total, name="total_solar_capacity_min")
    else:
        logger.warning("No generators with carrier 'PV ground'; solar constraint skipped.")

    # ---- 2) Wind onshore — total minimum ----
    onshore_gens = n.generators.index[n.generators.carrier == "wind"]
    if len(onshore_gens) > 0:
        expr = p_nom_gen.loc[onshore_gens].sum()
        m.add_constraints(expr == wind_onshore_total, name="total_onshore_capacity_min")
    else:
        logger.warning("No generators with carrier 'wind'; wind constraint skipped.")

    # ---- 3) Wind offshore — interval [min, max] ----
    # An upper cap is included here because offshore development is pipeline-
    # constrained; the lower bound ensures the model does not under-invest.
    offshore_gens       = n.generators.index[n.generators.carrier == "wind offshore"]
    wind_offshore_min   = float(cap_cfg.get("wind_offshore_min_mw", 30_000))
    if len(offshore_gens) > 0:
        expr = p_nom_gen.loc[offshore_gens].sum()
        m.add_constraints(expr >= wind_offshore_min,   name="total_offshore_capacity_min")
        m.add_constraints(expr <= wind_offshore_total, name="total_offshore_capacity_max")
    else:
        logger.warning("No generators with carrier 'wind offshore'; offshore wind constraints skipped.")

    # ---- 4) Natural gas — total minimum ----
    gas_gens = n.generators.index[n.generators.carrier == "Natural gas"]
    if len(gas_gens) > 0:
        expr = p_nom_gen.loc[gas_gens].sum()
        m.add_constraints(expr >= gas_total, name="total_gas_capacity_min")
    else:
        logger.warning("No generators with carrier 'Natural gas'; gas constraint skipped.")

    # ---- 5) Nuclear — total minimum ----
    nuclear_gens = n.generators.index[n.generators.carrier == "nuclear"]
    if len(nuclear_gens) > 0:
        expr = p_nom_gen.loc[nuclear_gens].sum()
        m.add_constraints(expr >= nuclear_total, name="total_nuclear_capacity_min")
    else:
        logger.warning("No generators with carrier 'nuclear'; nuclear constraint skipped.")

    # ---- StorageUnit p_nom minimum floors ----
    if "StorageUnit-p_nom" not in m.variables:
        logger.warning("Model variable 'StorageUnit-p_nom' not found; storage constraints skipped.")
        p_nom_su = None
    else:
        p_nom_su = m.variables["StorageUnit-p_nom"]

        def _min_storage_by_carrier(carrier: str, min_val: float, cname: str) -> None:
            """Add a minimum capacity floor constraint for storage units of a given carrier."""
            su = n.storage_units.index[n.storage_units.carrier == carrier]
            if len(su) == 0:
                logger.warning("No storage_units with carrier '%s'; %s skipped.", carrier, cname)
                return
            expr = p_nom_su.loc[su].sum()
            m.add_constraints(expr >= min_val, name=cname)

        _min_storage_by_carrier("hydrogen storage",       h2_storage_min,       "min_hydrogen_storage_power")
        _min_storage_by_carrier("hydrogen storage other", h2_storage_other_min, "min_hydrogen_storage_other_power")
        _min_storage_by_carrier("battery",                battery_min,          "min_battery_storage_power")
        _min_storage_by_carrier("flow",                   flow_min,             "min_flow_storage_power")
        _min_storage_by_carrier("PSH",                    psh_min,              "min_PSH_storage_power")
        _min_storage_by_carrier("hot_water",              heat_storage_min,     "min_hot_water_storage_power")

    # ---- Link p_nom upper caps ----
    if "Link-p_nom" not in m.variables:
        logger.warning("Model variable 'Link-p_nom' not found; link cap constraints skipped.")
    else:
        p_nom_link = m.variables["Link-p_nom"]

        # Cap total H2 pipeline capacity (links whose name ends with "hydrogen").
        hydrogen_links = [name for name in n.links.index if str(name).endswith("hydrogen")]
        if hydrogen_links:
            expr = p_nom_link.loc[hydrogen_links].sum()
            m.add_constraints(expr <= h2_links_cap, name="cap_links_ending_hydrogen")
        else:
            logger.warning("No links ending with 'hydrogen'; hydrogen link cap skipped.")

        # Cap total electrolyser capacity (links whose name ends with "electrolyzer").
        electrolyzer_links = [name for name in n.links.index if str(name).endswith("electrolyzer")]
        if electrolyzer_links:
            expr = p_nom_link.loc[electrolyzer_links].sum()
            m.add_constraints(expr <= ely_links_cap, name="cap_links_ending_electrolyzer")
        else:
            logger.warning("No links ending with 'electrolyzer'; electrolyzer link cap skipped.")

    # ---- H2 cavern minimum state-of-charge (seasonal robustness) ----
    _add_h2_min_soc_constraint(n, snapshots, cfg, p_nom_su)


# ---------------------------------------------------------------------------
# H2 cavern minimum state-of-charge constraint
# ---------------------------------------------------------------------------

def _add_h2_min_soc_constraint(
    n: pypsa.Network,
    snapshots,
    cfg: dict,
    p_nom_su,
) -> None:
    """
    Enforce a minimum state-of-charge floor on specified H2 cavern storage units.

    This constraint prevents the model from over-drawing caverns late in the
    year. Without it, the optimiser can freely drain caverns to zero by December,
    treating the initial SOC as a free gift that never needs replenishing —
    making the system appear more robust than it really is.

    The floor is applied from snapshot index `enforce_from_snapshot_index`
    onwards (default 360, roughly two weeks in), not from snapshot 0, to give
    the model freedom to ramp up initially without being immediately constrained.

    BUGS FIXED vs old code
    ----------------------
    1. The old bare ``except Exception as e: logger.warning(str(e))`` swallowed
       the full traceback, making constraint failures invisible. This version
       logs at ERROR level with the full traceback and re-raises so the solve
       aborts rather than producing a silently wrong result.

    2. The old code hardcoded the linopy dimension name "StorageUnit" when
       selecting from p_nom. In PyPSA 0.35 / linopy, extendable storage units
       use "StorageUnit-ext" instead, causing a KeyError. This version detects
       the actual dimension name at runtime from each variable's .dims attribute.

    3. The RHS no longer uses expand_dims — linopy broadcasting handles it.
    """
    m = n.model
    soc_cfg = cfg.get("operational_constraints", {}).get("hydrogen_min_soc", {})

    if not soc_cfg.get("enabled", True):
        logger.info("H2 min SOC constraint disabled in config; skipping.")
        return

    if p_nom_su is None:
        logger.warning("H2 min SOC: StorageUnit-p_nom variable unavailable; skipping.")
        return

    if "StorageUnit-state_of_charge" not in m.variables:
        logger.warning("H2 min SOC: 'StorageUnit-state_of_charge' variable not in model; skipping.")
        return

    # Check that the configured storage units actually exist in the network.
    units_cfg = soc_cfg.get("units", ["Hydrogen_Storage_PL KP_hydrogen"])
    units = [u for u in units_cfg if u in n.storage_units.index]
    if not units:
        logger.warning(
            "H2 min SOC: none of the configured units %s found in storage_units; skipping.",
            units_cfg,
        )
        return

    min_soc_frac  = float(soc_cfg.get("min_soc_fraction", 0.3))
    enforce_from  = int(soc_cfg.get("enforce_from_snapshot_index", 360))

    # Guard against enforce_from pointing beyond the end of the snapshot index.
    n_snaps = len(n.snapshots)
    if enforce_from >= n_snaps:
        logger.warning(
            "H2 min SOC: enforce_from_snapshot_index=%d >= number of snapshots=%d; "
            "constraint would cover zero snapshots. Skipping.",
            enforce_from, n_snaps,
        )
        return

    enforce_snapshots = n.snapshots[enforce_from:]
    logger.info(
        "H2 min SOC: enforcing %.0f%% floor on %s from snapshot %d (%s) to %s — %d snapshots.",
        min_soc_frac * 100,
        units,
        enforce_from,
        enforce_snapshots[0],
        enforce_snapshots[-1],
        len(enforce_snapshots),
    )

    try:
        soc_var = m.variables["StorageUnit-state_of_charge"]

        def _storage_dim(var) -> str:
            """
            Return the storage-unit dimension name from a linopy Variable.

            In PyPSA 0.35 / linopy the extendable storage p_nom variable uses
            "StorageUnit-ext" as its dimension name; operational variables may
            use a different name. Reading dims at runtime avoids hardcoding.
            """
            for dim in var.dims:
                if str(dim).startswith("StorageUnit"):
                    return str(dim)
            raise ValueError(
                f"Cannot find a StorageUnit* dimension in variable dims {var.dims}. "
                "Check PyPSA/linopy version compatibility."
            )

        pnom_dim = _storage_dim(p_nom_su)   # e.g. "StorageUnit-ext"
        soc_dim  = _storage_dim(soc_var)    # e.g. "StorageUnit" or "StorageUnit-ext"

        logger.info("H2 min SOC: p_nom dim='%s', soc dim='%s'.", pnom_dim, soc_dim)

        # Validate that all requested units exist in both variable coordinate arrays.
        pnom_coords = list(p_nom_su.coords[pnom_dim].values)
        soc_coords  = list(soc_var.coords[soc_dim].values)

        units_in_pnom = [u for u in units if u in pnom_coords]
        units_in_soc  = [u for u in units if u in soc_coords]
        units_ok      = [u for u in units if u in units_in_pnom and u in units_in_soc]
        units_missing = [u for u in units if u not in units_ok]

        if units_missing:
            logger.warning(
                "H2 min SOC: units %s not found in both p_nom (%s dim) and "
                "soc (%s dim) variables; they will be skipped.",
                units_missing, pnom_dim, soc_dim,
            )
        if not units_ok:
            logger.warning("H2 min SOC: no valid units remain after coordinate check; skipping.")
            return

        # Validate that the SOC variable carries a 'snapshot' coordinate.
        soc_snap_coords = soc_var.coords.get("snapshot", None)
        if soc_snap_coords is None:
            raise ValueError(
                "Linopy SOC variable has no 'snapshot' coordinate. "
                "Check PyPSA/linopy version compatibility."
            )

        # Slice the SOC and p_nom variables to the relevant units and snapshots.
        soc_sel   = soc_var.sel({"snapshot": enforce_snapshots, soc_dim:  units_ok})
        p_nom_sel = p_nom_su.sel({pnom_dim: units_ok})

        # Build RHS: min_soc_frac * max_hours * p_nom.
        # Shape is [StorageUnit] and linopy broadcasts across the snapshot dimension.
        max_hours_da = xr.DataArray(
            n.storage_units.loc[units_ok, "max_hours"].values,
            coords={pnom_dim: units_ok},
            dims=[pnom_dim],
        )
        rhs = min_soc_frac * max_hours_da * p_nom_sel

        m.add_constraints(soc_sel >= rhs, name="min_soc_h2_cavern")
        logger.info(
            "H2 min SOC constraint 'min_soc_h2_cavern' added successfully "
            "(%d unit(s) × %d snapshots).",
            len(units_ok),
            len(enforce_snapshots),
        )

    except Exception:
        # Log the FULL traceback at ERROR level — never swallow silently.
        # Re-raise so the solve aborts rather than producing a wrong result.
        logger.error(
            "H2 min SOC constraint failed and was NOT added to the model. "
            "Full traceback:\n%s",
            traceback.format_exc(),
        )
        raise

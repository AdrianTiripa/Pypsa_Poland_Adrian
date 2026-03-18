# src/pypsa_poland/components/supply.py

from __future__ import annotations

import logging
import pypsa
import numpy as np

from .profile_io import read_profile_csv

logger = logging.getLogger(__name__)


def _ensure_carrier(
    n: pypsa.Network,
    carrier: str,
    *,
    co2_emissions: float = 0.0,
    nice_name: str | None = None,
    color: str | None = None,
) -> None:
    if carrier in n.carriers.index:
        return
    kwargs = {"co2_emissions": co2_emissions}
    if nice_name is not None:
        kwargs["nice_name"] = nice_name
    if color is not None:
        kwargs["color"] = color
    n.add("Carrier", carrier, **kwargs)


def _unlock_thermal_generators(n: pypsa.Network, cfg: dict) -> None:
    """
    Make existing thermal generators (imported from data/) investable/usable.

    Typical issue: generators.csv has gas/biogas rows with p_nom=0 and p_nom_extendable=FALSE,
    which hard-locks them at zero capacity => they never appear in dispatch/results.
    """
    supply_cfg = cfg.get("supply", {})
    unlock_cfg = supply_cfg.get("thermal_unlock", {}) or {}

    carriers = unlock_cfg.get("carriers", ["Natural gas", "Biogas plant"])
    carriers = [str(c) for c in carriers]

    set_extendable = bool(unlock_cfg.get("set_extendable", True))
    default_p_nom_max = unlock_cfg.get("default_p_nom_max", None)  # e.g. 1e9

    # Optional per-carrier max/min overrides from YAML if you want later
    p_nom_max_by_carrier = unlock_cfg.get("p_nom_max_by_carrier", {}) or {}
    p_nom_min_by_carrier = unlock_cfg.get("p_nom_min_by_carrier", {}) or {}

    for car in carriers:
        mask = (n.generators.carrier.astype(str) == car)
        if not mask.any():
            logger.warning("Thermal unlock: no generators with carrier='%s' found in imported data.", car)
            continue

        if set_extendable:
            n.generators.loc[mask, "p_nom_extendable"] = True

        # If you imported p_nom=0 for everything, that's OK IF extendable=True and p_nom_max allows investment.
        if default_p_nom_max is not None and "p_nom_max" in n.generators.columns:
            bad = mask & (n.generators["p_nom_max"].replace([np.inf, -np.inf], np.nan).fillna(0.0) <= 0.0)
            if bad.any():
                n.generators.loc[bad, "p_nom_max"] = float(default_p_nom_max)

        # Optional explicit caps in YAML per carrier
        if car in p_nom_max_by_carrier and "p_nom_max" in n.generators.columns:
            n.generators.loc[mask, "p_nom_max"] = float(p_nom_max_by_carrier[car])

        if car in p_nom_min_by_carrier and "p_nom_min" in n.generators.columns:
            n.generators.loc[mask, "p_nom_min"] = float(p_nom_min_by_carrier[car])

        # Thermal should be fully available unless you intentionally constrain it.
        # If your imported p_max_pu column exists and is 0, fix it.
        if hasattr(n, "generators_t") and hasattr(n.generators_t, "p_max_pu"):
            cols = n.generators.index[mask].tolist()
            existing = [c for c in cols if c in n.generators_t.p_max_pu.columns]
            if existing:
                # Only fix zeros/NaNs; keep any intentional profiles
                pm = n.generators_t.p_max_pu[existing]
                bad_ts = pm.isna() | (pm <= 0.0)
                if bad_ts.any().any():
                    n.generators_t.p_max_pu.loc[:, existing] = pm.mask(bad_ts, 1.0)

        logger.info("Thermal unlock applied for carrier='%s' (%d generators).", car, int(mask.sum()))


def add_generators(n: pypsa.Network, cfg: dict) -> pypsa.Network:
    supply_cfg = cfg.get("supply", {})
    nuclear_cfg = supply_cfg.get("nuclear", {}) or {}
    biogas_cfg = supply_cfg.get("biogas", {}) or {}

    # -------------------------
    # Nuclear (as you had)
    # -------------------------
    _ensure_carrier(
        n,
        "nuclear",
        co2_emissions=float(nuclear_cfg.get("co2_emissions", 0.0)),
        nice_name="Nuclear Power",
        color=nuclear_cfg.get("color", "#808080"),
    )

    buses = nuclear_cfg.get("buses", "all")
    if buses == "all":
        buses = list(n.buses.index)

    skip_missing = bool(nuclear_cfg.get("skip_missing_buses", True))

    default_p_nom_max = float(nuclear_cfg.get("default_p_nom_max", 0))
    p_nom_max_by_bus = nuclear_cfg.get("p_nom_max_by_bus", {}) or {}
    p_nom_min_by_bus = nuclear_cfg.get("p_nom_min_by_bus", {}) or {}

    added = 0
    for bus in buses:
        if bus not in n.buses.index:
            if skip_missing:
                logger.warning("Nuclear: bus=%s not in network, skipping", bus)
                continue
            raise ValueError(f"Nuclear: bus '{bus}' not found in network.buses")

        name = f"nuclear_{bus}"
        if name in n.generators.index:
            continue

        p_nom_max = float(p_nom_max_by_bus.get(bus, default_p_nom_max))
        p_nom_min = p_nom_min_by_bus.get(bus, None)

        n.add(
            "Generator",
            name=name,
            bus=bus,
            carrier="nuclear",
            p_nom_extendable=True,
            capital_cost=float(nuclear_cfg["capital_cost"]),
            marginal_cost=float(nuclear_cfg["marginal_cost"]),
            efficiency=float(nuclear_cfg.get("efficiency", 0.33)),
            ramp_limit_up=float(nuclear_cfg.get("ramp_limit_up", 0.075)),
            ramp_limit_down=float(nuclear_cfg.get("ramp_limit_down", 0.075)),
            p_nom_max=p_nom_max,
            **({"p_nom_min": float(p_nom_min)} if p_nom_min is not None else {}),
        )
        added += 1

    logger.info("Added %d nuclear generators", added)

    # Min stable generation for nuclear
    nuclear_pmin = float(nuclear_cfg.get("p_min_pu", 0.65))
    mask_nuclear = (n.generators.carrier.astype(str) == "nuclear")
    if mask_nuclear.any():
        n.generators.loc[mask_nuclear, "p_min_pu"] = nuclear_pmin

    # Min stable generation for biogas (if biogas exists in imported generators)
    biogas_carrier = str(biogas_cfg.get("carrier_name", "Biogas plant"))
    biogas_pmin = float(biogas_cfg.get("p_min_pu", 0.55))
    mask_biogas = (n.generators.carrier.astype(str) == biogas_carrier)
    if mask_biogas.any():
        n.generators.loc[mask_biogas, "p_min_pu"] = biogas_pmin

    # -------------------------
    # CRITICAL: Unlock gas + biogas (investment/dispatch)
    # -------------------------
    _unlock_thermal_generators(n, cfg)

    # -------------------------
    # Renewable CF (p_max_pu) multi-year
    # -------------------------
    pv_cf = read_profile_csv(cfg, "pv_cf", n.snapshots)
    on_cf = read_profile_csv(cfg, "onshore_cf", n.snapshots)
    off_cf = read_profile_csv(cfg, "offshore_cf", n.snapshots)

    n.generators_t.p_max_pu = n.generators_t.p_max_pu.reindex(index=n.snapshots)

    pv_map = {col: f"PL {col} PV2" for col in pv_cf.columns}
    pv_cols = [pv_map[c] for c in pv_cf.columns if pv_map[c] in n.generators.index]
    if pv_cols:
        n.generators_t.p_max_pu.loc[:, pv_cols] = pv_cf.rename(columns=pv_map)[pv_cols]
        logger.info("Set PV p_max_pu for %d generators (PV2). Example: %s", len(pv_cols), pv_cols[0])
    else:
        logger.warning("PV CF: no generators matched pattern 'PL <region> PV2'")

    on_map = {col: f"PL {col} VESTAS V90" for col in on_cf.columns}
    on_cols = [on_map[c] for c in on_cf.columns if on_map[c] in n.generators.index]
    if on_cols:
        n.generators_t.p_max_pu.loc[:, on_cols] = on_cf.rename(columns=on_map)[on_cols]
        logger.info("Set onshore p_max_pu for %d generators. Example: %s", len(on_cols), on_cols[0])
    else:
        logger.warning("Onshore CF: no generators matched pattern 'PL <region> VESTAS V90'")

    def _off_region(col: str) -> str:
        c = str(col).strip()
        if "_" in c:
            c = c.split("_", 1)[0]
        c = c.replace("PL ", "").replace("PL_", "")
        return c

    off_map = {col: f"offshore_wind_PL_{_off_region(col)}" for col in off_cf.columns}
    off_cols = [off_map[c] for c in off_cf.columns if off_map[c] in n.generators.index]
    if off_cols:
        n.generators_t.p_max_pu.loc[:, off_cols] = off_cf.rename(columns=off_map)[off_cols]
        logger.info("Set offshore p_max_pu for %d generators. Example: %s", len(off_cols), off_cols[0])
    else:
        logger.warning("Offshore CF: no generators matched pattern 'offshore_wind_PL_XX'")

    return n
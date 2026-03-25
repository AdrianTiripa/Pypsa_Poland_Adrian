from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path

import pandas as pd
import pypsa

from .profile_io import read_profile_csv, read_excel_timeseries

logger = logging.getLogger(__name__)


PROVINCE_ORDER = [
    "DS", "KP", "LD", "LU", "LB", "MA", "MZ", "OP",
    "PK", "PD", "PM", "SL", "SK", "WN", "WP", "ZP",
]

REGION_ALIASES = {
    "ds": "DS",
    "pl ds": "DS",
    "pl-ds": "DS",
    "dolnoslaskie": "DS",
    "dolno slaskie": "DS",

    "kp": "KP",
    "pl kp": "KP",
    "pl-kp": "KP",
    "kujawsko pomorskie": "KP",
    "kujawsko-pomorskie": "KP",

    "ld": "LD",
    "pl ld": "LD",
    "pl-ld": "LD",
    "lodzkie": "LD",

    "lu": "LU",
    "pl lu": "LU",
    "pl-lu": "LU",
    "lubelskie": "LU",

    "lb": "LB",
    "pl lb": "LB",
    "pl-lb": "LB",
    "lubuskie": "LB",

    "ma": "MA",
    "pl ma": "MA",
    "pl-ma": "MA",
    "malopolskie": "MA",

    "mz": "MZ",
    "pl mz": "MZ",
    "pl-mz": "MZ",
    "mazowieckie": "MZ",

    "op": "OP",
    "pl op": "OP",
    "pl-op": "OP",
    "opolskie": "OP",

    "pk": "PK",
    "pl pk": "PK",
    "pl-pk": "PK",
    "podkarpackie": "PK",

    "pd": "PD",
    "pl pd": "PD",
    "pl-pd": "PD",
    "podlaskie": "PD",

    "pm": "PM",
    "pl pm": "PM",
    "pl-pm": "PM",
    "pomorskie": "PM",

    "sl": "SL",
    "pl sl": "SL",
    "pl-sl": "SL",
    "slaskie": "SL",

    "sk": "SK",
    "pl sk": "SK",
    "pl-sk": "SK",
    "swietokrzyskie": "SK",

    "wn": "WN",
    "pl wn": "WN",
    "pl-wn": "WN",
    "warminsko mazurskie": "WN",
    "warminsko-mazurskie": "WN",

    "wp": "WP",
    "pl wp": "WP",
    "pl-wp": "WP",
    "wielkopolskie": "WP",

    "zp": "ZP",
    "pl zp": "ZP",
    "pl-zp": "ZP",
    "zachodniopomorskie": "ZP",
}


def _slugify(text: str) -> str:
    s = str(text).strip()
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.replace("_", " ").replace(";", " ").replace("/", " ")
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def _normalize_region_code(col: str) -> str:
    raw = str(col).strip()

    if raw.lower().startswith("unnamed"):
        return raw

    parts = [p.strip() for p in raw.split(";") if p.strip()]
    for part in reversed(parts):
        part_up = part.upper()
        if re.fullmatch(r"[A-Z]{2}", part_up):
            return part_up
        if re.fullmatch(r"PL[-_ ]?[A-Z]{2}", part_up):
            return part_up[-2:]

    raw_up = raw.upper().replace("_", " ").strip()
    if re.fullmatch(r"[A-Z]{2}", raw_up):
        return raw_up
    if re.fullmatch(r"PL[-_ ]?[A-Z]{2}", raw_up):
        return raw_up[-2:]

    slug = _slugify(raw)
    return REGION_ALIASES.get(slug, raw.strip())


def _normalize_profile_columns(df: pd.DataFrame, label: str) -> pd.DataFrame:
    out = df.copy()

    keep_cols = [c for c in out.columns if not str(c).lower().startswith("unnamed")]
    out = out.loc[:, keep_cols]

    new_cols = [_normalize_region_code(c) for c in out.columns]

    if len(new_cols) != len(set(new_cols)):
        dupes = sorted({c for c in new_cols if new_cols.count(c) > 1})
        raise ValueError(f"{label}: duplicate columns after normalization: {dupes}")

    out.columns = new_cols
    return out


def _assign_fixed_province_order_if_needed(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    New COP / heat-demand files are raw numeric matrices with no headers.
    If pandas reads them with default integer columns 0..15, assign the
    known province order explicitly.
    """
    out = df.copy()

    expected_numeric_cols = list(range(out.shape[1]))
    actual_cols = list(out.columns)

    if actual_cols == expected_numeric_cols:
        if out.shape[1] != len(PROVINCE_ORDER):
            raise ValueError(
                f"{label}: file has {out.shape[1]} columns, expected {len(PROVINCE_ORDER)} "
                f"for the fixed province order."
            )
        out.columns = PROVINCE_ORDER

    return out


def add_heat(n: pypsa.Network, cfg: dict) -> pypsa.Network:
    """
    Add low-temperature (non-industrial) heat demand and heat pumps.

    - Creates a "{bus}_heat" bus for each electricity bus b (if not exists).
    - Adds a heat pump Link from elec bus -> heat bus for each region in the demand table.
    - Adds a Load on each heat bus with time series from profile CSV.
    """
    logger.info("Adding low-temp heat demand started.")

    heat_demand = read_profile_csv(cfg, "heat_demand_multi", n.snapshots)
    heat_demand = _assign_fixed_province_order_if_needed(heat_demand, "heat demand")
    heat_demand = _normalize_profile_columns(heat_demand, "heat demand")

    demand_scale = float(cfg.get("heat", {}).get("demand_scale", 1.0))
    if demand_scale != 1.0:
        heat_demand = heat_demand * demand_scale
        logger.info("Scaled heat demand by factor %s", demand_scale)

    for b in n.buses.index:
        heat_bus = f"{b}_heat"
        if heat_bus not in n.buses.index:
            n.add("Bus", heat_bus, carrier="heat")

    cop_default = float(cfg.get("heat", {}).get("heat_pump", {}).get("cop_default", 3.0))

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

    heat_demand.columns = [f"PL {col}_heat" for col in heat_demand.columns]

    for bus in heat_demand.columns:
        if bus not in n.buses.index:
            n.add("Bus", bus, carrier="heat")

    for load_name in heat_demand.columns:
        if load_name not in n.loads.index:
            n.add("Load", name=load_name, bus=load_name)

    n.loads_t.p_set[heat_demand.columns] = heat_demand

    logger.info("Adding low-temp heat demand done.")
    return n


def add_heat_storage(n: pypsa.Network, cfg: dict) -> pypsa.Network:
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
    data_folder = Path(cfg["paths"]["data_folder"])

    carrier = cfg.get("heat", {}).get("high_temp_heat", {}).get("carrier", "high_temp_heat")
    if carrier not in n.carriers.index:
        n.add("Carrier", carrier)

    chp_cfg = cfg.get("heat", {}).get("high_temp_heat", {}).get("chp_h2_plant", {})

    for bus in n.buses.index:
        if bus.endswith("_heat") or bus.endswith("_hydrogen") or bus.endswith("_high_temp_heat"):
            continue

        high_temp_heat_bus = f"{bus}_high_temp_heat"
        if high_temp_heat_bus not in n.buses.index:
            n.add("Bus", high_temp_heat_bus, carrier=carrier)

    heat_path = data_folder / cfg["files"]["heat_demand_industry"]
    heat_demand = read_excel_timeseries(heat_path, cfg, n.snapshots)
    heat_demand = _normalize_profile_columns(heat_demand, "industrial heat demand")

    for region in heat_demand.columns:
        elec_bus = f"PL {region}"
        h2_bus = f"PL {region}_hydrogen"
        heat_bus = f"PL {region}_high_temp_heat"

        if elec_bus not in n.buses.index or h2_bus not in n.buses.index:
            logger.warning("Skipping high-grade heat region '%s': required buses missing.", region)
            continue

        if heat_bus not in n.buses.index:
            n.add("Bus", heat_bus, carrier=carrier)

        link_name = f"PL {region}_chp_hydrogen"
        if link_name not in n.links.index:
            n.add(
                "Link",
                name=link_name,
                bus0=h2_bus,
                bus1=elec_bus,
                bus2=heat_bus,
                efficiency=float(chp_cfg.get("efficiency_el", 0.3)),
                efficiency2=float(chp_cfg.get("efficiency_heat", 0.6)),
                carrier="hydrogen",
                capital_cost=float(chp_cfg.get("capital_cost", 20_000)),
                p_nom=float(chp_cfg.get("p_nom", 0)),
                p_nom_extendable=bool(chp_cfg.get("p_nom_extendable", True)),
            )

    heat_demand.columns = [f"PL {col}_high_temp_heat" for col in heat_demand.columns]

    for load_name in heat_demand.columns:
        if load_name not in n.loads.index:
            n.add("Load", name=load_name, bus=load_name)

    n.loads_t.p_set[heat_demand.columns] = heat_demand
    return n


def add_cop(n: pypsa.Network, cfg: dict) -> pypsa.Network:
    """
    Set time-dependent COP (efficiency) for heat pump links from profile CSV.

    Expects region-like columns and maps them to:
      "PL <region>_heat_pump"
    """
    cop = read_profile_csv(cfg, "cop_multi", n.snapshots)
    cop = _assign_fixed_province_order_if_needed(cop, "COP")
    cop = _normalize_profile_columns(cop, "COP")
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
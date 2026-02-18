from __future__ import annotations

import logging
import pypsa

logger = logging.getLogger(__name__)


def _ensure_carrier(n: pypsa.Network, carrier: str, *, co2_emissions: float = 0.0,
                    nice_name: str | None = None, color: str | None = None) -> None:
    if carrier in n.carriers.index:
        return
    kwargs = {"co2_emissions": co2_emissions}
    if nice_name is not None:
        kwargs["nice_name"] = nice_name
    if color is not None:
        kwargs["color"] = color
    n.add("Carrier", carrier, **kwargs)


def add_generators(n: pypsa.Network, cfg: dict) -> pypsa.Network:
    supply_cfg = cfg.get("supply", {})
    nuclear_cfg = supply_cfg.get("nuclear", {})
    biogas_cfg = supply_cfg.get("biogas", {})

    # Ensure carrier exists
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

    # Add nuclear generators
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
            logger.info("Generator %s already exists; skipping", name)
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

    # Min stable generation
    nuclear_pmin = float(nuclear_cfg.get("p_min_pu", 0.65))
    mask_nuclear = n.generators.carrier == "nuclear"
    if mask_nuclear.any():
        n.generators.loc[mask_nuclear, "p_min_pu"] = nuclear_pmin

    biogas_carrier = biogas_cfg.get("carrier_name", "Biogas plant")
    biogas_pmin = float(biogas_cfg.get("p_min_pu", 0.55))
    mask_biogas = n.generators.carrier == biogas_carrier
    if mask_biogas.any():
        n.generators.loc[mask_biogas, "p_min_pu"] = biogas_pmin

    return n

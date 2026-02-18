from __future__ import annotations

import logging
import pypsa

logger = logging.getLogger(__name__)


def add_co2_constraint(n: pypsa.Network, cfg: dict) -> pypsa.Network:
    """
    Add global CO2 constraint. Keeps your constant as-is (cleanup-only).
    """
    co2_limit = 0.005 * 475e6  # 23.75e6 tCO2 (as in your current script)

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

from __future__ import annotations

import logging
import pypsa

logger = logging.getLogger(__name__)


def add_dc_links(n: pypsa.Network, cfg: dict) -> pypsa.Network:
    """
    Add fixed DC links. (Cleanup-only: keep your single hard-coded link.)
    """
    if "DC" not in n.carriers.index:
        n.add("Carrier", "DC")

    link_name = "DC - PL ZP - PL SL"
    if link_name not in n.links.index:
        n.add(
            "Link",
            link_name,
            bus0="PL ZP",
            bus1="PL SL",
            p_nom=4000,
            p_nom_extendable=False,
            p_min_pu=-1.0,
            efficiency=1.0,
            carrier="DC",
        )
    else:
        logger.info("DC link '%s' already exists; skipping.", link_name)

    return n

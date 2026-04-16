# src/pypsa_poland/components/__init__.py
from __future__ import annotations

from . import supply, heat, hydrogen, transport, network, constraints

REGISTRY = {
    # matches configs/default.yaml pipeline keys
    "generators": supply.add_generators,
    "heat": heat.add_heat,
    "hydrogen": hydrogen.add_hydrogen,
    "cop": heat.add_cop,
    "hydrogen_storage": hydrogen.add_hydrogen_storage,
    "heat_storage": heat.add_heat_storage,
    "high_grade_heat": heat.add_high_grade_heat,
    "transport": transport.add_transport,
    "dc_link": network.add_dc_links,
    "co2_constraint": constraints.add_co2_constraint,
}

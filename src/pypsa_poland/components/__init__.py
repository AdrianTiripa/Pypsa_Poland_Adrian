# src/pypsa_poland/components/__init__.py
#
# Component registry for the pypsa-poland pipeline.
#
# Each key matches a pipeline step name defined in configs/default.yaml.
# The orchestration layer calls REGISTRY[step](n, cfg) for each step in
# sequence, passing the live PyPSA network and the full config dict.
# Adding a new component only requires registering its function here.

from __future__ import annotations

from . import supply, heat, hydrogen, transport, network, constraints

REGISTRY = {
    "generators":       supply.add_generators,
    "heat":             heat.add_heat,
    "hydrogen":         hydrogen.add_hydrogen,
    "cop":              heat.add_cop,
    "hydrogen_storage": hydrogen.add_hydrogen_storage,
    "heat_storage":     heat.add_heat_storage,
    "high_grade_heat":  heat.add_high_grade_heat,
    "transport":        transport.add_transport,
    "dc_link":          network.add_dc_links,
    "co2_constraint":   constraints.add_co2_constraint,
}

from vivarium.core.process import Step
from wholecell.utils import units

from ecoli.processes.registries import topology_registry

TOPOLOGY = {
    "external_glucose": ("boundary", "external", "GLC[p]"),
    "glucose_constraint": ("environment", "exchange_data", "constrained", "GLC[p]"),
    "first_update": ("first_update", "environment-glucose-limit",)
}
NAME = "environment-glucose-limit"

topology_registry.register(NAME, TOPOLOGY)

class EnvironmentGlucoseLimit(Step):
    """Ensure that cells are no longer allowed to uptake glucose when environmental
    concentration hits 0."""
    name = NAME
    topology = TOPOLOGY
    
    def ports_schema(self):
        return {
            "external_glucose": {"_default": 0},
            "glucose_constraint": {
                "default": 20.0 * units.mmol / (units.g * units.h),
                "_updater": "set"
            },
            "first_update": {
                "_default": True,
                "_updater": "set",
                "_divider": {"divider": "set_value", "config": {"value": True}},
            }
        }
    
    def next_update(self, timestep, states):
        if states["first_update"]:
            return {"first_update": False}
        elif states["external_glucose"]==0:
            return {
                "glucose_constraint": 0.0 * units.mmol / (units.g * units.h)
            }
        return {}

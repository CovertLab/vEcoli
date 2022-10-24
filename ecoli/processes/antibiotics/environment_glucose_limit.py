from vivarium.core.process import Step
from vivarium.library.units import units

from ecoli.processes.registries import topology_registry

TOPOLOGY = {
    "external_glucose": ("boundary", "external", "GLC[p]"),
    "glucose_constraint": ("environment", "exchange_data", "constrained", "GLC[p]")
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
            }
        }
    
    def next_update(self, timestep, states):
        if states["external_glucose"] == 0:
            return {
                "glucose_constraint": 0.0 * units.mmol / (units.g * units.h)
            }
        return {}

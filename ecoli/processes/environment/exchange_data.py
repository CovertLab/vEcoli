from typing import Any

from ecoli.processes.registries import topology_registry
from vivarium.core.process import Step
from vivarium.library.units import units

NAME = "exchange_data"
TOPOLOGY = {
    "boundary": ("boundary",),
    "environment": ("environment",),
}
topology_registry.register(NAME, TOPOLOGY)


class ExchangeData(Step):
    """
    Update metabolism exchange constraints according to environment concs.
    """

    name = NAME
    topology = TOPOLOGY
    defaults: dict[str, Any] = {
        "external_state": None,
        "environment_molecules": [],
        "saved_media": {},
        "time_step": 1,
    }

    def __init__(self, parameters=None):
        super().__init__(parameters)
        self.external_state = self.parameters["external_state"]
        self.environment_molecules = self.parameters["environment_molecules"]

    def ports_schema(self):
        return {
            "boundary": {"external": {"*": {"_default": 0 * units.mM}}},
            "environment": {
                "exchange_data": {
                    "constrained": {"_default": {}, "_updater": "set"},
                    "unconstrained": {"_default": set(), "_updater": "set"},
                }
            },
        }

    def next_update(self, timestep, states):
        # Set exchange constraints for metabolism
        env_concs = {
            mol: states["boundary"]["external"][mol]
            for mol in self.environment_molecules
        }

        # Converting threshold is faster than converting all of env_concs
        self.external_state.import_constraint_threshold *= units.mM
        exchange_data = self.external_state.exchange_data_from_concentrations(env_concs)
        self.external_state.import_constraint_threshold = (
            self.external_state.import_constraint_threshold.magnitude
        )

        unconstrained = exchange_data["importUnconstrainedExchangeMolecules"]
        constrained = exchange_data["importConstrainedExchangeMolecules"]
        return {
            "environment": {
                "exchange_data": {
                    "constrained": constrained,
                    "unconstrained": list(unconstrained),
                }
            }
        }

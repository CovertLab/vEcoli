from vivarium.core.process import Step

from ecoli.library.schema import bulk_schema


class CephaloridineAntagonism(Step):
    defaults = {
        "murein_name": "CPD-12261[p]",
        "PBP": {  # penicillin-binding proteins
            "PBP1A": "CPLX0-7717[m]",  # transglycosylase-transpeptidase ~100
            "PBP1B": "CPLX0-3951[i]",  # transglycosylase-transpeptidase ~100
        },
    }

    def __init__(self, parameters=None):
        super().__init__(parameters)

        self.murein = parameters["murein_name"]

    def ports_schema(self):
        return {
            "total_murein": bulk_schema([self.parameters["murein_name"]]),
            "murein_state": bulk_schema(
                ["incorporated_murein", "unincorporated_murein"]
            ),
            "concentrations": {
                "cephaloridine": {"_default": 0.0},
            },
            "bulk": bulk_schema(list(self.parameters["PBP"].values())),
        }

    def next_update(self, timestep, states):
        update = {}

        # New murein to allocate
        new_murein = states["total_murein"][self.murein] - sum(states["murein_state"].values())

        # How many PBPs active vs. inactive
        active_fraction = 0.5

        # Allocate incorporated vs. unincorporated based on
        # what fraction of PBPs are active
        incorporated = round(active_fraction * new_murein)
        update["murein_state"] = {
            "incorporated_murein": incorporated,
            "unincorporated_murein": new_murein - incorporated
        }


        return update

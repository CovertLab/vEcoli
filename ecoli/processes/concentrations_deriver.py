from scipy.constants import N_A

from vivarium.core.process import Step
from vivarium.library.units import units, Quantity
from ecoli.library.schema import bulk_name_to_idx, numpy_schema, counts

AVOGADRO = N_A / units.mol


class ConcentrationsDeriver(Step):
    defaults: dict[str, list[str]] = {
        # Bulk molecule names supplied separately so
        # they can be pulled out the Numpy array
        "bulk_variables": [],
        "variables": [],
    }
    name = "concentrations_deriver"

    def __init__(self, parameters):
        super().__init__(parameters)
        self.bulk_var = self.parameters["bulk_variables"]
        self.var = self.parameters["variables"]
        # Helper indices for Numpy indexing
        self.bulk_var_idx = None

    def ports_schema(self):
        schema = {
            "bulk": numpy_schema("bulk"),
            "counts": {
                variable: {
                    "_default": 0,  # In counts
                }
                for variable in self.parameters["variables"]
            },
            "concentrations": {
                variable: {
                    "_default": 0 * units.mM,
                    "_updater": "set",
                }
                for variable in self.parameters["variables"]
            },
            "volume": {
                "_default": 0 * units.fL,
            },
        }
        return schema

    def next_update(self, timestep, states):
        if self.bulk_var_idx is None:
            self.bulk_var_idx = bulk_name_to_idx(self.bulk_var, states["bulk"]["id"])
        volume = states["volume"]
        assert isinstance(volume, Quantity)
        var_concs = {
            var: (count * units.count / AVOGADRO / volume).to(units.millimolar)
            for var, count in states["counts"].items()
        }
        bulk_counts = counts(states["bulk"], self.bulk_var_idx)
        bulk_concs = (bulk_counts * units.count / AVOGADRO / volume).to(
            units.millimolar
        )
        new_concs = {**var_concs, **dict(zip(self.bulk_var, bulk_concs))}
        update = {"concentrations": new_concs}
        return update

from re import L
from time import time
import numpy as np

from vivarium.core.process import Process

from ecoli.library.parameters import param_store
from ecoli.processes.registries import topology_registry

NAME = "ecoli-lysis-initiation"
TOPOLOGY = {
    "cracked": ("wall_state", "cracked"),
    "lysis_trigger": None  # TODO: connect with Lysis
}
topology_registry.register(NAME, TOPOLOGY)


class LysisInitiation(Process):
    name = NAME

    defaults = {
        "lysis_rate": param_store.get(("lysis_initiation", "lysis_rate")),
        "seed": 0,
    }

    def __init__(self, parameters=None):
        super().__init__(parameters)

        lysis_rate = self.parameters["lysis_rate"]
        rng = np.random.default_rng(self.parameters["seed"])
        self.remaining_time = rng.exponential(1 / lysis_rate)

    def ports_schema(self):
        return {"cracked": {"_default": False}, "lysis_trigger": {"_default": True}}

    def next_update(self, timestep, states):
        if states["cracked"] and not states["lysis_trigger"]:
            self.remaining_time -= timestep
            if self.remaining_time <= 0:
                return {"lysis_trigger": True}

        return {}

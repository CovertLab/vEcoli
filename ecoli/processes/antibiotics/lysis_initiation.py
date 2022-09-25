from re import L
from time import time
import numpy as np

from vivarium.core.process import Process

from ecoli.library.parameters import param_store
from ecoli.processes.registries import topology_registry

NAME = "ecoli-lysis-initiation"
TOPOLOGY = {
    "cracked": ("wall_state", "cracked"),
    "lysis_trigger": None,  # TODO: connect with Lysis
}
topology_registry.register(NAME, TOPOLOGY)


class LysisInitiation(Process):
    name = NAME

    defaults = {
        "mean_lysis_time": param_store.get(("lysis_initiation", "lysis_rate")),
        "seed": 0,
    }

    def __init__(self, parameters=None):
        super().__init__(parameters)

        mean_lysis_time = self.parameters["mean_lysis_time"]
        rng = np.random.default_rng(self.parameters["seed"])
        self.remaining_time = rng.exponential(mean_lysis_time)

    def ports_schema(self):
        return {"cracked": {"_default": False}, "lysis_trigger": {"_default": True}}

    def next_update(self, timestep, states):
        if states["cracked"] and not states["lysis_trigger"]:
            self.remaining_time -= timestep
            if self.remaining_time <= 0:
                return {"lysis_trigger": True}

        return {}


def test_lysis_initiation():
    lysis_initiation = LysisInitiation()


def main():
    test_lysis_initiation()


if __name__ == "__main__":
    main()

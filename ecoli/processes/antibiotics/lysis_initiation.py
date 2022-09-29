import numpy as np

from vivarium.core.composer import Composite
from vivarium.core.engine import Engine
from vivarium.core.process import Process

from ecoli.library.parameters import param_store
from ecoli.processes.registries import topology_registry
from ecoli.processes.environment.lysis import Lysis

NAME = "ecoli-lysis-initiation"
TOPOLOGY = {
    "cracked": ("wall_state", "cracked"),
    "lysis_trigger": ("lysis_trigger",),
}
topology_registry.register(NAME, TOPOLOGY)


class LysisInitiation(Process):
    name = NAME

    defaults = {
        "mean_lysis_time": param_store.get(("lysis_initiation", "mean_lysis_time")),
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
    lysis = Lysis({"agent_id": 0})

    composite = Composite(
        {
            "processes": {"lysis_initiation": lysis_initiation, "lysis": lysis},
            "topology": {
                "lysis_initiation": TOPOLOGY,
                "lysis": {
                    "trigger": ("lysis_trigger",),
                    "agents": ("agents",),
                    "internal": ("bulk",),
                    "fields": ("fields",),
                    "location": ("location",),
                    "dimensions": {
                        "bounds": ("dimensions", "bounds"),
                        "n_bins": ("dimensions", "n_bins"),
                        "depth": ("dimensions", "depth"),
                    },
                },
            },
        }
    )

    sim = Engine(composite=composite, initial_state={"wall_state": {"cracked": True}})
    sim.update(100)
    data = sim.emitter.get_data()

    # Validate data
    # expect lysis to have occurred
    print(data)


def main():
    test_lysis_initiation()


if __name__ == "__main__":
    main()

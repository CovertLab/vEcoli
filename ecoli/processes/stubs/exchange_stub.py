"""
=============
Exchange Stub
=============

Exchanges molecules at pre-set rates through a single port
"""

import numpy as np
from vivarium.core.process import Process
from vivarium.core.composition import simulate_process

from ecoli.library.schema import numpy_schema, bulk_name_to_idx


class Exchange(Process):
    name = "ecoli-exchange"
    defaults: dict[str, dict] = {"exchanges": {}}

    def __init__(self, parameters=None):
        """Exchange Stub Process

        Ports:
        * **molecules**: reads the current state of all molecules to be exchanged

        Arguments:
            parameters (dict): Accepts the following configuration keys:
                * **exchanges** (:py:class:`dict`): a dictionary with molecule ids
                mapped to the exchange rate, in counts/second.
        """
        super().__init__(parameters)
        self.exchange_mol = np.array(list(self.parameters["exchanges"].keys()))
        self.exchange_rate = np.array(list(self.parameters["exchanges"].values()))
        self.exchange_mol_idx = None

    def ports_schema(self):
        return {"bulk": numpy_schema("bulk")}

    def next_update(self, timestep, states):
        if self.exchange_mol_idx is None:
            self.exchange_mol_idx = bulk_name_to_idx(
                self.exchange_mol, states["bulk"]["id"]
            )
        exchange = self.exchange_rate * timestep
        return {"bulk": [(self.exchange_mol_idx, exchange)]}


def test_exchanger():
    parameters = {
        "exchanges": {
            "A": -1.0,
        }
    }
    process = Exchange(parameters)

    # declare the initial state
    initial_state = {
        "bulk": np.array([("A", 10.0)], dtype=[("id", "U40"), ("count", float)])
    }

    # run the simulation
    sim_settings = {"total_time": 10, "initial_state": initial_state}
    output = simulate_process(process, sim_settings)
    print(output)


if __name__ == "__main__":
    test_exchanger()

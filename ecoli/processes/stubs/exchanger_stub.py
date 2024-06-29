"""Exchanger Stub Process

Exchanges molecules at pre-set rates through a single port
"""

from typing import Any

from vivarium.core.process import Process
from vivarium.core.composition import simulate_process


class Exchanger(Process):
    defaults: dict[str, Any] = {"exchanges": {}}

    def __init__(self, parameters=None):
        """Exchanger Stub Process

        Ports:
        * **molecules**: reads the current state of all molecules to be exchanged

        Arguments:
            parameters (dict): Accepts the following configuration keys:
                * **exchanges** (:py:class:`dict`): a dictionary with molecule ids
                mapped to the exchange rate, in counts/second.
        """
        super().__init__(parameters)

    def ports_schema(self):
        return {
            "molecules": {
                mol_id: {"_default": 0}
                for mol_id in self.parameters["exchanges"].keys()
            }
        }

    def next_update(self, timestep, states):
        exchange = {
            mol_id: rate * timestep
            for mol_id, rate in self.parameters["exchanges"].items()
        }
        return {"molecules": exchange}


def test_exchanger():
    parameters = {
        "exchanges": {
            "A": -1.0,
        },
        # override emit
        "_schema": {"molecules": {"A": {"_emit": True}}},
    }
    process = Exchanger(parameters)

    # declare the initial state
    initial_state = {"molecules": {"A": 10.0}}

    # run the simulation
    sim_settings = {"total_time": 10, "initial_state": initial_state}
    output = simulate_process(process, sim_settings)
    print(output)


if __name__ == "__main__":
    test_exchanger()

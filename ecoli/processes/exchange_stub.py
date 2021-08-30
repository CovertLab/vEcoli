"""Exchange Stub Process

Exchanges molecules at pre-set rates through a single port
"""

from vivarium.core.process import Process
from vivarium.core.composition import simulate_process
import numpy as np



class Exchange(Process):
    defaults = {'exchanges': {}}

    def __init__(self, parameters=None):
        """ Exchange Stub Process

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
            'molecules': {
                mol_id: {'_default': 0, '_updater': 'accumulate'}
                for mol_id in self.parameters['exchanges'].keys()
            },
            'export': {
                mol_id: {'_default': 0, '_updater': 'set', '_emit': True}
                for mol_id in self.parameters['exchanges'].keys()
            },
            'listeners': {'enzyme_kinetics': {'countsToMolar': {'_default': 1.0, '_updater': 'set'}}}
        }

    def next_update(self, timestep, states):
        countsToMolar = states['listeners']['enzyme_kinetics']['countsToMolar']
        export = {
            mol_id: (rate * timestep * np.random.uniform(0.95, 1.05)) / countsToMolar
            for mol_id, rate in self.parameters['exchanges'].items()}
        return {'export': export, 'molecules': export} # in mmol/L


def test_exchanger():
    parameters = {
        'exchanges': {
            'A': -1.0,
        },
        # override emit
        '_schema': {
            'molecules': {
                'A': {'_emit': True}
            }}}
    process = Exchange(parameters)

    # declare the initial state
    initial_state = {
        'molecules': {
            'A': 10.0
        }}

    # run the simulation
    sim_settings = {
        'total_time': 10,
        'initial_state': initial_state}
    output = simulate_process(process, sim_settings)
    print(output)


if __name__ == '__main__':
    test_exchanger()

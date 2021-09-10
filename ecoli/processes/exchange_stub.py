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
        self.first = True

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
            'listeners': {
                'enzyme_kinetics': {
                    'countsToMolar': {'_default': 1.0, '_updater': 'set'}
                },
                'mass': {
                    'cell_mass': {'_default': 1.0},
                    'dry_mass': {'_default': 1.0}
                }
            }
        }

    def next_update(self, timestep, states):
        countsToMolar = states['listeners']['enzyme_kinetics']['countsToMolar']
        mass = states['listeners']['mass']['cell_mass']

        if self.first:
            self.first_mass = mass
            self.first = False

        mass_increase = mass / self.first_mass
        export = {
            mol_id: (rate * mass_increase * timestep * np.random.uniform(0.98, 1.02)) / countsToMolar
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

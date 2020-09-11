from __future__ import absolute_import, division, print_function

import numpy as np
from arrow import StochasticSystem

from vivarium.core.process import Process
from vivarium.core.composition import simulate_process_in_experiment

# Maximum unsigned int value + 1 for randint() to seed srand from C stdlib
RAND_MAX = 2**31

class Complexation(Process):
    name = "complexation-ecoli"

    defaults = {
        'stoichiometry': np.array([[]]),
        'rates': np.array([]),
        'molecule_names': [],
        'seed': 0}

    def __init__(self, initial_parameters=None):
        if not initial_parameters:
            initial_parameters = {}

        super(Complexation, self).__init__(initial_parameters)

        self.stoichiometry = self.parameters['stoichiometry']
        self.rates = self.parameters['rates']
        self.molecule_names = self.parameters['molecule_names']
        self.seed = self.parameters['seed']

        self.system = StochasticSystem(self.stoichiometry, random_seed=self.seed)

    def ports_schema(self):
        return {
            'molecules': {
                molecule: {
                    '_default': 0,
                    '_emit': True}
                for molecule in self.molecule_names}}

    def next_update(self, timestep, states):
        molecules = states['molecules']

        substrate = np.zeros(len(molecules), dtype=np.int64)
        for index, molecule in enumerate(self.molecule_names):
            substrate[index] = molecules[molecule]

        result = self.system.evolve(timestep, substrate, self.rates)
        outcome = result['outcome'] - substrate

        molecules_update = {
            molecule: outcome[index]
            for index, molecule in enumerate(self.molecule_names)}

        update = {
            'molecules': molecules_update}

        # # Write outputs to listeners
        # self.writeToListener("ComplexationListener", "complexationEvents", events)

        return update


def test_complexation():
    test_config = {
        'stoichiometry': np.array([
            [-1, 1, 0],
            [0, -1, 1],
            [1, 0, -1],
            [-1, 0, 1],
            [1, -1, 0],
            [0, 1, -1]], np.int64),
        'rates': np.array([1, 1, 1, 1, 1, 1], np.float64),
        'molecule_names': ['A', 'B', 'C'],
        'seed': 1}

    complexation = Complexation(test_config)

    state = {
        'molecules': {
            'A': 10,
            'B': 20,
            'C': 30}}

    settings = {
        'total_time': 10,
        'initial_state': state}

    data = simulate_process_in_experiment(complexation, settings)

    print(data)


if __name__ == "__main__":
    test_complexation()

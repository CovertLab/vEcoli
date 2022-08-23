"""
============
Complexation
============

This process encodes molecular simulation of macromolecular complexation,
in which monomers are assembled into complexes. Macromolecular complexation
is done by identifying complexation reactions that are possible (which are
reactions that have sufﬁcient counts of all sub-components), performing one
randomly chosen possible reaction, and re-identifying all possible complexation
reactions. This process assumes that macromolecular complexes form spontaneously,
and that complexation reactions are fast and complete within the time step of the
simulation.
"""

# TODO(wcEcoli):
# - allow for shuffling when appropriate (maybe in another process)
# - handle protein complex dissociation

import numpy as np
from arrow import StochasticSystem

from vivarium.core.composition import simulate_process

from ecoli.library.schema import array_to, bulk_schema
from ecoli.processes.registries import topology_registry
from ecoli.processes.partition import PartitionedProcess

# Register default topology for this process, associating it with process name
NAME = 'ecoli-complexation'
TOPOLOGY = {
    "molecules": ("bulk",),
    "listeners": ("listeners",)
}
topology_registry.register(NAME, TOPOLOGY)


class Complexation(PartitionedProcess):
    """ Complexation PartitionedProcess """

    name = NAME
    topology = TOPOLOGY
    defaults = {
        'stoichiometry': np.array([[]]),
        'rates': np.array([]),
        'molecule_names': [],
        'seed': 0,
        'numReactions': 0,
    }

    def __init__(self, parameters=None):
        super().__init__(parameters)

        self.stoichiometry = self.parameters['stoichiometry']
        self.rates = self.parameters['rates']
        self.molecule_names = self.parameters['molecule_names']
        self.seed = self.parameters['seed']

        self.system = StochasticSystem(self.stoichiometry, random_seed=self.seed)

        num_reactions = self.parameters['numReactions']
        self.complexationEvents = np.zeros(num_reactions, np.int64)

    def ports_schema(self):
        return {
            'molecules': bulk_schema(self.molecule_names),
            'listeners': {
                'complexation_events': {
                        '_default': [],
                        '_updater': 'set',
                        '_emit': True},
            },
        }

    def calculate_request(self, timestep, states):
        # The int64 dtype is important (can break otherwise)
        moleculeCounts = np.array(list(states['molecules'].values()),
                                  dtype = np.int64)

        result = self.system.evolve(
            timestep, moleculeCounts, self.rates)
        updatedMoleculeCounts = result['outcome']
        requests = {}
        requests['molecules'] = array_to(states['molecules'], np.fmax(
            moleculeCounts - updatedMoleculeCounts, 0))
        return requests

    def evolve_state(self, timestep, states):
        molecules = states['molecules']

        substrate = np.zeros(len(molecules), dtype=np.int64)
        for index, molecule in enumerate(self.molecule_names):
            substrate[index] = molecules[molecule]

        result = self.system.evolve(timestep, substrate, self.rates)
        self.complexationEvents = result['occurrences']
        outcome = result['outcome'] - substrate
        molecules_update = array_to(self.molecule_names, outcome)

        # Write outputs to listeners
        update = {
            'molecules': molecules_update,
            'listeners': {
                'complexation_events': self.complexationEvents
            }
        }

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

    data = simulate_process(complexation, settings)
    assert (type(data['listeners']['complexation_events'][0]) == list)
    assert (type(data['listeners']['complexation_events'][1]) == list)
    print(data)


if __name__ == "__main__":
    test_complexation()

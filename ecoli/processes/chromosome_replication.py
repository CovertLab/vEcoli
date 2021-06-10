"""
Complexation

Macromolecular complexation sub-model. Encodes molecular simulation of macromolecular complexation

TODO:
- allow for shuffling when appropriate (maybe in another process)
- handle protein complex dissociation
"""

import numpy as np
from arrow import StochasticSystem

from wholecell.utils import units

from vivarium.core.process import Process
from vivarium.core.composition import simulate_process



class ChromosomeReplication(Process):
    name = 'ecoli-chromosome_replication'

    defaults = {
        'max_time_step': 2.0,
        'get_dna_critical_mass': lambda doubling_time: units.Unum,
        'criticalInitiationMass': 975 * units.fg,
        'nutrientToDoublingTime': {},
        'replichore_lengths': np.array([]),
        'sequences': np.array([]),
        'polymerized_dntp_weights': [],
        'replication_coordinate': np.array([]),
        'D_period': np.array([]),
        'no_child_place_holder': -1,
        'basal_elongation_rate': 967,
        'make_elongation_rates': lambda random, replisomes, base, time_step: units.Unum,
        'mechanistic_replisome': True,
    }

    def __init__(self, parameters=None):
        super().__init__(parameters)

        self.max_time_step = self.parameters['max_time_step']

        # Load parameters
        self.get_dna_critical_mass = self.parameters['get_dna_critical_mass']
        self.criticalInitiationMass = self.parameters['criticalInitiationMass']
        self.nutrientToDoublingTime = self.parameters['nutrientToDoublingTime']
        self.replichore_lengths = self.parameters['replichore_lengths']
        self.sequences = self.parameters['sequences']
        self.polymerized_dntp_weights = self.parameters['polymerized_dntp_weights']
        self.replication_coordinate = self.parameters['replication_coordinate']
        self.D_period = self.parameters['D_period']
        self.no_child_place_holder = self.parameters['no_child_place_holder']
        self.basal_elongation_rate = self.parameters['basal_elongation_rate']
        self.make_elongation_rates = self.parameters['make_elongation_rates']

        # Sim options
        self.mechanistic_replisome = self.parameters['mechanistic_replisome']


        import ipdb; ipdb.set_trace()

        # # Create molecule views for replisome subunits, active replisomes,
        # # origins of replication, chromosome domains, and free active TFs
        # self.replisome_trimers = self.bulkMoleculesView(
        #     sim_data.molecule_groups.replisome_trimer_subunits)
        # self.replisome_monomers = self.bulkMoleculesView(
        #     sim_data.molecule_groups.replisome_monomer_subunits)
        # self.active_replisomes = self.uniqueMoleculesView('active_replisome')
        # self.oriCs = self.uniqueMoleculesView('oriC')
        # self.chromosome_domains = self.uniqueMoleculesView('chromosome_domain')
        #
        # # Create bulk molecule views for polymerization reaction
        # self.dntps = self.bulkMoleculesView(sim_data.molecule_groups.dntps)
        # self.ppi = self.bulkMoleculeView(sim_data.molecule_ids.ppi)
        #
        # # Create molecules views for full chromosomes
        # self.full_chromosomes = self.uniqueMoleculesView('full_chromosome')





    def ports_schema(self):
        return {}

    def next_update(self, timestep, states):

        update = {}

        return update


def test_chromosome_replication():
    test_config = {}

    process = ChromosomeReplication(test_config)

    initial_state = {}

    settings = {
        'total_time': 10,
        'initial_state': initial_state}

    data = simulate_process(process, settings)

    print(data)


if __name__ == "__main__":
    test_chromosome_replication()

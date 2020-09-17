from six.moves import cPickle
import numpy as np

from vivarium.core.process import Generator
from vivarium.core.composition import simulate_compartment_in_experiment

from ecoli.processes.complexation import Complexation
from ecoli.processes.protein_degradation import ProteinDegradation

from wholecell.utils import units


RAND_MAX = 2**31

class Ecoli(Generator):

    defaults = {
        'seed': 0,
        'sim_data_path': '../wcEcoli/out/manual/kb/simData.cPickle'}

    def __init__(self, config):
        super(Ecoli, self).__init__(config)

        self.seed = np.uint32(self.config['seed'] % np.iinfo(np.uint32).max)
        self.random_state = np.random.RandomState(seed = self.seed)

    def initialize_complexation(self, sim_data):
        complexation_config = {
            'stoichiometry': sim_data.process.complexation.stoichMatrix().astype(np.int64).T,
            'rates': sim_data.process.complexation.rates,
            'molecule_names': sim_data.process.complexation.moleculeNames,
            'seed': self.random_state.randint(RAND_MAX)}

        complexation = Complexation(complexation_config)
        return complexation

    def initialize_protein_degradation(self, sim_data):
        protein_degradation_config = {
            'raw_degradation_rate': sim_data.process.translation.monomerData['degRate'].asNumber(1 / units.s),
            'shuffle_indexes': sim_data.process.translation.monomerDegRateShuffleIdxs if hasattr(
                sim_data.process.translation, "monomerDegRateShuffleIdxs") else None,
            'water_id': sim_data.moleculeIds.water,
            'amino_acid_ids': sim_data.moleculeGroups.amino_acids,
            'amino_acid_counts': sim_data.process.translation.monomerData["aaCounts"].asNumber(),
            'protein_ids': sim_data.process.translation.monomerData['id'],
            'protein_lengths': sim_data.process.translation.monomerData['length'].asNumber(),
            'seed': self.random_state.randint(RAND_MAX)}

        protein_degradation = ProteinDegradation(protein_degradation_config)
        return protein_degradation

    def generate_processes(self, config):
        sim_data_path = config['sim_data_path']
        with open(sim_data_path, 'rb') as sim_data_file:
            sim_data = cPickle.load(sim_data_file)

        complexation = self.initialize_complexation(sim_data)
        protein_degradation = self.initialize_protein_degradation(sim_data)

        return {
            'complexation': complexation,
            'protein_degradation': protein_degradation}

    def generate_topology(self, config):
        return {
            'complexation': {
                'molecules': ('bulk',)},
            'protein_degradation': {
                'metabolites': ('bulk',),
                'proteins': ('bulk',)}}


def test_ecoli():
    ecoli = Ecoli({})

    initial_state = {
        'bulk': {}}

    settings = {
        'timestep': 1,
        'total_time': 10,
        'initial_state': initial_state}

    data = simulate_compartment_in_experiment(ecoli, settings)

    print(data)


if __name__ == '__main__':
    test_ecoli()

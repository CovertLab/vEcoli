import numpy as np

from vivarium.core.process import Generator

from ecoli.processes.complexation import Complexation

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

    def generate_processes(self, config):
        sim_data_path = config['sim_data_path']
        sim_data = cPickle.load(sim_data_path)

        complexation = self.initialize_complexation(sim_data)

    def generate_topology(self, config):
        return {
            'complexation': {
                'molecules': ('bulk',)}}


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

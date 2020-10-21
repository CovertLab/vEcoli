
import numpy as np

from six.moves import cPickle

from vivarium.core.process import Generator
from vivarium.core.composition import simulate_compartment_in_experiment
from vivarium.core.experiment import pp

# processes
from vivarium_cell.processes.growth_rate import GrowthRate
# from vivarium_cell.processes.diffusion_network import DiffusionNetwork

from ecoli.composites.ecoli_master import get_initial_state

SIM_DATA_PATH = '../wcEcoli/out/underscore/kb/simData.cPickle'


class EcoliSpatial(Generator):

    defaults = {
        'time_step': 2.0,
        'seed': 0,
        'sim_data_path': SIM_DATA_PATH,
        'diffusion_network': {},
    }

    def __init__(self, config):
        super(EcoliSpatial, self).__init__(config)
        self.seed = np.uint32(self.config['seed'] % np.iinfo(np.uint32).max)
        self.random_state = np.random.RandomState(seed = self.seed)
        self.sim_data_path = self.config['sim_data_path']

    def initial_state(self, config=None):

        # load sim_data
        with open(self.sim_data_path, 'rb') as sim_data_file:
            sim_data = cPickle.load(sim_data_file)

        initial_state = get_initial_state()
        # do string parsing of state by compartment and distribute state to spatial nodes

        return initial_state

    def generate_processes(self, config):

        return {
            # 'diffusion_network': DiffusionNetwork(config['diffusion_network'])
            'growth_rate': GrowthRate({})
        }

    def generate_topology(self, config):

        return {
            # 'diffusion_network': {},
            'growth_rate': {
                'molecules': ('molecules',),
                'global': ('boundary',)
            },
        }



def test_spatial_ecoli():
    ecoli = EcoliSpatial({})

    settings = {
        'total_time': 10,
        'initial_state': ecoli.initial_state()}

    data = simulate_compartment_in_experiment(ecoli, settings)

    # TODO -- add assert here for the test to check

    return data


def run_spatial_ecoli():
    output = test_spatial_ecoli()

    pp(output['molecules'])


if __name__ == '__main__':
    run_spatial_ecoli()

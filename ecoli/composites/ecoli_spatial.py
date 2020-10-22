
import numpy as np
import math
from wholecell.utils import units
from scipy import constants

from six.moves import cPickle

from vivarium.core.process import Generator
from vivarium.core.composition import simulate_compartment_in_experiment
from vivarium.core.experiment import pp

# processes
# from vivarium_cell.processes.growth_rate import GrowthRate
from vivarium_cell.processes.diffusion_network import DiffusionNetwork

from ecoli.composites.ecoli_master import get_initial_state

#TODO: change this before merging
SIM_DATA_PATH = '../wcEcoli/out/for_vivarium/kb/simData.cPickle'


class EcoliSpatial(Generator):

    defaults = {
        'time_step': 2.0,
        'seed': 0,
        'sim_data_path': SIM_DATA_PATH,
        'nodes': {
        },
        'edges': {
        },
        'mesh_size': 0,
    }

    def __init__(self, config):
        super(EcoliSpatial, self).__init__(config)
        self.seed = np.uint32(self.config['seed'] % np.iinfo(np.uint32).max)
        self.random_state = np.random.RandomState(seed = self.seed)
        self.sim_data_path = self.config['sim_data_path']
        self.nodes = self.config['nodes']
        self.edges = self.config['edges']
        self.mesh_size = self.config['mesh_size']
        self.time_step = self.config['time_step']

        # load sim_data
        with open(self.sim_data_path, 'rb') as sim_data_file:
            sim_data = cPickle.load(sim_data_file)

        bulk_ids = sim_data.internal_state.bulk_molecules.bulk_data['id']

        # molecule weight is converted to femtograms/mol
        self.molecular_weights = {
            molecule_id: (sim_data.getter.get_mass([molecule_id]) / constants.N_A).asNumber(
                units.fg / units.mol)[0]
            for molecule_id in bulk_ids}

    def initial_state(self, config=None):

        initial_state = get_initial_state()
        bulk = initial_state['bulk']

        # TODO: Make this not hardcoded
        cytosol_front = {
            mol_id: (math.ceil(value / 2) if '[c]' in mol_id else 0)
            for mol_id, value in bulk.items()
        }
        cytosol_rear = {
            mol_id: (math.floor(value / 2) if '[c]' in mol_id else 0)
            for mol_id, value in bulk.items()
        }
        # this is currently all 0s
        nucleoid = {
            mol_id: (value if '[n]' in mol_id else 0)
            for mol_id, value in bulk.items()
        }
        self.nodes['cytosol_front']['molecules'] = cytosol_front
        self.nodes['nucleoid']['molecules'] = nucleoid
        self.nodes['cytosol_rear']['molecules'] = cytosol_rear

        initial_spatial_state = self.nodes
        return initial_spatial_state

    def generate_processes(self, config):
        diffusion_config = {
            # this is the initial parameters
            'nodes': list(self.nodes.keys()),
            'edges': self.edges,
            'mw': self.molecular_weights,
            'mesh_size': self.mesh_size,
            'time_step': self.time_step,
        }
        return {
            'diffusion_network': DiffusionNetwork(diffusion_config)
        }

    def generate_topology(self, config):
        # connect ports
        return {
            'diffusion_network': {
                node: (node,) for node in config['nodes']
            },
        }



def test_spatial_ecoli():
    ecoli_config = {
        'nodes': {
            'cytosol_front': {
                'length': 0.75,
                'volume': 0.3,
                'molecules': {}
            },
            'nucleoid': {
                'length': 0.75,
                'volume': 0.3,
                'molecules': {}
            },
            'cytosol_rear': {
                'length': 0.75,
                'volume': 0.3,
                'molecules': {}
            },
        },
        'edges': {
            '1': {
                'nodes': ['cytosol_front', 'nucleoid'],
            },
            '2': {
                'nodes': ['nucleoid', 'cytosol_rear'],
            },
        },
        'mesh_size': 50,
        'time_step': 1,
    }
    ecoli = EcoliSpatial(ecoli_config)

    settings = {
        'total_time': 10,
        'initial_state': ecoli.initial_state()}

    data = simulate_compartment_in_experiment(ecoli, settings)

    # TODO -- add assert here for the test to check

    return data


def run_spatial_ecoli():
    output = test_spatial_ecoli()

    # import ipdb; ipdb.set_trace()

if __name__ == '__main__':
    run_spatial_ecoli()

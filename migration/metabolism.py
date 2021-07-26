"""
Metabolism process migration tests
"""
import argparse

# vivarium imports
from vivarium.core.engine import Engine
from vivarium.core.composer import Composer
from vivarium.library.dict_utils import deep_merge

# ecoli imports
from ecoli.library.sim_data import LoadSimData
from ecoli.states.wcecoli_state import get_state_from_file
from ecoli.composites.ecoli_master import SIM_DATA_PATH, ECOLI_TOPOLOGY, AA_MEDIA_ID
from ecoli.processes import Metabolism, Exchange

# migration imports
from migration.migration_utils import run_ecoli_process


# load sim_data
load_sim_data = LoadSimData(
            sim_data_path=SIM_DATA_PATH,
            seed=0)

# get topology from ecoli master
metabolism_topology = ECOLI_TOPOLOGY['metabolism']



# make a composite with Exchange
class MetabolismExchange(Composer):
    defaults = {
        'metabolism': {},
        'exchanges': {},  # dict with {molecule: exchange rate}
        'sim_data_path': SIM_DATA_PATH,
        'seed': 0,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.load_sim_data = LoadSimData(
            sim_data_path=self.config['sim_data_path'],
            seed=self.config['seed'])

    def generate_processes(self, config):

        # configure metabolism
        metabolism_config = self.load_sim_data.get_metabolism_gd_config()
        metabolism_config = deep_merge(metabolism_config, self.config['metabolism'])
        metabolism_process = Metabolism(metabolism_config)

        # configure exchanger stub process
        exchanger_process = Exchange({'exchanges': config['exchanges']})

        return {
            'metabolism': metabolism_process,
            'exchange': exchanger_process,
        }

    def generate_topology(self, config):
        return {
            'metabolism': metabolism_topology,
            'exchange': {
                'molecules': ('bulk',),
            }
        }



def test_metabolism_migration():
    # Create process, experiment, loading in initial state from file.
    config = load_sim_data.get_metabolism_config()
    metabolism_process = Metabolism(config)

    # run the process and get an update
    actual_update = run_ecoli_process(
        metabolism_process,
        metabolism_topology,
        total_time=2)


def run_metabolism(
        total_time=10,
        initial_time=0,
        config=None,
        initial_state=None,
):
    # get parameters from sim data
    metabolism_config = load_sim_data.get_metabolism_config()
    if config:
        metabolism_config = deep_merge(metabolism_config, config)

    # get initial state from file
    state = get_state_from_file(
        path=f'data/wcecoli_t{initial_time}.json')
    if initial_state:
        state = deep_merge(state, initial_state)

    #initialize Metabolism
    metabolism = Metabolism(metabolism_config)

    # initialize a simulation
    metabolism_composite = metabolism.generate()
    simulation = Engine({
        'processes': metabolism_composite['processes'],
        'topology': {metabolism.name: metabolism_topology},
        'initial_state': state
    })

    # run the simulation
    simulation.update(total_time)

    # get data
    data = simulation.emitter.get_data()
    return data


def test_metabolism():
    data = run_metabolism(total_time=10)


def test_metabolism_aas():
    config = {
        'media_id': AA_MEDIA_ID
    }
    initial_state = {
        'environment': {
            'media_id': AA_MEDIA_ID
        }
    }
    data = run_metabolism(
        total_time=10,
        config=config,
        initial_state=initial_state,
    )



def run_metabolism_composite():
    # configure exchange stub process with molecules to exchange
    config = {
        'exchanges': {
            'ARG[c]': 0.0,
            'ASN[c]': 0.0,
            'CYS[c]': 0.0,
            'GLN[c]': 0.0,
            'GLT[c]': 0.0,
            'GLY[c]': 0.0,
            'HIS[c]': 0.0,
            'ILE[c]': 0.0,
            'L-ALPHA-ALANINE[c]': 0.0,
            'L-ASPARTATE[c]': 0.0,
            'L-SELENOCYSTEINE[c]': 0.0,
            'LEU[c]': 0.0,
            'LYS[c]': 0.0,
            'MET[c]': 0.0,
            'PHE[c]': 0.0,
            'PRO[c]': 0.0,
            'SER[c]': 0.0,
            'THR[c]': 0.0,
            'TRP[c]': 0.0,
            'TYR[c]': 0.0,
            'VAL[c]': 0.0,
        }
    }

    composer = MetabolismExchange(config)
    metabolism_composite = composer.generate()

    # get initial state
    initial_state = get_state_from_file(
        path=f'data/wcecoli_t1000.json')

    # run a simulation
    experiment = Engine({
        'processes': metabolism_composite['processes'],
        'topology': metabolism_composite['topology'],
        'initial_state': initial_state})
    experiment.update(10)
    data = experiment.emitter.get_data()



# functions to run from the command line
test_library = {
    '0': test_metabolism_migration,
    '1': test_metabolism,
    '2': test_metabolism_aas,
    '3': run_metabolism_composite,
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='metabolism process migration')
    parser.add_argument(
        '--name', '-n', default=[], nargs='+', help='test ids to run')
    args = parser.parse_args()
    run_all = not args.name

    for name in args.name:
        test_library[name]()
    if run_all:
        for name, test in test_library.items():
            test()

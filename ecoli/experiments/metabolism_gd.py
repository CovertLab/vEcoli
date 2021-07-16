"""
===========================================
Metabolism using Gradient Descent-based FBA
===========================================
"""
import argparse

# vivarium-core imports
from vivarium.core.engine import Engine
from vivarium.core.composer import Composer
from vivarium.library.dict_utils import deep_merge

# vivarium-ecoli imports
from ecoli.library.sim_data import LoadSimData
from ecoli.states.wcecoli_state import get_state_from_file
from ecoli.composites.ecoli_master import SIM_DATA_PATH, ECOLI_TOPOLOGY
from ecoli.processes.metabolism_gd import MetabolismGD
from ecoli.processes import Exchanger


# get topology from ecoli_master
metabolism_topology = ECOLI_TOPOLOGY['metabolism']


# make a composite with Exchange
class MetabolismExchange(Composer):
    defaults = {
        'metabolism': {},
        'exchanger': {},
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
        metabolism_process = MetabolismGD(metabolism_config)

        # configure exchanger stub process
        # TODO -- this needs a dictionary with {mol_id: exchanged counts/sec}
        exchanger_config = {'exchanges': {}}
        exchanger_process = Exchanger(exchanger_config)

        return {
            'metabolism': metabolism_process,
            'exchanger': exchanger_process,
        }

    def generate_topology(self, config):
        return {
            'metabolism': metabolism_topology,
            'exchanger': {
                'molecules': ('bulk',),
            }
        }


def run_metabolism():
    # load the sim data
    load_sim_data = LoadSimData(
        sim_data_path=SIM_DATA_PATH,
        seed=0)

    # Create process, experiment, loading in initial state from file.
    config = load_sim_data.get_metabolism_gd_config()
    metabolism_process = MetabolismGD(config)

    initial_state = get_state_from_file(
        path=f'data/wcecoli_t1000.json')

    # TODO -- add perturbations to initial_state to test impact on metabolism

    metabolism_composite = metabolism_process.generate()
    experiment = Engine({
        'processes': metabolism_composite['processes'],
        'topology': {metabolism_process.name: metabolism_topology},
        'initial_state': initial_state
    })

    experiment.update(10)

    data = experiment.emitter.get_data()

    import ipdb; ipdb.set_trace()


def run_metabolism_composite():
    composer = MetabolismExchange()
    metabolism_composite = composer.generate()

    initial_state = get_state_from_file(
        path=f'data/wcecoli_t1000.json')


    experiment = Engine({
        'processes': metabolism_composite['processes'],
        'topology': metabolism_composite['topology'],
        'initial_state': initial_state
    })

    experiment.update(10)

    data = experiment.emitter.get_data()




experiment_library = {
    '0': run_metabolism,
    '1': run_metabolism_composite,
}


# run experiments with command line arguments: python ecoli/experiments/metabolism_gd.py -n exp_id
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='metabolism with gd')
    parser.add_argument('--name', '-n', default=[], nargs='+', help='test ids to run')
    args = parser.parse_args()
    run_all = not args.name

    for name in args.name:
        experiment_library[name]()
    if run_all:
        for name, test in experiment_library.items():
            test()

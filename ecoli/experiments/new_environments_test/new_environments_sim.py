"""
==================================
Metabolism using user-friendly FBA
==================================
"""
import argparse

# vivarium-core imports
import pytest
from vivarium.core.engine import Engine
from vivarium.core.composer import Composer
from vivarium.library.dict_utils import deep_merge

# vivarium-ecoli imports
from ecoli.experiments.ecoli_master_sim import EcoliSim, CONFIG_DIR_PATH
from ecoli.library.sim_data import LoadSimData
from ecoli.states.wcecoli_state import get_state_from_file
from ecoli.composites.ecoli_master import SIM_DATA_PATH
from ecoli.processes.metabolism_redux import MetabolismRedux
from ecoli.processes.stubs.exchange_stub import Exchange
from ecoli.processes.registries import topology_registry

import numpy as np
import pathlib, datetime
import dill
import pandas as pd

# get topology from ecoli_master
metabolism_topology = topology_registry.access('ecoli-metabolism')


# make a composite with Exchange
class MetabolismExchange(Composer):
    defaults = {
        'metabolism': {
            'kinetic_rates': [],
        },
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
        metabolism_config = self.load_sim_data.get_metabolism_redux_config()
        metabolism_config = deep_merge(metabolism_config, config['metabolism'])
        metabolism_process = MetabolismRedux(metabolism_config)

        example_update = {'ATP[c]': 9064, 'DATP[c]': 2222, 'DCTP[c]': 1649, 'DGTP[c]': 1647, 'FAD[c]': 171,
                          'GTP[c]': 20122, 'LEU[c]': 325, 'METHYLENE-THF[c]': 223, 'NAD[c]': 769,
                          'PHENYL-PYRUVATE[c]': 996, 'REDUCED-MENAQUINONE[c]': 240, 'UTP[c]': 14648}

        # configure exchanger stub process
        # TODO -- this needs a dictionary with {mol_id: exchanged counts/sec}
        exchanger_config = {'exchanges': example_update, 'time_step': metabolism_config['time_step']}
        exchanger_process = Exchange(exchanger_config)

        return {
            'metabolism': metabolism_process,
            'exchanger': exchanger_process,
        }

    def generate_topology(self, config):
        return {
            'metabolism': metabolism_topology,
            'exchanger': {
                'bulk': ('bulk',),
            }
        }

def run_ecoli_with_metabolism_redux(
        filename='fba_new_environments',
        total_time=50,
        divide=True,
        initial_state_file='wcecoli_t0',
        progress_bar=True,
        log_updates=False,
        emitter='timeseries',
        name='fba_new_environments',
        raw_output=False,
        save=False,        # save_times=4,
):
    condition_df = pd.read_excel("ecoli/experiments/new_environments_test/new_env.xlsx")
    condition_df["Add"] = condition_df["Add"].apply(lambda x: set(str(x).split(", ") if x == x else set()))
    condition_df["Remove"] = condition_df["Remove"].apply(lambda x: set(str(x).split(", ") if x == x else set()))
    condition_df["Demand"] = condition_df["Demand"].apply(lambda x: set(str(x).split(", ") if x == x else set()))

    for i in range(len(condition_df)):
        env_name = f'{condition_df.loc[i, "Environment"]}'
        if env_name[:4] == 'SKIP':
            continue
        # filename = 'default'
        sim = EcoliSim.from_file(CONFIG_DIR_PATH + filename + '.json')
        sim.total_time = total_time
        sim.divide = divide
        sim.progress_bar = progress_bar
        sim.log_updates = log_updates
        sim.emitter = emitter
        sim.initial_state = get_state_from_file(path=f'data/{initial_state_file}.json')
        sim.raw_output = raw_output
        sim.save = save
        # sim.save_times = [4]
        to_add = condition_df.loc[i, "Add"]
        to_remove = condition_df.loc[i, "Remove"]

        sim.initial_state['environment']['exchange_data']['constrained'] = {}
        sim.initial_state['environment']['exchange_data']['unconstrained'].add('GLC[p]')
        sim.initial_state['environment']['exchange_data']['unconstrained'].add('AMMONIUM[p]')
        sim.initial_state['environment']['exchange_data']['unconstrained'].remove('AMMONIUM[c]')

        for mol in to_add:
            sim.initial_state['environment']['exchange_data']['unconstrained'].add(mol)

        for mol in to_remove:
            sim.initial_state['environment']['exchange_data']['unconstrained'].remove(mol)

        try:
            sim.build_ecoli()
            sim.run()

            query = []
            folder = f'out/fba_new_env/{name}_{env_name}_{total_time}_{datetime.date.today()}/'
            save_sim_output(folder, query, sim, save_model=True)
        except ValueError:
            pass
@pytest.mark.slow
def test_ecoli_with_metabolism_redux(
        filename='fba_redux',
        total_time=4,
        divide=False,
        progress_bar=True,
        log_updates=False,
        emitter='timeseries',
):
    sim = EcoliSim.from_file(CONFIG_DIR_PATH + filename + '.json')
    sim.total_time = total_time
    sim.divide = divide
    sim.progress_bar = progress_bar
    sim.log_updates = log_updates
    sim.emitter = emitter
    sim.build_ecoli()

    # run simulation and add asserts to output
    sim.run()

    # assert 'ecoli-metabolism-gradient-descent' in sim.ecoli['processes']
    # assert 'ecoli-metabolism' not in sim.ecoli['processes']
    # assert 'ecoli-metabolism-gradient-descent' in sim.ecoli['topology']
    # assert 'ecoli-metabolism' not in sim.ecoli['topology']

experiment_library = {
    '0': run_ecoli_with_metabolism_redux,
    '1': test_ecoli_with_metabolism_redux,
}


def save_sim_output(folder, query, sim, save_model=False):
    agents = sim.query()['agents'].keys()
    for agent in agents:
        query.extend([('agents', agent, 'listeners', 'fba_results'),
                      ('agents', agent, 'listeners', 'mass'),
                      ('agents', agent, 'bulk')])
    output = sim.query(query)
    pathlib.Path(folder).mkdir(parents=True, exist_ok=True)
    np.save(folder + 'output.npy', output)

    if save_model:
        f = open(folder + 'agent_processes.pkl', 'wb')
        dill.dump(sim.ecoli['processes']['agents'][agent], f)
        f.close()

        f = open(folder + 'agent_steps.pkl', 'wb')
        dill.dump(sim.ecoli['steps']['agents'][agent], f)
        f.close()

# run experiments with command line arguments: python ecoli/experiments/metabolism_redux_sim.py -n exp_id
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='user-friendly metabolism')
    parser.add_argument('--name', '-n', default=[], nargs='+', help='test ids to run')
    args = parser.parse_args()
    run_all = not args.name

    for name in args.name:
        experiment_library[name]()
    if run_all:
        for name, test in experiment_library.items():
            test()

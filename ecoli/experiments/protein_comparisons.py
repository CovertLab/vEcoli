"""
==================================
Metabolism using user-friendly FBA
==================================
"""
import argparse

# vivarium-core imports
import pytest

# vivarium-ecoli imports
from ecoli.experiments.ecoli_master_sim import EcoliSim, CONFIG_DIR_PATH
from ecoli.states.wcecoli_state import get_state_from_file

from wholecell.utils import units

import numpy as np
import pathlib, datetime
import dill


# disables growth rate control
def validation_experiment(
        filename='metabolism_redux_classic',
        total_time=1300,
        divide=True,
        # initial_state_file='wcecoli_t0', # 'met_division_test_state',
        progress_bar=True,
        log_updates=False,
        emitter='timeseries', # 'timeseries',
        name='validation_experiment',
        raw_output=False,
        condition = "with_aa", # basal, with_aa
        fixed_media = "minimal_plus_amino_acids" # minimal, minimal_plus_amino_acids
):
    # filename = 'default'
    sim = EcoliSim.from_file(CONFIG_DIR_PATH + filename + ".json")
    sim.total_time = total_time
    sim.divide = divide
    sim.progress_bar = progress_bar
    sim.log_updates = log_updates
    sim.emitter = emitter
    # sim.initial_state = get_state_from_file(path=f'data/{initial_state_file}.json')
    sim.raw_output = raw_output

    sim.condition = condition
    sim.fixed_media = fixed_media

    sim.seed = 12

    # this means that sims will not create conflicting random indices when loading from saved state
    # if initial_state_file == 'wcecoli_t0':
    #     sim.seed += 1
    # else:
    #     sim.seed += int(sim.initial_state['agents']['0']['global_time'])

    sim.build_ecoli()

    sim.run()

    query = []
    folder = f'out/cofactors/{name}_{total_time}_{datetime.date.today()}/'
    save_sim_output(folder, query, sim, save_model=True)


experiment_library = {
    '1': validation_experiment,
}


def save_sim_output(folder, query, sim, save_model=False):
    agents = sim.query()['agents'].keys()
    for agent in agents:
        query = []
        query.extend([('agents', agent, 'listeners', 'fba_results'),
                      ('agents', agent, 'listeners', 'mass'),
                      ('agents', agent, 'listeners', 'unique_molecule_counts'),
                      ('agents', agent, 'bulk')])
        output = sim.query(query)
        pathlib.Path(folder).mkdir(parents=True, exist_ok=True)
        np.save(folder +  f'{agent}_output.npy', output)

    if save_model:
        f = open(folder + 'agent_steps.pkl', 'wb')
        dill.dump(sim.ecoli_experiment.steps['agents'][agent], f)
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

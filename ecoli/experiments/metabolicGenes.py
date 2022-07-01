"""
===========================================
Experiment for testing functionality of new genes
===========================================
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
from ecoli.processes.registries import topology_registry

import numpy as np
import pathlib, datetime

# get topology from ecoli_maste


def run_ecoli(
        filename='default',
        total_time=4,
        divide=True,
        initial_state_file='vivecoli_t2',
        progress_bar=True,
        log_updates=False,
        emitter='timeseries',
):
    sim = EcoliSim.from_file(CONFIG_DIR_PATH + filename + '.json')
    sim.total_time = total_time
    sim.raw_output = False
    sim.divide = divide
    sim.progress_bar = progress_bar
    sim.log_updates = log_updates
    sim.emitter = emitter
    sim.initial_state = get_state_from_file(path=f'data/{initial_state_file}.json')

    sim.run()

    query = []
    agents = sim.query()['agents'].keys()
    for agent in agents:
        query.extend([('agents', agent, 'listeners', 'fba_results'),
                      ('agents', agent, 'bulk')])
    output = sim.query(query)

    folder = f'out/geneRxnVerifData'
    pathlib.Path(folder).mkdir(parents=True, exist_ok=True)
    np.save(folder + 'output.npy', output)


experiment_library = {
    '0': run_ecoli,
}

# run experiments with command line arguments: python ecoli/experiments/metabolism_gd.py -n exp_id
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='testing metabolic genes')
    parser.add_argument('--name', '-n', default=[], nargs='+', help='test ids to run')
    args = parser.parse_args()
    run_all = not args.name

    for name in args.name:
        experiment_library[name]()
    if run_all:
        for name, test in experiment_library.items():
            test()

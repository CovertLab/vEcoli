from ecoli.library.sim_data import LoadSimData
from ecoli.composites.ecoli_master import SIM_DATA_PATH
from ecoli.processes.complexation import Complexation
from vivarium.library.dict_utils import deep_merge
from ecoli.states.wcecoli_state import get_state_from_file

from migration.migration_utils import (run_ecoli_process,
                                       array_diffs_report_test)
import json
import os
from matplotlib import pyplot as plt
import numpy as np
from ecoli.library.schema import array_from

load_sim_data = LoadSimData(
            sim_data_path=SIM_DATA_PATH,
            seed=0)

def test_complexation_migration():
    def test(initial_time):
        total_time = 2
        
        # Create process, experiment, loading in initial state from file.
        config = load_sim_data.get_complexation_config()
        config['seed'] = 0
        complexation = Complexation(config)

        topology = {
            'molecules': ('bulk',)
            }
        
        with open(f"data/complexation/complexation_partitioned_t{total_time+initial_time}.json") as f:
            partitioned_counts = json.load(f)
            
        initial_state = get_state_from_file(
            path=f'data/complexation/wcecoli_t{initial_time}.json')
        
        deep_merge(initial_state, {'bulk': partitioned_counts})

        # run the process and get an update
        actual_update = run_ecoli_process(complexation, topology,
                                        total_time=total_time,
                                        initial_time=initial_time,
                                        initial_state=initial_state)
        
        with open(f"data/complexation/complexation_update_t{total_time+initial_time}.json") as f:
            wc_update = json.load(f)

        plots(actual_update, wc_update, total_time+initial_time)
        assertions(actual_update,wc_update, total_time+initial_time)
    
    times = [0, 2, 8, 100]
    for initial_time in times:
        test(initial_time)

def plots(actual_update, expected_update, time):
    os.makedirs("out/migration/complexation/", exist_ok=True)

    molecules_update = actual_update['molecules']
    wc_molecules_update = expected_update['molecules']
    
    n_molecules = len(molecules_update)
    differences = array_from(molecules_update) - array_from(wc_molecules_update)
    plt.scatter(np.arange(n_molecules), differences, 0.2, c="b")
    plt.ylabel('Vivarium update - wcEcoli update')
    plt.xlabel('Molecules (unlabelled for space)')
    plt.title(f'Update Dictionary Differences at t = {time}')
    plt.tight_layout()
    plt.savefig(f"out/migration/complexation/complexation_{time}.png")
    plt.close()

def assertions(actual_update, expected_update, time):
    # Create report with exact differences between molecule count updates
    test = array_diffs_report_test(f"out/migration/complexation/complexation_{time}.txt")
    test(array_from(actual_update['molecules']), array_from(expected_update['molecules']))
    
    # check number of molecules
    assert len(actual_update['molecules']) == len(expected_update['molecules']), \
        "# of molecules not equal!"

if __name__ == "__main__":
    test_complexation_migration()

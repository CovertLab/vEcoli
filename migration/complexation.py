import json
import os
import numpy as np
from matplotlib import pyplot as plt

from vivarium.library.dict_utils import deep_merge
from ecoli.library.schema import array_from
from ecoli.processes.complexation import Complexation
from ecoli.states.wcecoli_state import get_state_from_file
from migration.migration_utils import (run_ecoli_process,
                                       array_equal,
                                       stochastic_equal,
                                       array_diffs_report_test)


TOPOLOGY = Complexation.topology


def complexation_migration(sim_data):
    def test(initial_time):
        total_time = 2
        
        # Create process, experiment, loading in initial state from file.
        config = sim_data.get_complexation_config()
        config['seed'] = 0
        complexation = Complexation(config)

        
        with open(f"data/complexation/complexation_partitioned_t"
            f"{total_time+initial_time}.json") as f:
            partitioned_counts = json.load(f)
            
        initial_state = get_state_from_file(
            path=f'data/complexation/wcecoli_t{initial_time}.json')
        
        deep_merge(initial_state, {'bulk': partitioned_counts})

        # run the process and get an update
        actual_update = run_ecoli_process(complexation, TOPOLOGY,
                                        total_time=total_time,
                                        initial_time=initial_time,
                                        initial_state=initial_state)
        
        with open("data/complexation/complexation_update_t"
            f"{total_time+initial_time}.json") as f:
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
    diffs = array_from(molecules_update) - array_from(wc_molecules_update)
    plt.scatter(np.arange(n_molecules), diffs, 0.2, c="b")
    plt.ylabel('Vivarium update - wcEcoli update')
    plt.xlabel('Molecules (unlabelled for space)')
    plt.title(f'Update Dictionary Differences at t = {time}')
    plt.tight_layout()
    plt.savefig(f"out/migration/complexation/complexation_{time}.png")
    plt.close()

def assertions(actual_update, expected_update, time):
    vivarium_deltas = array_from(actual_update['molecules'])
    wcecoli_deltas = array_from(expected_update['molecules'])
    
    # check that molecule count changes are exactly equal (must use seeded 
    # update dictionaries)
    assert array_equal(vivarium_deltas, wcecoli_deltas)
    
    # check that molecule count changes likely have the same underlying 
    # distribution (for use with unseeded update dictionaries)
    assert stochastic_equal(vivarium_deltas, wcecoli_deltas)
    
    # create report to troubleshoot any observed differences between 
    # molecule count updates
    test = array_diffs_report_test("out/migration/complexation/"
                                   f"complexation_{time}.txt")
    test(vivarium_deltas, wcecoli_deltas)
    
    # check number of molecules included in update dictionary
    assert len(vivarium_deltas) == len(wcecoli_deltas), \
        "# of molecules not equal!"

if __name__ == "__main__":
    from ecoli.library.sim_data import LoadSimData
    from ecoli.composites.ecoli_nonpartition import SIM_DATA_PATH
    sim_data = LoadSimData(
        sim_data_path=SIM_DATA_PATH,
        seed=0)

    complexation_migration(sim_data)

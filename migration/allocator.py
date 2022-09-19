"""
tests that vivarium-ecoli allocator process update is the same as saved wcEcoli updates
"""
import os
import json
import pytest

from ecoli.library.sim_data import LoadSimData
from ecoli.composites.ecoli_nonpartition import SIM_DATA_PATH
from ecoli.composites.ecoli_configs import CONFIG_DIR_PATH
from ecoli.processes.allocator import Allocator
from migration.migration_utils import run_ecoli_process, recursive_compare
from ecoli.states.wcecoli_state import get_state_from_file


load_sim_data = LoadSimData(
            sim_data_path=SIM_DATA_PATH,
            seed=0)

with open('data/proc_to_name.json', 'r') as f:
    proc_to_name = json.load(f)
name_to_proc = {v:k for k,v in proc_to_name.items()}

with open(os.path.join(CONFIG_DIR_PATH, 'default.json'), 'r') as f:
    default_config = json.load(f)
process_names = default_config['processes']

TOPOLOGY = Allocator.topology

@pytest.mark.master
def test_allocator_migration():
    def test(initial_time):
        # Set time parameters
        total_time = 2
        initial_time = initial_time

        # Create process, experiment, loading in initial state from file.
        config = load_sim_data.get_allocator_config(process_names=process_names)
        allocator_process = Allocator(config)
        allocator_process.first_update = False
        allocator_process.is_step = lambda: False

        initial_state = get_state_from_file(
            path=f'data/migration/wcecoli_t{initial_time}_before_layer_0.json')
        
        with open(
            f'data/migration/allocator/request_t{total_time+initial_time}.json',
            'r'
        ) as f:
            initial_request = json.load(f)
        
        actual_request = {'request': {}}
        for process in initial_request['request']:
            actual_request['request'][name_to_proc[process]] = initial_request['request'][process]
        
        with open(
            f"data/migration/allocator/partitioned_t{total_time+initial_time}.json",
            'w'
        ) as f:
            json.dump(actual_request, f)

        # run the process and get an update
        actual_update = run_ecoli_process(
            allocator_process, TOPOLOGY, initial_time = initial_time, 
            initial_state=initial_state, folder_name='allocator')
        
        actual_allocated = {'allocate': {}}
        for process in actual_update['allocate']:
            actual_allocated['allocate'][proc_to_name[process]] = actual_update['allocate'][process]
        
        with open(
            f"data/migration/allocator/update_t{total_time+initial_time}.json",
            'r'
        ) as f:
            wc_update = json.load(f)
        assert recursive_compare(actual_allocated, wc_update, check_keys_strict=False)

    times = [0, 2072]
    for initial_time in times:
        test(initial_time)

if __name__ == "__main__":
    test_allocator_migration()

"""
tests that vivarium-ecoli allocator process update is the same as saved wcEcoli updates
"""
import os
import json
import pytest
import numpy as np

from ecoli.composites.ecoli_configs import CONFIG_DIR_PATH
from ecoli.processes.allocator import Allocator
from migration import LOAD_SIM_DATA
from migration.migration_utils import run_non_partitioned_process, recursive_compare
from ecoli.states.wcecoli_state import get_state_from_file


with open('data/proc_to_name.json', 'r') as f:
    viv_to_wc_proc = json.load(f)
wc_to_viv_proc = {v:k for k,v in viv_to_wc_proc.items()}

with open(os.path.join(CONFIG_DIR_PATH, 'default.json'), 'r') as f:
    default_config = json.load(f)
process_names = default_config['processes']

skip_processes_viv = ['ecoli-metabolism', 'ecoli-chromosome-structure']
skip_processes_wc = ['Metabolism', 'ChromosomeStructure']


def run_and_compare(init_time):
    # Set time parameters
    init_time = init_time

    # Create process, experiment, loading in initial state from file.
    config = LOAD_SIM_DATA.get_allocator_config(process_names=process_names)
    allocator_process = Allocator(config)
    allocator_process.is_step = lambda: False

    initial_state = get_state_from_file(
        path=f'data/migration/wcecoli_t{init_time}_before_layer_0.json')
    bulk_idx = np.arange(len(initial_state['bulk']))
    
    # Load requests from wcEcoli into initial state
    with open(f'data/migration/bulk_requested_t{init_time}.json', 'r') as f:
        initial_request = json.load(f)
    initial_state['request'] = {}
    for process, proc_req in initial_request.items():
        proc_name = wc_to_viv_proc.get(process, None)
        if proc_name is None or proc_name in skip_processes_viv:
            continue
        initial_state['request'][proc_name] = {'bulk': [(bulk_idx, proc_req)]}

    # Run the process and get an update
    actual_update = run_non_partitioned_process(allocator_process, Allocator.topology, 
        initial_state=initial_state)
    actual_allocated = {}
    for process, proc_alloc in actual_update['allocate'].items():
        proc_name = viv_to_wc_proc.get(process, None)
        if proc_name is None or proc_name in skip_processes_wc:
            continue
        actual_allocated[proc_name] = proc_alloc['bulk']
    
    # Compare to wcEcoli partitioned counts
    with open(f"data/migration/bulk_partitioned_t{init_time}.json", 'r') as f:
        wc_update = json.load(f)
    assert recursive_compare(actual_allocated, wc_update, check_keys_strict=False)

@pytest.mark.master
def test_allocator_migration():
    times = [0, 2132]
    for initial_time in times:
        run_and_compare(initial_time)

if __name__ == "__main__":
    test_allocator_migration()

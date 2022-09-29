"""
tests that vivarium-ecoli rna_degradation process update is the same as saved wcEcoli updates
"""
import json
import pytest

from ecoli.library.sim_data import LoadSimData
from ecoli.composites.ecoli_nonpartition import SIM_DATA_PATH
from ecoli.processes.two_component_system import TwoComponentSystem
from migration.migration_utils import (run_custom_partitioned_process, 
                                       recursive_compare)
from ecoli.states.wcecoli_state import get_state_from_file

load_sim_data = LoadSimData(
            sim_data_path=SIM_DATA_PATH,
            seed=0)

TOPOLOGY = TwoComponentSystem.topology

@pytest.mark.master
def test_two_component_system_migration():
    def test(initial_time):
        # Set time parameters
        total_time = 2
        initial_time = initial_time

        # Create process, experiment, loading in initial state from file.
        config = load_sim_data.get_two_component_system_config()
        two_component_system_process = TwoComponentSystem(config)

        initial_state = get_state_from_file(
            path=f'data/migration/wcecoli_t{initial_time}_before_layer_0.json')

        # run the process and get an update
        actual_request, actual_update = run_custom_partitioned_process(
            two_component_system_process, TOPOLOGY, initial_time = initial_time,
            initial_state=initial_state, folder_name='two_component_system')

        with open(
            f"data/migration/two_component_system/request_t{total_time+initial_time}.json",
            'r'
        ) as f:
            wc_request = json.load(f)
        assert recursive_compare(actual_request, wc_request)
        
        with open(
            f"data/migration/two_component_system/update_t{total_time+initial_time}.json",
            'r'
        ) as f:
            wc_update = json.load(f)
        assert recursive_compare(actual_update, wc_update)

    times = [0, 2072]
    for initial_time in times:
        test(initial_time)

if __name__ == "__main__":
    test_two_component_system_migration()

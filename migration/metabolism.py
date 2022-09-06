"""
tests that vivarium-ecoli metabolism process update is the same as saved wcEcoli updates
"""
import json
import pytest

from ecoli.library.sim_data import LoadSimData
from ecoli.composites.ecoli_nonpartition import SIM_DATA_PATH
from ecoli.processes.metabolism import Metabolism
from migration.migration_utils import run_ecoli_process, recursive_compare
from ecoli.states.wcecoli_state import get_state_from_file

load_sim_data = LoadSimData(
            sim_data_path=SIM_DATA_PATH,
            seed=0)

TOPOLOGY = Metabolism.topology

@pytest.mark.master
def test_metabolism_migration():
    def test(initial_time):
        # Set time parameters
        total_time = 2
        initial_time = initial_time

        # Create process, experiment, loading in initial state from file.
        config = load_sim_data.get_metabolism_config()
        metabolism_process = Metabolism(config)
        metabolism_process.first_update = False
        metabolism_process.is_step = lambda: False

        initial_state = get_state_from_file(
            path=f'data/migration/wcecoli_t{initial_time}_before_layer_2.json')
        
        # run the process and get an update
        actual_update = run_ecoli_process(
            metabolism_process, TOPOLOGY, initial_time=initial_time, 
            initial_state=initial_state, folder_name='metabolism')
        
        with open(
            f"data/migration/metabolism/update_t{total_time+initial_time}.json",
            'r'
        ) as f:
            wc_update = json.load(f)
        assert recursive_compare(actual_update, wc_update, ignore_keys={
            'estimated_fluxes', 'target_dmdt', 'estimated_dmdt', 'estimated_exchange_dmdt'})

    times = [0, 2072]
    for initial_time in times:
        test(initial_time)

if __name__ == "__main__":
    test_metabolism_migration()

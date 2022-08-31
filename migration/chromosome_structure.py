"""
tests that vivarium-ecoli chromosome_structure process update is the same as saved wcEcoli updates
"""
import json
import pytest

from ecoli.library.sim_data import LoadSimData
from ecoli.composites.ecoli_nonpartition import SIM_DATA_PATH
from ecoli.processes.chromosome_structure import ChromosomeStructure
from migration.migration_utils import run_ecoli_process, recursive_compare
from ecoli.states.wcecoli_state import get_state_from_file

load_sim_data = LoadSimData(
            sim_data_path=SIM_DATA_PATH,
            seed=0)

TOPOLOGY = ChromosomeStructure.topology

@pytest.mark.master
def test_chromosome_structure_migration():
    def test(initial_time):
        # Set time parameters
        total_time = 2
        initial_time = initial_time

        # Create process, experiment, loading in initial state from file.
        config = load_sim_data.get_chromosome_structure_config()
        chromosome_structure_process = ChromosomeStructure(config)
        chromosome_structure_process.first_update = False
        chromosome_structure_process.is_step = lambda: False

        initial_state = get_state_from_file(
            path=f'data/migration/wcecoli_t{initial_time}_before_layer_1.json')

        # run the process and get an update
        actual_update = run_ecoli_process(
            chromosome_structure_process, TOPOLOGY, initial_time = initial_time, 
            initial_state=initial_state)
        
        with open(f"data/migration/chromosome_structure/update_t{total_time+initial_time}.json") as f:
            wc_update = json.load(f)
        assert recursive_compare(actual_update, wc_update, ignore_keys={'key', 'unique_index'})

    times = [0, 2072]
    for initial_time in times:
        test(initial_time)

if __name__ == "__main__":
    test_chromosome_structure_migration()

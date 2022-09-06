"""
tests that vivarium-ecoli mRNA_counts_listener process update is the same as saved wcEcoli updates
"""
import json
import pytest

from ecoli.library.sim_data import LoadSimData
from ecoli.composites.ecoli_nonpartition import SIM_DATA_PATH
from ecoli.processes.listeners.mRNA_counts import mRNACounts
from migration.migration_utils import run_ecoli_process, recursive_compare
from ecoli.states.wcecoli_state import get_state_from_file

load_sim_data = LoadSimData(
            sim_data_path=SIM_DATA_PATH,
            seed=0)

TOPOLOGY = mRNACounts.topology

@pytest.mark.master
def test_mRNA_counts_migration():
    def test(initial_time):
        # Set time parameters
        total_time = 2
        initial_time = initial_time

        # Create process, experiment, loading in initial state from file.
        config = load_sim_data.get_mrna_counts_listener_config()
        mRNA_counts_process = mRNACounts(config)
        mRNA_counts_process.is_step = lambda: False

        initial_state = get_state_from_file(
            path=f'data/migration/wcecoli_t{initial_time}_before_post.json')

        # run the process and get an update
        actual_update = run_ecoli_process(
            mRNA_counts_process, TOPOLOGY, initial_time=initial_time, 
            initial_state=initial_state)
        
        with open(
            f"data/migration/mRNA_counts/update_t{total_time+initial_time}.json"
            'r'
        ) as f:
            wc_update = json.load(f)
        assert recursive_compare(actual_update, wc_update)

    times = [0, 2072]
    for initial_time in times:
        test(initial_time)

if __name__ == "__main__":
    test_mRNA_counts_migration()

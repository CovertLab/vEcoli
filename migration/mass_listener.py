"""
tests that vivarium-ecoli mass_listener process update is the same as saved wcEcoli updates
"""
import json
import pytest

from ecoli.library.sim_data import LoadSimData
from ecoli.composites.ecoli_nonpartition import SIM_DATA_PATH
from ecoli.processes.listeners.mass_listener import MassListener
from migration.migration_utils import run_ecoli_process, recursive_compare
from ecoli.states.wcecoli_state import get_state_from_file

load_sim_data = LoadSimData(
            sim_data_path=SIM_DATA_PATH,
            seed=0)

TOPOLOGY = MassListener.topology

@pytest.mark.master
def test_mass_listener_migration():
    def test(initial_time):
        # Set time parameters
        total_time = 2
        initial_time = initial_time

        # Create process, experiment, loading in initial state from file.
        config = load_sim_data.get_mass_listener_config()
        mass_listener_process = MassListener(config)
        mass_listener_process.is_step = lambda: False

        initial_state = get_state_from_file(
            path=f'data/migration/wcecoli_t{initial_time}_before_post.json')

        # run the process and get an update
        actual_update = run_ecoli_process(
            mass_listener_process, TOPOLOGY, initial_time=initial_time, 
            initial_state=initial_state)
        
        with open(
            f"data/migration/mass_listener/update_t{total_time+initial_time}.json",
            'r') as f:
            wc_update = json.load(f)
        assert recursive_compare(actual_update, wc_update, ignore_keys={
            'dryMassFoldChange', 'proteinMassFoldChange', 'rnaMassFoldChange', 
            'smallMoleculeFoldChange', 'expectedMassFoldChange', 'instantaniousGrowthRate',
            'process_mass_diffs', 'growth'     
        })

    times = [0, 2072]
    for initial_time in times:
        test(initial_time)

if __name__ == "__main__":
    test_mass_listener_migration()

from ecoli.library.sim_data import LoadSimData
from ecoli.composites.ecoli_master import SIM_DATA_PATH
from ecoli.processes.two_component_system import TwoComponentSystem
from migration.migration_utils import run_ecoli_process

load_sim_data = LoadSimData(
            sim_data_path=SIM_DATA_PATH,
            seed=0)

TOPOLOGY = {
        'listeners': ('listeners',),
        'molecules': ('bulk',)}

def test_two_component_system_migration():
    # Create process, experiment, loading in initial state from file.
    config = load_sim_data.get_two_component_system_config()
    two_component_system_process = TwoComponentSystem(config)

    # run the process and get an update
    actual_update = run_ecoli_process(two_component_system_process, TOPOLOGY, total_time=2, initial_time=1000)

    #ipdb breakpoint - can import in the middle!
    import ipdb; ipdb.set_trace()

if __name__ == "__main__":
    test_two_component_system_migration()

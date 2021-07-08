import json

from ecoli.library.sim_data import LoadSimData
from ecoli.composites.ecoli_master import SIM_DATA_PATH
from ecoli.processes.two_component_system import TwoComponentSystem
from migration.migration_utils import run_ecoli_process

load_sim_data = LoadSimData(
            sim_data_path=SIM_DATA_PATH,
            seed=0)

def test_two_component_system_migration():
    # Create process, experiment, loading in initial state from file.
    config = load_sim_data.get_two_component_system_config()
    two_component_system_process = TwoComponentSystem(config)

    topology = {
        'listeners': ('listeners',),
        'molecules': ('bulk',)}

    # run the process and get an update
    actual_update = run_ecoli_process(two_component_system_process, topology, total_time=2, initial_time=0)
    """
    actual_updates = {}
    for x in range(10):
        config = load_sim_data.get_two_component_system_config(random_seed=x)
        two_component_system_process = TwoComponentSystem(config)

        actual_update = run_ecoli_process(two_component_system_process, topology, total_time=2, initial_time=0)
        actual_updates[x] = actual_update
    """
    with open("data/two_component_system_update_t2.json") as f:
        wc_update = json.load(f)

    # plot all the actual_updates and compare plots
    # save the stoichiometry matrix and compare

    # #ipdb breakpoint - can import in the middle!
    import ipdb; ipdb.set_trace()

if __name__ == "__main__":
    test_two_component_system_migration()

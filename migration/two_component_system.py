import json

from ecoli.library.sim_data import LoadSimData
from ecoli.composites.ecoli_master import SIM_DATA_PATH
from ecoli.processes.two_component_system import TwoComponentSystem
from migration.migration_utils import *

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
    actual_update = run_ecoli_process(two_component_system_process, topology, total_time=2, initial_time=10)

    with open("data/two_component_system_update_t12.json") as f:
        wc_update = json.load(f)

    assertions(actual_update, wc_update)


def assertions(actual_update, expected_update):
    test_structure = {
        'molecules': {key: scalar_equal for key in actual_update['molecules'].keys()}
    }

    tests = ComparisonTestSuite(test_structure, fail_loudly=False)
    tests.run_tests(actual_update, expected_update, verbose=True)

    #print(tests.report)
    tests.dump_report()

    #tests.fail()

if __name__ == "__main__":
    test_two_component_system_migration()
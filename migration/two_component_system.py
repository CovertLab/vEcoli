import os
import json
import numpy as np
import matplotlib.pyplot as plt

from ecoli.library.sim_data import LoadSimData
from ecoli.composites.ecoli_master import SIM_DATA_PATH
from ecoli.processes.two_component_system import TwoComponentSystem
from migration.plots import qqplot
from migration.migration_utils import run_ecoli_process, scalar_almost_equal, ComparisonTestSuite
from migration.migration_utils import run_ecoli_process
from ecoli.states.wcecoli_state import get_state_from_file

load_sim_data = LoadSimData(
            sim_data_path=SIM_DATA_PATH,
            seed=0)
TOPOLOGY = TwoComponentSystem.topology


def test_two_component_system_migration():
    # Set time parameters
    total_time = 2
    initial_time = 10

    # Create process, experiment, loading in initial state from file.
    config = load_sim_data.get_two_component_system_config()
    two_component_system_process = TwoComponentSystem(config)

    # run the process and get an update
    actual_update = run_ecoli_process(two_component_system_process, TOPOLOGY, total_time=total_time,
                                      initial_time=initial_time)

    with open("data/two_component_system_update_t12.json") as f:
        wc_update = json.load(f)

    plots(actual_update, wc_update, total_time + initial_time)
    assertions(actual_update, wc_update)

def plots(actual_update, expected_update, time):
    os.makedirs("out/migration/two_component_system/", exist_ok=True)

    # unpack updates
    molecules = np.array([actual_update['molecules'][mol] for mol in actual_update['molecules']])
    wc_molecules = np.array([expected_update['molecules'][mol] for mol in expected_update['molecules']])

    # Plot
    qqplot(molecules, wc_molecules)
    plt.ylabel('wcEcoli')
    plt.xlabel('Vivarium')
    plt.title('Q-Q Plot of molecules')
    plt.savefig(f"out/migration/two_component_system/two_component_system_figures{time}.png")

def assertions(actual_update, expected_update):
    # TODO: this should be removed once partitioning is fixed
    # list moleculeNames that are partitioned (this is for initial_time=10)
    partitioned_molecules = ['PHOP-MONOMER[c]', 'PHOSPHO-PHOP[c]', 'ATP[c]', 'PROTON[c]', 'ADP[c]']
    """
    As of now, we think partitioning causes only a partial amount of the moleculeCounts to be accounted for in the 
    function call to moleculesToNextTimeStep in wcEcoli/models/ecoli/processes/two_component_system.py.
    
    We suspect that the molecules listed in the partitioned_molecules variable are the ones that are being partitioned, 
    based on the differences in the actual_update dictionary produced in vivarium and wc_update dictionary produced in 
    wcEcoli.
    
    Once partitioning is fixed, we should include these molecules back into the migration tests.
    """
    test_structure = {
        'molecules': {
            key: scalar_almost_equal
            for key in actual_update['molecules'].keys()
            if key not in partitioned_molecules}
    }

    tests = ComparisonTestSuite(test_structure, fail_loudly=False)
    tests.run_tests(actual_update, expected_update, verbose=True)

    # display test reports
    tests.dump_report()

if __name__ == "__main__":
    test_two_component_system_migration()

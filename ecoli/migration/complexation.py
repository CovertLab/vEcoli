from ecoli.library.sim_data import LoadSimData
from ecoli.composites.ecoli_master import SIM_DATA_PATH
from ecoli.processes.complexation import Complexation

from ecoli.migration.migration_utils import run_ecoli_process


load_sim_data = LoadSimData(
            sim_data_path=SIM_DATA_PATH,
            seed=0)


def test_complexation():
    config = load_sim_data.get_complexation_config()
    complexation = Complexation(config)

    topology = {
        'molecules': ('bulk',)
        }

    # run the process and get an update
    actual_update = run_ecoli_process(complexation, topology)


if __name__ == "__main__":
    test_complexation()

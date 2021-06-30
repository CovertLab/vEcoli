from vivarium.core.engine import Engine
from ecoli.library.sim_data import LoadSimData
from ecoli.composites.ecoli_master import SIM_DATA_PATH
from ecoli.processes.two_component_system import TwoComponentSystem
from migration.migration_utils import run_ecoli_process
from ecoli.composites.ecoli_master import get_state_from_file


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
    actual_update = run_ecoli_process(two_component_system_process, TOPOLOGY, total_time=2)


def test_two_component_system():
    # Create process, experiment, loading in initial state from file.
    config = load_sim_data.get_two_component_system_config()
    two_component_system_process = TwoComponentSystem(config)

    initial_state = get_state_from_file(
        path=f'data/wcecoli_t0.json')

    two_component_system_composite = two_component_system_process.generate()

    experiment = Engine({
        'processes': two_component_system_composite['processes'],
        'topology': {two_component_system_process.name: TOPOLOGY},
        'initial_state': initial_state
    })

    experiment.update(10)

    data = experiment.emitter.get_data()

    return data

def run_two_component_system():
    data = test_two_component_system()
    import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    # test_two_component_system_migration()
    run_two_component_system()

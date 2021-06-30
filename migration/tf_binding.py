from vivarium.core.engine import Engine
from ecoli.library.sim_data import LoadSimData
from ecoli.composites.ecoli_master import SIM_DATA_PATH
from ecoli.processes.tf_binding import TfBinding
from migration.migration_utils import run_ecoli_process
from ecoli.composites.ecoli_master import get_state_from_file


load_sim_data = LoadSimData(
            sim_data_path=SIM_DATA_PATH,
            seed=0)

TF_BINDING_TOPOLOGY = {
    'promoters': ('unique', 'promoter'),
    'active_tfs': ('bulk',),
    'inactive_tfs': ('bulk',),
    'listeners': ('listeners',)}


def test_tf_binding_migration():
    # Create process, experiment, loading in initial state from file.
    config = load_sim_data.get_tf_config()
    tf_binding_process = TfBinding(config)

    # run the process and get an update
    actual_update = run_ecoli_process(tf_binding_process, TF_BINDING_TOPOLOGY, total_time=2)


def run_tf_binding():
    # Create process, experiment, loading in initial state from file.
    config = load_sim_data.get_tf_config()
    tf_binding_process = TfBinding(config)

    initial_state = get_state_from_file(
        path=f'data/wcecoli_t0.json')

    tf_binding_composite = tf_binding_process.generate()
    experiment = Engine({
        'processes': tf_binding_composite['processes'],
        'topology': {tf_binding_process.name: TF_BINDING_TOPOLOGY},
        'initial_state': initial_state
    })

    experiment.update(10)

    data = experiment.emitter.get_data()
    import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    # test_tf_binding_migration()
    run_tf_binding()

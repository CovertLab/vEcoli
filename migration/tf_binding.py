"""
tests that vivarium-ecoli tf_binding processupdate is the same as saved wcEcoli updates
"""

from vivarium.core.engine import Engine
from ecoli.library.sim_data import LoadSimData
from ecoli.composites.ecoli_master import SIM_DATA_PATH
from ecoli.processes.tf_binding import TfBinding
from migration.migration_utils import run_ecoli_process
from ecoli.composites.ecoli_master import get_state_from_file


load_sim_data = LoadSimData(
            sim_data_path=SIM_DATA_PATH,
            seed=0)

# copy topology from ecoli_master.py, under generate_topology
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

    # separate the update to its ports
    active_tfs = actual_update['active_tfs']
    promoters = actual_update['promoters']
    p_promoter_bound = actual_update['listeners']['rna_synth_prob']['pPromoterBound']
    n_promoter_bound = actual_update['listeners']['rna_synth_prob']['nPromoterBound']
    n_actual_bound = actual_update['listeners']['rna_synth_prob']['nActualBound']
    n_available_promoters = actual_update['listeners']['rna_synth_prob']['n_available_promoters']
    n_bound_TF_per_TU = actual_update['listeners']['rna_synth_prob']['n_bound_TF_per_TU']

    assert len(p_promoter_bound) == len(promoters), ("Update malformed: number of promoter"
                                                     "binding probabilities != number of"
                                                     "promoters")

    assert len(n_promoter_bound) == len(promoters), ("Update malformed: number of promoters"
                                                     "with binding info != number of"
                                                     "promoters")



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


if __name__ == "__main__":
    test_tf_binding_migration()
    # run_tf_binding()

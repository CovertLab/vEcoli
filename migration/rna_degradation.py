from vivarium.core.engine import Engine
from ecoli.processes.rna_degradation import RnaDegradation
from migration.migration_utils import run_ecoli_process
from ecoli.states.wcecoli_state import get_state_from_file


TOPOLOGY = RnaDegradation.topology


def run_rna_degradation_migration(sim_data):
    # Create process, experiment, loading in initial state from file.
    config = sim_data.get_rna_degradation_config()
    rna_degradation_process = RnaDegradation(config)

    # run the process and get an update
    actual_update = run_ecoli_process(rna_degradation_process, TOPOLOGY, total_time=2)


def run_rna_degradation_default(sim_data):
    # Create process, experiment, loading in initial state from file.
    config = sim_data.get_rna_degradation_config()
    rna_degradation_process = RnaDegradation(config)

    initial_state = get_state_from_file(
        path=f'data/wcecoli_t0.json')

    rna_degradation_composite = rna_degradation_process.generate()

    experiment = Engine(**{
        'processes': rna_degradation_composite['processes'],
        'topology': {rna_degradation_process.name: TOPOLOGY},
        'initial_state': initial_state
    })

    experiment.update(10)

    data = experiment.emitter.get_data()

    return data

def run_rna_degradation():
    data = run_rna_degradation_default()
    import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    # run_rna_degradation_migration()
    run_rna_degradation()

import pytest

from vivarium.core.engine import Engine
from ecoli.processes.rna_degradation import RnaDegradation
from migration.migration_utils import run_ecoli_process
from ecoli.states.wcecoli_state import get_state_from_file
from migration import load_sim_data


TOPOLOGY = RnaDegradation.topology


@pytest.mark.master
def test_rna_degradation_migration():
    # Create process, experiment, loading in initial state from file.
    config = load_sim_data.get_rna_degradation_config()
    rna_degradation_process = RnaDegradation(config)

    # run the process and get an update
    actual_update = run_ecoli_process(rna_degradation_process, TOPOLOGY, total_time=2)


@pytest.mark.master
def test_rna_degradation():
    # Create process, experiment, loading in initial state from file.
    config = load_sim_data.get_rna_degradation_config()
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
    data = test_rna_degradation()
    import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    # test_rna_degradation_migration()
    run_rna_degradation()

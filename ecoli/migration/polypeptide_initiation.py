from vivarium.core.experiment import Experiment
from ecoli.library.sim_data import LoadSimData
from ecoli.composites.ecoli_master import SIM_DATA_PATH
from ecoli.processes.polypeptide_initiation import PolypeptideInitiation
from ecoli.migration.migration_utils import run_ecoli_process
from ecoli.composites.ecoli_master import get_state_from_file


load_sim_data = LoadSimData(
            sim_data_path=SIM_DATA_PATH,
            seed=0)

PI_TOPOLOGY = {
    'environment': ('environment',),
    'listeners': ('listeners',),
    'active_ribosome': ('unique', 'active_ribosome'),
    'RNA': ('unique', 'RNA'),
    'subunits': ('bulk',)}


def test_polypeptide_initiation_migration():
    # Create process, experiment, loading in initial state from file.
    config = load_sim_data.get_polypeptide_initiation_config()
    polypeptide_initiation_process = PolypeptideInitiation(config)

    # run the process and get an update
    actual_update = run_ecoli_process(polypeptide_initiation_process, PI_TOPOLOGY, total_time=2)


def run_polypeptide_initiation():
    # Create process, experiment, loading in initial state from file.
    config = load_sim_data.get_polypeptide_initiation_config()
    polypeptide_initiation_process = PolypeptideInitiation(config)

    initial_state = get_state_from_file(
        path=f'data/wcecoli_t0.json')

    polypeptide_initiation_composite = polypeptide_initiation_process.generate()

    experiment = Experiment({
        'processes': polypeptide_initiation_composite['processes'],
        'topology': {polypeptide_initiation_process.name: PI_TOPOLOGY},
        'initial_state': initial_state
    })

    experiment.update(10)

    data = experiment.emitter.get_data()
    import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    # test_polypeptide_initiation_migration()
    run_polypeptide_initiation()

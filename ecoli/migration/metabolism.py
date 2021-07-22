from vivarium.core.experiment import Experiment
from ecoli.library.sim_data import LoadSimData
from ecoli.composites.ecoli_master import SIM_DATA_PATH
from ecoli.processes.metabolism import Metabolism
from ecoli.migration.migration_utils import run_ecoli_process
from ecoli.composites.ecoli_master import get_state_from_file


load_sim_data = LoadSimData(
            sim_data_path=SIM_DATA_PATH,
            seed=0)


def test_metabolism_migration():
    # Create process, experiment, loading in initial state from file.
    config = load_sim_data.get_metabolism_config()
    metabolism_process = Metabolism(config)

    # topology from ecoli_master
    topology = {
            'metabolites': ('bulk',),
            'catalysts': ('bulk',),
            'kinetics_enzymes': ('bulk',),
            'kinetics_substrates': ('bulk',),
            'amino_acids': ('bulk',),
            'listeners': ('listeners',),
            'environment': ('environment',),
            'polypeptide_elongation': ('process_state', 'polypeptide_elongation')}

    # run the process and get an update
    actual_update = run_ecoli_process(metabolism_process, topology, total_time=2)

    # print(actual_update)
    # import ipdb; ipdb.set_trace()

def run_metabolism():
    # Create process, experiment, loading in initial state from file.
    config = load_sim_data.get_metabolism_config()
    metabolism_process = Metabolism(config)

    # topology from ecoli_master
    topology = {
        'metabolites': ('bulk',),
        'catalysts': ('bulk',),
        'kinetics_enzymes': ('bulk',),
        'kinetics_substrates': ('bulk',),
        'amino_acids': ('bulk',),
        'listeners': ('listeners',),
        'environment': ('environment',),
        'polypeptide_elongation': ('process_state', 'polypeptide_elongation')}

    initial_state = get_state_from_file(
        path=f'data/wcecoli_t1000.json')
    # TODO -- add perturbations to initial_state to test impact on metabolism

    metabolism_composite = metabolism_process.generate()
    experiment = Experiment({
        'processes': metabolism_composite['processes'],
        'topology': {metabolism_process.name: topology},
        'initial_state': initial_state
    })

    experiment.update(10)

    data = experiment.emitter.get_data()
    import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    # test_metabolism_migration()
    run_metabolism()

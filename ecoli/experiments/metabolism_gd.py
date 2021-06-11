from vivarium.core.experiment import Experiment
from ecoli.library.sim_data import LoadSimData
from ecoli.composites.ecoli_master import SIM_DATA_PATH
from ecoli.processes.metabolism_gd import MetabolismGD
from ecoli.composites.ecoli_master import get_state_from_file


load_sim_data = LoadSimData(
            sim_data_path=SIM_DATA_PATH,
            seed=0)

# topology from ecoli_master
TOPOLOGY = {
    'metabolites': ('bulk',),
    'catalysts': ('bulk',),
    'kinetics_enzymes': ('bulk',),
    'kinetics_substrates': ('bulk',),
    'amino_acids': ('bulk',),
    'listeners': ('listeners',),
    'environment': ('environment',),
    'polypeptide_elongation': ('process_state', 'polypeptide_elongation')}



def run_metabolism():
    # Create process, experiment, loading in initial state from file.
    config = load_sim_data.get_metabolism_config()
    metabolism_process = MetabolismGD(config)

    initial_state = get_state_from_file(
        path=f'data/wcecoli_t1000.json')

    # TODO -- add perturbations to initial_state to test impact on metabolism

    metabolism_composite = metabolism_process.generate()
    experiment = Experiment({
        'processes': metabolism_composite['processes'],
        'topology': {metabolism_process.name: TOPOLOGY},
        'initial_state': initial_state
    })

    experiment.update(10)

    data = experiment.emitter.get_data()

    import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    run_metabolism()

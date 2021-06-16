
import numpy as np

from vivarium.core.experiment import pf
from ecoli.library.sim_data import LoadSimData
from ecoli.composites.ecoli_master import SIM_DATA_PATH
from ecoli.processes.chromosome_replication import ChromosomeReplication
from ecoli.composites.ecoli_master import get_state_from_file
from ecoli.migration.migration_utils import run_ecoli_process, get_process_state


load_sim_data = LoadSimData(
            sim_data_path=SIM_DATA_PATH,
            seed=0)

topology = {
    # bulk molecules
    'replisome_trimers': ('bulk',),
    'replisome_monomers': ('bulk',),
    'dntps': ('bulk',),
    'ppi': ('bulk',),

    # unique molecules
    'active_replisomes': ('unique', 'active_replisome',),
    'oriCs': ('unique', 'oriC',),
    'chromosome_domains': ('unique', 'chromosome_domain',),
    'full_chromosomes': ('unique', 'full_chromosome',),

    # other
    'listeners': ('listeners',),
    'environment': ('environment',),
}


def test_chromosome_replication_default():
    config = load_sim_data.get_chromosome_replication_config()
    chromosome_replication = ChromosomeReplication(config)

    # run the process and get an update
    actual_update = run_ecoli_process(
        chromosome_replication,
        topology,
        total_time=2,
        initial_time=1000)

    print(actual_update)



def test_initiate_replication():
    config = load_sim_data.get_chromosome_replication_config()
    chromosome_replication = ChromosomeReplication(config)

    # get the initial state
    initial_state = get_state_from_file(
        path=f'data/wcecoli_t1000.json')

    # increase cell_mass to trigger replication initiation
    cell_mass = 2000.0
    initial_state['listeners']['mass']['cell_mass'] = cell_mass

    # get relevant initial state and experiment
    state_before, experiment = get_process_state(chromosome_replication, topology, initial_state)

    # run experiment
    experiment.update(4)
    data = experiment.emitter.get_data()

    # print(f'BEFORE: {pf(state_before)}')
    print(pf(data))
    import ipdb; ipdb.set_trace()



def test_fork_termination():
    config = load_sim_data.get_chromosome_replication_config()

    # change replichore_length parameter to force termination
    config['replichore_lengths'] = np.array([1897318, 1897318])


    chromosome_replication = ChromosomeReplication(config)

    # get the initial state
    initial_state = get_state_from_file(
        path=f'data/wcecoli_t1000.json')

    # # increase cell_mass to trigger replication initiation
    # cell_mass = 2000.0
    # initial_state['listeners']['mass']['cell_mass'] = cell_mass

    # get relevant initial state and experiment
    state_before, experiment = get_process_state(chromosome_replication, topology, initial_state)

    # run experiment
    experiment.update(4)
    data = experiment.emitter.get_data()

    # print(f'BEFORE: {pf(state_before)}')
    print(pf(data))
    import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    # test_chromosome_replication_default()
    # test_initiate_replication()
    test_fork_termination()
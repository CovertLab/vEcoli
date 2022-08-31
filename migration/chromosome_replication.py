import argparse
import json
import pytest

from vivarium.core.engine import pf

from ecoli.library.sim_data import LoadSimData
from ecoli.composites.ecoli_nonpartition import SIM_DATA_PATH
from ecoli.states.wcecoli_state import get_state_from_file
from ecoli.processes.chromosome_replication import ChromosomeReplication

from migration.migration_utils import *

load_sim_data = LoadSimData(
            sim_data_path=SIM_DATA_PATH,
            seed=0)


TOPOLOGY = ChromosomeReplication.topology


@pytest.mark.master
def test_actual_update():
    def test(initial_time):
        # Set time parameters
        total_time = 2
        initial_time = initial_time

        # Create process, experiment, loading in initial state from file.
        config = load_sim_data.get_chromosome_replication_config()
        chromosome_replication_process = ChromosomeReplication(config)

        initial_state = get_state_from_file(
            path=f'data/migration/wcecoli_t{initial_time}_before_layer_0.json')

        # run the process and get an update
        actual_request, actual_update = run_custom_partitioned_process(
            chromosome_replication_process, TOPOLOGY, initial_time = initial_time, 
            initial_state=initial_state, folder_name='chromosome_replication')

        with open(f"data/migration/chromosome_replication/request_t{total_time+initial_time}.json") as f:
            wc_request = json.load(f)
        # Ignore differences in unique IDs
        assert recursive_compare(actual_request, wc_request)
        
        with open(f"data/migration/chromosome_replication/update_t{total_time+initial_time}.json") as f:
            wc_update = json.load(f)
        assert recursive_compare(actual_update, wc_update, ignore_keys={'key'})

    times = [0, 2072]
    for initial_time in times:
        test(initial_time)


@pytest.mark.master
def test_chromosome_replication_default():
    config = load_sim_data.get_chromosome_replication_config()
    chromosome_replication = ChromosomeReplication(config)

    # get the initial state
    initial_state = get_state_from_file(
        path=f'data/wcecoli_t1000.json')

    # get relevant initial state and experiment
    state_before, experiment = get_process_state(chromosome_replication, TOPOLOGY, initial_state)

    # run experiment
    experiment.update(4)
    data = experiment.emitter.get_data()

    print(pf(data))


@pytest.mark.master
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
    state_before, experiment = get_process_state(chromosome_replication, TOPOLOGY, initial_state)

    # run experiment
    experiment.update(4)
    data = experiment.emitter.get_data()

    print(pf(data))

    # TODO -- test whether oriC delete is coming through?
    # data[1.0]['oriC']


@pytest.mark.master
def test_fork_termination():
    config = load_sim_data.get_chromosome_replication_config()

    # change replichore_length parameter to force early termination
    config['replichore_lengths'] = np.array([1897318, 1897318])

    chromosome_replication = ChromosomeReplication(config)

    # get the initial state
    initial_state = get_state_from_file(
        path=f'data/wcecoli_t1000.json')

    # get relevant initial state and experiment
    state_before, experiment = get_process_state(chromosome_replication, TOPOLOGY, initial_state)

    # run experiment
    experiment.update(4)
    data = experiment.emitter.get_data()

    print(pf(data))


test_library = {
    '0': test_actual_update,
    '1': test_chromosome_replication_default,
    '2': test_initiate_replication,
    '3': test_fork_termination,
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='chromosome replication migration')
    parser.add_argument(
        '--name', '-n', default=[], nargs='+', help='test ids to run')
    args = parser.parse_args()
    run_all = not args.name

    for name in args.name:
        test_library[name]()
    if run_all:
        for name, test in test_library.items():
            test()

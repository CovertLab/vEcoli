import argparse
import numpy as np
import pytest

from vivarium.core.engine import pf

from ecoli.library.wcecoli_state import get_state_from_file
from ecoli.processes.chromosome_replication import ChromosomeReplication

from migration.migration_utils import get_process_state, run_and_compare
from migration import LOAD_SIM_DATA


@pytest.mark.master
def test_actual_update():
    times = [0, 1870]
    for initial_time in times:
        run_and_compare(initial_time, ChromosomeReplication, layer=3)
        run_and_compare(initial_time, ChromosomeReplication, layer=3, operons=False)


@pytest.mark.master
def test_fork_termination():
    config = LOAD_SIM_DATA.get_chromosome_replication_config()

    # change replichore_length parameter to force early termination
    config["replichore_lengths"] = np.array([930280, 930280])

    chromosome_replication = ChromosomeReplication(config)

    # get the initial state
    initial_state = get_state_from_file(path="data/migration/wcecoli_t0.json")

    # get relevant initial state and experiment
    state_before, experiment = get_process_state(
        chromosome_replication, ChromosomeReplication.topology, initial_state
    )
    chromosome_replication.calculate_request(2, state_before)
    chromosome_replication.evolve_only = True

    # run experiment
    experiment.update(4)
    data = experiment.emitter.get_data()

    print(pf(data))


test_library = {
    "0": test_actual_update,
    "1": test_fork_termination,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="chromosome replication migration")
    parser.add_argument("--name", "-n", default=[], nargs="+", help="test ids to run")
    args = parser.parse_args()
    run_all = not args.name

    for name in args.name:
        test_library[name]()
    if run_all:
        for name, test in test_library.items():
            test()

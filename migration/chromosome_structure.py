"""
tests that vivarium-ecoli chromosome_structure process update is the same as saved wcEcoli updates
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import os
import pytest

from vivarium.core.engine import Engine
from ecoli.processes.chromosome_structure import ChromosomeStructure
from migration.migration_utils import (ComparisonTestSuite, equal,
                                       equal_len, scalar_equal)
from ecoli.states.wcecoli_state import get_state_from_file

from ecoli.processes.registries import topology_registry
from migration import load_sim_data



CHROMOSOME_STRUCTURE_TOPOLOGY = topology_registry.access(ChromosomeStructure.name)

def custom_run_process(
    process,
    topology,
    total_time=2,
    initial_time=0,
    initial_state=None,
):
    # make an experiment
    experiment_config = {
        'processes': {process.name: process},
        'topology': {process.name: topology},
        'initial_state': initial_state}
    experiment = Engine(**experiment_config)

    # Get update from process.
    path = (process.name,)
    store = experiment.state.get_path(path)

    # translate the values from the tree structure into the form
    # that this process expects, based on its declared topology
    states = store.outer.schema_topology(process.schema, store.topology)

    # Make process see partitioned molecule counts
    with open(f"data/chromosome_structure/chromosome_structure_partitioned_t"
              f"{total_time+initial_time}.json") as f:
        partitioned_counts = json.load(f)

    states = partitioned_counts

    update = experiment._invoke_process(
        process,
        total_time,
        states)

    actual_update = update.get_command_result()
    return actual_update

@pytest.mark.master
def test_chromosome_structure_migration():
    def test(initial_time):
        # Set time parameters
        total_time = 2
        initial_time = initial_time

        # Create process, experiment, loading in initial state from file.
        config = load_sim_data.get_chromosome_structure_config()
        config['seed'] = 0
        chromosome_structure_process = ChromosomeStructure(config)

        initial_state = get_state_from_file(
            path=f'data/chromosome_structure/wcecoli_t{initial_time}.json')

        # run the process and get an update
        actual_update = custom_run_process(chromosome_structure_process, CHROMOSOME_STRUCTURE_TOPOLOGY,
                                        total_time=total_time, initial_time = initial_time,
                                        initial_state=initial_state)

        with open(f"data/chromosome_structure/chromosome_structure_update_t{total_time+initial_time}.json") as f:
            wc_update = json.load(f)

        plots(actual_update, wc_update, total_time+initial_time)
        assertions(actual_update,wc_update, total_time+initial_time)

    times = [0, 2, 10, 100]
    for initial_time in times:
        test(initial_time)

def plots(actual_update, expected_update, time):
    os.makedirs("out/migration/chromosome_structure/", exist_ok=True)
    def unpack(update):
        fragmentBases = update['fragmentBases']
        molecules = update['molecules']
        active_tfs = update['active_tfs']
        subunits = update['subunits']
        amino_acids = update['amino_acids']

        return (fragmentBases, molecules, active_tfs, subunits, amino_acids)

    # unpack updates
    actual_unpacked = unpack(actual_update)

    wc_unpacked = unpack(expected_update)

    # Plots ============================================================================
    data_to_plot = list(zip(actual_unpacked, wc_unpacked))
    plots_to_make = [True if data[0] else False for data in data_to_plot]
    total_plots = sum(plots_to_make)
    rows = int(np.ceil(total_plots/2))
    titles = ['Fragment Base Deltas', 'Molecule Deltas', 'Active TF Deltas',
              'Subunit Deltas', 'Amino Acid Deltas']
    index = 0
    for data_idx, data in enumerate(data_to_plot):
        if data[0]:
            plt.subplot(rows, 2, index+1)
            plt.bar(np.arange(len(data[0]))-0.1, data[0].values(), 0.2, label = "Vivarium")
            plt.bar(np.arange(len(data[1]))+0.1, data[1].values(), 0.2, label = "wcEcoli")
            plt.xticks(ticks = np.arange(len(data[0])), labels = data[0].keys(), rotation = 90)
            plt.title(titles[data_idx])
            plt.legend()
            index += 1

    plt.gcf().set_size_inches(16, 12)
    plt.tight_layout()
    plt.savefig(f"out/migration/chromosome_structure/chromosome_structure_figures{time}.png")
    plt.close()

def assertions(actual_update, expected_update, time):
    test_structure = {
        'fragmentBases' : {
            fragmentBase : scalar_equal
        for fragmentBase in actual_update['fragmentBases'].keys()},
        'molecules' : {
            molecule : scalar_equal
        for molecule in actual_update['molecules'].keys()},
        'active_tfs' : {
            active_tf : scalar_equal
        for active_tf in actual_update['active_tfs'].keys()},
        'subunits' : {
            subunit : scalar_equal
        for subunit in actual_update['subunits'].keys()},
        'amino_acids' : {
            amino_acid : scalar_equal
        for amino_acid in actual_update['amino_acids'].keys()}}
    unique_molecules = [
        'active_replisomes', 'oriCs', 'chromosome_domains',
        'active_RNAPs', 'RNAs', 'active_ribosome', 'full_chromosomes',
        'promoters', 'DnaA_boxes']
    for unique_molecule in unique_molecules:
        test_structure[unique_molecule] = {}
        if actual_update[unique_molecule]:
            if actual_update[unique_molecule].get('_add'):
                actual_update[unique_molecule]['_add'] = {
                    index: new_molecule
                    for index, new_molecule in enumerate(
                        actual_update[unique_molecule]['_add'])}
                expected_update[unique_molecule]['_add'] = {
                    index: new_molecule
                    for index, new_molecule in enumerate(
                        expected_update[unique_molecule]['_add'])}
                test_structure[unique_molecule]['_add'] = {
                    idx: {
                        'state': {
                            key: equal
                            for key in new['state'].keys()
                            if key != 'unique_index'}
                    }
                    for idx, new in actual_update[unique_molecule]['_add'].items()
                }
            if actual_update[unique_molecule].get('_delete'):
                test_structure[unique_molecule]['_delete'] = equal_len

    tests = ComparisonTestSuite(test_structure, fail_loudly=False)
    tests.run_tests(actual_update, expected_update, verbose=True)

    tests.fail()

if __name__ == "__main__":
    test_chromosome_structure_migration()

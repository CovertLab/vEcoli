import argparse
import json

import ipdb
import matplotlib.pyplot as plt
import os

from vivarium.core.engine import pf
from ecoli.library.sim_data import LoadSimData
from ecoli.composites.ecoli_master import SIM_DATA_PATH
from ecoli.processes.chromosome_replication import ChromosomeReplication
from migration.migration_utils import *

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


def test_actual_update():
    config = load_sim_data.get_chromosome_replication_config()
    chromosome_replication = ChromosomeReplication(config)
    total_time = 2
    initial_time = 3142

    # run the process and get an update
    actual_update = run_ecoli_process(
        chromosome_replication,
        topology,
        total_time=total_time,
        initial_time=initial_time)

    with open(f"data/chromosome_replication_update_t{initial_time + total_time}.json") as f:
        wc_update = json.load(f)

    print(pf(actual_update))
    plots(actual_update, wc_update, total_time + initial_time)
    assertions(actual_update, wc_update)


def plots(actual_update, expected_update, time):
    os.makedirs("out/migration/chromosome_replication/", exist_ok=True)

    # Plots ============================================================================

    for key in actual_update:

        if key == 'replisome_trimers':
            plt.subplot(3, 3, 1)
            plt.bar(np.arange(len(actual_update[key].keys())) - 0.1, actual_update[key].values(), 0.2, label="Vivarium")
            plt.bar(np.arange(len(expected_update[key].keys())) + 0.1, expected_update[key].values(), 0.2,
                    label="wcEcoli")
            plt.xticks(ticks=np.arange(len(actual_update[key].keys())), labels=actual_update[key].keys(), rotation=90)
            plt.ylabel('Change in Replisome Trimers')
            plt.title('Replisome Trimer Deltas')
            plt.legend()

        if key == 'replisome_monomers':
            plt.subplot(3, 3, 2)
            plt.bar(np.arange(len(actual_update[key].keys())) - 0.1, actual_update[key].values(), 0.2, label="Vivarium")
            plt.bar(np.arange(len(expected_update[key].keys())) + 0.1, expected_update[key].values(), 0.2,
                    label="wcEcoli")
            plt.xticks(ticks=np.arange(len(actual_update[key].keys())), labels=actual_update[key].keys(), rotation=90)
            plt.ylabel('Change in Replisome Monomers')
            plt.title('Replisome Monomer Deltas')
            plt.legend()

        if key == 'active_replisomes':
            plt.subplot(3, 3, 3)
            actual_coords = []
            expected_coords = []
            actual_dna_mass = []
            expected_dna_mass = []
            rep_labels = []
            for unique_index in actual_update[key]:
                if unique_index != '_add' and unique_index != '_delete':
                    actual_coords.append(actual_update[key][unique_index]['coordinates'])
                    expected_coords.append(expected_update[key][unique_index]['coordinates'])
                    actual_dna_mass.append(actual_update[key][unique_index]['dna_mass'])
                    expected_dna_mass.append(actual_update[key][unique_index]['dna_mass'])
                    rep_labels.append(unique_index)

            plt.bar(np.arange(len(actual_coords)) - 0.1, actual_coords, 0.2,
                    label="Vivarium")
            plt.bar(np.arange(len(expected_coords)) + 0.1, expected_coords, 0.2,
                    label="wcEcoli")
            plt.xticks(ticks=np.arange(len(rep_labels)), labels=rep_labels, rotation=90)
            plt.ylabel('Change in Active Replisomes Coordinates')
            plt.title('Active Replisomes Coordinates Deltas')
            plt.legend()

            plt.subplot(3, 3, 4)
            plt.bar(np.arange(len(actual_dna_mass)) - 0.1, actual_dna_mass, 0.2, label="Vivarium")
            plt.bar(np.arange(len(expected_dna_mass)) + 0.1, expected_dna_mass, 0.2,
                    label="wcEcoli")
            plt.xticks(ticks=np.arange(len(rep_labels)),
                       labels=rep_labels, rotation=90)
            plt.ylabel('Change in Active Replisomes DNA Mass')
            plt.title('Active Replisomes DNA Mass Deltas')
            plt.legend()

        if key == 'dntps':
            plt.subplot(3, 3, 5)
            plt.bar(np.arange(len(actual_update[key].keys())) - 0.1, actual_update[key].values(), 0.2, label="Vivarium")
            plt.bar(np.arange(len(expected_update[key].keys())) + 0.1, expected_update[key].values(), 0.2,
                    label="wcEcoli")
            plt.xticks(ticks=np.arange(len(actual_update[key].keys())), labels=actual_update[key].keys(), rotation=90)
            plt.ylabel('Change in DNTPs')
            plt.title('DNTPs Deltas')
            plt.legend()

    plt.gcf().set_size_inches(16, 12)
    plt.tight_layout()
    plt.savefig(f"out/migration/chromosome_replication/chromosome_replication_figures_t{time}.png")


def assertions(actual_update, expected_update):
    test_structure = {
        'replisome_trimers': {key: scalar_equal for key in actual_update['replisome_trimers'].keys()},
        'replisome_monomers': {key: scalar_equal for key in actual_update['replisome_monomers'].keys()},
        'listeners': {
            'replication_data': {},
        }
    }
    if 'criticalMassPerOriC' in actual_update['listeners']['replication_data']:
        test_structure['listeners']['replication_data']['criticalMassPerOriC'] = scalar_equal
    if 'criticalInitiationMass' in actual_update['listeners']['replication_data']:
        test_structure['listeners']['replication_data']['criticalInitiationMass'] = scalar_equal
    if 'dntps' in actual_update:
        test_structure['dntps'] = {key: scalar_equal for key in actual_update['dntps'].keys()}
    if 'ppi' in actual_update:
        test_structure['ppi'] = {key: scalar_equal for key in actual_update['ppi'].keys()}
    tests = ComparisonTestSuite(test_structure, fail_loudly=False)
    tests.run_tests(actual_update, expected_update, verbose=True)

    def compare_dict_data(actual_dicts, expected_dicts, comparison, keys, data_type=None):
        actual_data = []
        expected_data = []
        for i in range(len(actual_dicts)):
            actual_value = actual_dicts[i]
            for key in keys:
                actual_value = actual_value[key]
            if data_type:
                actual_value = data_type(actual_value)
            actual_data.append(actual_value)
        for j in range(len(expected_dicts)):
            expected_value = expected_dicts[j]
            for key in keys:
                expected_value = expected_value[key]
            if data_type:
                expected_value = data_type(expected_value)
            expected_data.append(expected_value)
        assert comparison(np.array(actual_data), expected_data)

    if '_delete' in actual_update['active_replisomes']:
        assert scalar_equal(len(actual_update['active_replisomes']['_delete']),
                            len(expected_update['active_replisomes']['_delete']))
    if '_add' in actual_update['active_replisomes']:
        assert scalar_equal(len(actual_update['active_replisomes']['_add']),
                            len(expected_update['active_replisomes']['_add']))
        compare_dict_data(actual_update['active_replisomes']['_add'], expected_update['active_replisomes']['_add'],
                          array_equal, ['state', 'coordinates'])
        compare_dict_data(actual_update['active_replisomes']['_add'], expected_update['active_replisomes']['_add'],
                          array_equal, ['state', 'right_replichore'], int)

    def compare_index_data(actual_dict, expected_dict, comparison, data_key, data_type):
        actual_data = []
        expected_data = []
        for key in actual_dict:
            if key != '_add' and key != '_delete':
                actual_data.append(data_type(actual_dict[key][data_key]))
        for key in expected_dict:
            if key != '_add' and key != '_delete':
                expected_data.append(data_type(actual_dict[key][data_key]))
        if actual_data and expected_data:
            assert comparison(np.array(actual_data), expected_data)

    compare_index_data(actual_update['active_replisomes'], expected_update['active_replisomes'],
                       array_almost_equal, 'coordinates', int)
    compare_index_data(actual_update['active_replisomes'], expected_update['active_replisomes'],
                       array_equal, 'dna_mass', float)

    if 'oriCs' in actual_update:
        assert scalar_equal(len(actual_update['oriCs']['_delete']), len(expected_update['oriCs']['_delete']))
        assert scalar_equal(len(actual_update['oriCs']['_add']), len(actual_update['oriCs']['_add']))

    if 'full_chromosomes' in actual_update:
        num_chromosome_keys = 0
        for key in actual_update['full_chromosomes']:
            if key != '_add':
                num_chromosome_keys += 1
        num_wc_chromosome_keys = 0
        for key in expected_update['full_chromosomes']:
            if key != '_add':
                num_wc_chromosome_keys += 1
        assert scalar_equal(num_chromosome_keys, num_wc_chromosome_keys)
        if '_add' in actual_update['full_chromosomes']:
            compare_dict_data(actual_update['full_chromosomes']['_add'], expected_update['full_chromosomes']['_add'],
                              array_equal, ['state', 'division_time'])
            compare_dict_data(actual_update['full_chromosomes']['_add'], expected_update['full_chromosomes']['_add'],
                              scalar_equal, ['state', 'has_triggered_division'], int)

    if 'chromosome_domains' in actual_update:
        # TODO: use compare_index_data
        compare_index_data(actual_update['active_replisomes'], expected_update['active_replisomes'],
                           array_almost_equal, 'coordinates', int)
        chromo_dom_keys = []
        chromo_dom_children = []
        for key in actual_update['chromosome_domains']:
            if key != '_add':
                chromo_dom_keys.append(int(key))
                for i in range(len(actual_update['chromosome_domains'][key]['child_domains'])):
                    chromo_dom_children.append(int(actual_update['chromosome_domains'][key]['child_domains'][i]))
        wc_chromo_dom_keys = []
        wc_chromo_dom_children = []
        for key in expected_update['chromosome_domains']:
            if key != '_add':
                wc_chromo_dom_keys.append(int(key))
                for i in range(len(expected_update['chromosome_domains'][key]['child_domains'])):
                    wc_chromo_dom_children.append(int(expected_update['chromosome_domains'][key]['child_domains'][i]))
        assert array_equal(np.array(chromo_dom_keys), wc_chromo_dom_keys)
        assert array_equal(np.array(chromo_dom_children), wc_chromo_dom_children)

        def create_child_list(add_dict):
            total_child_domains = []
            for i in range(len(add_dict)):
                child_domains = add_dict[i]['state']['child_domains']
                for j in range(len(child_domains)):
                    total_child_domains.append(int(child_domains[j]))
            return total_child_domains

        assert array_equal(np.array(create_child_list(actual_update['chromosome_domains']['_add'])),
                           create_child_list(expected_update['chromosome_domains']['_add']))


def test_chromosome_replication_default():
    config = load_sim_data.get_chromosome_replication_config()
    chromosome_replication = ChromosomeReplication(config)

    # get the initial state
    initial_state = get_state_from_file(
        path=f'data/wcecoli_t1000.json')

    # get relevant initial state and experiment
    state_before, experiment = get_process_state(chromosome_replication, topology, initial_state)

    # run experiment
    experiment.update(4)
    data = experiment.emitter.get_data()

    print(pf(data))


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

    print(pf(data))

    # TODO -- test whether oriC delete is coming through?
    # data[1.0]['oriC']


def test_fork_termination():
    config = load_sim_data.get_chromosome_replication_config()

    # change replichore_length parameter to force early termination
    config['replichore_lengths'] = np.array([1897318, 1897318])

    chromosome_replication = ChromosomeReplication(config)

    # get the initial state
    initial_state = get_state_from_file(
        path=f'data/wcecoli_t1000.json')

    # get relevant initial state and experiment
    state_before, experiment = get_process_state(chromosome_replication, topology, initial_state)

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

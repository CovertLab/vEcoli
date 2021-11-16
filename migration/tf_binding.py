"""
tests that vivarium-ecoli tf_binding process update is the same as saved wcEcoli updates
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from migration.plots import qqplot
import os

from vivarium.core.engine import Engine, view_values
from ecoli.library.sim_data import LoadSimData
from ecoli.composites.ecoli_nonpartition import SIM_DATA_PATH
from ecoli.processes.tf_binding import TfBinding
from migration.migration_utils import (ComparisonTestSuite, array_equal, 
                                       array_diffs_report_test, scalar_equal)
from ecoli.states.wcecoli_state import get_state_from_file

from ecoli.processes.registries import topology_registry


load_sim_data = LoadSimData(
            sim_data_path=SIM_DATA_PATH,
            seed=0)

TF_BINDING_TOPOLOGY = TfBinding.topology

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
    path, process = list(experiment.process_paths.items())[0]
    store = experiment.state.get_path(path)

    # translate the values from the tree structure into the form
    # that this process expects, based on its declared topology
    topology_view = store.outer.schema_topology(process.schema, store.topology)
    states = view_values(topology_view)

    # Make process see partitioned molecule counts
    with open(f"data/tf_binding/tf_binding_partitioned_t"
              f"{total_time+initial_time}.json") as f:
        partitioned_counts = json.load(f)
    
    states['active_tfs'] = partitioned_counts['bulk']
    
    update = experiment.invoke_process(
        process,
        path,
        total_time,
        states)

    actual_update = update.get()
    return actual_update

def test_tf_binding_migration():
    def test(initial_time):
        # Set time parameters
        total_time = 2
        initial_time = initial_time

        # Create process, experiment, loading in initial state from file.
        config = load_sim_data.get_tf_config()
        config['seed'] = 0
        tf_binding_process = TfBinding(config)
                
        initial_state = get_state_from_file(
            path=f'data/tf_binding/wcecoli_t{initial_time}.json')

        # run the process and get an update
        actual_update = custom_run_process(tf_binding_process, TF_BINDING_TOPOLOGY, 
                                        total_time=total_time, initial_time = initial_time,
                                        initial_state=initial_state)

        with open(f"data/tf_binding/tf_binding_update_t{total_time+initial_time}.json") as f:
            wc_update = json.load(f)

        plots(actual_update, wc_update, total_time+initial_time)
        assertions(actual_update,wc_update, total_time+initial_time)

    times = [0, 2, 100]
    for initial_time in times:
        test(initial_time)

def plots(actual_update, expected_update, time):
    os.makedirs("out/migration/tf_binding/", exist_ok=True)
    def unpack(update):
        promoters = [promoter for promoter in update['promoters']]
        active_tfs = update['active_tfs']
        pPromoterBound = update['listeners']['rna_synth_prob']['pPromoterBound']
        nPromoterBound = update['listeners']['rna_synth_prob']['nPromoterBound']
        nAcutalBound = update['listeners']['rna_synth_prob']['nPromoterBound']
        n_available_promoters = update['listeners']['rna_synth_prob']['n_available_promoters']
        n_bound_TF_per_TU = update['listeners']['rna_synth_prob']['n_bound_TF_per_TU']
        
        return (promoters, active_tfs, pPromoterBound, nPromoterBound, nAcutalBound, 
                n_available_promoters, n_bound_TF_per_TU)

    # unpack updates
    (promoters, active_tfs, pPromoterBound, nPromoterBound, nActualBound, 
        n_available_promoters, n_bound_TF_per_TU) = unpack(actual_update)

    (wc_promoters, wc_active_tfs, wc_pPromoterBound, wc_nPromoterBound, wc_nActualBound, 
        wc_n_available_promoters, wc_n_bound_TF_per_TU) = unpack(expected_update)

    # Plots ============================================================================

    plt.subplot(3, 2, 1)
    plt.bar(np.arange(24)-0.1, active_tfs.values(), 0.2, label = "Vivarium")
    plt.bar(np.arange(24)+0.1, wc_active_tfs.values(), 0.2, label = "wcEcoli")
    plt.xticks(ticks = np.arange(24), labels = active_tfs.keys(), rotation = 90)
    plt.ylabel('Change in Active TFs')
    plt.title('Active TF Deltas')
    plt.legend()

    plt.subplot(3, 2, 2)
    qqplot(pPromoterBound, wc_pPromoterBound)
    plt.ylabel('wcEcoli')
    plt.xlabel('Vivarium')
    plt.title('Q-Q Plot of pPromoterBound')

    plt.subplot(3, 2, 3)
    qqplot(nPromoterBound, wc_nPromoterBound)
    plt.ylabel('wcEcoli')
    plt.xlabel('Vivarium')
    plt.title('Q-Q Plot of nPromoterBound')

    plt.subplot(3, 2, 4)
    qqplot(nActualBound, wc_nActualBound)
    plt.ylabel('wcEcoli')
    plt.xlabel('Vivarium')
    plt.title('Q-Q Plot of nActualBound')

    plt.subplot(3, 2, 5)
    qqplot(n_available_promoters, wc_n_available_promoters)
    plt.ylabel('wcEcoli')
    plt.xlabel('Vivarium')
    plt.title('Q-Q Plot of n_available_promoters')

    plt.gcf().set_size_inches(16, 12)
    plt.tight_layout()
    plt.savefig(f"out/migration/tf_binding/tf_binding_figures{time}.png")
    plt.close()

def assertions(actual_update, expected_update, time):
    test_structure = {
        'active_tfs' : {
            tf_index : scalar_equal
        for tf_index in actual_update['active_tfs'].keys()},
        'listeners' : {
            'rna_synth_prob' : {
                'pPromoterBound' : [
                    array_equal,
                    array_diffs_report_test(
                        f"out/migration/tf_binding/pPromoterBound_comp{time}.txt")],
                'nPromoterBound' : [
                    array_equal,
                    array_diffs_report_test(
                        f"out/migration/tf_binding/nPromoterBound_comp{time}.txt")],
                'nActualBound' : [
                    array_equal,
                    array_diffs_report_test(
                        f"out/migration/tf_binding/nActualBound_comp{time}.txt")],
                'n_available_promoters' : [
                    array_equal,
                    array_diffs_report_test(
                        f"out/migration/tf_binding/n_available_promoters_comp{time}.txt")]
            }
        }
    }

    tests = ComparisonTestSuite(test_structure, fail_loudly=False)
    tests.run_tests(actual_update, expected_update, verbose=True)

    tests.fail()
    
    # Sanity checks for randomly sampled TF-promoter binding

    bound_TF = np.array([promoter['bound_TF'] for promoter in actual_update['promoters'].values()])
    bound_tf_totals = np.sum(bound_TF, axis=0)
    wc_bound_TF = np.array([promoter['bound_TF'] for promoter in actual_update['promoters'].values()])
    wc_bound_tf_totals = np.sum(wc_bound_TF, axis=0)

    assert array_equal(bound_tf_totals, wc_bound_tf_totals), "Counts of bound TFs not consistent!"

    bound_TF_submass = np.array([promoter['submass'] for promoter in actual_update['promoters'].values()])
    bound_submass_per_TF = np.sum(bound_TF_submass, axis=0)
    wc_bound_TF_submass = np.array([promoter['submass'] for promoter in actual_update['promoters'].values()])
    wc_bound_submass_per_TF = np.sum(wc_bound_TF_submass, axis=0)

    assert array_equal(bound_submass_per_TF, wc_bound_submass_per_TF), "Sums of bound TF submasses not consistent!"

    n_bound_TF_per_TU = actual_update['listeners']['rna_synth_prob']['n_bound_TF_per_TU']
    n_bound_TU_per_TF = np.sum(n_bound_TF_per_TU, axis=0)
    wc_n_bound_TF_per_TU = np.array(expected_update['listeners']['rna_synth_prob']['n_bound_TF_per_TU'])
    wc_n_bound_TU_per_TF = np.sum(wc_n_bound_TF_per_TU, axis=0)
    

    assert array_equal(n_bound_TU_per_TF, wc_n_bound_TU_per_TF), "Counts of bound TUs per TF not consistent!"
        
def run_tf_binding():
    # Create process, experiment, loading in initial state from file.
    config = load_sim_data.get_tf_config()
    tf_binding_process = TfBinding(config)

    initial_state = get_state_from_file(
        path=f'data/wcecoli_t0.json')

    tf_binding_composite = tf_binding_process.generate()
    experiment = Engine(**{
        'processes': tf_binding_composite['processes'],
        'topology': {tf_binding_process.name: TF_BINDING_TOPOLOGY},
        'initial_state': initial_state
    })

    experiment.update(10)

    data = experiment.emitter.get_data()


if __name__ == "__main__":
    test_tf_binding_migration()
    # run_tf_binding()

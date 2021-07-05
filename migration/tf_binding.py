"""
tests that vivarium-ecoli tf_binding processupdate is the same as saved wcEcoli updates
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from migration.plots import qqplot

from vivarium.core.engine import Engine
from ecoli.library.sim_data import LoadSimData
from ecoli.composites.ecoli_master import SIM_DATA_PATH
from ecoli.processes.tf_binding import TfBinding
from migration.migration_utils import *
from ecoli.composites.ecoli_master import get_state_from_file


load_sim_data = LoadSimData(
            sim_data_path=SIM_DATA_PATH,
            seed=0)

# copy topology from ecoli_master.py, under generate_topology
TF_BINDING_TOPOLOGY = {
    'promoters': ('unique', 'promoter'),
    'active_tfs': ('bulk',),
    'inactive_tfs': ('bulk',),
    'listeners': ('listeners',)}


def test_tf_binding_migration():
    # Set time parameters
    total_time = 2
    initial_time = 100
    
    # Create process, experiment, loading in initial state from file.
    config = load_sim_data.get_tf_config()
    tf_binding_process = TfBinding(config)

    # run the process and get an update
    actual_update = run_ecoli_process(tf_binding_process, TF_BINDING_TOPOLOGY, 
                                      total_time=total_time, initial_time = initial_time)

    with open(f"data/tf_binding_update_t{total_time+initial_time}.json") as f:
        wc_update = json.load(f)
    
    plots(actual_update, wc_update, total_time+initial_time)
    assertions(actual_update,wc_update, total_time+initial_time)

def plots(actual_update, expected_update, time):
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
    plt.ylabel('Active TFs')
    plt.title('Active TF Counts')
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
    
    #plt.subplot(3, 2, 6)
    #qqplot(n_bound_TF_per_TU, wc_n_bound_TF_per_TU)
    
    plt.gcf().set_size_inches(16, 12)
    plt.tight_layout()
    plt.savefig(f"out/migration/tf_binding/tf_binding_figures{time}.png")

def assertions(actual_update, expected_update, time):
    test_structure = {
        'active_tfs' : {
            tf_index : scalar_almost_equal
        for tf_index in actual_update['active_tfs'].keys()},
        'listeners' : {
            'rna_synth_prob' : {
                'pPromoterBound' : [array_almost_equal,
                                    array_diffs_report_test(f"out/migration/tf_binding/pPromoterBound_comp{time}.txt")],
                'nPromoterBound' : [array_almost_equal,
                                    array_diffs_report_test(f"out/migration/tf_binding/nPromoterBound_comp{time}.txt")],
                'nActualBound' : [array_almost_equal,
                                array_diffs_report_test(f"out/migration/tf_binding/nActualBound_comp{time}.txt")],
                'n_available_promoters' : [array_almost_equal,
                                        array_diffs_report_test(f"out/migration/tf_binding/n_available_promoters_comp{time}.txt")]
            }
        }
    }

    tests = ComparisonTestSuite(test_structure, fail_loudly=False)
    tests.run_tests(actual_update, expected_update, verbose=True)

    #tests.fail()
    
    def stochastic_equal_array(arr1, arr2):
        u_stats, p_vals = mannwhitneyu(arr1, arr2)
        return p_vals > PVALUE_THRESHOLD, f"Mann-Whitney U, (U,p) = {list(zip(u_stats, [round(p, 4) for p in p_vals]))}"
    
    # Test similarity of underlying distributions for randomly sampled promoter binding
    bound_TF = np.array([promoter['bound_TF'] for promoter in actual_update['promoters'].values()])
    bound_promoters = np.transpose(bound_TF)
    wc_bound_TF = np.array([promoter['bound_TF'] for promoter in actual_update['promoters'].values()])
    wc_bound_promoters = np.transpose(wc_bound_TF)
    
    assert all(stochastic_equal_array(bound_promoters, wc_bound_promoters)[0]), "Distributions of bound promoters are not similar!"
    
    # Test similarity of underlying TF submass distributions for randomly sampled promoter binding
    bound_TF_submass = np.array([promoter['submass'] for promoter in actual_update['promoters'].values()])
    bound_promoters_submass = np.transpose(bound_TF_submass)
    wc_bound_TF_submass = np.array([promoter['submass'] for promoter in actual_update['promoters'].values()])
    wc_bound_promoters_submass = np.transpose(wc_bound_TF_submass)
    
    assert all(stochastic_equal_array(bound_promoters_submass, wc_bound_promoters_submass)[0]), "Distributions of bound TF submasses are not similar!"
    
    # Test similarity of underlying TF-to-TU distributions for randomly sampled promoter binding
    n_bound_TF_per_TU = actual_update['listeners']['rna_synth_prob']['n_bound_TF_per_TU']
    n_bound_TU_per_TF = np.transpose(n_bound_TF_per_TU)
    wc_n_bound_TF_per_TU = expected_update['listeners']['rna_synth_prob']['n_bound_TF_per_TU']
    wc_n_bound_TU_per_TF = np.transpose(wc_n_bound_TF_per_TU)
    
    assert all(stochastic_equal_array(n_bound_TU_per_TF, wc_n_bound_TU_per_TF)[0]), "Distrubutions of bound TUs are not similar!"
    
def run_tf_binding():
    # Create process, experiment, loading in initial state from file.
    config = load_sim_data.get_tf_config()
    tf_binding_process = TfBinding(config)

    initial_state = get_state_from_file(
        path=f'data/wcecoli_t0.json')

    tf_binding_composite = tf_binding_process.generate()
    experiment = Engine({
        'processes': tf_binding_composite['processes'],
        'topology': {tf_binding_process.name: TF_BINDING_TOPOLOGY},
        'initial_state': initial_state
    })

    experiment.update(10)

    data = experiment.emitter.get_data()


if __name__ == "__main__":
    test_tf_binding_migration()
    # run_tf_binding()

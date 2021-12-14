"""
Metabolism process migration tests
"""
import json
import os
import numpy as np
from matplotlib import pyplot as plt
from migration.plots import qqplot

# vivarium imports
from vivarium.core.engine import Engine
from vivarium.core.composer import Composer
from vivarium.library.dict_utils import deep_merge
from vivarium.core.control import run_library_cli

# ecoli imports
from ecoli.library.sim_data import LoadSimData
from ecoli.states.wcecoli_state import get_state_from_file
from ecoli.composites.ecoli_nonpartition import SIM_DATA_PATH, AA_MEDIA_ID
from ecoli.processes import Metabolism, Exchange
from ecoli.library.schema import array_from

# migration imports
from migration.migration_utils import (run_ecoli_process, ComparisonTestSuite,
                                       scalar_almost_equal, transform_and_run, 
                                       array_diffs_report_test)
from migration import load_sim_data


TOPOLOGY = Metabolism.topology

# make a composite with Exchange
class MetabolismExchange(Composer):
    defaults = {
        'metabolism': {},
        'exchanges': {},  # dict with {molecule: exchange rate}
        'sim_data_path': SIM_DATA_PATH,
        'seed': 0,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.load_sim_data = load_sim_data

    def generate_processes(self, config):

        # configure metabolism
        metabolism_config = self.load_sim_data.get_metabolism_gd_config()
        metabolism_config = deep_merge(metabolism_config, self.config['metabolism'])
        metabolism_process = Metabolism(metabolism_config)

        # configure exchanger stub process
        exchanger_process = Exchange({'exchanges': config['exchanges']})

        return {
            'metabolism': metabolism_process,
            'exchange': exchanger_process,
        }

    def generate_topology(self, config):
        return {
            'metabolism': TOPOLOGY,
            'exchange': {
                'molecules': ('bulk',),
            }
        }



def test_metabolism_migration():
    # Create process, experiment, loading in initial state from file.
    config = load_sim_data.get_metabolism_config()
    metabolism_process = Metabolism(config)

    # run the process and get an update
    actual_update = run_ecoli_process(
        metabolism_process,
        TOPOLOGY,
        total_time=2)


def run_metabolism(
        total_time=10,
        initial_time=0,
        config=None,
        initial_state=None,
):
    # get parameters from sim data
    metabolism_config = load_sim_data.get_metabolism_config()
    if config:
        metabolism_config = deep_merge(metabolism_config, config)

    # initialize Metabolism
    metabolism = Metabolism(metabolism_config)

    # get initial state from file
    state = get_state_from_file(
        path=f'data/wcecoli_t{initial_time}.json')
    if initial_state:
        state = deep_merge(state, initial_state)

    # initialize a simulation
    metabolism_composite = metabolism.generate()
    simulation = Engine(**{
        'processes': metabolism_composite['processes'],
        'topology': {metabolism.name: metabolism.topology},
        'initial_state': state
    })

    # run the simulation
    simulation.update(total_time)

    # get data
    data = simulation.emitter.get_data()
    return data


def test_metabolism():
    def test(initial_time=0):
        initial_time = initial_time
        # get parameters from sim data
        metabolism_config = load_sim_data.get_metabolism_config()
        
        # initialize Metabolism
        metabolism = Metabolism(metabolism_config)

        # get initial state from file
        state = get_state_from_file(
            path=f'data/wcecoli_t{initial_time}.json')
        # get partitioned molecule counts from file
        with open('data/metabolism/metabolism_partitioned_'
                  f't{initial_time+2}.json') as f:
            partitioned = json.load(f)
        
        deep_merge(state, partitioned)

        # run the process and get an update
        actual_update = run_ecoli_process(metabolism, metabolism.topology,
                                        initial_time=initial_time,
                                        initial_state=state)
        
        with open('data/metabolism/metabolism_update_'
                  f't{initial_time+2}.json') as f:
            expected_update = json.load(f)
        
        plots(actual_update, expected_update, initial_time+2)
        assertions(actual_update, expected_update, initial_time+2)
    
    os.makedirs('out/migration/metabolism/', exist_ok=True)
    initial_times = [0, 2, 100]
    for time in initial_times:
        test(time)
    
def plots(actual_update, expected_update, time):
    os.makedirs("out/migration/metabolism/", exist_ok=True)
    def unpack(update):
        return {
            'environment_exhange': array_from(update['environment']['exchange']),
            'conc_updates': update['listeners']['fba_results']['conc_updates'],
            'catalyst_counts': update['listeners']['fba_results']['catalyst_counts'],
            'delta_metabolites': update['listeners']['fba_results']['deltaMetabolites'],
            'reaction_fluxes': update['listeners']['fba_results']['reactionFluxes'],
            'external_exchange_fluxes': update['listeners']['fba_results']['externalExchangeFluxes'],
            'shadow_prices': update['listeners']['fba_results']['shadowPrices'],
            'reduced_costs': update['listeners']['fba_results']['reducedCosts'],
            'target_concentrations': update['listeners']['fba_results']['targetConcentrations'],
            'homeostatic_objective_values': update['listeners']['fba_results']['homeostaticObjectiveValues'],
            'kinetic_objective_values': update['listeners']['fba_results']['kineticObjectiveValues'],
            'metabolite_counts_init': update['listeners']['enzyme_kinetics']['metaboliteCountsInit'],
            'metabolite_counts_final': update['listeners']['enzyme_kinetics']['metaboliteCountsFinal'],
            'enzyme_counts_init': update['listeners']['enzyme_kinetics']['enzymeCountsInit'],
            'actual_fluxes': update['listeners']['enzyme_kinetics']['actualFluxes'],
            'target_fluxes': update['listeners']['enzyme_kinetics']['targetFluxes'],
            'target_fluxes_upper': update['listeners']['enzyme_kinetics']['targetFluxesUpper'],
            'target_fluxes_lower': update['listeners']['enzyme_kinetics']['targetFluxesLower'],
        }

    # unpack updates
    actual_update = unpack(actual_update)

    expected_update = unpack(expected_update)

    # Plots ============================================================================
    plot_num = 0
    for key, dist in actual_update.items():
        plot_num += 1
        plt.subplot(6, 3, plot_num)
        qqplot(dist, expected_update[key])
        plt.ylabel('wcEcoli')
        plt.xlabel('Vivarium')
        plt.title(key)

    plt.gcf().set_size_inches(16, 12)
    plt.tight_layout()
    plt.savefig(f"out/migration/metabolism/metabolism_figures{time}.png")
    plt.close()

def assertions(actual_update, expected_update, time):
    def array_close(a, b):
        return np.allclose(a, b, rtol=0.05, atol=1)
    
    test_structure = {
        'environment': {
            'exchange': {
                molecule: scalar_almost_equal
                for molecule in actual_update['environment']['exchange']}},
        
        'listeners': {
            'fba_results': {
                'conc_updates': transform_and_run(np.array, array_close),
                'catalyst_counts': transform_and_run(np.array, array_close),
                'translation_gtp': scalar_almost_equal,
                'coefficient': scalar_almost_equal,
                'deltaMetabolites': [transform_and_run(np.array, array_close),
                                 transform_and_run(np.array, array_diffs_report_test(
                                f'out/migration/metabolism/delta_metabolites_t{time}.txt'))],
                'reactionFluxes': [transform_and_run(np.array, array_close),
                                 transform_and_run(np.array, array_diffs_report_test(
                                f'out/migration/metabolism/reaction_fluxes_t{time}.txt'))],
                'externalExchangeFluxes': transform_and_run(np.array, array_close),
                'objectiveValue': scalar_almost_equal,
                'shadowPrices': [transform_and_run(np.array, array_close),
                                 transform_and_run(np.array, array_diffs_report_test(
                                     f'out/migration/metabolism/shadow_prices_t{time}.txt'))],
                'reducedCosts': [transform_and_run(np.array, array_close),
                                 transform_and_run(np.array, array_diffs_report_test(
                                     f'out/migration/metabolism/reduced_costs_t{time}.txt'))],
                'targetConcentrations': transform_and_run(np.array, array_close),
                'homeostaticObjectiveValues': transform_and_run(np.array, array_close),
                'kineticObjectiveValues': [transform_and_run(np.array, array_close),
                                            transform_and_run(np.array, array_diffs_report_test(
                                            f'out/migration/metabolism/kinetic_objective_value_t{time}.txt'))],
            },
            
            'enzyme_kinetics': {
                'metaboliteCountsInit': transform_and_run(np.array, array_close),
                'metaboliteCountsFinal': transform_and_run(np.array, array_close),
                'enzymeCountsInit': transform_and_run(np.array, array_close),
                'countsToMolar': scalar_almost_equal,
                'actualFluxes': [transform_and_run(np.array, array_close),
                                 transform_and_run(np.array, array_diffs_report_test(
                                f'out/migration/metabolism/actual_fluxes_t{time}.txt'))],
                'targetFluxes': transform_and_run(np.array, array_close),
                'targetFluxesUpper': transform_and_run(np.array, array_close),
                'targetFluxesLower': transform_and_run(np.array, array_close),
            }
        }
    }

    tests = ComparisonTestSuite(test_structure, fail_loudly=False)
    tests.run_tests(actual_update, expected_update)

    tests.dump_report()
    
    assert (actual_update['listeners']['fba_results']['media_id'] ==
            expected_update['listeners']['fba_results']['media_id']),\
            'Media IDs not consistent!'

def test_metabolism_aas():
    config = {
        'media_id': AA_MEDIA_ID
    }
    initial_state = {
        'environment': {
            'media_id': AA_MEDIA_ID
        }
    }
    data = run_metabolism(
        total_time=10,
        config=config,
        initial_state=initial_state,
    )



def run_metabolism_composite():
    # configure exchange stub process with molecules to exchange
    config = {
        'exchanges': {
            'ARG[c]': 0.0,
            'ASN[c]': 0.0,
            'CYS[c]': 0.0,
            'GLN[c]': 0.0,
            'GLT[c]': 0.0,
            'GLY[c]': 0.0,
            'HIS[c]': 0.0,
            'ILE[c]': 0.0,
            'L-ALPHA-ALANINE[c]': 0.0,
            'L-ASPARTATE[c]': 0.0,
            'L-SELENOCYSTEINE[c]': 0.0,
            'LEU[c]': 0.0,
            'LYS[c]': 0.0,
            'MET[c]': 0.0,
            'PHE[c]': 0.0,
            'PRO[c]': 0.0,
            'SER[c]': 0.0,
            'THR[c]': 0.0,
            'TRP[c]': 0.0,
            'TYR[c]': 0.0,
            'VAL[c]': 0.0,
        }
    }

    composer = MetabolismExchange(config)
    metabolism_composite = composer.generate()

    # get initial state
    initial_state = get_state_from_file(
        path=f'data/wcecoli_t1000.json')

    # run a simulation
    experiment = Engine(**{
        'processes': metabolism_composite['processes'],
        'topology': metabolism_composite['topology'],
        'initial_state': initial_state})
    experiment.update(10)
    data = experiment.emitter.get_data()



# functions to run from the command line
test_library = {
    '0': test_metabolism_migration,
    '1': test_metabolism,
    '2': test_metabolism_aas,
    '3': run_metabolism_composite
}

if __name__ == '__main__':
    run_library_cli(test_library)
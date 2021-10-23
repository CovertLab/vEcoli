import os
import json
import numpy as np
import matplotlib.pyplot as plt

from vivarium.core.engine import Engine
from ecoli.states.wcecoli_state import get_state_from_file, MASSDIFFS
from ecoli.processes.polypeptide_elongation import PolypeptideElongation
from ecoli.library.schema import array_from
from migration.plots import qqplot
from migration.migration_utils import run_ecoli_process
from migration.migration_utils import scalar_almost_equal, array_almost_equal, ComparisonTestSuite, transform_and_run


PE_TOPOLOGY = PolypeptideElongation.topology


def run_polypeptide_elongation_migration(sim_data):
    # Create process, experiment, loading in initial state from file.
    config = sim_data.get_polypeptide_elongation_config()
    polypeptide_elongation_process = PolypeptideElongation(config)

    # initialize time parameters
    total_time = 2
    initial_times = [0, 2, 4, 10, 1000]

    for initial_time in initial_times:
        # run the process and get an update
        actual_update = run_ecoli_process(polypeptide_elongation_process, PE_TOPOLOGY, total_time=total_time,
                                          initial_time=initial_time)
        with open(f"data/polypeptide_elongation_update_t{total_time + initial_time}.json") as f:
            wc_update = json.load(f)
        plots(actual_update, wc_update, total_time + initial_time)
        assertions(actual_update, wc_update)

def plots(actual_update, expected_update, time):
    os.makedirs("out/migration/polypeptide_elongation/", exist_ok=True)

    # unpack updates and plot
    if 'molecules' in actual_update:
        molecules = np.array([actual_update['molecules'][mol] for mol in actual_update['molecules']])
        wc_molecules = np.array([expected_update['molecules'][mol] for mol in expected_update['molecules']])

        plt.subplot(2, 2, 1)
        qqplot(molecules, wc_molecules)
        plt.ylabel('wcEcoli')
        plt.xlabel('Vivarium')
        plt.title('Q-Q Plot of molecules')

    if 'subunits' in actual_update:
        subunits = np.array([actual_update['subunits'][mol] for mol in actual_update['subunits']])
        wc_subunits = np.array([expected_update['subunits'][mol] for mol in expected_update['subunits']])

        plt.subplot(2, 2, 2)
        qqplot(subunits, wc_subunits)
        plt.ylabel('wcEcoli')
        plt.xlabel('Vivarium')
        plt.title('Q-Q Plot of subunits')


    if 'amino_acids' in actual_update:
        amino_acids = np.array([actual_update['amino_acids'][mol] for mol in actual_update['amino_acids']])
        wc_amino_acids = np.array([expected_update['amino_acids'][mol] for mol in expected_update['amino_acids']])

        plt.subplot(2, 2, 3)
        qqplot(amino_acids, wc_amino_acids)
        plt.ylabel('wcEcoli')
        plt.xlabel('Vivarium')
        plt.title('Q-Q Plot of amino acids')

    if 'monomers' in actual_update:
        monomers = np.array([actual_update['monomers'][mol] for mol in actual_update['monomers']])
        wc_monomers = np.array([expected_update['monomers'][mol] for mol in expected_update['monomers']])

        plt.subplot(2, 2, 4)
        qqplot(monomers, wc_monomers)
        plt.ylabel('wcEcoli')
        plt.xlabel('Vivarium')
        plt.title('Q-Q Plot of monomers')

    plt.gcf().set_size_inches(16, 12)
    plt.tight_layout()
    plt.savefig(f"out/migration/polypeptide_elongation/polypeptide_elongation_figures{time}.png")
    plt.close()

def assertions(actual_update, expected_update):
    test_structure = {
        'molecules': {key: scalar_almost_equal for key in actual_update['molecules'].keys()},
        'listeners': {
            'growth_limits': {
                'aa_pool_size': array_almost_equal,
                'aa_request_size': array_almost_equal,
                'active_ribosomes_allocated': scalar_almost_equal,
                'fraction_trna_charged': array_almost_equal,
                'net_charged': array_almost_equal
            },
            'ribosome_data': {
                'aaCountInSequence': array_almost_equal,
                'aaCounts': array_almost_equal,
                'actualElongationHist': array_almost_equal,
                'actualElongations': scalar_almost_equal,
                'didTerminate': scalar_almost_equal,
                'effective_elongation_rate': scalar_almost_equal,
                'elongationsNonTerminatingHist': array_almost_equal,
                'numTrpATerminated': scalar_almost_equal,
                'processElongationRate': scalar_almost_equal,
                'terminationLoss': scalar_almost_equal,
                'translation_supply': array_almost_equal
            }
        }
    }
    if 'monomers' in actual_update:
        test_structure['monomers'] = transform_and_run(array_from, array_almost_equal)
        test_structure['monomers'] = {key: scalar_almost_equal for key in actual_update['monomers'].keys()}
    if 'subunits' in actual_update:
        test_structure['subunits'] = {key: scalar_almost_equal for key in actual_update['subunits'].keys()}
    if 'amino_acids' in actual_update:
        test_structure['amino_acids'] = {key: scalar_almost_equal for key in actual_update['amino_acids'].keys()}
    if 'polypeptide_elongation' in actual_update:
        test_structure['polypeptide_elongation'] = {
            'aa_count_diff': {},
            'gtp_to_hydrolyze': scalar_almost_equal
        }
    tests = ComparisonTestSuite(test_structure, fail_loudly=False)
    tests.run_tests(actual_update, expected_update, verbose=True)
    tests.dump_report()

    if 'active_ribosome' in actual_update:
        assert scalar_almost_equal(len(actual_update['active_ribosome']['_delete']),
                            len(expected_update['active_ribosome']['_delete'])), "Del ribosomes not consistent"

        actual_update_indices = []
        for key in actual_update['active_ribosome']:
            if key != '_delete':
                actual_update_indices.append(key)

        expected_update_indices = []
        for key in expected_update['active_ribosome']:
            if key != '_delete':
                expected_update_indices.append(key)

        for i in range(min(len(expected_update_indices), len(actual_update_indices))):
            actual_key = actual_update_indices[i]
            expected_key = expected_update_indices[i]
            assert scalar_almost_equal(actual_update['active_ribosome'][actual_key]['peptide_length'], expected_update['active_ribosome'][expected_key]['peptide_length'])
            assert scalar_almost_equal(actual_update['active_ribosome'][actual_key]['pos_on_mRNA'], expected_update['active_ribosome'][expected_key]['pos_on_mRNA'])
            assert scalar_almost_equal(actual_update['active_ribosome'][actual_key]['submass'][MASSDIFFS['massDiff_protein']], 
                                       expected_update['active_ribosome'][expected_key]['submass']['protein'])

def run_polypeptide_elongation(sim_data):
    # Create process, experiment, loading in initial state from file.
    config = sim_data.get_polypeptide_elongation_config()
    polypeptide_elongation_process = PolypeptideElongation(config)

    initial_state = get_state_from_file(
        path=f'data/wcecoli_t0.json')

    polypeptide_elongation_composite = polypeptide_elongation_process.generate()
    experiment = Engine(**{
        'processes': polypeptide_elongation_composite['processes'],
        'topology': {polypeptide_elongation_process.name: PE_TOPOLOGY},
        'initial_state': initial_state
    })

    experiment.update(10)

    data = experiment.emitter.get_data()


if __name__ == "__main__":
    from ecoli.library.sim_data import LoadSimData
    from ecoli.composites.ecoli_nonpartition import SIM_DATA_PATH
    sim_data = LoadSimData(
        sim_data_path=SIM_DATA_PATH,
        seed=0)

    run_polypeptide_elongation_migration(sim_data)
    # run_polypeptide_elongation()

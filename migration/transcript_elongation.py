"""
tests that vivarium-ecoli process update are the same as saved wcEcoli updates

"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, chisquare

from ecoli.library.sim_data import LoadSimData
from ecoli.composites.ecoli_master import SIM_DATA_PATH
from migration.migration_utils import *

from ecoli.processes.transcript_elongation import TranscriptElongation

from ecoli.library.schema import arrays_from
from ecoli.composites.ecoli_master import get_state_from_file
from migration.plots import qqplot
from migration.migration_utils import array_diffs_report


load_sim_data = LoadSimData(sim_data_path=SIM_DATA_PATH,
                            seed=0)

def test_transcription_elongation():

    # Create process, experiment, loading in initial state from file.
    config = load_sim_data.get_transcript_elongation_config()
    process = TranscriptElongation(config)

    topology = {'environment': ('environment',),
                'RNAs': ('unique', 'RNA'),
                'active_RNAPs': ('unique', 'active_RNAP'),
                'molecules': ('bulk',),
                'bulk_RNAs': ('bulk',),
                'ntps': ('bulk',),
                'listeners': ('listeners',)}

    actual_update = run_ecoli_process(process, topology)

    with open("data/transcript_elongation_update_t2.json") as f:
        wc_update = json.load(f)

    plots(actual_update, wc_update)
    assertions(actual_update, wc_update)


def plots(actual_update, expected_update):
    # unpack update
    trans_lengths = [r['transcript_length'] for k, r in actual_update["RNAs"].items()
                     if k != "_delete"]
    rnas_synthesized = actual_update['listeners']['transcript_elongation_listener']['countRnaSynthesized']
    # bulk_16SrRNA = actual_update['bulk_RNAs']['16S rRNA']
    # bulk_5SrRNA = actual_update['bulk_RNAs']['5S rRNA']
    # bulk_23SrRNA = actual_update['bulk_RNAs']['23S rRNA']
    # bulk_mRNA = actual_update['bulk_RNAs']['mRNA']

    ntps_used = actual_update['listeners']['growth_limits']['ntpUsed']
    total_ntps_used = actual_update['listeners']['transcript_elongation_listener']['countNTPsUsed']
    ntps = actual_update['ntps']

    RNAP_coordinates = [v['coordinates'] for k, v in actual_update['active_RNAPs'].items()
                        if k != "_delete"]
    RNAP_elongations = actual_update['listeners']['rnap_data']['actualElongations']
    terminations = actual_update['listeners']['rnap_data']['didTerminate']

    ppi = actual_update['molecules']['PPI[c]']
    inactive_RNAP = actual_update['molecules']['APORNAP-CPLX[c]']

    # unpack wcEcoli update
    # TODO: wc_trans_lengths = ...
    wc_rnas_synthesized = expected_update['listeners']['transcript_elongation_listener']['countRnaSynthesized']

    wc_ntps_used = expected_update['listeners']['growth_limits']['ntpUsed']
    wc_total_ntps_used = expected_update['listeners']['transcript_elongation_listener']['countNTPsUsed']
    wc_ntps = expected_update['ntps']

    # TODO: wc_RNAP_coordinates = ...
    wc_RNAP_elongations = expected_update['listeners']['rnap_data']['actualElongations']
    wc_terminations = expected_update['listeners']['rnap_data']['didTerminate']

    wc_ppi = expected_update['molecules']['PPI[c]']
    wc_inactive_RNAP = expected_update['molecules']['APORNAP-CPLX[c]']

    # Plots ======================================================

    # rnas synthesized follow same distribution
    plt.subplot(3, 1, 1)
    qqplot(rnas_synthesized, wc_rnas_synthesized)

    # ntps used of each type
    plt.subplot(3, 1, 2)
    plt.bar(np.arange(4)-0.1, ntps_used, 0.2)
    plt.bar(np.arange(4)+0.1, wc_ntps_used, 0.2)

    # wc_ntps
    plt.subplot(3, 1, 3)
    plt.bar(np.arange(4) - 0.1, [v for v in ntps.values()], 0.2)
    plt.bar(np.arange(4) + 0.1, [v for v in wc_ntps.values()], 0.2)

    plt.gcf().set_size_inches(8, 6)
    plt.savefig("out/migration/transcript_elongation_figures.png")


def assertions(actual_update, expected_update):
    test_structure = {
        'listeners' : {
            'transcript_elongation_listener' : {
                'countRnaSynthesized' : stochastic_equal,
                'countNTPsUsed' : scalar_almost_equal
            },
            'growth_limits' : {
                'ntpUsed' : run_all(array_almost_equal,
                                    array_diffs_report_test("out/migration/ntpUsed_comparison.txt"))
            }
        }
    }

    # Note: While writing tests, use fail_loudly=False, verbose=True
    #       Once all tests are passing, consider fail_loudly=True, verbose=False for efficiency

    tests = ComparisonTestSuite(test_structure, fail_loudly=False)
    tests.run_tests(actual_update,
                    expected_update,
                    verbose=True)

    #pretty_print(tests.report)
    #print(tests.report)
    #tests.dump_report()

    tests.fail()


def save_test_sequences(config):
    save_idx = np.array([config['idx_16S_rRNA'][0],
                         config['idx_23S_rRNA'][0],
                         config['idx_5S_rRNA'][0],
                         config['is_mRNA'].tolist().index(True)])

    save_seq = config['rnaSequences'][save_idx]

    with open('data/elongation_sequences.npy', 'wb') as f:
        np.save(f, save_seq)

if __name__ == "__main__":
    test_transcription_elongation()

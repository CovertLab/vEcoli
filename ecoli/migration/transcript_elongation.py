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
from ecoli.migration.migration_utils import run_ecoli_process, percent_error

from ecoli.processes.transcript_elongation import TranscriptElongation

from ecoli.library.schema import arrays_from
from ecoli.composites.ecoli_master import get_state_from_file
from ecoli.migration.plots import qqplot
from ecoli.migration.migration_utils import array_diffs_report


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

    plots(actual_update, None)
    assertions(actual_update, None)


def plots(actual_update, expected_update):
    # unpack update
    rnas_synthesized = actual_update['listeners']['transcript_elongation_listener']['countRnaSynthesized']
    ntps_used = actual_update['listeners']['growth_limits']['ntpUsed']
    total_ntps_used = actual_update['listeners']['transcript_elongation_listener']['countNTPsUsed']

    ntps = actual_update['ntps']

    plt.subplot(2, 1, 1)
    plt.bar(range(len(rnas_synthesized)), rnas_synthesized, width=1)
    plt.xlabel("TU")
    plt.ylabel("Count")
    plt.title("Counts synthesized")

    plt.subplot(2, 1, 2)
    import ipdb; ipdb.set_trace()
    plt.bar(range(ntps_used.size), ntps_used)
    plt.xticks(ticks=range(len(ntps.keys())), labels=list(ntps.keys()))
    plt.ylabel('Count')
    plt.title('NTP Counts Used')

    plt.subplots_adjust(hspace=0.5)
    plt.gcf().set_size_inches(8, 6)
    plt.savefig("out/migration/transcript_elongation_toymodel.png")


def assertions(actual_update, expected_update):
    # unpack update
    rnas_synthesized = actual_update['listeners']['transcript_elongation_listener']['countRnaSynthesized']
    ntps_used = actual_update['listeners']['growth_limits']['ntpUsed']
    total_ntps_used = actual_update['listeners']['transcript_elongation_listener']['countNTPsUsed']

    ntps = actual_update['ntps']

    # NTPs used match sequences of which RNAs were elongated

    # total NTPS used matches sum of NTP-used types

    assert sum(ntps_used) == total_ntps_used



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
    plt.ylabel("# synthesized")

"""
tests that vivarium-ecoli process update are the same as saved wcEcoli updates

TODO:
    - get wcEcoli state at time 0, so that the comparison is fair.
"""

import os
import json
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

from ecoli.library.sim_data import LoadSimData
from ecoli.composites.ecoli_master import SIM_DATA_PATH
from ecoli.migration.migration_utils import run_ecoli_process, percent_error

from ecoli.processes.transcript_initiation import TranscriptInitiation

from ecoli.migration.plots import qqplot


load_sim_data = LoadSimData(
            sim_data_path=SIM_DATA_PATH,
            seed=0)


def test_transcript_initiation():
    # Create process, experiment, loading in initial state from file.
    config = load_sim_data.get_transcript_initiation_config()
    process = TranscriptInitiation(config)

    # get ids from config

    # copy topology from ecoli_master, under generate_topology
    topology = {
            'environment': ('environment',),
            'full_chromosomes': ('unique', 'full_chromosome'),
            'RNAs': ('unique', 'RNA'),
            'active_RNAPs': ('unique', 'active_RNAP'),
            'promoters': ('unique', 'promoter'),
            'molecules': ('bulk',),
            'listeners': ('listeners',)
    }

    # run the process and get an update
    actual_update = run_ecoli_process(process, topology)

    # separate the update to its ports

    # compare to collected update from ecEcoli
    with open("data/transcript_initiation_update_t2.json") as f:
        wc_data = json.load(f)

    # unpack wc_data

    # Sanity checks: ...

    # Numerical tests =======================================================================

    # Write test log to file
    log_file = "out/migration/transcript_initiation.txt"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, "w") as f:
        report = []
        f.writelines(report)

    # Plots =========================================================================
    plt.subplots_adjust(hspace=0.5)

    plt.savefig("out/migration/transcript_initiation_figures.png")

    # Asserts for numerical tests:



if __name__ == "__main__":
    test_transcript_initiation()

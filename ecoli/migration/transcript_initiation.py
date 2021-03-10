"""
tests that vivarium-ecoli process update are the same as save wcEcoli updates

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
    topology = {}

    # run the process and get an update
    actual_update = run_ecoli_process(process, topology)

    # separate the update to its ports

    # compare to collected update from ecEcoli
    with open("data/transcript_initiation_update_t2.json") as f:
        wc_data = json.load(f)

    # unpack wc_data

    # Sanity checks: wcEcoli and vivarium-ecoli match in number, names of proteins, metabolites
    assert len(d_proteins) == len(wc_proteins) == len(protein_ids), (
        f"Mismatch in lengths: vivarium-ecoli protein update has length {len(d_proteins)}\n"
        f"while wcecoli has {len(protein_ids)} proteins with {len(wc_proteins)} values.")

    assert len(d_metabolites) == len(wc_metabolites) == len(metabolite_ids), (
        f"Mismatch in lengths: vivarium-ecoli metabolite update has length {len(d_metabolites)}\n"
        f"while wcecoli has {len(metabolite_ids)} metabolites with {len(wc_metabolites)} values.")

    assert set(d_proteins.keys()) == set(protein_ids), (
        "Mismatch between protein ids in vivarium-ecoli and wcEcoli.")

    assert set(d_metabolites.keys()) == set(metabolite_ids), (
        "Mismatch between metabolite ids in vivarium-ecoli and wcEcoli.")

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

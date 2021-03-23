"""
tests that vivarium-ecoli process update are the same as saved wcEcoli updates

TODO:
    - get wcEcoli state at time 0, so that the comparison is fair.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, chisquare

from ecoli.library.sim_data import LoadSimData
from ecoli.composites.ecoli_master import SIM_DATA_PATH
from ecoli.migration.migration_utils import run_ecoli_process, percent_error

from ecoli.processes.transcript_initiation import TranscriptInitiation

from ecoli.migration.plots import qqplot
from ecoli.migration.migration_utils import array_diffs_report


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
    rna_synth_prob = actual_update['listeners']['rna_synth_prob']['rna_synth_prob']

    rib_data = actual_update['listeners']['ribosome_data']
    n_5SrRNA_prod = rib_data['rrn5S_produced']
    n_16SrRNA_prod = rib_data['rrn16S_produced']
    n_23SrRNA_prod = rib_data['rrn23S_produced']
    init_prob_5SrRNA = rib_data['rrn5S_init_prob']
    init_prob_16SrRNA = rib_data['rrn16S_init_prob']
    init_prob_23SrRNA = rib_data['rrn23S_init_prob']
    total_rna_init = rib_data['total_rna_init']

    rnap_data = actual_update['listeners']['rnap_data']
    assert total_rna_init == rnap_data['didInitialize'], ("Update malformed: number of inits does not "
                                                          "match change in active rnaps")
    rna_inits = rnap_data['rnaInitEvent']
    assert rna_inits.shape[0] == rna_synth_prob.shape[0], ("Update malformed: TUs in rna_inits array "
                                                           "do not match TUs in rna synthesis probabilities")

    active_RNAPs = actual_update['active_RNAPs']['_add']
    d_inactive_RNAPs = actual_update['molecules'][config['inactive_RNAP']]
    assert d_inactive_RNAPs == -total_rna_init, ("Update malformed: change in inactive RNAPs does not match ",
                                                 "total rnas initiated.")
    RNAs = actual_update['RNAs']['_add']

    # compare to collected update from wcEcoli
    with open("data/transcript_initiation_update_t2.json") as f:
        wc_data = json.load(f)

    # unpack wc_data
    wc_rna_synth_prob = np.array(wc_data['listeners']['rna_synth_prob']['rna_synth_prob'])

    wc_rib_data = wc_data['listeners']['ribosome_data']
    wc_n_5SrRNA_prod = wc_rib_data['rrn5S_produced']
    wc_n_16SrRNA_prod = wc_rib_data['rrn16S_produced']
    wc_n_23SrRNA_prod = wc_rib_data['rrn23S_produced']
    wc_init_prob_5SrRNA = wc_rib_data['rrn5S_init_prob']
    wc_init_prob_16SrRNA = wc_rib_data['rrn16S_init_prob']
    wc_init_prob_23SrRNA = wc_rib_data['rrn23S_init_prob']
    wc_total_rna_init = wc_rib_data['total_rna_init']

    wc_rnap_data = wc_data['listeners']['rnap_data']
    assert wc_total_rna_init == wc_rnap_data['didInitialize'], ("Update malformed: number of inits does not "
                                                                "match change in active rnaps")
    wc_rna_inits = np.array(wc_rnap_data['rnaInitEvent'])
    assert wc_rna_inits.shape[0] == wc_rna_synth_prob.shape[0], ("Update malformed: TUs in rna_inits array "
                                                                 "do not match TUs in rna synthesis probabilities")

    wc_active_RNAPs = wc_data['active_RNAPs']['_add']
    wc_d_inactive_RNAPs = wc_data['molecules']['APORNAP-CPLX[c]']
    assert wc_d_inactive_RNAPs == -wc_total_rna_init, ("Update malformed: change in inactive RNAPs does not "
                                                       "match total rnas initiated.")
    wc_RNAs = wc_data['RNAs']['_add']

    # Sanity checks:

    assert len(rna_inits) == len(wc_rna_inits), "Number of TUs differs between vivarium-ecoli and wcEcoli."

    # Numerical tests =======================================================================

    # Compare synthesis probabilities
    # Using max percent error of 5% - however, this should probably be ~0%.
    # TODO: Possible origin of differences:
    #  1) differing basal probabilities
    #  2) differing TFs bound (in initial state of promoters)
    #  3) differing mRNA/tRNA/rRNA set points, in media data of initial state
    #  4) differing fixed synthesis set points, in media data of initial state
    #  5) differing calculation (code itself)
    #  6) rounding issues?

    np.testing.assert_allclose(rna_synth_prob, wc_rna_synth_prob,
                               rtol=0.05,
                               err_msg="Vivarium-ecoli calculates different synthesis probabilities than wcEcoli")

    # Compare synthesis of rRNAs
    n_rRNA_testresult = chisquare(np.array([n_5SrRNA_prod, n_16SrRNA_prod, n_23SrRNA_prod]),
                                  np.array([wc_n_5SrRNA_prod, wc_n_16SrRNA_prod, wc_n_23SrRNA_prod]))
    assert n_rRNA_testresult.pvalue > 0.05

    n_rRNA_testresult = chisquare(np.array([init_prob_5SrRNA, init_prob_16SrRNA, init_prob_23SrRNA]) * total_rna_init,
                                  np.array([wc_init_prob_5SrRNA, wc_init_prob_16SrRNA, wc_init_prob_23SrRNA]) * total_rna_init)
    assert n_rRNA_testresult.pvalue > 0.05

    # Compare fixed synthesis probabilities
    fixed_synths = np.array([sum(rna_inits[config['idx_rnap']]),
                             sum(rna_inits[config['idx_rprotein']]),
                             total_rna_init])
    fixed_synths[2] -= fixed_synths[0] + fixed_synths[1]
    wc_fixed_synths = np.array([sum(wc_rna_inits[config['idx_rnap']]),
                                sum(wc_rna_inits[config['idx_rprotein']]),
                                wc_total_rna_init])
    wc_fixed_synths[2] -= wc_fixed_synths[0] + wc_fixed_synths[1]

    fixed_test_result = mannwhitneyu(fixed_synths, wc_fixed_synths)
    assert fixed_test_result.pvalue > 0.05

    # Write test log to file
    log_file = "out/migration/transcript_initiation.txt"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, "w") as f:
        pe_synths = (rna_synth_prob - wc_rna_synth_prob) / wc_rna_synth_prob
        report = ["Compared synthesis probabilities, finding a maximum percent-error difference of",
                  f" {max(pe_synths[~np.isnan(pe_synths)])}.\n",
                  "\nCompared inits for fixed-synthesis-probability TUs, using a chi-square GOF test \n",
                  f"(p = {fixed_test_result.pvalue})",
                  ""]
        f.writelines(report)

    synth_prob_file = "out/migration/synth_prob_comparison.txt"
    with open(synth_prob_file, "w") as f:
        f.write(array_diffs_report(rna_synth_prob, wc_rna_synth_prob,
                                   names=[x[0] for x in config['rna_data']]))

    # Plots =========================================================================

    plt.subplot(2, 2, 1)
    qqplot(rna_synth_prob, wc_rna_synth_prob, quantile_precision=0.0001)
    plt.xlabel("vivarium-ecoli")
    plt.ylabel("wcEcoli")
    plt.title("QQ-Plot of Synthesis Probabilities")

    plt.subplot(2, 2, 2)
    qqplot(rna_inits, wc_rna_inits, quantile_precision=0.0001)
    plt.xlabel("vivarium-ecoli")
    plt.ylabel("wcEcoli")
    plt.title("QQ-Plot of # of Initiations per TU")

    plt.subplot(2, 1, 2)
    diffplot = -np.sort(wc_rna_synth_prob - rna_synth_prob)
    diffplot = diffplot[np.nonzero(diffplot)]
    plt.hist(diffplot, bins=1000, rwidth=1)
    plt.yscale("log")
    plt.xlabel("$P_{Vivarium} - P_{wcEcoli}$")
    plt.ylabel("TUs")
    plt.title("Histogram of Synthesis Probability Differences")

    plt.subplots_adjust(hspace=0.5)

    plt.gcf().set_size_inches(8, 6)
    plt.savefig("out/migration/transcript_initiation_figures.png")


if __name__ == "__main__":
    test_transcript_initiation()

"""
tests that vivarium-ecoli process update are the same as save wcEcoli updates

TODO:
    - get wcEcoli state at time 0, so that the comparison is fair.
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2_contingency

from ecoli.library.sim_data import LoadSimData
from ecoli.composites.ecoli_nonpartition import SIM_DATA_PATH
from migration.migration_utils import run_ecoli_process, percent_error

from ecoli.processes.protein_degradation import ProteinDegradation

from migration.plots import qqplot


TOPOLOGY = ProteinDegradation.topology


def run_protein_degradation(sim_data):
    # Create process, experiment, loading in initial state from file.
    config = sim_data.get_protein_degradation_config()
    prot_deg_process = ProteinDegradation(config)

    # get ids from config
    amino_acid_ids = config["amino_acid_ids"]
    water_id = config["water_id"]

    # run the process and get an update
    actual_update = run_ecoli_process(prot_deg_process, TOPOLOGY, total_time=2, initial_time=0)

    # separate the update to its ports
    d_proteins = actual_update['proteins']
    d_metabolites = actual_update['metabolites']

    viv_amino_acids = [d_metabolites[aa] for aa in amino_acid_ids]
    viv_water = d_metabolites[water_id]

    # compare to collected update from wcEcoli
    with open("data/prot_deg_update_t2.json") as f:
        wc_data = json.load(f)

    # unpack wc_data
    protein_ids = wc_data['protein_ids']
    wc_proteins = wc_data['proteins_to_degrade']
    metabolite_ids = wc_data['metabolite_ids']
    wc_metabolites = wc_data['metabolite_update']
    wc_amino_acids = [metabolite for id, metabolite in zip(metabolite_ids, wc_metabolites)
                      if id != water_id]
    wc_water = wc_metabolites[metabolite_ids.index(water_id)]

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
    # Perform tests of homogeneity (or more precisely, inability to reject homogeneity)
    # in distributions of number of proteins degraded, amino acids released:
    test_threshold = 0.01

    def hist(dist):
        return [sum(np.array(dist) == k) for k in range(0, max(dist))]

    def match_shape(dist1, dist2):
        l = max(len(dist1), len(dist2))
        return [[dist1[k] if k < len(dist1) else 0 for k in range(l)],
                [dist2[k] if k < len(dist2) else 0 for k in range(l)]]

    viv_proteins = [-p for p in d_proteins.values()]
    test_protein = chi2_contingency(match_shape(hist(wc_proteins), hist(viv_proteins)))
    test_aa = chi2_contingency(np.array([wc_amino_acids, viv_amino_acids]) + 1) # using pseudo-counts to prevent divide-by-zero

    # Find percent errors between total numbers of
    # proteins degraded, amino acids released, and water molecules consumed
    # between wcEcoli and vivarium-ecoli.
    percent_error_threshold = 0.5

    protein_error = percent_error(-sum(d_proteins.values()), sum(wc_proteins))
    aa_error = percent_error(sum(viv_amino_acids), sum(wc_amino_acids))
    water_error = percent_error(viv_water, wc_water)

    # Write test log to file
    log_file = "out/migration/protein_degradation.txt"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, "w") as f:
        report = []

        report += 'COMPARING MEDIANS ===========================\n\n'
        report += ('Distribution of number of degradations per protein \n' +
                   ("differed significantly " if test_protein[1] <= test_threshold
                    else "did not differ significantly ") +
                   'between wcEcoli and vivarium-ecoli \n' +
                   f'(Chi-Squared test, X^2={test_protein[0]}, p={test_protein[1]}).\n\n')
        report += ('Number of molecules released for each amino acid type \n' +
                   ('differed significantly ' if test_aa[1] <= test_threshold
                    else 'did not differ significantly ') +
                   'between wcEcoli and vivarium-ecoli \n' +
                   f'(Chi-Squared test, X^2={test_aa[0]}, p={test_aa[1]}).\n\n')

        report += 'PERCENT ERROR ===============================\n\n'
        report += ('Percent error in total number of proteins degraded \n' +
                   f'was {protein_error * 100}% ' + ('<= ' if protein_error <= percent_error_threshold else '> ') +
                   f'{percent_error_threshold * 100}% (threshold).\n\n')
        report += ('Percent error in total number of amino acids released \n' +
                   f'was {aa_error * 100}% ' + ('<= ' if aa_error <= percent_error_threshold else '> ') +
                   f'{percent_error_threshold * 100}% (threshold).\n\n')
        report += ('Percent error in total number of water molecules consumed \n' +
                   f'was {water_error * 100}% ' + ('<= ' if water_error <= percent_error_threshold else '> ') +
                   f'{percent_error_threshold * 100}% (threshold).\n\n')

        f.writelines(report)

    # QQ-plot of degradation per protein (wcEcoli vs. vivarium-ecoli)
    plt.subplot(2, 1, 1)
    qqplot(wc_proteins, [-p for p in d_proteins.values()], quantile_precision=0.0001)
    plt.xlabel("wcEcoli")
    plt.ylabel("vivarium-ecoli")
    plt.title("QQ-plot of degradations per protein type")

    plt.subplot(2, 1, 2)
    qqplot(wc_amino_acids, viv_amino_acids)
    plt.xlabel("wcEcoli")
    plt.ylabel("vivarium-ecoli")
    plt.title("QQ-plot of release events per amino acid type")

    plt.subplots_adjust(hspace=0.5)

    plt.savefig("out/migration/protein_degradation_figures.png")

    # Asserts for numerical tests:
    assert test_protein[1] > test_threshold, (
        "Distribution of #proteins degraded is different between wcEcoli and vivarium-ecoli"
        f"(p={test_protein[1]} <= {test_threshold}) ")
    assert test_aa[1] > test_threshold, (
        "Distribution of #amino acids released is different between wcEcoli and vivarium-ecoli"
        f"(p={test_aa[1]} <= {test_threshold})")

    assert protein_error < percent_error_threshold, (
        "Total # of proteins degraded differs between wcEcoli and vivarium-ecoli"
        f"(percent error = {protein_error})")
    assert aa_error < percent_error_threshold, (
        "Total # of amino acids released differs between wcEcoli and vivarium-ecoli"
        f"(percent error = {aa_error})")
    assert water_error < percent_error_threshold, (
        "Total # of water molecules used differs between wcEcoli and vivarium-ecoli"
        f"(percent error = {water_error})")


if __name__ == "__main__":
    run_protein_degradation()

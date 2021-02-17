"""
tests that vivarium-ecoli process update are the same as save wcEcoli updates

TODO:
    - get wcEcoli state at time 0, so that the comparison is fair.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from scipy.stats import mannwhitneyu
from vivarium.core.experiment import Experiment

from ecoli.library.sim_data import LoadSimData
from ecoli.composites.ecoli_master import SIM_DATA_PATH, get_state_from_file

from ecoli.processes.protein_degradation import ProteinDegradation
from ecoli.processes.complexation import Complexation

from utils.plots import qqplot


load_sim_data = LoadSimData(
            sim_data_path=SIM_DATA_PATH,
            seed=0)


def run_ecoli_process(process, topology, total_time=2):
    """
    load a single ecoli process, run it, and return the update

    Args:
        process: an initialized Process
        topology: (dict) topology for the Process, from ecoli_master
        total_time: (optional) run time. defaults at 2 seconds -- the default time of wcEcoli

    Returns:
        an update from the perspective of the Process.
    """

    # get initial state from file
    # TODO -- get wcecoli_t0
    initial_state = get_state_from_file(path='data/wcecoli_t10.json')

    # make an experiment
    experiment_config = {
        'processes': {process.name: process},
        'topology': {process.name: topology},
        'initial_state': initial_state}
    experiment = Experiment(experiment_config)

    # Get update from process.
    # METHOD 1
    path, process = list(experiment.process_paths.items())[0]
    store = experiment.state.get_path(path)

    # translate the values from the tree structure into the form
    # that this process expects, based on its declared topology
    states = store.outer.schema_topology(process.schema, store.topology)

    update = experiment.invoke_process(
        process,
        path,
        total_time,
        states)

    actual_update = update.get()
    return actual_update


def percent_error(actual, expected):
    return abs((actual - expected) / expected)


def test_protein_degradation():
    # Create process, experiment, loading in initial state from file.
    config = load_sim_data.get_protein_degradation_config()
    prot_deg_process = ProteinDegradation(config)

    # get ids from config
    amino_acid_ids = config["amino_acid_ids"]
    water_id = config["water_id"]

    # copy topology from ecoli_master, under generate_topology
    topology = {
        'metabolites': ('bulk',),
        'proteins': ('bulk',)}

    # run the process and get an update
    actual_update = run_ecoli_process(prot_deg_process, topology)

    # separate the update to its ports
    d_proteins = actual_update['proteins']
    d_metabolites = actual_update['metabolites']

    viv_amino_acids = [d_metabolites[aa] for aa in amino_acid_ids]
    viv_water = d_metabolites[water_id]

    # compare to collected update from ecEcoli
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
    # Perform tests of equal-medians (or more precisely, failure to reject non-equal medians)
    # in distributions of number of proteins degraded, amino acids released:
    utest_threshold = 0.05

    utest_protein = mannwhitneyu(wc_proteins, [-p for p in d_proteins.values()], alternative="two-sided")
    utest_aa = mannwhitneyu(wc_amino_acids, viv_amino_acids, alternative="two-sided")

    # Find percent errors between total numbers of
    # proteins degraded, amino acids released, and water molecules consumed
    # between wcEcoli and vivarium-ecoli.
    percent_error_threshold = 0.05

    protein_error = percent_error(-sum(d_proteins.values()), sum(wc_proteins))
    aa_error = percent_error(sum(viv_amino_acids), sum(wc_amino_acids))
    water_error = percent_error(viv_water, wc_water)

    # Write test log to file
    log_file = "out/migration/protein_degradation.txt"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, "w") as f:
        report = []

        report += 'COMPARING MEDIANS ===========================\n\n'
        report += ('Median number of degradations per protein \n' +
                   ("differed significantly " if utest_protein.pvalue <= utest_threshold
                    else "did not differ significantly ") +
                   'between wcEcoli and vivarium-ecoli \n' +
                   f'(Mann-Whitney U, U={utest_protein.statistic}, p={utest_protein.pvalue}).\n\n')
        report += ('Median number of molecules released per amino acid \n' +
                   ('differed significantly ' if utest_aa.pvalue <= utest_threshold
                    else 'did not differ significantly ') +
                   'between wcEcoli and vivarium-ecoli \n' +
                   f'(Mann-Whitney U, U={utest_aa.statistic}, p={utest_aa.pvalue}).\n\n')

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
    assert utest_protein.pvalue > utest_threshold, (
        "Distribution of #proteins degraded is different between wcEcoli and vivarium-ecoli"
        f"(p={utest_protein.pvalue} <= {threshold}) ")
    assert utest_aa.pvalue > utest_threshold, (
        "Distribution of #amino acids released is different between wcEcoli and vivarium-ecoli"
        f"(p={utest_aa.pvalue} <= {threshold})")

    assert protein_error < percent_error_threshold, (
        "Total # of proteins degraded differs between wcEcoli and vivarium-ecoli"
        f"(percent error = {protein_error})")
    assert aa_error < percent_error_threshold, (
        f"Total # of amino acids released differs between wcEcoli and vivarium-ecoli"
        "(percent error = {aa_error})")
    assert water_error < percent_error_threshold, (
        f"Total # of water molecules used differs between wcEcoli and vivarium-ecoli"
        "(percent error = {water_error})")

def test_complexation():
    config = load_sim_data.get_complexation_config()
    complexation = Complexation(config)



    molecules = config["molecule_names"]

    topology = {
        'molecules': ('bulk',)
        }


    # run the process and get an update
    actual_update = run_ecoli_process(complexation, topology)




    with open("data/complexation_update3_t2.json") as f:
        wc_complexation_data = json.load(f)

    d_molecules = actual_update["molecules"]
    wc_molecules = wc_complexation_data["molecules"]
    wc_moleculesCounts = wc_complexation_data["migrationCounts"]




    assert len(wc_molecules) == len(d_molecules), (
        f"Mismatch in lengths: vivarium-ecoli molecule update has length {len(d_molecules)}\n"
        f"while wcecoli has {len(wc_molecules)}.")

    assert set(d_molecules.keys()) == set(wc_molecules), (
        "Mismatch between moleules in vivarium-ecoli and wcEcoli.")


    utest_threshold = 0.05
    utest_protein = mannwhitneyu(wc_moleculesCounts, [-p for p in d_molecules.values()], alternative="two-sided")
    percent_error_threshold = 0.05
    molecules_error = percent_error(-sum(d_molecules.values()), sum(wc_moleculesCounts))

    import ipdb
    ipdb.set_trace()


if __name__ == "__main__":
    test_complexation()

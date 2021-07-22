import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from scipy.stats import mannwhitneyu
from vivarium.core.experiment import Experiment

from ecoli.library.sim_data import LoadSimData
from ecoli.composites.ecoli_master import SIM_DATA_PATH, get_state_from_file
from ecoli.processes.equilibrium import Equilibrium

from ecoli.migration.plots import qqplot
from ecoli.migration.migration_utils import run_ecoli_process, percent_error

import scipy.constants
from wholecell.utils import units
import pandas

load_sim_data = LoadSimData(
            sim_data_path=SIM_DATA_PATH,
            seed=0)



def test_equilibrium():
    config = load_sim_data.get_equilibrium_config()
    equil = Equilibrium(config)

    timestep = 2
    topology = {
        'molecules': ('bulk',)
        }

    # run the process and get an update
    actual_update = run_ecoli_process(equil, topology, timestep, 0, None)


    update_molecules = actual_update['molecules']
    update_listeners = actual_update['listeners']
    with open("data/equilibrium_update_t62.json") as f:
        wc_data = json.load(f)

    wc_moleculeNames = wc_data["molecules"]
    wc_moleculesCounts = wc_data["moleculeCounts"]
    wc_rxn = wc_data["rxnFluxes"]
    wc_req_molecules = wc_data['requested']
    wc_partitioned = wc_data['partitioned_molecules']
    wc_d_molecules = wc_data['evolveResult']
    wc_rxnEvolve = wc_data['rcnFluxesEvolve']
    wc_rates = wc_data["wc_rates"]

    moleculeNames = update_molecules.keys()
    moleculeChanges = update_molecules.values()
    listeners = update_listeners['equilibrium_listener']
    rates = listeners['reaction_rates']

    # Sanity checks: wcEcoli and vivarium-ecoli match
    assert len(moleculeNames) == len(wc_moleculeNames), (
        f"Mismatch in lengths: vivarium-ecoli molecule_list update has length {len(moleculeNames)}\n"
        f"while wcecoli has {len(wc_moleculeNames)} molecules.")

    assert len(moleculeChanges) == len(wc_d_molecules), (
        f"Mismatch in lengths: vivarium-ecoli moleculeCounts update has length {len(moleculeChanges)}\n"
        f"while wcecoli has {len(wc_d_molecules)}.")

    assert len(wc_rxnEvolve) == len(rates), (
        f"Mismatch in lengths: vivarium-ecoli metabolite update has length {len(d_metabolites)}\n"
        f"while wcecoli has {len(metabolite_ids)} metabolites with {len(wc_metabolites)} values.")


    assert set(wc_moleculeNames) == set(moleculeNames), (
        "Mismatch between protein ids in vivarium-ecoli and wcEcoli.")


    utest_threshold = 0.05

    utest_partioned_molecules = mannwhitneyu(wc_d_molecules, [d for d in moleculeChanges], alternative="two-sided")
    utest_rates = mannwhitneyu(rates, wc_rates, alternative="two-sided")

    percent_error_threshold = 0.05
    molecule_partition_error = percent_error(sum(wc_d_molecules), sum(moleculeChanges))
    rates_error = percent_error(sum(rates), sum(wc_rates))

    assert utest_partioned_molecules.pvalue > utest_threshold, (
        "Distribution of molecules_partioned is different between wcEcoli and vivarium-ecoli"
        f"(p={utest_partioned_molecules.pvalue} <= {utest_threshold}) ")

    assert utest_rates.pvalue > utest_threshold, (
        "Distribution of rates is different between wcEcoli and vivarium-ecoli"
        f"(p={utest_rates.pvalue} <= {utest_threshold}) ")


    assert molecule_partition_error < percent_error_threshold, (
        "Total # of molecules_partioned differs between wcEcoli and vivarium-ecoli"
        f"(percent error = {molecule_partition_error})")

    assert rates_error < percent_error_threshold, (
        "Total Rates differs between wcEcoli and vivarium-ecoli"
        f"(percent error = {rates_error})")

    plt.figure(figsize=(10,8))
    plt.subplot(2, 1, 1, aspect='equal')
    qqplot(np.array(wc_d_molecules).astype(int), np.array(list(moleculeChanges)).astype(int), quantile_precision=0.0001)
    plt.xlabel("wcEcoli")
    plt.ylabel("vivarium-ecoli")
    plt.title("QQ-plot of changes per molecule")
    plt.tight_layout()
    plt.subplot(2, 1, 2, aspect='equal')
    qqplot(np.array(wc_rates).astype(int), np.array(rates).astype(int), quantile_precision=0.0001)
    plt.xlabel("wcEcoli")
    plt.ylabel("vivarium-ecoli")
    plt.title("QQ-plot of rate changes per partitioned molecule")

    plt.savefig("out/migration/equilibrium_figures.png")

if __name__ == "__main__":
    test_equilibrium()

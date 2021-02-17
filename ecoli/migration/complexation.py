import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from scipy.stats import mannwhitneyu
from vivarium.core.experiment import Experiment

from ecoli.library.sim_data import LoadSimData
from ecoli.composites.ecoli_master import SIM_DATA_PATH, get_state_from_file
from ecoli.processes.complexation import Complexation

from ecoli.migration.plots import qqplot
from ecoli.migration.migration_utils import run_ecoli_process, percent_error


load_sim_data = LoadSimData(
            sim_data_path=SIM_DATA_PATH,
            seed=0)



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

if __name__ == "__main__":
    test_complexation()

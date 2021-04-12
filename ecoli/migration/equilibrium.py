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

#from ecoli.migration.write_json import write_json
import pandas

load_sim_data = LoadSimData(
            sim_data_path=SIM_DATA_PATH,
            seed=0)



def test_equilibrium():
    config = load_sim_data.get_equilibrium_config()
    equil = Equilibrium(config)

    #molecules = config["molecule_names"]

    topology = {
        'molecules': ('bulk',)
        }

    # run the process and get an update
    actual_update = run_ecoli_process(equil, topology)

    import ipdb; ipdb.set_trace()
    with open("data/complexation_update9_t2.json") as f:
        wc_complexation_data = json.load(f)

    d_molecules = actual_update["molecules"]
    wc_molecules = wc_complexation_data["molecules"]
    wc_moleculesCounts = wc_complexation_data["migrationCounts"]
    wc_ocurrences = wc_moleculesCounts['outcome']
    wc_outcome = wc_complexation_data['moleculeCounts']
    wc_test = (np.array(wc_outcome) - np.array(wc_ocurrences))
    wc_test2 = wc_complexation_data["test2"]
    wc_test3 = wc_complexation_data["test3"]


    fixing_viv =  list(d_molecules.values())
    fixviv = np.abs(fixing_viv)

    fixing_wc = np.abs(np.abs(wc_test3) - fixviv)
    fakefixWC = fixing_wc
    printMold = {}
    for key in d_molecules:
        for x in fakefixWC:
            #import ipdb; ipdb.set_trace()
            printMold[key] = x
            fakefixWC = np.delete(fakefixWC, 0)
            break

    dictForPrint = sorted(printMold.items(), key=lambda x: x[1], reverse=True)

    #update_to_save = {}
    #saved = False
    #update_to_save["moleculeDifference"] = dictForPrint
    #update_to_save["vivmolecules"] = d_molecules
    #update_to_save["wc_values"] = wc_test2
    #if not saved and "wc_values" in update_to_save.keys():
    #    write_json('out/migration/complexation_ecoliT00.json', update_to_save)
    #    saved = True
    import ipdb; ipdb.set_trace()


    assert len(wc_molecules) == len(d_molecules), (
        f"Mismatch in lengths: vivarium-ecoli molecule update has length {len(d_molecules)}\n"
        f"while wcecoli has {len(wc_molecules)}.")

    assert set(d_molecules.keys()) == set(wc_molecules), (
        "Mismatch between moleules in vivarium-ecoli and wcEcoli.")

    utest_threshold = 0.05
    utest_molecules = mannwhitneyu(wc_test, [-p for p in d_molecules.values()], alternative="two-sided")
    percent_error_threshold = 0.05
    molecules_error = percent_error(-sum(d_molecules.values()), sum(wc_test))
    import ipdb; ipdb.set_trace()
    #assert molecules_error < percent_error_threshold, (
    #    "Total # of molecules changed differs between wcEcoli and vivarium-ecoli"
    #    f"(percent error = {molecules_error})" )


if __name__ == "__main__":
    test_equilibrium()

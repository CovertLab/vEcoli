"""
====================================
Tetracycline-Induced Gene Regulation
====================================

This file contains functions to plot the output of simulations where tetracycline 
induces a change in gene expression via the enabling of the `mar_regulon` option.
- runDefault(): run and save data for a default simulation for plotting
- includeTetracycline(x): run a simulation with x molecules of tetracycline
- allPlots(): plots graphs comparing protein, mRNA, and mRNA synthesis probabilities
              of two saved experiment IDs
- All other methods: helper functions to make plots
"""
import gc
import ast
import json
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from vivarium.core.emitter import timeseries_from_data
from vivarium.plots.simulation_output import plot_variables

from ecoli.analysis.analyze_db_experiment import access
from ecoli.experiments.ecoli_master_sim import EcoliSim

def runDefault():
    sim = EcoliSim.from_file()
    sim.raw_output = False
    sim.total_time = 3000
    sim.emitter = "database"
    
    sim.run()
    
def includeTetracycline(tet=None):
    sim = EcoliSim.from_file()
    sim.raw_output = False
    sim.total_time = 3000
    sim.emitter = "database"
    sim.config['mar_regulon'] = True
    if tet != None:
        with open('data/wcecoli_t0.json') as f:
            initial_state = json.load(f)
        # Pre-seed tetracycline and add marR-tet complex
        initial_state['bulk']['tetracycline[c]'] = int(tet)
        initial_state['bulk']['marR-tet[c]'] = 0
        # Add promoter binding data for marA and marR
        for promoter_data in initial_state['unique']['promoter'].values():
            promoter_data['bound_TF'] += [False, False]
        with open('data/wcecoli_marA.json', 'w') as f:
            json.dump(initial_state, f)
    sim.initial_state_file = "wcecoli_marA"
    
    sim.run()

def count_by_tu_idx(data):
    tu_idx = np.zeros(len(data))
    for i, rna in enumerate(data.values()):
        tu_idx[i] = rna["TU_index"]
    tu_idx = tu_idx.astype(int)
    return np.bincount(tu_idx, minlength=4687)

def plot_mrnas(mrna_ts, baseline_mrna_ts, name, idx, genes, fc):
    mrna_data = np.array(mrna_ts["unique"]["RNA"])[:, idx]
    baseline_mrna_data = np.array(baseline_mrna_ts["unique"]["RNA"])[:, idx]
    
    n_mrnas = len(idx)
    fig, axes = plt.subplots(n_mrnas, 1, figsize=(8, 4 * n_mrnas), tight_layout=True)

    for i, gene in enumerate(genes):
        title = gene
        x = np.arange(start=0, stop=len(mrna_data[:,i])*2, step=2)
        y = mrna_data[:,i]
        axes[i].plot(x, y, label="marA")
        x_exp = np.arange(start=0, stop=len(baseline_mrna_data[:,i])*2, step=2)
        y_exp = baseline_mrna_data[:, i]
        axes[i].plot(x_exp, y_exp, 'r--', label="base")
        target = baseline_mrna_data[:, i] * float(fc[i])
        axes[i].plot(x_exp, target, 'g--', label="target")
        axes[i].set_title(title)
    plt.tight_layout()
    plt.savefig(f"ecoli/experiments/marA_binding/{name}.png")
    
def plot_rna_synth_prob(timeseries, baseline_ts, name, idx, genes, fc):
    rna_synth_prob = timeseries["listeners"]["rna_synth_prob"]["rna_synth_prob"][2:]
    rna_synth_prob = np.array(rna_synth_prob)[:, idx]
    baseline_rna_synth_prob = baseline_ts["listeners"]["rna_synth_prob"]["rna_synth_prob"][2:]
    baseline_rna_synth_prob = np.array(baseline_rna_synth_prob)[:, idx]
    
    n_mrnas = len(idx)
    fig, axes = plt.subplots(n_mrnas, 1, figsize=(8, 4 * n_mrnas), tight_layout=True)

    for i, gene in enumerate(genes):
        title = gene
        x = np.arange(start=0, stop=len(rna_synth_prob[:,i])*2, step=2)
        y = rna_synth_prob[:,i]
        axes[i].plot(x, y, label="marA")
        x_exp = np.arange(start=0, stop=len(baseline_rna_synth_prob[:,i])*2, step=2)
        y_exp = baseline_rna_synth_prob[:, i]
        axes[i].plot(x_exp, y_exp, 'r--', label="base")
        target = baseline_rna_synth_prob[:, i] * float(fc[i])
        axes[i].plot(x_exp, target, 'g--', label="target")
        axes[i].set_title(title)
    plt.tight_layout()
    plt.savefig(f"ecoli/experiments/marA_binding/{name}.png")
    
def ids_of_interest():
    model_degenes = pd.read_csv("ecoli/experiments/marA_binding/model_degenes.csv")
    bulk_paths = []
    for bulk_names, common_id, fold_change in zip(
            model_degenes["bulk_ids"], model_degenes["common_name"],
            model_degenes["Fold change"]):
        common_names = [common_id]*len(bulk_names)
        bulk_names = ast.literal_eval(bulk_names)
        bulk_paths += [
            {
                "variable": ('bulk', bulk_name),
                "display": f"{common_name}: {fold_change+1}x baseline"
            }
            for bulk_name, common_name in zip(
                bulk_names, common_names)
        ]
    complex_paths = []
    for complex_names, monomers_used, common_id in zip(
            model_degenes["complex_ids"], model_degenes["monomers_used"], 
            model_degenes["common_name"]):
        if len(complex_names) == 0:
            continue
        common_names = [common_id]*len(complex_names)
        complex_names = ast.literal_eval(complex_names)
        monomers_used = np.array(ast.literal_eval(monomers_used))
        monomers_used[monomers_used == None] = 0
        complex_paths += [
            {
                "variable": ('bulk', complex_name),
                "color": "tab:green",
                "display": f"{complex_name} " +
                    f"({common_name} x {-monomer_used})"
            }
            for complex_name, common_name, monomer_used in zip(
                complex_names, common_names, monomers_used)
        ]

    return bulk_paths + complex_paths

def extract_data(timeseries, variable_paths):
    fig = plot_variables(
        timeseries, 
        variables=variable_paths
    )
    data = {}
    for axes in fig.axes:
        line = axes.get_lines()[0]
        title = axes.get_title()
        data.update({
            title: {
                "x_data": line.get_xdata(),
                "y_data": line.get_ydata()
            }
        })
    plt.close(fig="all")
    complex_names = []
    dict_keys = list(data.keys())
    for title in dict_keys:
        if ":" in title:
            fold_change = title.split(": ")[-1].split("x ")[0]
            data[title.split(": ")[0]] = data.pop(title)
            data[title.split(": ")[0]]['fold_change'] = fold_change
            continue
        if "(" in title:
            monomer = title.split("(")[-1].split(" x")[0]
            counts = title.split("(")[-1].split("x ")[-1].split(")")[0]
            data[monomer]["y_data"] += int(counts)*data[title]["y_data"]
            complex_names.append(title)
    for key in complex_names:
        data.pop(key)
    
    return data

def plot_degenes(timeseries, baseline, name, variable_paths):
    data = extract_data(timeseries, variable_paths)
    marR_tet = -1
    for i, variable_path in enumerate(variable_paths):
        if variable_path['variable'] == ('bulk', 'marR-tet[c]'):
            marR_tet = i         
    variable_paths.pop(marR_tet)
    baseline_data = extract_data(baseline, variable_paths)
    
    n_monomers = len(data)
    fig, axes = plt.subplots(n_monomers, 1, figsize=(8, 4 * n_monomers), tight_layout=True)

    for i, key in enumerate(data.keys()):
        monomer_data = data[key]
        baseline_monomer_data = baseline_data[key]
        title = key
        x = monomer_data['x_data']
        y = monomer_data['y_data']
        y_exp = baseline_monomer_data['y_data']
        axes[i].plot(x, y, label="marA")
        axes[i].plot(x, y_exp, 'r--', label="base")  
        axes[i].set_title(title)
    plt.tight_layout()
    plt.savefig(f"ecoli/experiments/marA_binding/{name}.png")
    
def all_plots(marA_id, baseline_id, name):
    # Get metadata
    degenes = pd.read_csv("ecoli/experiments/marA_binding/model_degenes.csv")
    TU_idx = degenes["TU_idx"].to_list()
    genes = degenes["Gene name"]
    fc = degenes["Fold change"]
    variable_paths = ids_of_interest()
    
    # Perform regular garbage collection to free memory
    # Plot monomer counts (including monomers in complexes)
    paths = [('agents', '0', ) + i['variable'] for i in variable_paths]
    marA_bulk_data = access(marA_id, paths)[0]
    marA_bulk_timeseries = timeseries_from_data(marA_bulk_data)
    time = marA_bulk_timeseries['time']
    marA_bulk_timeseries = marA_bulk_timeseries['agents']['0']
    marA_bulk_timeseries['time'] = time
    del marA_bulk_data
    gc.collect()
    baseline_bulk_data = access(baseline_id, paths)[0]
    baseline_bulk_timeseries = timeseries_from_data(baseline_bulk_data)
    time = baseline_bulk_timeseries['time']
    baseline_bulk_timeseries = baseline_bulk_timeseries['agents']['0']
    baseline_bulk_timeseries['time'] = time
    del baseline_bulk_data
    gc.collect()
    plot_degenes(marA_bulk_timeseries, baseline_bulk_timeseries, f"genes_{name}", variable_paths)
    del marA_bulk_timeseries, baseline_bulk_timeseries, variable_paths
    gc.collect()
    
    # # Plot mRNA counts
    # # NOTE: Requires changes to `emitter.py` from the `memory` branch of vivarium-core
    # marA_mrna_data = access(marA_id, [("agents", "0", "unique", "RNA")], f=count_by_tu_idx)[0]
    # marA_mrna_timeseries = timeseries_from_data(marA_mrna_data)
    # del marA_mrna_data
    # gc.collect()
    # baseline_mrna_data = access(baseline_id, [("agents", "0", "unique", "RNA")], f=count_by_tu_idx)[0]
    # baseline_mrna_timeseries = timeseries_from_data(baseline_mrna_data)
    # del baseline_mrna_data
    # gc.collect()
    # plot_mrnas(marA_mrna_timeseries, baseline_mrna_timeseries, f"mrna_{name}", TU_idx, genes, fc)
    # del marA_mrna_timeseries, baseline_mrna_timeseries
    # gc.collect()
    
    # Plot RNA synthesis probabilities
    rna_synth_prob_data = access(marA_id, [("agents", "0", "listeners", "rna_synth_prob", "rna_synth_prob")])[0]
    rna_synth_prob_ts = timeseries_from_data(rna_synth_prob_data)
    rna_synth_prob_ts = rna_synth_prob_ts['agents']['0']
    del rna_synth_prob_data
    gc.collect()
    baseline_rna_synth_prob_data = access(baseline_id, [("agents", "0", "listeners", "rna_synth_prob", "rna_synth_prob")])[0]
    baseline_rna_synth_prob_ts = timeseries_from_data(baseline_rna_synth_prob_data)
    baseline_rna_synth_prob_ts = baseline_rna_synth_prob_ts['agents']['0']
    del baseline_rna_synth_prob_data
    gc.collect()
    plot_rna_synth_prob(rna_synth_prob_ts, baseline_rna_synth_prob_ts, f"synth_prob_{name}", TU_idx, genes, fc)
    del rna_synth_prob_ts, baseline_rna_synth_prob_ts
    gc.collect()

def main():
    # Generate baseline/experimental data
    # runDefault()
    # includeTetracycline(2200)
    # Experimental id for marA simulation
    marA_id = "eb39ea8a-09fe-11ed-ab2f-9cfce8b9977c"
    # Experimental id for baseline simulation
    baseline_id = "cf37764a-09fe-11ed-80ab-9cfce8b9977c"
    # Suffix for output graphs
    name = "full_integration"
        
    all_plots(marA_id, baseline_id, name)

if __name__=="__main__":
    main()

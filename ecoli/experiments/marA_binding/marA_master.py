import ast
import json
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from os.path import exists

from vivarium.core.emitter import timeseries_from_data
from vivarium.plots.simulation_output import plot_variables

from ecoli.analysis.analyze_db_experiment import access
from ecoli.experiments.ecoli_master_sim import EcoliSim

def runDefault():
    sim = EcoliSim.from_file()
    sim.raw_output = False
    sim.total_time = 500
    sim.emitter = "database"
    
    sim.run()

def includeMarA():
    sim = EcoliSim.from_file()
    sim.raw_output = False
    sim.total_time = 500
    sim.emitter = "database"
    sim.config['mar_regulon'] = True
    if not exists('data/wcecoli_marA.json'):
        with open('data/wcecoli_t0.json') as f:
            initial_state = json.load(f)
        # Add promoter binding data for marA and marR
        for promoter_data in initial_state['unique']['promoter'].values():
            promoter_data['bound_TF'] += [False, False]
        with open('data/wcecoli_marA.json', 'w') as f:
            json.dump(initial_state, f)
    sim.initial_state_file = "wcecoli_marA"
    
    sim.run()
    
def includeTetracycline():
    sim = EcoliSim.from_file()
    sim.raw_output = False
    sim.total_time = 500
    sim.emitter = "database"
    sim.config['mar_regulon'] = True
    if not exists('data/wcecoli_tet.json'):
        with open('data/wcecoli_marA.json') as f:
            initial_state = json.load(f)
        # Pre-seed tetracycline and add tet-marR complex
        initial_state['bulk']['tetracycline[c]'] = 200
        initial_state['bulk']['tet-marR[c]'] = 0
        with open('data/wcecoli_tet.json', 'w') as f:
            json.dump(initial_state, f)
    sim.initial_state_file = "wcecoli_tet"
    
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
        y_exp = baseline_mrna_data[:, i] * float(fc[i])
        y_exp = baseline_mrna_data[:, i] 
        axes[i].plot(x_exp, y_exp, 'r--', label="base")
        axes[i].set_title(title)
    plt.tight_layout()
    plt.savefig(f"ecoli/experiments/marA_binding/{name}.png")
    
def plot_rna_synth_prob(timeseries, baseline_ts, name, idx, genes, fc):
    rna_synth_prob = timeseries["listeners"]["rna_synth_prob"]["rna_synth_prob"][1:]
    rna_synth_prob = np.array(rna_synth_prob)[:, idx]
    baseline_rna_synth_prob = baseline_ts["listeners"]["rna_synth_prob"]["rna_synth_prob"][1:]
    baseline_rna_synth_prob = np.array(baseline_rna_synth_prob)[:, idx]
    
    n_mrnas = len(idx)
    fig, axes = plt.subplots(n_mrnas, 1, figsize=(8, 4 * n_mrnas), tight_layout=True)

    for i, gene in enumerate(genes):
        title = gene
        x = np.arange(start=0, stop=len(rna_synth_prob[:,i])*2, step=2)
        y = rna_synth_prob[:,i]
        axes[i].plot(x, y, label="marA")
        x_exp = np.arange(start=0, stop=len(baseline_rna_synth_prob[:,i])*2, step=2)
        y_exp = baseline_rna_synth_prob[:, i] * float(fc[i])
        y_exp = baseline_rna_synth_prob[:, i]
        axes[i].plot(x_exp, y_exp, 'r--', label="base")
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
    for complex_names, monomers_used, common_id, complex_common_names in zip(
            model_degenes["complex_ids"], model_degenes["monomers_used"], 
            model_degenes["common_name"], model_degenes["complex_common_names"]):
        if len(complex_names) == 0:
            continue
        common_names = [common_id]*len(complex_names)
        complex_names = ast.literal_eval(complex_names)
        monomers_used = np.array(ast.literal_eval(monomers_used))
        monomers_used[monomers_used == None] = 0
        complex_common_names = ast.literal_eval(complex_common_names)
        complex_paths += [
            {
                "variable": ('bulk', complex_name),
                "color": "tab:green",
                "display": f"{complex_name} aka {complex_common_name[:45]} " +
                    f"({common_name} x {-monomer_used})"
            }
            for complex_name, common_name, monomer_used, complex_common_name in zip(
                complex_names, common_names, monomers_used, complex_common_names)
        ]

    return bulk_paths + complex_paths

def extract_data(timeseries, variable_paths):
    fig = plot_variables(
        timeseries, 
        variables=variable_paths,
        # Get list of misbehaving molecules to report at meetings
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
    baseline_data = extract_data(baseline, variable_paths)
    
    n_monomers = len(data)
    fig, axes = plt.subplots(n_monomers, 1, figsize=(8, 4 * n_monomers), tight_layout=True)

    for i, key in enumerate(data.keys()):
        monomer_data = data[key]
        baseline_monomer_data = baseline_data[key]
        title = key
        x = monomer_data['x_data']
        assert (x == baseline_monomer_data['x_data']).all()
        y = monomer_data['y_data']
        fold_change = monomer_data['fold_change']
        assert fold_change == baseline_monomer_data['fold_change']
        y_exp = baseline_monomer_data['y_data']
        axes[i].plot(x, y, label="marA")
        axes[i].plot(x, y_exp*float(fold_change), 'r--', label="base")
        axes[i].set_title(title)
    plt.tight_layout()
    plt.savefig(f"ecoli/experiments/marA_binding/{name}.png")

def main():
    # Generate baseline/experimental data
    # runDefault()
    # includeMarA()
    # includeTetracycline()
    
    # Get metadata
    degenes = pd.read_csv("ecoli/experiments/marA_binding/model_degenes.csv")
    TU_idx = degenes["TU_idx"].to_list()
    genes = degenes["Gene name"]
    fc = degenes["Fold change"] + 1

    # f67f750e-afa4-11ec-b92c-9cfce8b9977c: control (500 seconds)
    # 4a361a0e-b04f-11ec-80b7-9cfce8b9977c: control (2600 seconds)    
    # cb78bdea-afa7-11ec-8c9a-9cfce8b9977c: deltaV = first guess (500 seconds)
    # 060d0b24-b52e-11ec-8c7d-9cfce8b9977c: added marR to "cancel out" basal marA expression
    # cf00d53c-b9f8-11ec-bb30-9cfce8b9977c: adjusted deltaVs, with marR
    # 61e43714-ba02-11ec-a54c-9cfce8b9977c: adjusted deltas further and changed marA mRNA to marR in initial state
    # 29947084-ba09-11ec-a834-9cfce8b9977c: account for random marA appearing at t=10 ruining things (recapitulate control!)
    # 1cdc4df0-ba20-11ec-bc53-9cfce8b9977c: add tetracycline-marR complexation inactivation (50 sec)
    # cc06c8a0-ba20-11ec-97e2-9cfce8b9977c: add tetracycline-marR complexation inactivation (500 sec, 12 tet)
    # 27e76e74-ba6a-11ec-bac3-9cfce8b9977c: presence of active marR is binary switch for marA binding (500 sec, 12 tet)
    # 2cecaae2-ba9c-11ec-9728-9cfce8b9977c: make adj to deltaV (500 sec, 12 tet)
    # bf8dd480-baa0-11ec-9b39-9cfce8b9977c: even more adj to deltaV (500 sec, 200 tet)
    # 7d119bc8-baa5-11ec-bf40-9cfce8b9977c: even, even more adj to deltaV (500 sec, 200 tet)
    # f088e4ce-baa7-11ec-b7d6-9cfce8b9977c: final adj to deltaV (500 sec, 200 tet)
    # 3d85657a-baaa-11ec-b38c-9cfce8b9977c: recapitulate control w/ final adj (500 sec)
    
    marA_id = "3d85657a-baaa-11ec-b38c-9cfce8b9977c"
    marA_mrna_data = access(marA_id, [("unique", "RNA")], count_by_tu_idx)[0]
    marA_mrna_timeseries = timeseries_from_data(marA_mrna_data)
    rna_synth_prob_data = access(marA_id, [("listeners", "rna_synth_prob", "rna_synth_prob")])[0]
    rna_synth_prob_ts = timeseries_from_data(rna_synth_prob_data)
    
    baseline_id = "f67f750e-afa4-11ec-b92c-9cfce8b9977c"
    baseline_mrna_data = access(baseline_id, [("unique", "RNA")], count_by_tu_idx)[0]
    baseline_mrna_timeseries = timeseries_from_data(baseline_mrna_data)
    baseline_rna_synth_prob_data = access(baseline_id, [("listeners", "rna_synth_prob", "rna_synth_prob")])[0]
    baseline_rna_synth_prob_ts = timeseries_from_data(baseline_rna_synth_prob_data)
    
    tet_timeseries = access(marA_id, [("bulk", "tetracycline[c]")])[0]
    marR_timeseries = access(marA_id, [("bulk", "CPLX0-7710[c]")])[0]
    marA_timeseries = access(marA_id, [("bulk", "PD00365[c]")])[0]
    tet_marR_timeseries = access(marA_id, [("bulk", "tet-marR[c]")])[0]
    
    plot_mrnas(marA_mrna_timeseries, baseline_mrna_timeseries, "mrna_no_tet", TU_idx, genes, fc)
    plot_rna_synth_prob(rna_synth_prob_ts, baseline_rna_synth_prob_ts, "synth_prob_no_tet", TU_idx, genes, fc)

if __name__=="__main__":
    main()

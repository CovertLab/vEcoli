import pandas as pd
import numpy as np
import ast
import json
from matplotlib import pyplot as plt
from os.path import exists

from ecoli.experiments.ecoli_master_sim import EcoliSim
from vivarium.plots.simulation_output import plot_variables
from ecoli.analysis.analyze_db_experiment import access
from vivarium.core.emitter import timeseries_from_data

def runDefault(paths):
    sim = EcoliSim.from_file()
    sim.raw_output = False
    sim.total_time = 500
    sim.emitter = "database"
    
    query = [i['variable'] for i in paths]
    sim.run()
    
    return sim.ecoli_experiment.experiment_id


def includeMarA(paths, baseline_id):
    sim = EcoliSim.from_file()
    sim.raw_output = False
    sim.total_time = 500
    sim.emitter = "database"
    if not exists('data/wcecoli_marA.json') or not exists('data/wcecoli_marA_added.json'):
        with open('data/wcecoli_t0.json') as f:
            initial_state = json.load(f)
        for promoter in initial_state['unique']['promoter']:
            initial_state['unique']['promoter'][promoter]['bound_TF'] += [False]
        with open('data/wcecoli_marA.json', 'w') as f:
            json.dump(initial_state, f)
        initial_state['bulk']['PD00365[c]'] *= 4
        with open('data/wcecoli_marA_added.json', 'w') as f:
            json.dump(initial_state, f)
    sim.processes.pop('ecoli-tf-binding')
    sim.processes = {'ecoli-tf-binding-marA': None, **sim.processes}
    sim.initial_state_file = "wcecoli_marA"
    
    query = [i['variable'] for i in paths]
    sim.run()
    timeseries = sim.query(query)
    
    # TODO: Plot mRNA counts instead of final protein counts
    
    baseline_data = access(baseline_id, query)
    baseline_timeseries = timeseries_from_data(baseline_data[0])

    plot_degenes(timeseries, baseline_timeseries, "marA_1000", paths)

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
    
def timeseries_from_unique(data, path, f):
    times_vector = list(data.keys())
    embedded_timeseries: dict = {}
    for value in data.values():
        if isinstance(value, dict):
            embedded_timeseries = value_in_embedded_dict(
                value, embedded_timeseries, path, f)
    embedded_timeseries['time'] = times_vector
    return embedded_timeseries

def value_in_embedded_dict(
        data,
        timeseries,
        path,
        f):
    timeseries = timeseries or {}

    for key, value in data.items():
        if isinstance(value, dict):
            if key == path[0]:
                if len(path) == 1:
                    if key not in timeseries:
                        timeseries[key] = []
                    timeseries[key].append(f(value))
                else:
                    if key not in timeseries:
                        timeseries[key] = {}
                    path = path[1:]
                    timeseries[key] = value_in_embedded_dict(value, timeseries[key], path, f)

    return timeseries

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
        y_exp = baseline_mrna_data[:, i]
        axes[i].plot(x, y, label="marA")
        x_exp = np.arange(start=0, stop=len(baseline_mrna_data[:,i])*2, step=2)
        fold_change = fc[i]
        axes[i].plot(x_exp, y_exp*float(fold_change), 'r--', label="base")
        axes[i].set_title(title)
    plt.tight_layout()
    plt.savefig(f"ecoli/experiments/marA_binding/{name}.png")

def main():
    degenes = pd.read_csv("ecoli/experiments/marA_binding/model_degenes.csv")
    TU_idx = degenes["TU_idx"].to_list()
    genes = degenes["Gene name"]
    fc = degenes["Fold change"] + 1
    paths = ids_of_interest()
    runDefault()
    # query = [i['variable'] for i in paths]
    # exp_id = runDefault(paths)
    # includeMarA(paths, exp_id)
    # e28c1670-99af-11ec-a9df-9cfce8b9977c: control (500 seconds)
    # 75294312-99b1-11ec-93b7-9cfce8b9977c: deltaV = fold change - 1 (500 seconds)
    # e6a0f8da-99b3-11ec-be35-9cfce8b9977c: deltaV = (fold change - 1)/1000 (500 seconds)
    exp_id = "e28c1670-99af-11ec-a9df-9cfce8b9977c"
    # baseline_monomer_data = access(exp_id, query)[0]
    # baseline_monomer_timeseries = timeseries_from_data(baseline_monomer_data)
    # baseline_mrna_data = access(exp_id, [("unique", "RNA")])[0]
    # baseline_mrna_timeseries = timeseries_from_unique(baseline_mrna_data, ("unique", "RNA"), count_by_tu_idx)
    
    exp_id = "e6a0f8da-99b3-11ec-be35-9cfce8b9977c"
    #marA_monomer_data = access(exp_id, query)[0]
    #marA_monomer_timeseries = timeseries_from_data(marA_monomer_data)
    # marA_mrna_data = access(exp_id, [("unique", "RNA")])[0]
    # marA_mrna_timeseries = timeseries_from_unique(marA_mrna_data, ("unique", "RNA"), count_by_tu_idx)
    
    # TODO: Run several simulations with different seeds and longer times and plot an average

    # plot_mrnas(marA_mrna_timeseries, baseline_mrna_timeseries, "mrnas", TU_idx, genes, fc)

if __name__=="__main__":
    main()

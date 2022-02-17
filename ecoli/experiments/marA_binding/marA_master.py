import pandas as pd
import numpy as np
import ast
import json
from matplotlib import pyplot as plt
from os.path import exists

from ecoli.experiments.ecoli_master_sim import EcoliSim
from vivarium.plots.simulation_output import plot_variables

def runDefault(paths):
    sim = EcoliSim.from_file()
    sim.raw_output = False
    sim.total_time = 500
    sim.emitter = "database"
    
    query = [i['variable'] for i in paths]
    sim.run()
    timeseries = sim.query(query)

    plot_degenes(timeseries, "default", paths)


def includeMarA(paths):
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
    #query += [("bulk", "ACRB-MONOMER[p]")]
    sim.run()
    timeseries = sim.query(query)
    # Consider using "inherit_from" key (see lysis.json) to inherit from silent_unqiue.json once that is fixed
    # vivarium scripts to clear out MongoDB database
    #print(timeseries["bulk"]["ACRB-MONOMER[p]"])
    
    # TODO: Plot mRNA counts instead of final protein counts
    # TODO: Visualization tool
    #   - Plot data from baseline run, marA-added run, and baseline * expected fold change

    plot_degenes(timeseries, "marA", paths)

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

def plot_degenes(timeseries, name, variable_paths):
    
    fig = plot_variables(
        timeseries, 
        variables=variable_paths,
        # TODO: Try to find a way to overlay line plots from different runs
        # TODO: Put expected fold change in plot as dashed line
        # Get list of misbehaving molecules to report at meetings
        # Use mRNA fold changes from 10 mg/L tetracycline from PLoS ONE paper
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
    complex_names = []
    for title in data.keys():
        if "(" in title:
            monomer = title.split("(")[-1].split(" x")[0]
            counts = title.split("(")[-1].split("x ")[-1].split(")")[0]
            data[monomer]["y_data"] += int(counts)*data[title]["y_data"]
            complex_names.append(title)
        if ":" in title:
            fold_change = title.split(": ")[-1].split("x ")[0]
            data[title]["fold_change"] = fold_change
    for key in complex_names:
        data.pop(key)
    plt.close(fig="all")
    n_monomers = len(data)
    for i, monomer_data in enumerate(data.items()):
        plt.subplot(n_monomers, 1, i+1)
        title = monomer_data[0]
        x = monomer_data[1]['x_data']
        y = monomer_data[1]['y_data']
        fold_change = monomer_data[1]['fold_change']
        plt.plot(x, y)
        plt.title(title)
        # TODO: Extract baseline data from default sim for expected value
        plt.plot(x, y*fold_change, '-')
    plt.savefig(f"ecoli/experiments/marA_binding/{name}.png")
    

def main():
    paths = ids_of_interest()
    #runDefault(paths)
    includeMarA(paths)

if __name__=="__main__":
    main()

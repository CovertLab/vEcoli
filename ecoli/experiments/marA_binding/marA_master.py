import pandas as pd
import json
from os.path import exists

from ecoli.experiments.ecoli_master_sim import EcoliSim
from ecoli.states.wcecoli_state import get_state_from_file
from vivarium.plots.simulation_output import plot_variables

def testDefault():
    sim = EcoliSim.from_file()
    sim.raw_output = False
    timeseries = sim.run()

    plot_degenes(timeseries, "default")


def testMarA():
    sim = EcoliSim.from_file()
    sim.raw_output = False
    if not exists('data/wcecoli_marA_added.json'):
        with open('data/wcecoli_t0.json') as f:
            initial_state = json.load(f)
        for promoter in initial_state['unique']['promoter']:
            initial_state['unique']['promoter'][promoter]['bound_TF'] += [False]
        initial_state['bulk']['PD00365[c]'] *= 1
        with open('data/wcecoli_marA_added.json', 'w') as f:
            json.dump(initial_state, f)
    sim.processes.pop('ecoli-tf-binding')
    sim.processes = {'ecoli-tf-binding-marA': None, **sim.processes}
    sim.initial_state_file = "wcecoli_marA_added"

    timeseries = sim.run()

    plot_degenes(timeseries, "marA_added_1000")

def plot_degenes(timeseries, name):
    model_degenes = pd.read_csv("ecoli/experiments/marA_binding/model_degenes.csv")
    variable_paths = [
        ('bulk', bulk_name)
        for bulk_name in model_degenes["bulk_name"]
    ]
    fig = plot_variables(
        timeseries, 
        variables=variable_paths,
        # Change variable to use a dictionary with "display" key
        # Try to find a way to overlay line plots from different runs
        # Put expected fold change in plot as dashed line
        # Get list of misbehaving molecules to report at meetings
    )
    fig.tight_layout()
    fig.savefig(f"ecoli/experiments/marA_binding/{name}.png")
    

def main():
    #testDefault()
    testMarA()

if __name__=="__main__":
    main()

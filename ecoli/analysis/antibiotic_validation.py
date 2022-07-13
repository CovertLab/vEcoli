import os
import argparse
import numpy as np
from matplotlib import pyplot as plt

from vivarium.core.serialize import deserialize_value
from vivarium.core.emitter import timeseries_from_data
from vivarium.library.units import remove_units

from ecoli.composites.ecoli_engine_process import run_simulation
from ecoli.analysis.analyze_db_experiment import access
from ecoli.experiments.ecoli_master_sim import CONFIG_DIR_PATH, SimConfig


def do_plot(data, out_dir):
    os.makedirs(out_dir, exist_ok = True)
    data = timeseries_from_data(data)
    tetracycline = data['cytoplasm']['concentrations']['tetracycline']
    external = data['boundary']['external']['tetracycline']
    time = data['time']
    expected = np.array(external) * 4
    plt.plot(time, tetracycline)
    plt.plot(time, expected, 'r--')
    plt.savefig(os.path.join(out_dir, 'tet_transport.png'))
    

def run_plot():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--experiment_id', '-e', type=str,
        help='ID of experiment to plot data for.'
    )
    parser.add_argument(
        '--agent_id', '-a', type=str,
        help='ID of agent. If not specified, assume single-cell sim.'
    )
    args = parser.parse_args()
    
    query = [('cytoplasm', 'concentrations', 'tetracycline'), ('boundary', 'external', 'tetracycline')]
    
    if args.agent_id:
        query = [('agents', args.agent_id) + path for path in query]
    data, _, _ = access(args.experiment_id, query)
    if args.agent_id:
        data = {
            time: timepoint['agents'][args.agent_id]
            for time, timepoint in data.items()
            if args.agent_id in timepoint['agents']
        }
    data = deserialize_value(data)
    data = remove_units(data)
        
    out_dir = os.path.join('out',' analysis', args.experiment_id)
    if args.agent_id:
        out_dir = os.path.join(out_dir, args.agent_id)
    do_plot(data, out_dir)
    
def run_sim():
    config = SimConfig()
    config.update_from_json(os.path.join(CONFIG_DIR_PATH, "antibiotics_tetracycline_cephaloridine.json"))
    run_simulation(config)

if __name__ == "__main__":
    run_sim()
    # run_plot()

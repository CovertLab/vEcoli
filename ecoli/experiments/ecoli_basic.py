import numpy as np
from vivarium.core.engine import pf
from ecoli.experiments.ecoli_master_sim import EcoliSim, CONFIG_DIR_PATH


def run_default():
    sim = EcoliSim.from_file()
    sim.total_time = 2


    print(f'INITIAL PROCESS CONFIGS: {pf(sim.config["process_configs"])}')

    # update process configs
    sim.override_config({
        'process_configs': {
            'ecoli-metabolism': {
                'config': {}
            }
        }
    })
    print(f'NEW PROCESS CONFIGS: {pf(sim.config["process_configs"])}')

    # build the ecoli model
    sim.build_ecoli()

    # # inspect the sim object
    # process_configs = sim.config['process_configs']
    # sim.processes['ecoli-metabolism']


    # run the simulation
    sim.run()

    # get the data
    # sim.emitter.


if __name__ == "__main__":
    run_default()

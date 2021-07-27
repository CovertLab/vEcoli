"""
Run simulations of Ecoli Master
"""

import argparse
import json

from vivarium.core.engine import Engine
from ecoli.composites.ecoli_master import Ecoli, SIM_DATA_PATH
from ecoli.processes import process_registry


CONFIG_DIR_PATH = 'data/ecoli_master_configs/'

def ecoli_from_file(file_path=CONFIG_DIR_PATH + 'default.json'):
    with open(file_path) as json_file:
        ecoli_config = json.load(json_file)

    processes_dict = {}
    processes = ecoli_config['processes']
    for process_name in processes:
        try:
            processes_dict[process_name] = process_registry.access(process_name)
        except: #TODO: What error?
            raise ValueError(f"Unknown process with name {process_name}. "
                             "Did you call process_registry.register() in ecoli/processes/__init__.py?")
    ecoli_config['processes'] = processes_dict

    topology = ecoli_config['topology']

    return Ecoli(config)


class EcoliSim:

    def __init__(self, config):
        self.config = config

    @staticmethod
    def from_cli():
        parser = argparse.ArgumentParser(description='ecoli_master')
        parser.add_argument(
            '--config', '-c', action='store', default=CONFIG_DIR_PATH + 'default.json',
            help="Path to configuration file for the simulation."
        )
        parser.add_argument(
            '--emitter', '-e', action="store", default="RAM", choices = ["RAM", "database"],
            help="Emitter to use (either RAM or database)."
        )
        parser.add_argument(
            '--seed', '-s', action="store", default=0, type=int,
            help="Random seed."
        )
        parser.add_argument(
            '--runtime', '-t', action="store", default=100, type=float,
            help="Time to run the simulation for."
        )
        parser.add_argument(
            '--generations', '-g', action="store", default=None, type=int,
            help="Number of generations to run the simulation for."
        )
        parser.add_argument(
            '--log_updates', '-b', action="store_true",
            help="Save updates from each process if this flag is set, e.g. for use with blame plot."
        )
        parser.add_argument(
            'path_to_simdata', nargs="*", default=SIM_DATA_PATH,
            help="Path to the sim_data to use for this experiment."
        )
        args = parser.parse_args()


        # make the ecoli model
        self.ecoli = ecoli_from_file()

        self.total_time = args.total_time

        return EcoliSim(...)


    def run(self):
        # make the experiment
        ecoli_experiment = Engine({
            'processes': self.ecoli.processes,
            'topology': self.ecoli.topology,
            'initial_state': self.initial_state,
            'progress_bar': self.progress_bar,
        })

        # run the experiment
        ecoli_experiment.update(self.total_time)

        # retrieve the data
        if self.timeseries_output:
            return ecoli_experiment.emitter.get_timeseries()
        elif self.data_output:
            return ecoli_experiment.emitter.get_data()


if __name__ == '__main__':
    ecoli_sim = EcoliSim.from_cli()
    ecoli_sim.run()
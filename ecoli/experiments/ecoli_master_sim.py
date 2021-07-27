"""
Run simulations of Ecoli Master
"""

import argparse
import json

from vivarium.core.engine import Engine
from vivarium.library.dict_utils import deep_merge
from ecoli.composites.ecoli_master import Ecoli, SIM_DATA_PATH
from ecoli.processes import process_registry


CONFIG_DIR_PATH = 'data/ecoli_master_configs/'


class EcoliSim:
    def __init__(self,
                 config,
                 emitter="RAM",
                 seed=0,
                 total_time=100,
                 generations=None,
                 log_updates=False,
                 timeseries_output=True,
                 sim_data_path=SIM_DATA_PATH):

        # Get processes
        config['processes'] = self._retrieve_processes(config['processes'])
        config['topology'] = self._retrieve_topology(config['topology'])
        self.config = config
        self.processes = config['processes']
        self.topology = config['topology']  # TODO: do these need to be processed into tuples?
        self.emitter = emitter
        self.seed = seed
        self.total_time = total_time
        self.generations = generations
        self.log_updates = log_updates
        self.timeseries_output = timeseries_output
        self.sim_data_path = sim_data_path

    @staticmethod
    def from_cli():
        # TODO: Experiment id

        parser = argparse.ArgumentParser(description='ecoli_master')
        parser.add_argument(
            '--config', '-c', action='store', default=CONFIG_DIR_PATH + 'default.json',
            help="Path to configuration file for the simulation."
        )
        parser.add_argument(
            '--emitter', '-e', action="store", default="RAM", choices=["RAM", "database"],
            help="Emitter to use (either RAM or database)."
        )
        parser.add_argument(
            '--seed', '-s', action="store", default=0, type=int,
            help="Random seed."
        )
        parser.add_argument(
            '--total_time', '-t', action="store", default=100, type=float,
            help="Time to run the simulation for."
        )
        parser.add_argument(
            '--generations', '-g', action="store", default=None, type=int,
            help="Number of generations to run the simulation for."
        )
        parser.add_argument(
            '--log_updates', '-u', action="store_true",
            help="Save updates from each process if this flag is set, e.g. for use with blame plot."
        )
        parser.add_argument(
            '--timeseries', action="store_true",
            help="Whether to emit data in time-series format."
        )
        parser.add_argument(
            'path_to_sim_data', nargs="*", default=SIM_DATA_PATH,
            help="Path to the sim_data to use for this experiment."
        )
        args = parser.parse_args()

        # Load config, deep-merge with default config
        with open(args.config) as config_file:
            ecoli_config = json.load(config_file)

        with open(CONFIG_DIR_PATH + 'default.json') as default_file:
            default_config = json.load(default_file)

        # Use defaults for any attributes not supplied
        ecoli_config = deep_merge(dict(default_config), ecoli_config)

        # TODO: some of these items (e.g. seed) should overwrite values in config instead
        return EcoliSim(ecoli_config,
                        emitter=args.emitter,
                        seed=args.seed,
                        total_time=args.total_time,
                        generations=args.generations,
                        log_updates=args.log_updates,
                        timeseries_output=args.timeseries,
                        sim_data_path=args.path_to_sim_data)

    def _retrieve_processes(self, process_names):
        result = {}
        for process_name in process_names:
            process_class = process_registry.access(process_name)
           
            if not process_class:
                raise ValueError(f"Unknown process with name {process_name}. "
                                 "Did you call process_registry.register() in ecoli/processes/__init__.py?")
           
            result[process_name] = process_class

        return result
    
    def _retrieve_topology(self, topology):
        result = {}
        for process, process_topology in topology.items():
            result[process] = {
                k: tuple(v) for k, v in process_topology.items()
            }
        return result

    def run(self):
        # initialize the ecoli composer
        ecoli_composer = Ecoli(self.config)

        # set path at which agent is initialized
        path = tuple()
        if self.config['divide']:
            path = ('agents', self.config['agent_id'],)

        # get initial state
        initial_state = ecoli_composer.initial_state(path=path)

        # generate the composite at the path
        ecoli = ecoli_composer.generate(path=path)

        # make the experiment
        ecoli_experiment = Engine({
            'processes': ecoli.processes,
            'topology': ecoli.topology,
            'initial_state': initial_state,
            'progress_bar': True, #TODO: make configurable?
        })

        # run the experiment
        ecoli_experiment.update(self.total_time)

        # retrieve the data
        if self.timeseries_output:
            return ecoli_experiment.emitter.get_timeseries()
        else:
            return ecoli_experiment.emitter.get_data()


if __name__ == '__main__':
    ecoli_sim = EcoliSim.from_cli()
    ecoli_sim.run()

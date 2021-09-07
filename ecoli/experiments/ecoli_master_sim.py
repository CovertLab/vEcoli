"""
Run simulations of Ecoli Master
"""

import argparse
import json
import warnings
from datetime import datetime

from vivarium.core.engine import Engine
from vivarium.library.dict_utils import deep_merge
from ecoli.library.logging import write_json
from ecoli.composites.ecoli_master import SIM_DATA_PATH
# Two different Ecoli composites depending on partitioning
import ecoli.composites.ecoli_master
import ecoli.composites.ecoli_partition

from ecoli.processes import process_registry
from ecoli.processes.registries import topology_registry

from ecoli.composites.ecoli_configs import CONFIG_DIR_PATH


class EcoliSim:
    def __init__(self, config):
        # Do some datatype pre-processesing
        config['agents_path'] = tuple(config['agents_path'])
        config['processes'] = {
            process: None for process in config['processes']}

        # store config
        self.config = config

        # Unpack config using Descriptor protocol:
        # All of the entries in config are translated to properties
        # (of EcoliSim class) that get/set an entry in self.config.
        #
        # For example:
        #
        # >> sim = EcoliSim.from_file()
        # >> sim.total_time
        #    10
        # >> sim.config['total_time']
        #    10
        # >> sim.total_time = 100
        # >> sim.config['total_time']
        #    100

        class ConfigEntry():
            def __init__(self, name):
                self.name = name

            def __get__(self, sim, type=None):
                return sim.config[self.name]

            def __set__(self, sim, value):
                sim.config[self.name] = value

        for attr in self.config.keys():
            config_entry = ConfigEntry(attr)
            setattr(EcoliSim, attr, config_entry)

        if self.generations:
            warnings.warn("generations option is not yet implemented!")


    @staticmethod
    def from_file(filepath=CONFIG_DIR_PATH + 'default.json'):
        # Load config, deep-merge with default config
        with open(filepath) as config_file:
            ecoli_config = json.load(config_file)

        with open(CONFIG_DIR_PATH + 'default.json') as default_file:
            default_config = json.load(default_file)

        # Use defaults for any attributes not supplied
        ecoli_config = deep_merge(dict(default_config), ecoli_config)

        return EcoliSim(ecoli_config)

    @staticmethod
    def from_cli():
        parser = argparse.ArgumentParser(description='ecoli_master')
        parser.add_argument(
            '--config', '-c', action='store', default=CONFIG_DIR_PATH + 'default.json',
            help=f"Path to configuration file for the simulation. Defaults to {CONFIG_DIR_PATH + 'default.json'}."
        )
        parser.add_argument(
            '--experiment_id', '-id', action="store",
            help='ID for this experiment. A UUID will be generated if this argument is not used and "experiment_id" is null in the configuration file.'
        )
        parser.add_argument(
            '--emitter', '-e', action="store", choices=["timeseries", "database", "print"],
            help="Emitter to use. Timeseries uses RAMEmitter, database emits to MongoDB, and print emits to stdout."
        )
        parser.add_argument(
            '--seed', '-s', action="store", type=int,
            help="Random seed."
        )
        parser.add_argument(
            '--initial_time', '-t0', action="store",
            help="Time of the initial state to load from (corresponding inital state file must be present in data folder)."
        )
        parser.add_argument(
            '--total_time', '-t', action="store", type=float,
            help="Time to run the simulation for."
        )
        parser.add_argument(
            '--generations', '-g', action="store", type=int,
            help="Number of generations to run the simulation for."
        )
        parser.add_argument(
            '--log_updates', '-u', action="store_true",
            help="Save updates from each process if this flag is set, e.g. for use with blame plot."
        )
        parser.add_argument(
            '--raw_output', action="store_true",
            help="Whether to return data in raw format (dictionary where keys are times, values are states)."
        )
        parser.add_argument(
            'sim_data_path', nargs="*", default=None,
            help="Path to the sim_data to use for this experiment."
        )
        args = parser.parse_args()

        # Load config, deep-merge with default config
        with open(args.config) as config_file:
            ecoli_config = json.load(config_file)

        with open(CONFIG_DIR_PATH + 'default.json') as default_file:
            default_config = json.load(default_file)

        # add attributes from CLI to the config
        for setting, value in vars(args).items():
            if value and setting != "config":
                ecoli_config[setting] = value

        # Use defaults for any attributes not supplied
        ecoli_config = deep_merge(dict(default_config), ecoli_config)

        return EcoliSim(ecoli_config)

    def _retrieve_processes(self,
                            processes,
                            add_processes,
                            exclude_processes,
                            swap_processes):
        result = {}
        for process_name in list(processes.keys()) + list(add_processes):
            if process_name in exclude_processes:
                continue

            if process_name in swap_processes:
                process_name = swap_processes[process_name]

            process_class = process_registry.access(process_name)

            if not process_class:
                raise ValueError(f"Unknown process with name {process_name}. "
                                 "Did you call process_registry.register() in ecoli/processes/__init__.py?")

            result[process_name] = process_class

        return result

    def _retrieve_topology(self,
                           topology,
                           processes,
                           swap_processes,
                           log_updates,
                           divide):
        result = {}

        original_processes = {v: k for k, v in swap_processes.items()}
        for process in processes:
            # Start from default topology if it exists
            original_process = (process
                                if process not in swap_processes.values()
                                else original_processes[process])

            process_topology = topology_registry.access(original_process)
            if process_topology:
                process_topology = dict(process_topology)
            else:
                process_topology = {}

            # Allow the user to override default topology
            if original_process in topology.keys():
                deep_merge(process_topology, {k: tuple(v)
                           for k, v in topology[original_process].items()})

            # For swapped processes, do additional overrides if they are provided
            if process != original_process and process in topology.keys():
                deep_merge(process_topology, {k: tuple(v)
                           for k, v in topology[process].items()})

            result[process] = process_topology

        # Add log_update ports if log_updates is True
        if log_updates:
            for process, ports in result.items():
                result[process]['log_update'] = ('log_update', process,)

        # add division
        if divide:
            result['division'] = {
                'variable': ('listeners', 'mass', 'cell_mass'),
                'agents': self.agents_path}

        return result

    def _retrieve_process_configs(self, process_configs, processes):
        result = {}
        for process in processes:
            result[process] = process_configs.get(process)

            if result[process] == None:
                result[process] = "sim_data"

        return result

    def build_ecoli(self):
        """
        Build self.ecoli, the Ecoli composite, and self.initial_state, from current settings.
        """

        # build processes, topology, configs
        self.processes = self._retrieve_processes(self.processes,
                                                  self.add_processes,
                                                  self.exclude_processes,
                                                  self.swap_processes)
        self.topology = self._retrieve_topology(self.topology,
                                                self.processes,
                                                self.swap_processes,
                                                self.log_updates,
                                                self.divide)
        self.process_configs = self._retrieve_process_configs(self.process_configs,
                                                              self.processes)

        # initialize the ecoli composer
        if self.partition:
            ecoli_composer = ecoli.composites.ecoli_partition.Ecoli(
                self.config)
        else:
            ecoli_composer = ecoli.composites.ecoli_master.Ecoli(self.config)

        # set path at which agent is initialized
        path = tuple()
        if self.divide:
            path = ('agents', self.agent_id,)

        # get initial state
        self.initial_state = ecoli_composer.initial_state(
            config=self.config, path=path)

        # generate the composite at the path
        self.ecoli = ecoli_composer.generate(path=path)

    def run(self):
        # build self.ecoli and self.initial_state
        self.build_ecoli()

        # make the experiment
        experiment_config = {
            'description': self.description,
            'processes': self.ecoli.processes,
            'topology': self.ecoli.topology,
            'initial_state': self.initial_state,
            'progress_bar': self.progress_bar,
            'emit_topology': self.emit_topology,
            'emit_processes': self.emit_processes,
            'emit_config': self.emit_config,
            'emitter': self.emitter,
        }
        if self.experiment_id:
            experiment_config['experiment_id'] = self.experiment_id
            if self.suffix_time:
                experiment_config['experiment_id'] += datetime.now().strftime(
                    "_%d/%m/%Y %H:%M:%S")

        self.ecoli_experiment = Engine(**experiment_config)

        # run the experiment
        self.ecoli_experiment.update(self.total_time)

        # return the data
        if self.raw_output:
            return self.ecoli_experiment.emitter.get_data()
        else:
            return self.ecoli_experiment.emitter.get_timeseries()

    def merge(self, other):
        """
        Combine settings from this EcoliSim with another, overriding 
        current settings with those from the other EcoliSim.
        """

        deep_merge(self.config, other.config)

    
    def export_json(self, filename=CONFIG_DIR_PATH + "export.json"):
        export = dict(self.config)
        export['processes'] = [k for k in export['processes'].keys()]
        write_json(filename, export)


if __name__ == '__main__':
    ecoli_sim = EcoliSim.from_file()
    ecoli_sim.run()

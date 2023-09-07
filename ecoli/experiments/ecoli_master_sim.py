"""
============================
*E. coli* Master Simulations
============================

Run simulations of Ecoli Master
"""

import argparse
import copy
import os
import subprocess
import json
import warnings
from datetime import datetime
from typing import Optional, Dict, Any

from vivarium.core.engine import Engine
from vivarium.core.serialize import deserialize_value, serialize_value
from vivarium.library.dict_utils import deep_merge
from vivarium.library.topology import assoc_path
from ecoli.library.logging_tools import write_json
import ecoli.composites.ecoli_master
# Environment composer for spatial environment sim
import ecoli.composites.environment.lattice

from ecoli.processes import process_registry
from ecoli.processes.registries import topology_registry

from ecoli.composites.ecoli_configs import CONFIG_DIR_PATH
from ecoli.library.schema import not_a_process


def _tuplify_topology(topology):
    """transform an embedded topology with list paths to tuple paths"""
    tuplified_topology = {}
    for k, v in topology.items():
        if isinstance(v, dict):
            tuplified_topology[k] = _tuplify_topology(v)
        elif isinstance(v, str):
            tuplified_topology[k] = (v,)
        else:
            tuplified_topology[k] = tuple(v)
    return tuplified_topology


def get_git_revision_hash():
    return subprocess.check_output(
        ['git', 'rev-parse', 'HEAD']).decode('ascii').strip()


def get_git_status():
    status_str = subprocess.check_output(
        ['git', 'status', '--porcelain']).decode('ascii').strip()
    status = status_str.split('\n')
    return status


def report_profiling(stats):
    _, stats_keys = stats.get_print_list(
        ('(next_update)|(calculate_request)|(evolve_state)',))
    summed_stats = {}
    for key in stats_keys:
        key_stats = stats.stats[key]
        _, _, _, cumtime, _ = key_stats
        path, line, func = key
        path = os.path.basename(path)
        summed_stats[(path, line, func)] = summed_stats.get(
            (path, func), 0) + cumtime
    summed_stats_inverse_map = {
        time: key for key, time in summed_stats.items()
    }
    print('\nPer-process profiling:\n')
    for time in sorted(summed_stats_inverse_map.keys())[::-1]:
        path, line, func = summed_stats_inverse_map[time]
        print(f'{path}:{line} {func}(): {time}')
    print('\nOverall Profile:\n')
    stats.sort_stats('cumtime').print_stats(20)


def key_value_pair(argument_string):
    split = argument_string.split('=')
    if len(split) != 2:
        raise ValueError(
            'Key-value pair arguments must have exactly one `=`.')
    return split


class SimConfig:

    default_config_path = os.path.join(CONFIG_DIR_PATH, 'default.json')

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        '''Stores configuration options for a simulation.

        Attributes:
            config: Current configuration.
            parser: Argument parser for the command-line interface.

        Args:
            config: Configuration options. If provided, the default
                configuration is not loaded.
        '''
        self._config = config or {}
        if not self._config:
            self.update_from_json(self.default_config_path)

        self.parser = argparse.ArgumentParser(description='ecoli_master')
        self.parser.add_argument(
            '--config', '-c', action='store',
            default=self.default_config_path,
            help=(
                'Path to configuration file for the simulation. '
                f'Defaults to {self.default_config_path}.'))
        self.parser.add_argument(
            '--experiment_id', '-id', action="store",
            help=(
                'ID for this experiment. A UUID will be generated if '
                'this argument is not used and "experiment_id" is null '
                'in the configuration file.'))
        self.parser.add_argument(
            '--emitter', '-e', action='store',
            choices=["timeseries", "database", "print", "null",
                "shared_ram"],
            help=(
                "Emitter to use. Timeseries uses RAMEmitter, database "
                "emits to MongoDB, and print emits to stdout."))
        self.parser.add_argument(
            '--emitter_arg', '-ea', action='store', nargs='*',
            type=key_value_pair,
            help=(
                'Key-value pairs, separated by `=`, to include in '
                'emitter config.'))
        self.parser.add_argument(
            '--seed', '-s', action="store", type=int,
            help="Random seed.")
        self.parser.add_argument(
            '--initial_state', '-t0', action="store",
            help=(
                "Name of the initial state to load from (corresponding "
                "initial state file must be present in data folder)."))
        self.parser.add_argument(
            '--total_time', '-t', action="store", type=float,
            help="Time to run the simulation for.")
        self.parser.add_argument(
            '--generations', '-g', action="store", type=int,
            help="Number of generations to run the simulation for.")
        self.parser.add_argument(
            '--log_updates', '-u', action="store_true",
            help=(
                "Save updates from each process if this flag is set, "
                "e.g. for use with blame plot."))
        self.parser.add_argument(
            '--raw_output', action="store_true",
            help=(
                "Whether to return data in raw format (dictionary "
                "where keys are times, values are states)."))
        self.parser.add_argument(
            "--agent_id", action="store", type=str,
            help="Agent ID.")
        self.parser.add_argument(
            '--sim_data_path', nargs="*", default=None,
            help="Path to the sim_data to use for this experiment.")
        self.parser.add_argument(
            '--parallel', action='store_true', default=False,
            help='Run processes in parallel.')
        self.parser.add_argument(
            '--profile', action='store_true', default=False,
            help='Print profiling information at the end.')
        self.parser.add_argument(
            '--initial_state_file', action='store',
            default='',
            help='Name of initial state file (no ".json") under data/')
        self.parser.add_argument(
            '--initial_state_overrides', action='store', nargs='*',
            help='Name of initial state overrides (no ".json") under '
                'data/overrides')


    @staticmethod
    def merge_config_dicts(d1, d2):
        # Handle config keys that need special handling.
        LIST_KEYS_TO_MERGE = (
            'save_times', 'add_processes', 'exclude_processes',
            'processes', 'engine_process_reports', 'initial_state_overrides')
        for key in LIST_KEYS_TO_MERGE:
            d2.setdefault(key, [])
            d2[key].extend(d1.get(key, []))
            if key == 'engine_process_reports':
                d2[key] = [tuple(path) for path in d2[key]]
            # Ensures there are no duplicates in d2
            d2[key] = list(set(d2[key])) 
            d2[key].sort()
        deep_merge(d1, d2)


    def update_from_json(self, path):
        with open(path, 'r') as f:
            new_config = json.load(f)
        new_config = deserialize_value(new_config)
        for config_name in new_config.get('inherit_from', []):
            config_path = os.path.join(CONFIG_DIR_PATH, config_name)
            self.update_from_json(config_path)
        self.merge_config_dicts(self._config, new_config)


    def update_from_cli(self, cli_args=None):
        args = self.parser.parse_args(cli_args)
        # First load in a configuration file, if one was specified.
        config_path = getattr(args, 'config', None)
        if config_path:
            self.update_from_json(config_path)
        # Then override the configuration file with any command-line
        # options.
        cli_config = {
            key: value
            for key, value in vars(args).items()
            if value and key not in (
                'config', 'parallel', 'no_parallel')
        }
        cli_config['parallel'] = args.parallel
        self.merge_config_dicts(self._config, cli_config)


    def update_from_dict(self, dict_config):
        self.merge_config_dicts(self._config, dict_config)


    def __getitem__(self, key):
        return self._config[key]


    def get(self, key, default=None):
        return self._config.get(key, default)


    def __setitem__(self, key, val):
        self._config[key] = val


    def keys(self):
        return self._config.keys()


    def to_dict(self):
        return copy.deepcopy(self._config)


class EcoliSim:
    def __init__(self, config):
        # Do some datatype pre-processesing
        config['agents_path'] = tuple(config['agents_path'])
        config['processes'] = {
            process: None for process in config['processes']}

        # Keep track of base experiment id
        # in case multiple simulations are run with suffix_time = True.
        self.experiment_id_base = config['experiment_id']
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
        config = SimConfig()
        config.update_from_json(filepath)
        return EcoliSim(config.to_dict())


    @staticmethod
    def from_cli(cli_args=None):
        config = SimConfig()
        config.update_from_cli(cli_args)
        return EcoliSim(config.to_dict())


    def _retrieve_processes(self,
                            processes,
                            add_processes,
                            exclude_processes,
                            swap_processes,
                            ):
        result = {}
        for process_name in list(processes.keys()) + list(add_processes):
            if process_name in exclude_processes:
                continue
            if process_name in swap_processes:
                process_name = swap_processes[process_name]
            process_class = process_registry.access(process_name)
            if not process_class:
                raise ValueError(f"Unknown process with name {process_name}. "
                    "Did you call process_registry.register() in "
                    "ecoli/processes/__init__.py?")
            result[process_name] = process_class

        return result


    def _retrieve_topology(self,
                           topology,
                           processes,
                           swap_processes,
                           log_updates,
                           divide,
                           ):
        result = {}
        original_processes = {v: k for k, v in swap_processes.items()}
        for process in processes:
            # Start from default topology if it exists
            original_process = (process
                                if process not in swap_processes.values()
                                else original_processes[process])
            process_topology = topology_registry.access(original_process)
            if process_topology:
                process_topology = copy.deepcopy(process_topology)
            else:
                process_topology = {}
            # Allow the user to override default topology
            if original_process in topology.keys():
                deep_merge(process_topology, _tuplify_topology(
                    topology[original_process]))
            # For swapped processes, do additional overrides if provided
            if process != original_process and process in topology.keys():
                deep_merge(process_topology, _tuplify_topology(
                    topology[process]))
            result[process] = process_topology

        # Add log_update ports if log_updates is True
        if log_updates:
            for process, ports in result.items():
                result[process]['log_update'] = ('log_update', process,)

        return result


    def _retrieve_process_configs(self, process_configs, processes):
        result = {}
        for process in processes:
            result[process] = process_configs.get(process)
            if result[process] is None:
                result[process] = "sim_data"
        return result


    def build_ecoli(self):
        """
        Build self.ecoli, the Ecoli composite, and self.generated_initial_state
        """
        # build processes, topology, configs
        self.processes = self._retrieve_processes(
            self.processes, self.add_processes, self.exclude_processes,
            self.swap_processes)
        self.topology = self._retrieve_topology(
            self.topology, self.processes, self.swap_processes,
            self.log_updates, self.divide)
        self.process_configs = self._retrieve_process_configs(
            self.process_configs, self.processes)

        # Prevent clashing unique indices by reseeding when loading
        # a saved state (assumed to have name 'vivecoli_t{save time}')
        initial_state_path = self.config.get('initial_state_file', '')
        if initial_state_path.startswith('vivecoli'):
            time_str = initial_state_path[len('vivecoli_t'):]
            seed = int(float(time_str))
            self.config['seed'] += seed

        # initialize the ecoli composer
        ecoli_composer = ecoli.composites.ecoli_master.Ecoli(
            self.config)

        # set path at which agent is initialized
        path = tuple()
        if self.divide or self.spatial_environment:
            path = ('agents', self.agent_id,)

        # get initial state
        initial_cell_state = ecoli_composer.initial_state()
        initial_cell_state = assoc_path({}, path, initial_cell_state)

        # generate the composite at the path
        self.ecoli = ecoli_composer.generate(path=path)
        # Some processes define their own initial_state methods
        # Incoporate them into the generated initial state
        self.generated_initial_state = self.ecoli.initial_state({
            'initial_state': initial_cell_state})

        # merge a lattice composite for the spatial environment
        if self.spatial_environment:
            initial_state_config = self.spatial_environment_config.get(
                'initial_state_config')
            environment_composite = \
                ecoli.composites.environment.lattice.Lattice(
                    self.spatial_environment_config).generate()
            initial_environment = environment_composite.initial_state(
                initial_state_config)
            self.ecoli.merge(environment_composite)
            self.generated_initial_state = deep_merge(
                self.generated_initial_state, initial_environment)


    def save_states(self):
        """
        Runs the simulation while saving the states of specific
        timesteps to jsons.
        """
        for time in self.save_times:
            if time > self.total_time:
                raise ValueError(
                    f'Config contains save_time ({time}) > total '
                    f'time ({self.total_time})')

        for i in range(len(self.save_times)):
            if i == 0:
                time_to_next_save = self.save_times[i]
            else:
                time_to_next_save = self.save_times[i] - self.save_times[i-1]
            self.ecoli_experiment.update(time_to_next_save)
            time_elapsed = self.save_times[i]
            state = self.ecoli_experiment.state.get_value(
                condition=not_a_process)
            if self.divide:
                for agent_state in state['agents'].values():
                    # Will be set to true when starting sim
                    del agent_state['first_update']
                    # Processes can't be serialized
                    del agent_state['process']
                    # Bulk random state can't be serialized
                    del agent_state['allocator_rng']
                    # Save bulk and unique dtypes
                    agent_state['bulk_dtypes'] = str(agent_state['bulk'].dtype)
                    agent_state['unique_dtypes'] = {}
                    for name, mols in agent_state['unique'].items():
                        agent_state['unique_dtypes'][name] = str(mols.dtype)
            else:
                del state['first_update']
                del state['process']
                del state['allocator_rng']
                state['bulk_dtypes'] = str(state['bulk'].dtype)
                state['unique_dtypes'] = {}
                for name, mols in state['unique'].items():
                    state['unique_dtypes'][name] = str(mols.dtype)
            write_json('data/vivecoli_t' + str(time_elapsed) + '.json', state)
            print('Finished saving the state at t = ' + str(time_elapsed))
        time_remaining = self.total_time - self.save_times[-1]
        if time_remaining:
            self.ecoli_experiment.update(time_remaining)


    def run(self):
        """Create and run an EcoliSim experiment. Must run build_ecoli first!"""
        metadata = self.get_metadata()
        # make the experiment
        emitter_config = {'type': self.emitter}
        for key, value in self.emitter_arg:
            emitter_config[key] = value
        experiment_config = {
            'description': self.description,
            'metadata': metadata,
            'processes': self.ecoli.processes,
            'steps': self.ecoli.steps,
            'flow': self.ecoli.flow,
            'topology': self.ecoli.topology,
            'initial_state': self.generated_initial_state,
            'progress_bar': self.progress_bar,
            'emit_topology': self.emit_topology,
            'emit_processes': self.emit_processes,
            'emit_config': self.emit_config,
            'emitter': self.emitter,
        }
        if self.experiment_id:
            # Store backup of base experiment ID,
            # in case multiple experiments are run in a row
            # with suffix_time = True.
            if not self.experiment_id_base:
                self.experiment_id_base = self.experiment_id
            if self.suffix_time:
                self.experiment_id = datetime.now().strftime(
                    f"{self.experiment_id_base}_%d/%m/%Y %H:%M:%S")
            experiment_config['experiment_id'] = self.experiment_id
        experiment_config['profile'] = self.profile

        # Since unique numpy updater is an class method, internal
        # deepcopying in vivarium-core causes this warning to appear
        warnings.filterwarnings("ignore",
            message="Incompatible schema assignment at ")
        self.ecoli_experiment = Engine(**experiment_config)

        # Only emit designated stores if specified
        if self.config['emit_paths']:
            self.ecoli_experiment.state.set_emit_values([tuple()], False)
            self.ecoli_experiment.state.set_emit_values(
                self.config['emit_paths'],
                True,
            )
        
        # Clean up unnecessary references
        self.generated_initial_state = None
        self.ecoli_experiment.initial_state = None
        del metadata, experiment_config, emitter_config
        self.ecoli = None

        # run the experiment
        if self.save:
            self.save_states()
        else:
            self.ecoli_experiment.update(self.total_time)
        self.ecoli_experiment.end()
        if self.profile:
            report_profiling(self.ecoli_experiment.stats)


    def query(self, query=None):
        # Retrieve queried data (all if not specified)
        if self.raw_output:
            return self.ecoli_experiment.emitter.get_data(query)
        else:
            return self.ecoli_experiment.emitter.get_timeseries(query)


    def merge(self, other):
        """
        Combine settings from this EcoliSim with another, overriding
        current settings with those from the other EcoliSim.
        """
        deep_merge(self.config, other.config)


    def get_metadata(self):
        # create metadata of this experiment to be emitted,
        # namely the config of this EcoliSim object
        # with an additional key for the current git hash.
        # Goal is to save enough information to reproduce the experiment.
        metadata = dict(self.config)
        try:
            metadata["git_hash"] = get_git_revision_hash()
        except:
            warnings.warn("Unable to retrieve current git revision hash. "
                "Try making a note of this manually if your experiment may "
                "need to be replicated.")
        metadata['processes'] = [k for k in metadata['processes'].keys()]
        return metadata


    def to_json_string(self):
        return str(serialize_value(self.get_metadata()))


    def export_json(self, filename=CONFIG_DIR_PATH + "export.json"):
        with open(filename, 'w') as f:
            f.write(self.to_json_string())


def main():
    import multiprocessing; multiprocessing.set_start_method('spawn')
    ecoli_sim = EcoliSim.from_cli()
    ecoli_sim.build_ecoli()
    ecoli_sim.run()


# python ecoli/experiments/ecoli_master_sim.py
if __name__ == '__main__':
    main()

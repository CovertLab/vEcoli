"""
==============================
E. coli partitioning composite
==============================
NOTE: All ports with '_total' in their name are
automatically exempt from partitioning
"""

from copy import deepcopy

# vivarium-core
from vivarium.core.composer import Composer
from vivarium.plots.topology import plot_topology
from vivarium.library.dict_utils import deep_merge
from vivarium.core.control import run_library_cli

# sim data
from ecoli.library.sim_data import LoadSimData, RAND_MAX

# logging
from ecoli.library.logging import make_logging_process

# vivarium-ecoli processes
from ecoli.composites.ecoli_configs import (
    ECOLI_DEFAULT_PROCESSES, ECOLI_DEFAULT_TOPOLOGY)
from ecoli.plots.topology import get_ecoli_partition_topology_settings
from ecoli.processes.cell_division import Division
from ecoli.processes.allocator import Allocator
from ecoli.processes.partition import PartitionedProcess

# state
from ecoli.processes.partition import get_bulk_topo, Requester, Evolver
from ecoli.states.wcecoli_state import get_state_from_file


SIM_DATA_PATH = 'reconstruction/sim_data/kb/simData.cPickle'

MINIMAL_MEDIA_ID = 'minimal'
AA_MEDIA_ID = 'minimal_plus_amino_acids'
ANAEROBIC_MEDIA_ID = 'minimal_minus_oxygen'

COUNT_THRESHOLD = 20


class Ecoli(Composer):

    defaults = {
        'time_step': 2.0,
        'seed': 0,
        'sim_data_path': SIM_DATA_PATH,
        'daughter_path': tuple(),
        'agent_id': '0',
        'agents_path': ('..', '..', 'agents',),
        'division': {
            'threshold': 2220},  # fg
        'divide': False,
        'log_updates': False,
        'mar_regulon': False,
        'flow': {},
    }

    def __init__(self, config):
        super().__init__(config)

        self.load_sim_data = LoadSimData(
            sim_data_path=self.config['sim_data_path'],
            seed=self.config['seed'],
            mar_regulon=self.config['mar_regulon'])

        if not self.config.get('processes'):
            self.config['processes'] = deepcopy(ECOLI_DEFAULT_PROCESSES)
        if not self.config.get('process_configs'):
            self.config['process_configs'] = {process: "sim_data" for process in self.config['processes']}
        if not self.config.get('topology'):
            self.config['topology'] = deepcopy(ECOLI_DEFAULT_TOPOLOGY)

        self.processes = self.config['processes']
        self.topology = self.config['topology']

        self.processes_and_steps = None
        self.seed = None

    def initial_state(self, config=None):
        # Use initial state calculated with trna_charging and translationSupply disabled
        config = config or {}
        initial_state_file = config.get('initial_state_file', 'wcecoli_t0')
        initial_state_overrides = config.get('initial_state_overrides', [])
        initial_state = get_state_from_file(path=f'data/{initial_state_file}.json')

        for override_file in initial_state_overrides:
            override = get_state_from_file(path=f"data/{override_file}.json")
            deep_merge(initial_state, override)

        initial_state = super().initial_state({
            'initial_state': initial_state})
        return initial_state

    def _generate_processes_and_steps(self, config):
        config = deepcopy(config)
        time_step = config['time_step']

        process_order = list(config['processes'].keys())

        # get the configs from sim_data (except for allocator, built later)
        process_configs = config['process_configs']
        for process in process_configs.keys():
            if process_configs[process] == "sim_data":
                process_configs[process] = self.load_sim_data.get_config_by_name(
                    process)
            elif process_configs[process] == "default":
                process_configs[process] = None
            else:
                # user passed a dict, deep-merge with config from LoadSimData
                # if it exists, else, deep-merge with default
                try:
                    default = self.load_sim_data.get_config_by_name(process)
                except KeyError:
                    default = self.processes[process].defaults

                process_configs[process] = deep_merge(
                    deepcopy(default), process_configs[process])

                if 'seed' in process_configs[process]:
                    process_configs[process]['seed'] = (
                        process_configs[process]['seed'] +
                        config['seed']) % RAND_MAX
            process_configs[process]['_no_original_parameters'] = True

        # make the processes
        processes = {
            process_name: process(process_configs[process_name])
            if not config['log_updates']
            else make_logging_process(process)(process_configs[process_name])
            for process_name, process in config['processes'].items()
        }

        # Add allocator process
        process_configs['allocator'] = self.load_sim_data.get_allocator_config(
            process_names=[p for p in config['processes'].keys()
                           if not processes[p].is_deriver()],
        )
        process_configs['allocator']['_no_original_parameters'] = True

        config['processes']['allocator'] = Allocator
        processes['allocator'] = Allocator(process_configs['allocator'])

        # Store list of partition processes
        self.partitioned_processes = [
            process_name
            for process_name, process in processes.items()
            if isinstance(process, PartitionedProcess)
        ]

        # update schema overrides for evolvers and requesters
        update_override = {}
        delete_override = []
        for process_id, override in self.schema_override.items():
            if process_id in self.partitioned_processes:
                delete_override.append(process_id)
                update_override[f'{process_id}_evolver'] = override
                update_override[f'{process_id}_requester'] = override
        for process_id in delete_override:
            del self.schema_override[process_id]
        self.schema_override.update(update_override)

        # make the requesters
        requesters = {
            f'{process_name}_requester': Requester({
                'time_step': time_step,
                'process': process,
                '_no_original_parameters': True,
            })
            for (process_name, process) in processes.items()
            if process_name in self.partitioned_processes
        }

        # make the evolvers
        evolvers = {
            f'{process_name}_evolver': Evolver({
                'time_step': time_step,
                'process': process,
                '_no_original_parameters': True,
            })
            if not config['log_updates']
            else make_logging_process(Evolver)({
                'time_step': time_step,
                'process': process,
                '_no_original_parameters': True,
            })
            for (process_name, process) in processes.items()
            if process_name in self.partitioned_processes
        }

        processes.update(requesters)
        processes.update(evolvers)

        # add division process
        if config['divide']:
            division_name = 'division'
            division_config = dict(
                config['division'],
                agent_id=config['agent_id'],
                composer=self,
                seed=self.load_sim_data.random_state.randint(RAND_MAX),
                _no_original_parameters=True,
            )
            division_process = {division_name: Division(division_config)}
            processes.update(division_process)
            process_order.append(division_name)

        steps = {
            **requesters,
            'allocator': processes['allocator'],
        }
        processes_not_steps = {
            **evolvers,
        }
        flow = {
            'allocator': [],
        }

        for name in process_order:
            process = processes[name]
            if name in self.partitioned_processes:
                requester_name = f'{name}_requester'
                evolver_name = f'{name}_evolver'
                flow[requester_name] = [
                    ('ecoli-chromosome-structure',)]
                if config['divide']:
                    flow[requester_name].append(('division',))
                flow['allocator'].append((requester_name,))
                steps[requester_name] = processes[requester_name]
                processes_not_steps[evolver_name] = processes[
                    evolver_name]
            elif name == 'division':
                steps[name] = process
                flow[name] = [('ecoli-chromosome-structure',)]
            elif process.is_step():
                steps[name] = process
                flow[name] = []
            else:
                processes_not_steps[name] = process

        for name, dependencies in config['flow'].items():
            flow.setdefault(name, [])
            flow[name].extend([tuple(dep) for dep in dependencies])

        return processes_not_steps, steps, flow

    def generate_processes(self, config):
        if not self.processes_and_steps or self.seed != config['seed']:
            self.seed = config['seed']
            self.processes_and_steps = (
                self._generate_processes_and_steps(config))
        processes, _, _ = self.processes_and_steps
        return processes

    def generate_steps(self, config):
        if not self.processes_and_steps or self.seed != config['seed']:
            self.seed = config['seed']
            self.processes_and_steps = (
                self._generate_processes_and_steps(config))
        _, steps, _ = self.processes_and_steps
        return steps

    def generate_flow(self, config):
        if not self.processes_and_steps or self.seed != config['seed']:
            self.seed = config['seed']
            self.processes_and_steps = (
                self._generate_processes_and_steps(config))
        _, _, flow = self.processes_and_steps
        return flow

    def generate_topology(self, config):
        topology = {}

        # make the topology
        for process_id, ports in config['topology'].items():

            # make the partitioned processes' topologies
            if process_id in self.partitioned_processes:
                topology[f'{process_id}_requester'] = deepcopy(ports)
                topology[f'{process_id}_evolver'] = deepcopy(ports)
                if config['log_updates']:
                    topology[f'{process_id}_evolver']['log_update'] = (
                        'log_update', process_id,)
                    topology[f'{process_id}_requester']['log_update'] = (
                        'log_update', process_id,)
                bulk_topo = get_bulk_topo(ports)
                topology[f'{process_id}_requester']['request'] = {
                    '_path': ('request', process_id,),
                    **bulk_topo}
                topology[f'{process_id}_evolver']['allocate'] = {
                    '_path': ('allocate', process_id,),
                    **bulk_topo}
                topology[f'{process_id}_requester'][
                    'evolvers_ran'] = ('evolvers_ran',)
                topology[f'{process_id}_evolver'][
                    'evolvers_ran'] = ('evolvers_ran',)

            # make the non-partitioned processes' topologies
            else:
                topology[process_id] = ports
                if config['log_updates']:
                    topology[process_id]['log_update'] = (
                        'log_update', process_id,)

        # add division
        if config['divide']:
            topology['division'] = {
                'variable': ('listeners', 'mass', 'cell_mass'),
                'agents': config['agents_path']}

        topology['allocator'] = {
            'request': ('request',),
            'allocate': ('allocate',),
            'bulk': ('bulk',),
            'evolvers_ran': ('evolvers_ran',),
        }

        return topology


def run_ecoli(
        filename='default',
        total_time=10,
        divide=False,
        progress_bar=True,
        log_updates=False,
        emitter='timeseries',
):
    """Run ecoli_master simulations

    Arguments: TODO -- complete the arguments docstring
        * **total_time** (:py:class:`int`): the total runtime of the experiment
        * **config** (:py:class:`dict`):

    Returns:
        * output data
    """
    from ecoli.experiments.ecoli_master_sim import EcoliSim, CONFIG_DIR_PATH

    sim = EcoliSim.from_file(CONFIG_DIR_PATH + filename + '.json')
    sim.total_time = total_time
    sim.divide = divide
    sim.progress_bar = progress_bar
    sim.log_updates = log_updates
    sim.emitter = emitter

    sim.run()
    return sim.query()


def ecoli_topology_plot(config={}):
    """Make a topology plot of Ecoli"""
    agent_id_config = {'agent_id': '1'}
    ecoli = Ecoli({**agent_id_config, **config})
    settings = get_ecoli_partition_topology_settings()
    topo_plot = plot_topology(
        ecoli,
        filename='topology',
        out_dir='out/composites/ecoli_master',
        settings=settings
    )
    return topo_plot


test_library = {
    '0': run_ecoli,
    '1': ecoli_topology_plot,
}

# run experiments in test_library from the command line with:
# python ecoli/composites/ecoli_master.py -n [experiment id]
if __name__ == '__main__':
    run_library_cli(test_library)

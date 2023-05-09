"""
==============================
E. coli partitioning composite
==============================
Do not run standalone. Use EcoliSim interface.
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
from ecoli.library.logging_tools import make_logging_process

# vivarium-ecoli processes
from ecoli.composites.ecoli_configs import (
    ECOLI_DEFAULT_PROCESSES, ECOLI_DEFAULT_TOPOLOGY)
from ecoli.plots.topology import get_ecoli_partition_topology_settings
from ecoli.processes.cell_division import Division
from ecoli.processes.allocator import Allocator
from ecoli.processes.partition import PartitionedProcess
from ecoli.processes.unique_update import UniqueUpdate

# state
from ecoli.processes.partition import filter_bulk_topology, Requester, Evolver, Step
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
        'division_threshold': 668,  # fg
        'division_variable': ('listeners', 'mass', 'dry_mass'),
        'chromosome_path': ('unique',' full_chromosome'),
        'divide': False,
        'log_updates': False,
        'mar_regulon': False,
        'process_configs': {},
        'flow': {},
    }


    def __init__(self, config):
        super().__init__(config)
        self.load_sim_data = LoadSimData(
            sim_data_path=self.config['sim_data_path'],
            seed=self.config['seed'],
            mar_regulon=self.config['mar_regulon'],
            rnai_data=self.config['process_configs'].get(
                'ecoli-rna-interference'))

        if not self.config.get('processes'):
            self.config['processes'] = deepcopy(ECOLI_DEFAULT_PROCESSES)
        if not self.config.get('process_configs'):
            self.config['process_configs'] = {process: "sim_data"
                for process in self.config['processes']}
        if not self.config.get('topology'):
            self.config['topology'] = deepcopy(ECOLI_DEFAULT_TOPOLOGY)

        self.processes = self.config['processes']
        self.topology = self.config['topology']
        self.processes_and_steps = self._generate_processes_and_steps(self.config)

        self.seed = None


    def initial_state(self, config=None):
        """Either pass initial state dictionary in config['initial_state']
        or filename to load from in config['initial_state_file']. Optionally
        apply overrides in config['initial_state_overrides']. Automatically
        generates shared process states for partitioned processes. MUST be
        run BEFORE calling generate() on composer instance."""
        config = config or self.config
        # Allow initial state to be directly supplied instead of a file name
        # (e.g. when loading individual cells in a colony save file)
        initial_state = config.get('initial_state', None)
        if not initial_state:
            initial_state_file = config.get('initial_state_file', 'wcecoli_t0')
            initial_state = get_state_from_file(
                path=f'data/{initial_state_file}.json')

        initial_state_overrides = config.get('initial_state_overrides', [])
        if initial_state_overrides:
            bulk_map = {bulk_id: row_id for row_id, bulk_id
                in enumerate(initial_state['bulk']['id'])}
        for override_file in initial_state_overrides:
            override = get_state_from_file(path=f"data/{override_file}.json")
            # Apply bulk overrides of the form {molecule: count} to Numpy array
            bulk_overrides = override.pop('bulk', {})
            initial_state['bulk'].flags.writeable = True
            for molecule, count in bulk_overrides.items():
                initial_state['bulk']['count'][bulk_map[molecule]] = count
            initial_state['bulk'].flags.writeable = False
            deep_merge(initial_state, override)

        # Put shared process instances for partitioned processes into state
        processes, _, _ = self.processes_and_steps
        initial_state['process'] = {
            process.parameters['process'].name: (process.parameters['process'],)
            for process in processes.values()
            if 'process' in process.parameters
        }
        return initial_state


    def _generate_processes_and_steps(self, config):
        """Instantiates all processes and steps, creates separate Requester
        and Evolver for each PartitionedProcess as well as Allocator, adds
        division process if configured, ensures that Requesters run after
        all listeners and allocator after all Requesters, adds UniqueUpdate
        Steps to ensure that unique molecule states are updated before and
        after running each Step"""
        time_step = config['time_step']
        # get the configs from sim_data (except for allocator, built later)
        process_configs = config['process_configs']
        for process in process_configs.keys():
            if process_configs[process] == "sim_data":
                process_configs[process] = \
                    self.load_sim_data.get_config_by_name(process)
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

        # make the processes
        processes = {}
        steps = {}
        flow = {}
        self.partitioned_processes = []
        for process_name, process_class in config['processes'].items():
            process = process_class(process_configs[process_name])
            if isinstance(process, PartitionedProcess):
                if config['log_updates']:
                    processes[f'{process_name}_evolver'] = \
                        make_logging_process(Evolver)({
                            'time_step': time_step,
                            'process': process
                        })
                else:
                    processes[f'{process_name}_evolver'] = Evolver({
                        'time_step': time_step,
                        'process': process
                    })
                steps[f'{process_name}_requester'] = Requester({
                    'time_step': time_step,
                    'process': process
                })
                self.partitioned_processes.append(process_name)
            elif isinstance(process, Step):
                steps[process_name] = process
            else:
                processes[process_name] = process
            
        # Add allocator Step
        allocator_config = self.load_sim_data.get_allocator_config(
            process_names=self.partitioned_processes)
        steps['allocator'] = Allocator(allocator_config)

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

        # add division Step
        if config['divide']:
            division_config = {
                'division_threshold': config['division_threshold'],
                'agent_id': config['agent_id'],
                'composer': Ecoli,
                'composer_config': self.config,
                'dry_mass_inc_dict': \
                    self.load_sim_data.sim_data.expectedDryMassIncreaseDict,
                'seed': config['seed'],
            }
            steps['division'] = Division(division_config)

        # generate Step dependency dictionary
        flow = {'allocator': []}
        for name in steps:
            if '_requester' in name:
                # Requesters run after listeners, division, and shape
                flow[name] = [('monomer_counts_listener',)]
                if config['divide']:
                    flow[name].append(('division',))
                if 'ecoli-shape' in config['processes']:
                    flow[name].append(('ecoli-shape',))
                # Allocator runs after all Requesters
                flow['allocator'].append((name,))
            elif name == 'division':
                # Division runs after mass listener
                flow[name] = [('ecoli-mass-listener',)]
            else:
                flow.setdefault(name, [])
            # Update unique molecules before any Steps run
            flow[name].append(('unique-update',))

        # Add Step dependecies specified in config
        for name, dependencies in config['flow'].items():
            if name in steps:
                flow[name].extend([tuple(dep) for dep in dependencies])

        # TODO: Find a more robust way to get topology containing all
        # unique molecules
        unique_dict = steps['ecoli-chromosome-structure'].topology
        unique_dict = unique_dict.copy()
        unique_dict.pop('bulk')
        unique_dict.pop('listeners')
        unique_dict.pop('evolvers_ran')
        unique_dict.pop('first_update')
        flow_keys = list(flow.keys())
        params = {'unique_dict': unique_dict}
        steps['unique-update'] = UniqueUpdate(params)
        flow['unique-update'] = []
        # Update unique molecules after each Step runs
        for i in range(len(flow_keys)):
            flow[f'unique-update-{i+1}'] = [(flow_keys[i],)]
            steps[f'unique-update-{i+1}'] = UniqueUpdate(params)
        # Free memory by removing reference to sim_data object
        del self.load_sim_data
        return processes, steps, flow


    def generate_processes(self, config):
        processes, _, _ = self.processes_and_steps
        return processes


    def generate_steps(self, config):
        _, steps, _ = self.processes_and_steps
        return steps


    def generate_flow(self, config):
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
                # Only the bulk ports should be included in the request
                # and allocate topologies
                bulk_topo = filter_bulk_topology(ports)
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
                topology[f'{process_id}_requester'][
                    'process'] = ('process', process_id,)
                topology[f'{process_id}_evolver'][
                    'process'] = ('process', process_id,)
            # make the non-partitioned processes' topologies
            else:
                topology[process_id] = ports
                if config['log_updates']:
                    topology[process_id]['log_update'] = (
                        'log_update', process_id,)

        # add division
        if config['divide']:
            topology['division'] = {
                'division_variable': config['division_variable'],
                'full_chromosome': config['chromosome_path'],
                'agents': config['agents_path'],
                'media_id': ('environment', 'media_id'),
                'division_threshold': ('division_threshold',)}
        # add allocator
        topology['allocator'] = {
            'request': ('request',),
            'allocate': ('allocate',),
            'bulk': ('bulk',),
            'evolvers_ran': ('evolvers_ran',),
        }

        _, steps, _ = self.processes_and_steps
        # UniqueUpdate signals for collected unique molecule updates
        # to be applied before and after all Steps run
        for step_name in steps.keys():
            if 'unique-update' in step_name:
                topology[step_name] = steps['unique-update'
                    ].unique_dict.copy()
        # Do not keep an unnecessary reference to these
        self.processes_and_steps = None

        # Add clock process to facilitate unique molecule updates
        topology['clock'] = {
            'global_time': ('global_time',)
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
    # Import here to avoid circular import
    from ecoli.experiments.ecoli_master_sim import EcoliSim, CONFIG_DIR_PATH

    sim = EcoliSim.from_file(CONFIG_DIR_PATH + filename + '.json')
    sim.total_time = total_time
    sim.divide = divide
    sim.progress_bar = progress_bar
    sim.log_updates = log_updates
    sim.emitter = emitter
    sim.build_ecoli()
    sim.run()
    return sim.query()


def ecoli_topology_plot(config=None):
    if not config:
        config = {}
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

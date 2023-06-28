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
from vivarium.core.engine import _StepGraph

# sim data
from ecoli.library.sim_data import LoadSimData, RAND_MAX

# logging
from ecoli.library.logging_tools import make_logging_process

# vivarium-ecoli processes
from ecoli.composites.ecoli_configs import (
    ECOLI_DEFAULT_PROCESSES, ECOLI_DEFAULT_TOPOLOGY)
from ecoli.plots.topology import get_ecoli_partition_topology_settings
from ecoli.processes.cell_division import Division, MarkDPeriod
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
        'amp_lysis': False,
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
                'ecoli-rna-interference'),
            amp_lysis=self.config['amp_lysis'])

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
        """Instantiates all processes and steps
         * Creates separate Requester and Evolvers for each PartitionedProcess
         * Adds Allocator b/w Requesters and Evolvers for an execution layer
         * Adds division if configured
         * Adds UniqueUpdate Step between execution layers to ensure that
           unique molecule states are properly updated"""
        time_step = config['time_step']
        # get the configs from sim_data (except for allocator, built later)
        process_configs = config['process_configs']
        for process in process_configs.keys():
            if process_configs[process] == "sim_data":
                process_configs[process] = \
                    self.load_sim_data.get_config_by_name(process)
                process_configs[process] = None
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

        # make the processes
        processes = {}
        steps = {}
        flow = {}
        self.partitioned_processes = []
        for process_name, process_class in config['processes'].items():
            if issubclass(process_class, PartitionedProcess):
                process = process_class(process_configs[process_name])
                if config['log_updates']:
                    steps[f'{process_name}_evolver'] = \
                        make_logging_process(Evolver)({
                            'time_step': time_step,
                            'process': process
                        })
                    steps[f'{process_name}_requester'] = \
                        make_logging_process(Requester)({
                            'time_step': time_step,
                            'process': process
                        })
                else:
                    steps[f'{process_name}_evolver'] = Evolver({
                        'time_step': time_step,
                        'process': process
                    })
                    steps[f'{process_name}_requester'] = Requester({
                        'time_step': time_step,
                        'process': process
                    })
                self.partitioned_processes.append(process_name)
            elif issubclass(process_class, Step):
                if config['log_updates']:
                    process_class = make_logging_process(process_class)
                process = process_class(process_configs[process_name])
                steps[process_name] = process
                steps[process_name] = None
                continue
            else:
                process = process_class(process_configs[process_name])
                processes[process_name] = process
                continue

        # Parse flow to get execution layers
        step_graph = _StepGraph()
        allocator_counter = 1
        unique_update_counter = 1
        for process in config['processes']:
            # Get Step dependencies as tuple paths
            deps = config['flow'].get(process, [])
            tuplified_deps = []
            for dep_path in deps:
                # Use evolver for partitioned dependencies
                if dep_path[-1] in self.partitioned_processes:
                    tuplified_deps.append(tuple(dep_path[:-1])
                        + (f'{dep_path[-1]}_evolver',))
                else:
                    tuplified_deps.append(tuple(dep_path))
            # For partitioned steps, requesters must run before evolvers
            if process in self.partitioned_processes:
                step_graph.add((f'{process}_requester',), tuplified_deps)
                step_graph.add((f'{process}_evolver',),
                               [(f'{process}_requester',)])
            elif process in steps:
                step_graph.add((process,), tuplified_deps)
        
        # Build simulation flow with UniqueUpdate and Allocator layers
        layers = step_graph.get_execution_layers()
        allocator_counter = 1
        unique_update_counter = 1
        for layer_steps in layers:
            requesters = False
            for step_path in layer_steps:
                # Evolvers always go after the allocator for a given layer
                if 'evolver' in step_path[-1]:
                    flow[step_path[-1]] = [
                        (f'allocator_{allocator_counter - 1}',)]
                # Aside from first layer, all non-evolver layers will be
                # immediately preceeded by a UniqueUpdate layer
                elif unique_update_counter > 1:
                    flow[step_path[-1]] = [
                        (f'unique_update_{unique_update_counter - 1}',)]
                    if 'requester' in step_path[-1]:
                        requesters = True
            # Add Allocator layer right after requester layer
            if requesters:
                flow[f'allocator_{allocator_counter}'] = layer_steps
                allocator_counter += 1
            # Add UniqueUpdate layer after non-requester layers
            else:
                flow[f'unique_update_{unique_update_counter}'] = [step_path]
                unique_update_counter += 1

        # Add Allocator Steps
        allocator_config = self.load_sim_data.get_allocator_config(
            process_names=self.partitioned_processes)
        for i in range(1, allocator_counter):
            steps[f'allocator_{i}'] = Allocator(allocator_config)

        # Add UniqueUpdate Steps
        unique_mols = (self.load_sim_data.sim_data.internal_state
                       ).unique_molecule.unique_molecule_definitions.keys()
        unique_topo = {
            unique_mol: ('unique', unique_mol)
            for unique_mol in unique_mols}
        params = {'unique_topo': unique_topo}
        for i in range(1, unique_update_counter):
            steps[f'unique_update_{i}'] = UniqueUpdate(params)

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
            if config['d_period']:
                steps['mark_d_period'] = MarkDPeriod()
                flow['mark_d_period'] = [
                    (f'unique_update_{unique_update_counter - 1}',)]
                flow['division'] = [('mark_d_period',)]
            else:
                flow['division'] = [
                    (f'unique_update_{unique_update_counter - 1}',)]

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
                        'log_update', f'{process_id}_evolver',)
                    topology[f'{process_id}_requester']['log_update'] = (
                        'log_update', f'{process_id}_requester',)
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
            if config['d_period']:
                topology['mark_d_period'] = {
                    'full_chromosome': tuple(config['chromosome_path']),
                    'global_time': ('global_time',),
                    'divide': ('divide',)}
            topology['division'] = {
                'division_variable': tuple(config['division_variable']),
                'full_chromosome': tuple(config['chromosome_path']),
                'agents': tuple(config['agents_path']),
                'media_id': ('environment', 'media_id'),
                'division_threshold': ('division_threshold',)}

        # Add Allocator and UniqueUpdate topologies
        _, steps, _ = self.processes_and_steps
        allocator_topo = {
            'request': ('request',),
            'allocate': ('allocate',),
            'bulk': ('bulk',),
            'evolvers_ran': ('evolvers_ran',),
        }
        for step_name in steps.keys():
            if 'unique_update' in step_name:
                topology[step_name] = steps[step_name].unique_topo.copy()
            elif 'allocator' in step_name:
                topology[step_name] = allocator_topo.copy()

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
    time_series=True,
):
    """Run ecoli_master simulations

    Arguments:
        * **total_time** (:py:class:`int`): the total runtime of the experiment
        * **divide** (:py:class:`bool`): whether to incorporate division
        * **progress_bar** (:py:class:`bool`): whether to show a progress bar
        * **log_updates**  (:py:class:`bool`): whether to save updates from each process
        * **emitter** (:py:class:`str`): type of emitter to use
        * **time_series** (:py:class:`bool`): whether to return data in timeseries format

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
    sim.raw_output = not time_series
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

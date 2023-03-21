"""
==============================
E. coli partitioning composite
==============================
NOTE: All ports with '_total' in their name are
automatically exempt from partitioning
"""

import binascii
import numpy as np
from copy import deepcopy

# vivarium-core
from vivarium.core.composer import Composer
from vivarium.plots.topology import plot_topology
from vivarium.library.dict_utils import deep_merge
from vivarium.core.control import run_library_cli

# sim data
from wholecell.utils import units
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
        'division_threshold': 668,  # fg
        'division_variable': ('listeners', 'mass', 'dry_mass'),
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
            rnai_data=self.config['process_configs'].get('ecoli-rna-interference'))

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
        # Allow initial state to be directly supplied instead of a file name (useful when
        # loading individual cells in a colony save file)
        initial_state = config.get('initial_state', None)
        if not initial_state:
            initial_state_file = config.get('initial_state_file', 'wcecoli_t0')
            initial_state = get_state_from_file(path=f'data/{initial_state_file}.json')

        initial_state_overrides = config.get('initial_state_overrides', [])
        for override_file in initial_state_overrides:
            override = get_state_from_file(path=f"data/{override_file}.json")
            deep_merge(initial_state, override)

        initial_cell_state_list = list(initial_state["bulk"].items())
        total_tuple_list = []
        name_list = []
        for i in range(len(initial_cell_state_list)):
            name_list.append(initial_cell_state_list[i][0])
            total_tuple_list.append(tuple(list(initial_cell_state_list[i]) + [
                self.load_sim_data.sim_data.internal_state.bulk_molecules.bulk_data.fullArray()[i][
                    1].tolist()]))
        string_length = 'U' + str(len(max(name_list, key=len)))
        total_bulk_array = np.array(total_tuple_list,
                                    dtype=[('id', string_length), ('count', 'i'), ('submasses', 'd', (9,))])
        initial_state['bulk'] = total_bulk_array
        unique_DnaA_dt = [('submasses', 'd', (9,)), ('DnaA_bound', '?'), ('_entryState', 'i'), ('_globalIndex', 'i'),
                          ('coordinates', 'i'), ('domain_index', 'i'), ('unique_index', 'U4'), ('time', 'i'), ('_cached_entryState', 'i')]
        unique_DnaA_array = np.array(
            [tuple(list(i.values()) + [0, 0]) for i in list(initial_state['unique']["DnaA_box"].values())], dtype=unique_DnaA_dt)
        initial_state['unique']["DnaA_box"] = unique_DnaA_array
        unique_RNA_dt = [('submasses', 'd', (9,)), ('RNAP_index', 'U4'), ('TU_index', 'i'), ('_entryState', 'i'),
                         ('_globalIndex', 'i'), ('can_translate', '?'), ('is_full_transcript', '?'), ('is_mRNA', '?'),
                         ('transcript_length', 'i'), ('unique_index', 'U4'), ('time', 'i'), ('_cached_entryState', 'i')]
        unique_RNA_array = np.array(
            [tuple(list(i.values()) + [0, 0]) for i in list(initial_state['unique']["RNA"].values())], dtype=unique_RNA_dt) #worried about the extend in order to add time and cached entryState
        initial_state['unique']["RNA"] = unique_RNA_array
        unique_active_RNAP_dt = [('submasses', 'd', (9,)), ('_entryState', 'i'), ('_globalIndex', 'i'),
                                 ('_coordinates', 'i'),
                                 ('direction', '?'), ('domain_index', 'i'), ('unique_index', 'U4'), ('time', 'i'), ('_cached_entryState', 'i')]
        unique_active_RNAP_array = np.array(
            [tuple(list(i.values()) + [0, 0]) for i in list(initial_state['unique']["active_RNAP"].values())],
            dtype=unique_active_RNAP_dt)
        initial_state['unique']["active_RNAP"] = unique_active_RNAP_array
        unique_active_replisome_dt = [('submasses', 'd', (9,)), ('_entryState', 'i'), ('_globalIndex', 'i'),
                                      ('_coordinates', 'i'),
                                      ('domain_index', 'i'), ('right_replichore', '?'), ('unique_index', 'U1'), ('time', 'i'), ('_cached_entryState', 'i')]
        unique_active_replisome_array = np.array(
            [tuple(list(i.values()) + [0, 0]) for i in list(initial_state['unique']["active_replisome"].values())],
            dtype=unique_active_replisome_dt)
        initial_state['unique']["active_replisome"] = unique_active_replisome_array
        unique_active_ribosome_dt = [('submasses', 'd', (9,)), ('_entryState', 'i'), ('_globalIndex', 'i'),
                                     ('mRNA_index', 'U5'),
                                     ('peptide_length', 'i'), ('pos_on_mRNA', 'i'), ('protein_index', 'i'),
                                     ('unique_index', 'U5'), ('time', 'i'), ('_cached_entryState', 'i')]
        unique_active_ribosome_array = np.array(
            [tuple(list(i.values()) + [0, 0]) for i in list(initial_state['unique']["active_ribosome"].values())],
            dtype=unique_active_ribosome_dt)
        initial_state['unique']["active_ribosome"] = unique_active_ribosome_array
        unique_chromosomal_segment_dt = [('submasses', 'd', (9,)), ('unique_index', 'U5'), ('time', 'i'), ('_cached_entryState', 'i')]
        unique_chromosomal_segment_array = np.empty(shape=(4,), dtype=unique_chromosomal_segment_dt)
        initial_state['unique']["chromosomal_segment"] = unique_chromosomal_segment_array
        unique_chromosome_domain_dt = [('submasses', 'd', (9,)), ('_entryState', 'i'), ('_globalIndex', 'i'),
                                       ('child_domains', 'O'),
                                       ('domain_index', 'i'), ('unique_index', 'i'), ('time', 'i'), ('_cached_entryState', 'i')]
        unique_chromosome_domain_array = np.array(
            [tuple(list(i.values()) + [0, 0]) for i in list(initial_state['unique']["chromosome_domain"].values())],
            dtype=unique_chromosome_domain_dt)
        initial_state['unique']["chromosome_domain"] = unique_chromosome_domain_array
        unique_full_chromosome_dt = [('submasses', 'd', (9,)), ('_entryState', 'i'), ('_globalIndex', 'i'),
                                     ('division_time', 'float'),
                                     ('domain_index', 'i'), ('has_triggered_division', '?'), ('unique_index', 'U1'), ('time', 'i'), ('_cached_entryState', 'i')]
        unique_full_chromosome_array = np.array(
            [tuple(list(i.values()) + [0, 0]) for i in list(initial_state['unique']["full_chromosome"].values())],
            dtype=unique_full_chromosome_dt)
        initial_state['unique']["full_chromosome"] = unique_full_chromosome_array
        unique_oriC_dt = [('submasses', 'd', (9,)), ('_entryState', 'i'), ('_globalIndex', 'i'),
                          ('domain_index', 'i'), ('unique_index', 'U1'), ('time', 'i'), ('_cached_entryState', 'i')]
        unique_oriC_array = np.array(
            [tuple(list(i.values()) + [0, 0]) for i in list(initial_state['unique']["oriC"].values())],
            dtype=unique_oriC_dt)
        initial_state['unique']["oriC"] = unique_oriC_array
        unique_promoter_dt = [('submasses', 'd', (9,)), ('TU_index', 'i'), ('_entryState', 'i'), ('_globalIndex', 'i'),
                              ('bound_TF', 'O'),
                              ('coordinates', 'i'), ('domain_index', 'i'), ('unique_index', 'U1'), ('time', 'i'), ('_cached_entryState', 'i')]
        unique_promoter_array = np.array(
            [tuple(list(i.values()) + [0, 0]) for i in list(initial_state['unique']["promoter"].values())],
            dtype=unique_promoter_dt)
        initial_state['unique']["promoter"] = unique_promoter_array

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
            })
            for (process_name, process) in processes.items()
            if process_name in self.partitioned_processes
        }

        # make the evolvers
        evolvers = {
            f'{process_name}_evolver': Evolver({
                'time_step': time_step,
                'process': process,
            })
            if not config['log_updates']
            else make_logging_process(Evolver)({
                'time_step': time_step,
                'process': process,
            })
            for (process_name, process) in processes.items()
            if process_name in self.partitioned_processes
        }

        processes.update(requesters)
        processes.update(evolvers)
        
        # add division process
        if config['divide']:
            expectedDryMassIncreaseDict = self.load_sim_data.sim_data.expectedDryMassIncreaseDict
            division_name = 'division'
            division_config = {'threshold': config['division_threshold']}
            if config['division_threshold'] == 'massDistribution':
                division_random_seed = binascii.crc32(b'CellDivision', config['seed']) & 0xffffffff
                division_random_state = np.random.RandomState(seed=division_random_seed)
                division_mass_multiplier = division_random_state.normal(loc=1.0, scale=0.1)
                initial_state_file = config.get('initial_state_file', 'wcecoli_t0')
                initial_state_overrides = config.get('initial_state_overrides', [])
                initial_state = get_state_from_file(path=f'data/{initial_state_file}.json')
                for override_file in initial_state_overrides:
                    override = get_state_from_file(path=f"data/{override_file}.json")
                    deep_merge(initial_state, override)
                current_media_id = initial_state['environment']['media_id']
                division_config['threshold'] = (initial_state['listeners']['mass']['dry_mass'] + 
                    expectedDryMassIncreaseDict[current_media_id].asNumber(
                        units.fg) * division_mass_multiplier)
            division_config = dict(
                division_config,
                agent_id=config['agent_id'],
                composer=self,
                seed=self.load_sim_data.random_state.randint(RAND_MAX),
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
                    ('monomer_counts_listener',)]
                if config['divide']:
                    flow[requester_name].append(('division',))
                if 'ecoli-shape' in config['processes']:
                    flow[requester_name].append(('ecoli-shape',))
                flow['allocator'].append((requester_name,))
                steps[requester_name] = processes[requester_name]
                processes_not_steps[evolver_name] = processes[
                    evolver_name]
            elif name == 'division':
                steps[name] = process
                flow[name] = [('ecoli-mass-listener',)]
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
            self.load_sim_data.seed = config['seed']
            self.load_sim_data.random_state = np.random.RandomState(
                seed = config['seed'])
            self.processes_and_steps = (
                self._generate_processes_and_steps(config))
        processes, _, _ = self.processes_and_steps
        return processes

    def generate_steps(self, config):
        if not self.processes_and_steps or self.seed != config['seed']:
            self.seed = config['seed']
            self.load_sim_data.seed = config['seed']
            self.load_sim_data.random_state = np.random.RandomState(
                seed = config['seed'])
            self.processes_and_steps = (
                self._generate_processes_and_steps(config))
        _, steps, _ = self.processes_and_steps
        return steps

    def generate_flow(self, config):
        if not self.processes_and_steps or self.seed != config['seed']:
            self.seed = config['seed']
            self.load_sim_data.seed = config['seed']
            self.load_sim_data.random_state = np.random.RandomState(
                seed = config['seed'])
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
                'variable': config['division_variable'],
                'agents': config['agents_path']}

        topology['allocator'] = {
            'request': ('request',),
            'allocate': ('allocate',),
            'bulk': ('bulk',),
            'evolvers_ran': ('evolvers_ran',),
        }

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
    from ecoli.experiments.ecoli_master_sim import EcoliSim, CONFIG_DIR_PATH

    sim = EcoliSim.from_file(CONFIG_DIR_PATH + filename + '.json')
    sim.total_time = total_time
    sim.divide = divide
    sim.progress_bar = progress_bar
    sim.log_updates = log_updates
    sim.emitter = emitter

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

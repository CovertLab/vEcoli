"""
==============================
E. coli partitioning composite
==============================
NOTE: All ports with '_total' in their name are
automatically exempt from partitioning
"""

from copy import deepcopy

import pytest
from vivarium.core.composer import Composer
from vivarium.plots.topology import plot_topology
from vivarium.library.topology import assoc_path
from vivarium.library.dict_utils import deep_merge
from vivarium.core.control import run_library_cli
from vivarium.core.engine import Engine

# sim data
from ecoli.library.sim_data import LoadSimData

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

# tests
from ecoli.library.schema import get_domain_index_to_daughter
from migration.migration_utils import scalar_almost_equal

RAND_MAX = 2**31
SIM_DATA_PATH = 'reconstruction/sim_data/kb/simData.cPickle'

MINIMAL_MEDIA_ID = 'minimal'
AA_MEDIA_ID = 'minimal_plus_amino_acids'
ANAEROBIC_MEDIA_ID = 'minimal_minus_oxygen'

COUNT_THRESHOLD = 20


class Ecoli(Composer):

    defaults = {
        'time_step': 2.0,
        'parallel': False,
        'parallel_allocator': False,
        'seed': 0,
        'sim_data_path': SIM_DATA_PATH,
        'daughter_path': tuple(),
        'agent_id': '0',
        'agents_path': ('..', '..', 'agents',),
        'division': {
            'threshold': 2220},  # fg
        'divide': False,
        'log_updates': False,
    }

    def __init__(self, config):
        super().__init__(config)

        self.load_sim_data = LoadSimData(
            sim_data_path=self.config['sim_data_path'],
            seed=self.config['seed'])

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

    def initial_state(self, config=None, path=()):
        # Use initial state calculated with trna_charging and translationSupply disabled
        config = config or {}
        initial_state_file = config.get('initial_state_file', 'wcecoli_t0')
        initial_state = get_state_from_file(path=f'data/{initial_state_file}.json')
        embedded_state = {}
        assoc_path(embedded_state, path, initial_state)
        return embedded_state

    def _generate_processes_and_steps(self, config):
        time_step = config['time_step']
        parallel = config['parallel']

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
                    process_configs[process]['seed'] = process_configs[process]['seed'] + config['seed']

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
            parallel=config['parallel_allocator'],
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
            f'{process_name}_requester': Requester({'time_step': time_step,
                                                    'process': process})
            for (process_name, process) in processes.items()
            if process_name in self.partitioned_processes
        }

        # make the evolvers
        evolvers = {
            f'{process_name}_evolver': Evolver({'time_step': time_step,
                                                'process': process})
            if not config['log_updates']
            else make_logging_process(Evolver)({'time_step': time_step,
                                                'process': process})
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
                composer=self)
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
                topology[f'{process_id}_requester']['hidden_state'] = (
                    'hidden_state',)
                topology[f'{process_id}_evolver']['hidden_state'] = (
                    'hidden_state',)

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
            'bulk': ('bulk',)}

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

    return sim.run()



@pytest.mark.slow
def test_division(
        agent_id='1',
        total_time=60
):
    """tests that a cell can be divided and keep running"""

    # get initial mass from Ecoli composer
    initial_state = Ecoli({}).initial_state({'initial_state_file': 'vivecoli_t1840'})

    # make a new composer under an embedded path
    config = {
        'divide': True,
        'agent_id': agent_id,
        'division': {
            'threshold': 2220},  # fg
    }
    agent_path = ('agents', agent_id)
    ecoli_composer = Ecoli(config)
    ecoli_composite = ecoli_composer.generate(path=agent_path)

    # make and run the experiment
    experiment = Engine(
        processes=ecoli_composite.processes,
        steps=ecoli_composite.steps,
        flow=ecoli_composite.flow,
        topology=ecoli_composite.topology,
        initial_state={'agents': {agent_id: initial_state}},
    )
    experiment.update(total_time)

    # retrieve output
    output = experiment.emitter.get_data()

    # get the states of the daughter cells and the mother cell
    daughter_states = []
    for timestep in output:
        if len(output[timestep]['agents'].keys()) == 2:
            d1 = list(output[timestep]['agents'].keys())[0]
            daughter_states.append(output[timestep]['agents'][d1])
            d2 = list(output[timestep]['agents'].keys())[1]
            daughter_states.append(output[timestep]['agents'][d2])
            if timestep == 0.0:
                mother_state = initial_state
            else:
                mother_idx = list(output[timestep - 2.0]['agents'].keys())[0]
                mother_state = output[timestep - 2.0]['agents'][mother_idx]
            break

    # compare the counts of bulk molecules between the mother and daughters
    for bulk_molecule in mother_state['bulk']:
        if mother_state['bulk'][bulk_molecule] > COUNT_THRESHOLD:
            assert (scalar_almost_equal(mother_state['bulk'][bulk_molecule],
                                        daughter_states[0]['bulk'][bulk_molecule] +
                                        daughter_states[1]['bulk'][bulk_molecule],
                                        custom_threshold=0.1))

    # compare the counts of unique molecules between the mother and daughters
    idx_to_d = get_domain_index_to_daughter(mother_state['unique']['chromosome_domain'])
    for key in mother_state['unique']:
        num_divided = 0
        if key == 'promoter' or key == 'oriC' or key == 'DnaA_box' or key == 'chromosomal_segment' \
                or key == 'full_chromosome' or key == 'active_replisome':
            for unique_molecule in mother_state['unique'][key]:
                if idx_to_d[0][mother_state['unique'][key][unique_molecule]['domain_index']] != -1:
                    num_divided += 1
        elif key == 'RNA':
            for rna in mother_state['unique']['RNA']:
                if mother_state['unique']['RNA'][rna]['is_full_transcript']:
                    num_divided += 1
                else:
                    rnap_index = mother_state['unique']['RNA'][rna]['RNAP_index']
                    if idx_to_d[0][mother_state['unique']['active_RNAP'][rnap_index]['domain_index']] != -1:
                        num_divided += 1
        elif key == 'active_RNAP':
            for rnap in mother_state['unique']['active_RNAP']:
                if idx_to_d[0][mother_state['unique']['active_RNAP'][rnap]['domain_index']] != -1:
                    num_divided += 1
        elif key == 'active_ribosome':
            for ribosome in mother_state['unique']['active_ribosome']:
                mrna_index = mother_state['unique']['active_ribosome'][ribosome]['mRNA_index']
                if mother_state['unique']['RNA'][mrna_index]['is_full_transcript']:
                    num_divided += 1
                else:
                    rnap_index = mother_state['unique']['RNA'][mrna_index]['RNAP_index']
                    if idx_to_d[0][mother_state['unique']['active_RNAP'][rnap_index]['domain_index']] != -1:
                        num_divided += 1
        elif key == 'chromosome_domain':
            num_divided = len(mother_state['unique']['chromosome_domain'].keys()) - 1
        assert (scalar_almost_equal(num_divided,
                len(daughter_states[0]['unique'][key]) +
                len(daughter_states[1]['unique'][key]),
                custom_threshold=0.1))

    # asserts
    final_agents = output[total_time]['agents'].keys()
    print(f"initial agent id: {agent_id}")
    print(f"final agent ids: {final_agents}")
    assert len(final_agents) == 2


def test_division_topology():
    """test that the topology is correctly dividing"""
    timestep = 2

    # get initial mass from Ecoli composer
    initial_state = Ecoli({}).initial_state({'initial_state_file': 'vivecoli_t1840'})
    initial_mass = initial_state['listeners']['mass']['cell_mass']
    division_mass = initial_mass + 4.5
    print(f"DIVIDE AT {division_mass} fg")

    # make a new composer under an embedded path
    agent_id = '0'
    config = {
        'divide': True,
        'agent_id': agent_id,
        'division': {
            'threshold': division_mass},  # fg
    }
    agent_path = ('agents', agent_id)
    ecoli_composer = Ecoli(config)
    ecoli_composite = ecoli_composer.generate(path=agent_path)

    # make experiment
    experiment = Engine(
        processes=ecoli_composite.processes,
        steps=ecoli_composite.steps,
        flow=ecoli_composite.flow,
        topology=ecoli_composite.topology,
        initial_state={'agents': {agent_id: initial_state}},
    )

    full_topology = experiment.state.get_topology()
    mother_topology = full_topology['agents'][agent_id].copy()

    # update one time step at a time until division
    while len(full_topology['agents']) <= 1:
        experiment.update(timestep)
        full_topology = experiment.state.get_topology()

    # assert that the daughter topologies are the same as the mother topology
    daughter_ids = list(full_topology['agents'].keys())
    for daughter_id in daughter_ids:
        daughter_topology = full_topology['agents'][daughter_id]
        assert daughter_topology == mother_topology


def test_ecoli_generate():
    ecoli_composer = Ecoli({})
    ecoli_composite = ecoli_composer.generate()

    # asserts to ecoli_composite['processes'] and ecoli_composite['topology']
    assert all('_requester' in k or
               '_evolver' in k or
               k == 'allocator' or
               isinstance(v, ECOLI_DEFAULT_PROCESSES[k])
               for k, v in ecoli_composite['processes'].items())
    assert all(ECOLI_DEFAULT_TOPOLOGY[k] == v
               for k, v in ecoli_composite['topology'].items()
               if k in ECOLI_DEFAULT_TOPOLOGY)


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
    '1': test_division,
    '2': test_division_topology,
    '3': test_ecoli_generate,
    '4': ecoli_topology_plot,
}

# run experiments in test_library from the command line with:
# python ecoli/composites/ecoli_master.py -n [experiment id]
if __name__ == '__main__':
    run_library_cli(test_library)

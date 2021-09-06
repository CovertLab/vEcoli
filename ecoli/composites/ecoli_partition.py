"""
==============================
E. coli partitioning composite
==============================
NOTE: All ports with '_total' in their name are
automatically exempt from partitioning
"""

from copy import deepcopy

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
from ecoli.composites.ecoli_master_configs import (
    ECOLI_DEFAULT_PROCESSES, ECOLI_DEFAULT_TOPOLOGY)
from ecoli.plots.topology import get_partition_topology_settings
from ecoli.processes.cell_division import Division
from ecoli.processes.allocator import Allocator
from ecoli.processes.partition import PartitionedProcess

# state
from ecoli.processes.partition import get_bulk_topo, Requester, Evolver
from ecoli.states.wcecoli_state import get_state_from_file


RAND_MAX = 2**31
SIM_DATA_PATH = 'reconstruction/sim_data/kb/simData.cPickle'

MINIMAL_MEDIA_ID = 'minimal'
AA_MEDIA_ID = 'minimal_plus_amino_acids'
ANAEROBIC_MEDIA_ID = 'minimal_minus_oxygen'


class Ecoli(Composer):

    defaults = {
        'time_step': 2.0,
        'parallel': False,
        'seed': 0,
        'sim_data_path': SIM_DATA_PATH,
        'daughter_path': tuple(),
        'agent_id': '0',
        'agents_path': ('..', '..', 'agents',),
        'division': {
            'threshold': 2220},  # fg
        'divide': False,
        'log_updates': False
    }

    def __init__(self, config):
        super().__init__(config)

        self.load_sim_data = LoadSimData(
            sim_data_path=self.config['sim_data_path'],
            seed=self.config['seed'])

        if not self.config.get('processes'):
            self.config['processes'] = ECOLI_DEFAULT_PROCESSES.copy()
        if not self.config.get('process_configs'):
            self.config['process_configs'] = {process: "sim_data" for process in self.config['processes']}
        if not self.config.get('topology'):
            self.config['topology'] = ECOLI_DEFAULT_TOPOLOGY.copy()

        self.processes = self.config['processes']
        self.topology = self.config['topology']

    def initial_state(self, config=None, path=()):
        # Use initial state calculated with trna_charging and translationSupply disabled
        config = config or {}
        initial_time = config.get('initial_time', 0)
        initial_state = get_state_from_file(path=f'data/metabolism/wcecoli_t{initial_time}.json')
        embedded_state = {}
        assoc_path(embedded_state, path, initial_state)
        return embedded_state

    def generate_processes(self, config):
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
                    dict(default), process_configs[process])

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
                           if not processes[p].is_deriver()]
        )

        config['processes']['allocator'] = Allocator
        processes['allocator'] = (Allocator(process_configs['allocator'])
                                  if not config['log_updates']
                                  else make_logging_process(Allocator)(process_configs['allocator']))

        # Store list of partition processes
        self.partitioned_processes = [process_name
                         for process_name, process in processes.items()
                         if isinstance(process, PartitionedProcess)]

        # Update schema overrides to reflect name change for requesters/evolvers
        self.schema_override = {f'{p}_evolver': v for p, v in self.schema_override.items()
                                if p not in self.partitioned_processes}

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

        division = {}
        # add division
        if self.config['divide']:
            division_config = dict(
                config['division'],
                agent_id=self.config['agent_id'],
                composer=self)
            division = {'division': Division(division_config)}

        # Create final list of processes in the correct order.
        # Following process_order, except that:
        #   - All requesters appear before all evolvers
        #   - Allocator appears immediately after requesters and immediately before evolvers
        result = []
        last_requester = 0
        for i, process in enumerate(process_order):
            if process not in self.partitioned_processes:
                result.append(process)
            else:
                result.append(f'{process}_requester')
                last_requester = i

        result[last_requester+1:last_requester+1] = [f'{process}_evolver'
                                                     for process in process_order
                                                     if process in self.partitioned_processes and process != "allocator"]
        result.insert(last_requester+1, "allocator")

        result = {process: processes[process] for process in result}

        # Under default config, should look like
        # {**chromosome_structure, **metabolism, **requesters, **allocator, **evolvers, **division, **mrna_counts, **mass}
        return result

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
                bulk_topo = get_bulk_topo(ports)
                topology[f'{process_id}_requester']['request'] = {
                    '_path': ('request', process_id,),
                    **bulk_topo}
                topology[f'{process_id}_evolver']['allocate'] = {
                    '_path': ('allocate', process_id,),
                    **bulk_topo}

            # make the non-partitioned processes' topologies
            else:
                topology[process_id] = ports
                if config['log_updates']:
                    topology[process_id]['log_update'] = (
                        'log_update', process_id,)

        # add division
        if self.config['divide']:
            topology['division'] = {
                'variable': ('listeners', 'mass', 'cell_mass'),
                'agents': config['agents_path']}

        topology['allocator'] = {
            'request': ('request',),
            'allocate': ('allocate',),
            'bulk': ('bulk',)}

        if config['log_updates']:
            topology['allocator']['log_update'] = ('log_update', 'allocator',)

        return topology


def run_ecoli(
        total_time=10,
        divide=False,
        progress_bar=True,
        log_updates=False,
):
    """Run ecoli_master simulations

    Arguments: TODO -- complete the arguments docstring
        * **total_time** (:py:class:`int`): the total runtime of the experiment
        * **config** (:py:class:`dict`):

    Returns:
        * output data
    """
    from ecoli.experiments.ecoli_master_sim import EcoliSim, CONFIG_DIR_PATH

    sim = EcoliSim.from_file(CONFIG_DIR_PATH + "default.json")
    sim.total_time = total_time
    sim.divide = divide
    sim.progress_bar = progress_bar
    sim.log_updates = log_updates

    return sim.run()


def test_division():
    """
    Work in progress to get division working
    * TODO -- unique molecules need to be divided between daughter cells!!! This can get sophisticated
    """
    config = {
        'divide': True
    }
    ecoli_composer = Ecoli(config)
    ecoli_composite = ecoli_composer.generate()

    experiment = Engine(
        processes=ecoli_composite.processes,
        topology=ecoli_composite.topology,
    )

    experiment.update(10)
    # print(f"agents: {output['agents'].keys()}")
    import ipdb; ipdb.set_trace()


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
    settings = get_partition_topology_settings()
    topo_plot = plot_topology(
        ecoli,
        filename='topology',
        out_dir='out/composites/ecoli_partition',
        settings=settings
    )
    return topo_plot


test_library = {
    '0': run_ecoli,
    '1': test_division,
    '2': test_ecoli_generate,
    '3': ecoli_topology_plot,
}

# run experiments in test_library from the command line with:
# python ecoli/composites/ecoli_partition.py -n [experiment id]
if __name__ == '__main__':
    run_library_cli(test_library)

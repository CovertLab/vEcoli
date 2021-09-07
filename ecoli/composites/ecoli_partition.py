"""
==============================
E. coli partitioning composite
==============================
NOTE: All ports with '_total' in their name are
automatically exempt from partitioning
"""

import os
import argparse
from copy import deepcopy

from vivarium.core.composer import Composer
from vivarium.core.engine import Engine
from vivarium.plots.topology import plot_topology
from vivarium.library.topology import assoc_path
from vivarium.library.dict_utils import deep_merge
from vivarium.core.serialize import deserialize_value

# sim data
from ecoli.library.sim_data import LoadSimData

# logging
from ecoli.library.logging import make_logging_process

# vivarium-ecoli processes
from ecoli.composites.ecoli_master_configs import (
    ECOLI_DEFAULT_PROCESSES, ECOLI_DEFAULT_TOPOLOGY)
from ecoli.processes.cell_division import Division
from ecoli.processes.allocator import Allocator

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
        # initial_state = get_state_from_file(
        #      path='data/wcecoli_t0.json')
        initial_state = get_state_from_file(
            path='data/vivecoli_t10.json')

        # import json
        # with open('data/vivecoli_t10.json', 'r') as state_file:
        #     initial_state = json.load(state_file)
        # # initial_state = deserialize_value(initial_state)
        #
        # def infinitize(value):
        #     if value == '__INFINITY__':
        #         return float('inf')
        #     else:
        #         return value
        #
        # environment_state = {
        #     key: infinitize(value)
        #     for key, value in initial_state['environment'].items()}
        # from wholecell.utils import units
        # initial_state['environment'] = {
        #     'media_id': 'minimal',
        #     # TODO(Ryan): pull in environmental amino acid levels
        #     'amino_acids': {},
        #     'exchange_data': {
        #         'unconstrained': {
        #             'CL-[p]',
        #             'FE+2[p]',
        #             'CO+2[p]',
        #             'MG+2[p]',
        #             'NA+[p]',
        #             'CARBON-DIOXIDE[p]',
        #             'OXYGEN-MOLECULE[p]',
        #             'MN+2[p]',
        #             'L-SELENOCYSTEINE[c]',
        #             'K+[p]',
        #             'SULFATE[p]',
        #             'ZN+2[p]',
        #             'CA+2[p]',
        #             'Pi[p]',
        #             'NI+2[p]',
        #             'WATER[p]',
        #             'AMMONIUM[c]'},
        #         'constrained': {
        #             'GLC[p]': 20.0 * units.mmol / (units.g * units.h)}},
        #     'external_concentrations': environment_state}

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

        # Store list of derivers
        self.derivers = [process_name
                         for process_name, process in processes.items()
                         if process.is_deriver()]

        # Update schema overrides to reflect name change for requesters/evolvers
        self.schema_override = {f'{p}_evolver': v for p, v in self.schema_override.items()
                                if p not in self.derivers}

        # make the requesters
        requesters = {
            f'{process_name}_requester': Requester({'time_step': time_step,
                                                    'process': process})
            for (process_name, process) in processes.items()
            if process_name not in self.derivers
        }

        # make the evolvers
        evolvers = {
            f'{process_name}_evolver': Evolver({'time_step': time_step,
                                                'process': process})
            if not config['log_updates']
            else make_logging_process(Evolver)({'time_step': time_step,
                                                'process': process})
            for (process_name, process) in processes.items()
            if process_name not in self.derivers
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
            if process in self.derivers:
                result.append(process)
            else:
                result.append(f'{process}_requester')
                last_requester = i

        result[last_requester+1:last_requester+1] = [f'{process}_evolver'
                                                     for process in process_order
                                                     if process not in self.derivers and process != "allocator"]
        result.insert(last_requester+1, "allocator")

        result = {process: processes[process] for process in result}

        # Under default config, should look like
        # {**chromosome_structure, **metabolism, **requesters, **allocator, **evolvers, **division, **mrna_counts, **mass}
        return result

    def generate_topology(self, config):
        topology = {}

        # make the topology
        for process_id, ports in config['topology'].items():
            if process_id not in self.derivers:
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

    from ecoli.experiments.ecoli_master_sim import EcoliSim, CONFIG_DIR_PATH

    sim = EcoliSim.from_file(CONFIG_DIR_PATH + "default.json")
    sim.division = {'threshold': 1170}

    # Remove metabolism for now
    # (divison fails because cannot deepcopy metabolism process)
    # sim.exclude_processes.append("ecoli-metabolism")

    sim.total_time = 10
    sim.divide = True
    sim.progress_bar = True

    output = sim.run()


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


def get_partition_topology_settings():
    evolver_row = -6
    allocator_row = -7
    requester_row = -8
    process_distance = 0.9
    settings = {
        'graph_format': 'hierarchy',
        'dashed_edges': True,
        'show_ports': False,
        'node_size': 12000,
        'coordinates': {
            'tf_binding_evolver': (1 * process_distance, evolver_row),
            'tf_binding_requester': (1 * process_distance, requester_row),

            'transcript_initiation_evolver': (2 * process_distance, evolver_row),
            'transcript_initiation_requester': (2 * process_distance, requester_row),

            'transcript_elongation_evolver': (3 * process_distance, evolver_row),
            'transcript_elongation_requester': (3 * process_distance, requester_row),

            'rna_degradation_evolver': (4 * process_distance, evolver_row),
            'rna_degradation_requester': (4 * process_distance, requester_row),

            'polypeptide_initiation_evolver': (5 * process_distance, evolver_row),
            'polypeptide_initiation_requester': (5 * process_distance, requester_row),

            'polypeptide_elongation_evolver': (6 * process_distance, evolver_row),
            'polypeptide_elongation_requester': (6 * process_distance, requester_row),

            'complexation_evolver': (7 * process_distance, evolver_row),
            'complexation_requester': (7 * process_distance, requester_row),

            'two_component_system_evolver': (8 * process_distance, evolver_row),
            'two_component_system_requester': (8 * process_distance, requester_row),

            'equilibrium_evolver': (9 * process_distance, evolver_row),
            'equilibrium_requester': (9 * process_distance, requester_row),

            'protein_degradation_evolver': (10 * process_distance, evolver_row),
            'protein_degradation_requester': (10 * process_distance, requester_row),

            'chromosome_replication_evolver': (11 * process_distance, evolver_row),
            'chromosome_replication_requester': (11 * process_distance, requester_row),

            'metabolism': (12 * process_distance, evolver_row),
            'mass': (13 * process_distance, evolver_row),
            'mrna_counts': (14 * process_distance, evolver_row),
            'divide_condition': (15 * process_distance, evolver_row),

            'allocator': (6 * process_distance, allocator_row),
        },
        'node_labels': {
            # processes
            'tf_binding_requester': 'tf\nbinding\nrequester',
            'tf_binding_evolver': 'tf\nbinding\nevolver',

            'transcript_initiation_requester': 'transcript\ninitiation\nrequester',
            'transcript_initiation_evolver': 'transcript\ninitiation\nevolver',

            'transcript_elongation_requester': 'transcript\nelongation\nrequester',
            'transcript_elongation_evolver': 'transcript\nelongation\nevolver',

            'rna_degradation_requester': 'rna\ndegradation\nrequester',
            'rna_degradation_evolver': 'rna\ndegradation\nevolver',

            'polypeptide_initiation_requester': 'polypeptide\ninitiation\nrequester',
            'polypeptide_initiation_evolver': 'polypeptide\ninitiation\nevolver',

            'polypeptide_elongation_requester': 'polypeptide\nelongation\nrequester',
            'polypeptide_elongation_evolver': 'polypeptide\nelongation\nevolver',

            'complexation_requester': 'complexation\nrequester',
            'complexation_evolver': 'complexation\nevolver',

            'two_component_system_requester': 'two component\nsystem\nrequester',
            'two_component_system_evolver': 'two component\nsystem\nevolver',

            'equilibrium_requester': 'equilibrium\nrequester',
            'equilibrium_evolver': 'equilibrium\nevolver',

            'protein_degradation_requester': 'protein\ndegradation\nrequester',
            'protein_degradation_evolver': 'protein\ndegradation\nevolver',

            'chromosome_replication_requester': 'chromosome\nreplication\nrequester',
            'chromosome_replication_evolver': 'chromosome\nreplication\nevolver',

            'metabolism': 'metabolism',
            'mass': 'mass',
            'mrna_counts': 'mrna\ncounts',
            'divide_condition': 'division',
        },
    }
    return settings


def ecoli_topology_plot(config={}, filename=None, out_dir=None):
    """Make a topology plot of Ecoli"""
    agent_id_config = {'agent_id': '1'}
    ecoli = Ecoli({**agent_id_config, **config})
    settings = get_partition_topology_settings()
    topo_plot = plot_topology(
        ecoli,
        filename=filename,
        out_dir=out_dir,
        settings=settings
    )
    return topo_plot


test_library = {
    '0': run_ecoli,
    '1': test_division,
    '2': test_ecoli_generate
}


def main():
    out_dir = os.path.join('out', 'ecoli_partition')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    parser = argparse.ArgumentParser(description='ecoli_partition')
    parser.add_argument(
        '--name', '-n', default=[], nargs='+',
        help='test ids to run')
    parser.add_argument(
        '--topology', '-t', action='store_true', default=False,
        help='save a topology plot of ecoli master')
    parser.add_argument(
        '--blame', '-b', action='store_true', default=False,
        help='when running simulation, create a report of which processes affected which molecules')
    parser.add_argument(
        '--debug', '-d', action='store_true', default=False,
        help='run tests, generating a report of failures/successes')
    args = parser.parse_args()

    if args.topology:
        ecoli_topology_plot(filename='ecoli_partition', out_dir=out_dir)
    elif args.name:
        for name in args.name:
            test_library[name]()
    else:
        output = run_ecoli(
            blame=args.blame,
        )
        if args.debug:
            pass

        if args.blame:
            blame_plot(output, highlight_molecules=['PD00413[c]'])


if __name__ == '__main__':
    main()

"""
==============================
E. coli partitioning composite
==============================
"""

import os
import argparse
from copy import deepcopy

from vivarium.core.composer import Composer
from vivarium.core.engine import pp, Engine
from vivarium.plots.topology import plot_topology
from vivarium.library.topology import assoc_path
from vivarium.library.dict_utils import deep_merge
from vivarium.core.process import Process, Deriver

# sim data
from ecoli.library.sim_data import LoadSimData
from data.ecoli_master_configs import default

# logging
from ecoli.library.logging import make_logging_process
from ecoli.plots.blame import blame_plot

# vivarium-ecoli processes
from ecoli.plots.topology import get_ecoli_master_topology_settings
from ecoli.processes.cell_division import Division
from ecoli.processes.allocator import Allocator

# state
from ecoli.states.wcecoli_state import get_state_from_file

from ecoli.library.data_predicates import all_nonnegative

RAND_MAX = 2**31
SIM_DATA_PATH = 'reconstruction/sim_data/kb/simData.cPickle'

MINIMAL_MEDIA_ID = 'minimal'
AA_MEDIA_ID = 'minimal_plus_amino_acids'
ANAEROBIC_MEDIA_ID = 'minimal_minus_oxygen'


def change_bulk_updater(schema, new_updater):
    """Retrieve port schemas for all bulk molecules
    and modify their updater

    Args:
        schema (Dict): The ports schema to change
        new_updater (String): The new updater to use

    Returns:
        Dict: Ports schema that only includes bulk molecules 
        with the new updater
    """
    bulk_schema = {}
    if '_properties' in schema:
        if schema['_properties']['bulk']:
            topo_copy = schema.copy()
            topo_copy.update({'_updater': new_updater})
            return topo_copy
    for port, value in schema.items():
        if has_bulk_property(value):
            bulk_schema[port] = change_bulk_updater(value, new_updater)
    return bulk_schema

def has_bulk_property(schema):
    """Check to see if a subset of the ports schema contains
    a bulk molecule using {'_property': {'bulk': True}}

    Args:
        schema (Dict): Subset of ports schema to check for bulk

    Returns:
        Bool: Whether the subset contains a bulk molecule
    """
    if isinstance(schema, dict):
        if '_properties' in schema:
            if schema['_properties']['bulk']:
                return True

        for value in schema.values():
            if isinstance(value, dict):
                if has_bulk_property(value):
                    return True
    return False

def get_bulk_topo(topo):
    """Return topology of only bulk molecules
    NOTE: Does not work with '_path' key

    Args:
        topo (Dict): Experiment topology

    Returns:
        Dict: Experiment topology with non-bulk stores excluded
    """
    if 'bulk' in topo:
        return topo
    if isinstance(topo, dict):
        bulk_topo = {}
        for port, value in topo.items():
            if path_in_bulk(value):
                bulk_topo[port] = get_bulk_topo(value)
    return bulk_topo

def path_in_bulk(topo):
    """Check whether a subset of the topology is contained within
    the bulk store

    Args:
        topo (Dict): Subset of experiment topology

    Returns:
        Bool: Whether subset contains stores listed under 'bulk'
    """
    if 'bulk' in topo:
        return True
    if isinstance(topo, dict):
        for value in topo.values():
            if path_in_bulk(value):
                return True
    return False


class Requester(Deriver):
    defaults = {'process': None}

    def __init__(self, parameters=None):
        super().__init__(parameters)
        self.process = self.parameters['process']

    def ports_schema(self):
        ports = self.process.ports_schema()
        ports_copy = ports.copy()
        ports['request'] = change_bulk_updater(ports_copy, 'set')
        return ports

    def next_update(self, timestep, states):
        update = self.process.calculate_request(self.parameters['time_step'], states)
        # Ensure listeners are updated if passed by calculate_request
        listeners = update.pop('listeners', None)
        if listeners != None:
            return {'request': update, 'listeners': listeners}
        return {'request': update}
    


class Evolver(Process):
    defaults = {'process': None}

    def __init__(self, parameters=None):
        super().__init__(parameters)
        self.process = self.parameters['process']

    def ports_schema(self):
        ports = self.process.ports_schema()
        ports_copy = ports.copy()
        ports['allocate'] = change_bulk_updater(ports_copy, 'set')
        return ports

    def next_update(self, timestep, states):
        states = deep_merge(states, states.pop('allocate'))
        return self.process.evolve_state(timestep, states)


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
        'blame': False,
        'processes' : default.ECOLI_PROCESSES.copy(),
        'topology' : default.ECOLI_TOPOLOGY.copy()
    }

    def __init__(self, config):
        super().__init__(config)

        self.load_sim_data = LoadSimData(
            sim_data_path=self.config['sim_data_path'],
            seed=self.config['seed'])

    def initial_state(self, config=None, path=()):
        initial_state = get_state_from_file(path='data/metabolism/wcecoli_t0.json')
        embedded_state = {}
        assoc_path(embedded_state, path, initial_state)
        return embedded_state

    def generate_processes(self, config):
        time_step = config['time_step']
        parallel = config['parallel']     
        
        process_names = list(config['processes'].keys())
        process_names.remove('mass')
        process_names.remove('metabolism')
        
        config['processes']['allocator'] = Allocator   

        # get the configs from sim_data
        configs = {
            'tf_binding': self.load_sim_data.get_tf_config(time_step=time_step),
            'transcript_initiation': self.load_sim_data.get_transcript_initiation_config(time_step=time_step),
            'transcript_elongation': self.load_sim_data.get_transcript_elongation_config(time_step=time_step),
            'rna_degradation': self.load_sim_data.get_rna_degradation_config(time_step=time_step),
            'polypeptide_initiation': self.load_sim_data.get_polypeptide_initiation_config(time_step=time_step),
            'polypeptide_elongation': self.load_sim_data.get_polypeptide_elongation_config(time_step=time_step),
            'complexation': self.load_sim_data.get_complexation_config(time_step=time_step),
            'two_component_system': self.load_sim_data.get_two_component_system_config(time_step=time_step),
            'equilibrium': self.load_sim_data.get_equilibrium_config(time_step=time_step),
            'protein_degradation': self.load_sim_data.get_protein_degradation_config(time_step=time_step),
            'metabolism': self.load_sim_data.get_metabolism_config(time_step=time_step, deriver_mode=True),
            'chromosome_replication': self.load_sim_data.get_chromosome_replication_config(time_step=time_step),
            'mass': self.load_sim_data.get_mass_listener_config(time_step=time_step),
            'allocator': self.load_sim_data.get_allocator_config(time_step=time_step, process_names=process_names)
        }

        # make the processes
        processes = {
            process_name: process(configs[process_name])
            for (process_name, process) in config['processes'].items()
            if process_name not in [
                # 'polypeptide_elongation'
                # TODO: get this working again
            ]
        }
        
        derivers = ['metabolism', 'mass', 'allocator']
        
        # make the requesters
        requesters = {
            f'{process_name}_requester': Requester({'time_step': time_step,
                                                    'process': process})
            for (process_name, process) in processes.items()
            if process_name not in derivers
        }
        
        # make the evolvers
        evolvers = {
            f'{process_name}_evolver': Evolver({'time_step': time_step,
                                                'process': process})
                if not config['blame']
                else make_logging_process(Evolver)({'time_step': time_step,
                                                    'process': process})
            for (process_name, process) in processes.items()
            if process_name not in derivers
        }
        
        if config['blame']:
            processes['metabolism'] = make_logging_process(
                config['processes']['metabolism'])(configs['metabolism'])
        else:
            processes['metabolism'] = config['processes']['metabolism'](configs['metabolism'])
        
        division = {}
        # add division
        if self.config['divide']:
            division_config = dict(
                config['division'],
                agent_id=self.config['agent_id'],
                composer=self)
            division = {'division': Division(division_config)}
            
        allocator = {'allocator': processes['allocator']}
        mass = {'mass': processes['mass']}
        metabolism = {'metabolism': processes['metabolism']}

        all_procs = {**metabolism, **requesters, **allocator, **evolvers, **division, **mass}
        
        return all_procs

    def generate_topology(self, config):
        topology = {}
        
        derivers = ['metabolism', 'mass', 'allocator']

        # make the topology
        for process_id, ports in config['topology'].items():
            if process_id not in derivers:
                topology[f'{process_id}_requester'] = deepcopy(ports)
                topology[f'{process_id}_evolver'] = deepcopy(ports)
                if config['blame']:
                    topology[f'{process_id}_evolver']['log_update'] = ('log_update', process_id,)
                bulk_topo = get_bulk_topo(ports)
                topology[f'{process_id}_requester']['request'] = {
                    '_path': ('request', process_id,),
                    **bulk_topo}
                topology[f'{process_id}_evolver']['allocate'] = {
                    '_path': ('allocate', process_id,),
                    **bulk_topo}

        # add division
        if self.config['divide']:
            topology['division'] = {
                'variable': ('listeners', 'mass', 'cell_mass'),
                'agents': config['agents_path']}
            
        topology['allocator'] = {
            'request': ('request',),
            'allocate': ('allocate',),
            'bulk': ('bulk',)}
        
        topology['mass'] = config['topology']['mass']
        
        topology['metabolism'] = config['topology']['metabolism']
        
        if config['blame']:
            topology['metabolism']['log_update'] = ('log_update', 'metabolism',)

        return topology


def run_ecoli(
        total_time=10,
        config=None,
        divide=False,
        progress_bar=True,
        blame=False,
):
    """Run ecoli_master simulations

    Arguments: TODO -- complete the arguments docstring
        * **total_time** (:py:class:`int`): the total runtime of the experiment
        * **config** (:py:class:`dict`):

    Returns:
        * output data
    """
    # make the ecoli config dictionary
    agent_id = '0'
    ecoli_config = {
        'blame': blame,
        'agent_id': agent_id,
        # TODO -- remove schema override once values don't go negative
        '_schema': {
            'equilibrium_evolver': {
                'molecules': {
                    'PD00413[c]': {'_updater': 'nonnegative_accumulate'}
                }
            },
        },
    }
    if config:
        ecoli_config = deep_merge(ecoli_config, config)

    # initialize the ecoli composer
    ecoli_composer = Ecoli(ecoli_config)

    # set path at which agent is initialized
    path = tuple()
    if divide:
        path = ('agents', agent_id,)

    # get initial state
    initial_state = ecoli_composer.initial_state(path=path)

    # generate the composite at the path
    ecoli = ecoli_composer.generate(path=path)

    # make the experiment
    ecoli_experiment = Engine({
        'processes': ecoli.processes,
        'topology': ecoli.topology,
        'initial_state': initial_state,
        'progress_bar': progress_bar,
        'emit_config': False,
        # Not emitting every step is faster but breaks blame.py
        #'emit_step': 1000,
        #'emitter': 'database'
    })

    # run the experiment
    ecoli_experiment.update(total_time)

    # retrieve the data
    output = ecoli_experiment.emitter.get_timeseries()
    
    pp(output['listeners']['mass'])

    return output


def test_division():
    """
    Work in progress to get division working

    * TODO -- unique molecules need to be divided between daughter cells!!! This can get sophisticated
    """

    config = {
        'division': {
            'threshold': 1170}}
    output = run_ecoli(
        total_time=10,
        divide=True,
        config=config,
        progress_bar=False,
    )


def assertions(sim_output):

    test_structure = {
        'bulk' : ...,# transform_and_run(array_from, nonnegative),
        'process_state' : {
            'polypeptide_elongation' : {
                'aa_count_diff' : ...,
                'gtp_to_hydrolyze' : ...
            }
        },
        'listeners' : {
            'rna_synth_prob' : {
                'pPromoterBound' : ...,
                'nPromoterBound' : ...,
                'nActualBound' : ...,
                'n_available_promoters' : ...,
                'n_bound_TF_per_TU' : ...,
                'rna_synth_prob' : ...
            },
            'mass' : {
                'cell_mass' : ...,
                'dry_mass' : ...,
                'water_mass' : ...
            },
            'ribosome_data' : {
                'rrn16S_produced' : ...,
                'rrn23S_produced' : ...,
                'rrn5S_produced' : ...,
                'rrn16S_init_prob' : ...,
                'rrn23S_init_prob' : ...,
                'rrn5S_init_prob' : ...,
                'total_rna_init' : ...,
                'ribosomes_initialized' : ...,
                'prob_translation_per_transcript' : ...,
                'effective_elongation_rate' : ...,
                'translation_supply' : ...,
                'aaCountInSequence' : ...,
                'aaCounts' : ...,
                'actualElongations' : ...,
                'actualElongationHist' : ...,
                'elongationsNonTerminatingHist' : ...,
                'didTerminate' : ...,
                'terminationLoss' : ...,
                'numTrpATerminated' : ...,
                'processElongationRate' : ...
            },
            'rnap_data' : {
                'didInitialize' : ...,
                'rnaInitEvent' : ...,
                'actualElongations' : ...,
                'didTerminate' : ...,
                'terminationLoss' : ...,
                'didStall' : ...
            },
            'transcript_elongation_listener' : {
                'countNTPsUsed' : ...,
                'countRnaSynthesized' : ...,
                'attenuation_probability' : ...,
                'counts_attenuated' : ...
            },
            'growth_limits' : {
                'ntpUsed' : ...,
                'fraction_trna_charged': ...,
                'aa_pool_size' : ...,
                'aa_request_size' : ...,
                'aa_allocated' : ...,
                'active_ribosomes_allocated' : ...,
                'net_charged' : ...,
                'aasUsed' : ...
            },
            'rna_degradation_listener' : {
                'fraction_active_endo_rnases' : ...,
                'diff_relative_first_order_decay' : ...,
                'fract_endo_rrna_counts' : ...,
                'count_rna_degraded' : ...,
                'nucleotides_from_degradation' : ...,
                'fragment_bases_digested' : ...
            },
            'equilibrium_listener' : {},
            'fba_results' : {},
            'enzyme_kinetics' : {},
            'replication_data' : {}
        }
    }

    for molecule, timeseries in sim_output['bulk'].items():
        assert all_nonnegative(timeseries), f'{molecule} goes negative'


def ecoli_topology_plot(config={}, filename=None, out_dir=None):
    """Make a topology plot of Ecoli"""
    agent_id_config = {'agent_id': '1'}
    ecoli = Ecoli({**agent_id_config, **config})
    settings = get_ecoli_master_topology_settings()
    topo_plot = plot_topology(
        ecoli,
        filename=filename,
        out_dir=out_dir,
        settings=settings)
    return topo_plot


test_library = {
    '0': run_ecoli,
    '1': test_division,
}


def main():
    out_dir = os.path.join('out', 'ecoli_master')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    parser = argparse.ArgumentParser(description='ecoli_master')
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
        ecoli_topology_plot(filename='ecoli_master', out_dir=out_dir)
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

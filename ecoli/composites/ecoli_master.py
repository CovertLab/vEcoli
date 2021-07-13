"""
========================
E. coli master composite
========================
"""

import os
import argparse
import json
import uuid
from pprint import pformat

from vivarium.core.composer import Composer
from vivarium.core.engine import pp, Engine
from vivarium.plots.topology import plot_topology

# sim data
from ecoli.library.sim_data import LoadSimData

# logging
from ecoli.library.logging import make_logging_process

# vivarium processes
from vivarium.processes.divide_condition import DivideCondition

# vivarium-ecoli processes
from ecoli.processes.tf_binding import TfBinding
from ecoli.processes.transcript_initiation import TranscriptInitiation
from ecoli.processes.transcript_elongation import TranscriptElongation
from ecoli.processes.rna_degradation import RnaDegradation
from ecoli.processes.polypeptide_initiation import PolypeptideInitiation
from ecoli.processes.polypeptide_elongation import PolypeptideElongation
from ecoli.processes.complexation import Complexation
from ecoli.processes.two_component_system import TwoComponentSystem
from ecoli.processes.equilibrium import Equilibrium
from ecoli.processes.protein_degradation import ProteinDegradation
from ecoli.processes.metabolism import Metabolism
from ecoli.processes.chromosome_replication import ChromosomeReplication
from ecoli.processes.mass import Mass

from wholecell.utils import units

RAND_MAX = 2**31
SIM_DATA_PATH = 'reconstruction/sim_data/kb/simData.cPickle'

ECOLI_PROCESSES = {
    'tf_binding': TfBinding,
    'transcript_initiation': TranscriptInitiation,
    'transcript_elongation': TranscriptElongation,
    'rna_degradation': RnaDegradation,
    'polypeptide_initiation': PolypeptideInitiation,
    'polypeptide_elongation': PolypeptideElongation,
    'complexation': Complexation,
    'two_component_system': TwoComponentSystem,
    'equilibrium': Equilibrium,
    'protein_degradation': ProteinDegradation,
    'metabolism': Metabolism,
    'chromosome_replication': ChromosomeReplication,
    'mass': Mass,
    'divide_condition': DivideCondition,
}

ECOLI_TOPOLOGY = {
        'tf_binding': {
            'promoters': ('unique', 'promoter'),
            'active_tfs': ('bulk',),
            'inactive_tfs': ('bulk',),
            'listeners': ('listeners',)},

        'transcript_initiation': {
            'environment': ('environment',),
            'full_chromosomes': ('unique', 'full_chromosome'),
            'RNAs': ('unique', 'RNA'),
            'active_RNAPs': ('unique', 'active_RNAP'),
            'promoters': ('unique', 'promoter'),
            'molecules': ('bulk',),
            'listeners': ('listeners',)},

        'transcript_elongation': {
            'environment': ('environment',),
            'RNAs': ('unique', 'RNA'),
            'active_RNAPs': ('unique', 'active_RNAP'),
            'molecules': ('bulk',),
            'bulk_RNAs': ('bulk',),
            'ntps': ('bulk',),
            'listeners': ('listeners',)},

        'rna_degradation': {
            'charged_trna': ('bulk',),
            'bulk_RNAs': ('bulk',),
            'nmps': ('bulk',),
            'fragmentMetabolites': ('bulk',),
            'fragmentBases': ('bulk',),
            'endoRnases': ('bulk',),
            'exoRnases': ('bulk',),
            'subunits': ('bulk',),
            'molecules': ('bulk',),
            'RNAs': ('unique', 'RNA'),
            'active_ribosome': ('unique', 'active_ribosome'),
            'listeners': ('listeners',)},

        'polypeptide_initiation': {
            'environment': ('environment',),
            'listeners': ('listeners',),
            'active_ribosome': ('unique', 'active_ribosome'),
            'RNA': ('unique', 'RNA'),
            'subunits': ('bulk',)},

        'polypeptide_elongation': {
            'environment': ('environment',),
            'listeners': ('listeners',),
            'active_ribosome': ('unique', 'active_ribosome'),
            'molecules': ('bulk',),
            'monomers': ('bulk',),
            'amino_acids': ('bulk',),
            'ppgpp_reaction_metabolites': ('bulk',),
            'uncharged_trna': ('bulk',),
            'charged_trna': ('bulk',),
            'charging_molecules': ('bulk',),
            'synthetases': ('bulk',),
            'subunits': ('bulk',),
            'polypeptide_elongation': ('process_state', 'polypeptide_elongation')},

        'complexation': {
            'molecules': ('bulk',)},

        'two_component_system': {
            'listeners': ('listeners',),
            'molecules': ('bulk',)},

        'equilibrium': {
            'listeners': ('listeners',),
            'molecules': ('bulk',)},

        'protein_degradation': {
            'metabolites': ('bulk',),
            'proteins': ('bulk',)},

        'metabolism': {
            'metabolites': ('bulk',),
            'catalysts': ('bulk',),
            'kinetics_enzymes': ('bulk',),
            'kinetics_substrates': ('bulk',),
            'amino_acids': ('bulk',),
            'listeners': ('listeners',),
            'environment': ('environment',),
            'polypeptide_elongation': ('process_state', 'polypeptide_elongation')},

        'chromosome_replication': {
            'replisome_trimers': ('bulk',),
            'replisome_monomers': ('bulk',),
            'dntps': ('bulk',),
            'ppi': ('bulk',),
            'active_replisomes': ('unique', 'active_replisome',),
            'oriCs': ('unique', 'oriC',),
            'chromosome_domains': ('unique', 'chromosome_domain',),
            'full_chromosomes': ('unique', 'full_chromosome',),
            'listeners': ('listeners',),
            'environment': ('environment',)},

        'mass': {
            'bulk': ('bulk',),
            'unique': ('unique',),
            'listeners': ('listeners',)},

        'divide_condition': {
            'variable': ('listeners', 'mass', 'cell_mass'),
            'divide': ('globals', 'divide',),
        },
    }


class Ecoli(Composer):

    defaults = {
        'time_step': 2.0,
        'parallel': False,
        'seed': 0,
        'sim_data_path': SIM_DATA_PATH,
        'daughter_path': tuple(),
        'division': {'threshold': 2220},  # fg
        'blame' : False
    }

    def __init__(self, config):
        super().__init__(config)

        self.load_sim_data = LoadSimData(
            sim_data_path=self.config['sim_data_path'],
            seed=self.config['seed'])

    def initial_state(self, config=None):
        return get_state_from_file()

    def generate_processes(self, config):
        time_step = config['time_step']
        parallel = config['parallel']  # TODO (Eran) -- which processes can be parallelized?

        # get the configs from sim_data
        configs = {
            'tf_binding' : self.load_sim_data.get_tf_config(time_step=time_step),
            'transcript_initiation' : self.load_sim_data.get_transcript_initiation_config(time_step=time_step),
            'transcript_elongation' : self.load_sim_data.get_transcript_elongation_config(time_step=time_step),
            'rna_degradation' : self.load_sim_data.get_rna_degradation_config(time_step=time_step),
            'polypeptide_initiation' : self.load_sim_data.get_polypeptide_initiation_config(time_step=time_step),
            'polypeptide_elongation' : self.load_sim_data.get_polypeptide_elongation_config(time_step=time_step),
            'complexation' : self.load_sim_data.get_complexation_config(time_step=time_step),
            'two_component_system' : self.load_sim_data.get_two_component_system_config(time_step=time_step),
            'equilibrium' : self.load_sim_data.get_equilibrium_config(time_step=time_step),
            'protein_degradation' : self.load_sim_data.get_protein_degradation_config(time_step=time_step),
            'metabolism' : self.load_sim_data.get_metabolism_config(time_step=time_step),
            'chromosome_replication' : self.load_sim_data.get_chromosome_replication_config(time_step=time_step),
            'mass' : self.load_sim_data.get_mass_config(time_step=time_step),

            # additional processes
            'divide_condition' : config['division']
        }

        return {
            process_name: (process(configs[process_name])
                           if not config['blame']
                           else make_logging_process(process)(configs[process_name]))

            for (process_name, process) in ECOLI_PROCESSES.items()
            #if process_name != "polypeptide_elongation"  # TODO: get polypeptide elongation working again
        }

    def generate_topology(self, config):
        topology = {}
        for process_id, ports in ECOLI_TOPOLOGY.items():
            topology[process_id] = ports
            if config['blame']:
                topology[process_id]['log_update'] = ('log_update', process_id,)
        return topology


def infinitize(value):
    if value == '__INFINITY__':
        return float('inf')
    else:
        return value


def load_states(path):
    with open(path, 'r') as states_file:
        states = json.load(states_file)

    states['environment'] = {
        key: infinitize(value)
        for key, value in states['environment'].items()}

    return states


def get_state_from_file(path='data/wcecoli_t0.json'):

    states = load_states(path)

    initial_state = {
        'environment': {
            'media_id': 'minimal',
            # TODO(Ryan): pull in environmental amino acid levels
            'amino_acids': {},
            'exchange_data': {
                'unconstrained': {
                    'CL-[p]',
                    'FE+2[p]',
                    'CO+2[p]',
                    'MG+2[p]',
                    'NA+[p]',
                    'CARBON-DIOXIDE[p]',
                    'OXYGEN-MOLECULE[p]',
                    'MN+2[p]',
                    'L-SELENOCYSTEINE[c]',
                    'K+[p]',
                    'SULFATE[p]',
                    'ZN+2[p]',
                    'CA+2[p]',
                    'PI[p]',
                    'NI+2[p]',
                    'WATER[p]',
                    'AMMONIUM[c]'},
                'constrained': {
                    'GLC[p]': 20.0 * units.mmol / (units.g * units.h)}},
            'external_concentrations': states['environment']},
        # TODO(Eran): deal with mass
        # add mw property to bulk and unique molecules
        # and include any "submass" attributes from unique molecules
        'listeners': states['listeners'],
        'bulk': states['bulk'],
        'unique': states['unique'],
        'process_state': {
            'polypeptide_elongation': {}}}

    return initial_state


def run_ecoli(blame=False, total_time=10):
    # configure the composer
    ecoli_config = {
        'agent_id': '1',
        # TODO -- remove schema override once values don't go negative
        '_schema': {
            'equilibrium': {
                'molecules': {
                    'PD00413[c]': {
                        '_updater': 'nonnegative_accumulate'
                    }
                }
            }
        },
        'blame' : blame
    }
    ecoli_composer = Ecoli(ecoli_config)

    # get initial state
    initial_state = get_state_from_file()

    # make the experiment
    ecoli = ecoli_composer.generate()
    ecoli_experiment = Engine({
        'processes': ecoli.processes,
        'topology': ecoli.topology,
        'initial_state': initial_state,
        'progress_bar': True,
    })

    # run the experiment
    ecoli_experiment.update(total_time)

    # retrieve the data
    output = ecoli_experiment.emitter.get_timeseries()

    # separate data by port
    bulk = output['bulk']
    unique = output['unique']
    listeners = output['listeners']
    process_state = output['process_state']
    environment = output['environment']

    # print(bulk)
    # print(unique.keys())
    pp(listeners['mass'])

    return output


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

    from ecoli.library.data_predicates import all_nonnegative

    for molecule, timeseries in sim_output['bulk'].items():
        assert all_nonnegative(timeseries), f'{molecule} goes negative'


def ecoli_topology_plot(out_dir='out'):
    ecoli = Ecoli({'agent_id': '1'})

    process_row = -4
    process_distance = 0.9
    settings = {
        'graph_format': 'hierarchy',
        'dashed_edges': True,
        'show_ports': False,
        'node_size': 12000,
        'coordinates': {
            'tf_binding': (1*process_distance, process_row),
            'transcript_initiation': (2*process_distance, process_row),
            'transcript_elongation': (3*process_distance, process_row),
            'rna_degradation': (4*process_distance, process_row),
            'polypeptide_initiation': (5*process_distance, process_row),
            'polypeptide_elongation': (6*process_distance, process_row),
            'complexation': (7*process_distance, process_row),
            'two_component_system': (8*process_distance, process_row),
            'equilibrium': (9*process_distance, process_row),
            'protein_degradation': (10*process_distance, process_row),
            'metabolism': (11*process_distance, process_row),
            'mass': (12*process_distance, process_row),
            'divide_condition': (13*process_distance, process_row),
            'chromosome_replication': (14*process_distance, process_row),
            'mass': (13*process_distance, process_row),
            'divide_condition': (14*process_distance, process_row),
        },
        'node_labels': {
            # processes
            'tf_binding': 'tf\nbinding',
            'transcript_initiation': 'transcript\ninitiation',
            'transcript_elongation': 'transcript\nelongation',
            'rna_degradation': 'rna\ndegradation',
            'polypeptide_initiation': 'polypeptide\ninitiation',
            'polypeptide_elongation': 'polypeptide\nelongation',
            'complexation': 'complexation',
            'two_component_system': 'two component\nsystem',
            'equilibrium': 'equilibrium',
            'protein_degradation': 'protein\ndegradation',
            'metabolism': 'metabolism',
            'chromosome_replication': 'chromosome\nreplication',
            'mass': 'mass',
            'divide_condition': 'division',
        },
        'remove_nodes': [
            'listeners\nmass\ncell_mass',
            'process_state',
            'listeners\nfba_results',
            'listeners\nenzyme_kinetics',
            'listeners\nmass',
            'listeners\nribosome_data',
            'listeners\nfba_results',
            'listeners\nequilibrium_listener',
            'listeners\nrna_degradation_listener',
            'process_state\npolypeptide_elongation',
        ]
    }
    plot_topology(
        ecoli,
        filename='ecoli_master',
        out_dir=out_dir,
        settings=settings)


def main():
    out_dir = os.path.join('out', 'ecoli_master')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    parser = argparse.ArgumentParser(description='ecoli_master')
    parser.add_argument('-topology', '-t', action='store_true', default=False,
                        help='save a topology plot of ecoli master')
    parser.add_argument('-blame', '-b', action='store_true', default=False,
                        help='when running simulation, create a report of which processes affected which molecules')
    parser.add_argument('-debug', '-d', action='store_true', default=False,
                        help='run tests, generating a report of failures/successes')
    args = parser.parse_args()

    if args.topology:
        ecoli_topology_plot(out_dir)
    else:
        if args.debug:
            output = run_ecoli(args.blame)
            assertions(output)
        else:
            output = run_ecoli(args.blame)

        if args.blame:
            from ecoli.plots.blame import blame_plot
            blame_plot(output, highlight_molecules=['PD00413[c]'])


if __name__ == '__main__':
    main()

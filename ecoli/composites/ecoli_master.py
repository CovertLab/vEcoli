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

from vivarium.core.process import Composer
from vivarium.core.experiment import pp, Experiment
from vivarium.plots.topology import plot_topology

# sim data
from ecoli.library.sim_data import LoadSimData

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
from ecoli.processes.mass import Mass

from wholecell.utils import units

RAND_MAX = 2**31
SIM_DATA_PATH = 'reconstruction/sim_data/kb/simData.cPickle'


class Ecoli(Composer):

    defaults = {
        'time_step': 2.0,
        'parallel': False,
        'seed': 0,
        'sim_data_path': SIM_DATA_PATH,
        'daughter_path': tuple(),
        'division': {'threshold': 2220},  # fg
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
        tf_binding_config = self.load_sim_data.get_tf_config(time_step=time_step)
        transcript_initiation_config = self.load_sim_data.get_transcript_initiation_config(time_step=time_step)
        transcript_elongation_config = self.load_sim_data.get_transcript_elongation_config(time_step=time_step)
        rna_degradation_config = self.load_sim_data.get_rna_degradation_config(time_step=time_step)
        polypeptide_initiation_config = self.load_sim_data.get_polypeptide_initiation_config(time_step=time_step)
        polypeptide_elongation_config = self.load_sim_data.get_polypeptide_elongation_config(time_step=time_step)
        complexation_config = self.load_sim_data.get_complexation_config(time_step=time_step)
        two_component_system_config = self.load_sim_data.get_two_component_system_config(time_step=time_step)
        equilibrium_config = self.load_sim_data.get_equilibrium_config(time_step=time_step)
        protein_degradation_config = self.load_sim_data.get_protein_degradation_config(time_step=time_step)
        metabolism_config = self.load_sim_data.get_metabolism_config(time_step=time_step)
        mass_config = self.load_sim_data.get_mass_config(time_step=time_step)

        # additional processes
        divide_config = config['division']

        return {
            'tf_binding': TfBinding(tf_binding_config),
            'transcript_initiation': TranscriptInitiation(transcript_initiation_config),
            'transcript_elongation': TranscriptElongation(transcript_elongation_config),
            'rna_degradation': RnaDegradation(rna_degradation_config),
            'polypeptide_initiation': PolypeptideInitiation(polypeptide_initiation_config),
            # 'polypeptide_elongation': PolypeptideElongation(polypeptide_elongation_config),
            'complexation': Complexation(complexation_config),
            'two_component_system': TwoComponentSystem(two_component_system_config),
            'equilibrium': Equilibrium(equilibrium_config),
            'protein_degradation': ProteinDegradation(protein_degradation_config),
            'metabolism': Metabolism(metabolism_config),
            'mass': Mass(mass_config),
            'divide_condition': DivideCondition(divide_config),
        }

    def generate_topology(self, config):
        return {
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

            # 'polypeptide_elongation': {
            #     'environment': ('environment',),
            #     'listeners': ('listeners',),
            #     'active_ribosome': ('unique', 'active_ribosome'),
            #     'molecules': ('bulk',),
            #     'monomers': ('bulk',),
            #     'amino_acids': ('bulk',),
            #     'ppgpp_reaction_metabolites': ('bulk',),
            #     'uncharged_trna': ('bulk',),
            #     'charged_trna': ('bulk',),
            #     'charging_molecules': ('bulk',),
            #     'synthetases': ('bulk',),
            #     'subunits': ('bulk',),
            #     'polypeptide_elongation': ('process_state', 'polypeptide_elongation')},

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

            'mass': {
                'bulk': ('bulk',),
                'unique': ('unique',),
                'listeners': ('listeners',)},

            'divide_condition': {
                'variable': ('listeners', 'mass', 'cell_mass'),
                'divide': ('globals', 'divide',),
            },

        }



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


def test_ecoli(
        total_time=10,
        debug_config=False,
):

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
        }
    }
    ecoli_composer = Ecoli(ecoli_config)

    # get initial state
    initial_state = get_state_from_file()

    # make the experiment
    ecoli = ecoli_composer.generate()
    ecoli_experiment = Experiment({
        'processes': ecoli.processes,
        'topology': ecoli.topology,
        'initial_state': initial_state,
        'progress_bar': True,
    })

    if debug_config:
        print(pformat(ecoli_experiment.state.get_config(True)))
        import ipdb; ipdb.set_trace()

    # run the experiment
    ecoli_experiment.update(total_time)

    # retrieve the data
    data = ecoli_experiment.emitter.get_timeseries()
    return data


def run_ecoli():
    output = test_ecoli()

    # separate data by port
    bulk = output['bulk']
    unique = output['unique']
    listeners = output['listeners']
    process_state = output['process_state']
    environment = output['environment']

    # print(bulk)
    # print(unique.keys())
    pp(listeners['mass'])


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
    parser.add_argument('-topology', '-t', action='store_true', default=False, help='save a topology plot of ecoli master')
    parser.add_argument('-simulate', '-s', action='store_true', default=False, help='simulate ecoli master')
    args = parser.parse_args()


    if args.topology:
        ecoli_topology_plot(out_dir)
    else:
        run_ecoli()

if __name__ == '__main__':
    main()

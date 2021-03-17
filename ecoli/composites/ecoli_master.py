"""
========================
E. coli master composite
========================
"""

import os
import argparse
import json
import uuid

from vivarium.core.process import Composer
from vivarium.core.composition import simulate_composer
from vivarium.core.experiment import pp
from vivarium.plots.topology import plot_topology

# sim data
from ecoli.library.sim_data import LoadSimData

# vivarium processes
from vivarium.processes.divide_condition import DivideCondition
from vivarium.processes.meta_division import MetaDivision

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
    }

    def __init__(self, config):
        super(Ecoli, self).__init__(config)

        self.load_sim_data = LoadSimData(
            sim_data_path=self.config['sim_data_path'],
            seed=self.config['seed'])

    def initial_state(self, config=None):
        return get_state_from_file()

    def generate_processes(self, config):
        time_step = config['time_step']
        parallel = config['parallel']  # TODO (Eran) -- which processes can be parallelized?

        # initialize processes
        tf_binding = TfBinding(
            self.load_sim_data.get_tf_config(time_step=time_step))
        transcript_initiation = TranscriptInitiation(
            self.load_sim_data.get_transcript_initiation_config(time_step=time_step))
        transcript_elongation = TranscriptElongation(
            self.load_sim_data.get_transcript_elongation_config(time_step=time_step))
        rna_degradation = RnaDegradation(
            self.load_sim_data.get_rna_degradation_config(time_step=time_step))
        polypeptide_initiation = PolypeptideInitiation(
            self.load_sim_data.get_polypeptide_initiation_config(time_step=time_step))
        polypeptide_elongation = PolypeptideElongation(
            self.load_sim_data.get_polypeptide_elongation_config(time_step=time_step))
        complexation = Complexation(
            self.load_sim_data.get_complexation_config(time_step=time_step))
        two_component_system = TwoComponentSystem(
            self.load_sim_data.get_two_component_system_config(time_step=time_step))
        equilibrium = Equilibrium(
            self.load_sim_data.get_equilibrium_config(time_step=time_step))
        protein_degradation = ProteinDegradation(
            self.load_sim_data.get_protein_degradation_config(time_step=time_step))
        metabolism = Metabolism(
            self.load_sim_data.get_metabolism_config(time_step=time_step))
        mass = Mass(
            self.load_sim_data.get_mass_config(time_step=time_step))

        # Division
        # TODO -- get mass for division from sim_data
        # TODO -- set divider to binomial division
        divide_config = {'threshold': 2220}  # fg
        divide_condition = DivideCondition(divide_config)

        # daughter_path = config['daughter_path']
        # agent_id = config.get('agent_id', str(uuid.uuid1()))
        # division_config = dict(
        #     config.get('division', {}),
        #     daughter_path=daughter_path,
        #     agent_id=agent_id,
        #     compartment=self)
        # meta_division = MetaDivision(division_config)

        return {
            'tf_binding': tf_binding,
            'transcript_initiation': transcript_initiation,
            'transcript_elongation': transcript_elongation,
            'rna_degradation': rna_degradation,
            'polypeptide_initiation': polypeptide_initiation,
            'polypeptide_elongation': polypeptide_elongation,
            'complexation': complexation,
            'two_component_system': two_component_system,
            'equilibrium': equilibrium,
            'protein_degradation': protein_degradation,
            'metabolism': metabolism,
            'mass': mass,
            'divide_condition': divide_condition,
            # 'division': meta_division,
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

            'mass': {
                'bulk': ('bulk',),
                'unique': ('unique',),
                'listeners': ('listeners',)},

            'divide_condition': {
                'variable': ('listeners', 'mass', 'cell_mass'),
                'divide': ('globals', 'divide',),
            },

            # 'division': {
            #     'global': boundary_path,
            #     'agents': agents_path
            # },
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


def test_ecoli():
    ecoli = Ecoli({'agent_id': '1'})
    initial_state = get_state_from_file()
    settings = {
        'timestep': 1,
        'total_time': 10,
        'initial_state': initial_state}

    data = simulate_composer(ecoli, settings)

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


    parser = argparse.ArgumentParser(description='bioscrape_cobra')
    parser.add_argument('-topology', '-t', action='store_true', default=False, help='save a topology plot of ecoli master')
    parser.add_argument('-simulate', '-s', action='store_true', default=False, help='simulate ecoli master')
    args = parser.parse_args()


    if args.topology:
        ecoli_topology_plot(out_dir)
    else:
        run_ecoli()

if __name__ == '__main__':
    main()

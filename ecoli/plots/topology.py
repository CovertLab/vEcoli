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
from ecoli.processes.listeners.mass_listener import MassListener
from ecoli.processes.listeners.mRNA_counts import mRNACounts
from ecoli.processes.cell_division import Division
from ecoli.processes.chromosome_structure import ChromosomeStructure


def get_ecoli_master_topology_settings():
    """plot settings for ecoli_master's topology plot"""
    process_row = -4
    process_distance = 0.9
    settings = {
        'graph_format': 'hierarchy',
        'dashed_edges': True,
        'show_ports': False,
        'node_size': 12000,
        'font_size': 12,
        'coordinates': {
            TfBinding.name: (1*process_distance, process_row),
            TranscriptInitiation.name: (2*process_distance, process_row),
            TranscriptElongation.name: (3*process_distance, process_row),
            RnaDegradation.name: (4*process_distance, process_row),
            PolypeptideInitiation.name: (5*process_distance, process_row),
            PolypeptideElongation.name: (6*process_distance, process_row),
            Complexation.name: (7*process_distance, process_row),
            TwoComponentSystem.name: (8*process_distance, process_row),
            Equilibrium.name: (9*process_distance, process_row),
            ProteinDegradation.name: (10*process_distance, process_row),
            Metabolism.name: (11*process_distance, process_row),
            ChromosomeReplication.name: (12 * process_distance, process_row),
            ChromosomeStructure.name: (13 * process_distance, process_row),
            MassListener.name: (14*process_distance, process_row),
            mRNACounts.name: (15 * process_distance, process_row),
            Division.name: (16*process_distance, process_row),
        },
        'node_labels': {
            # processes
            TfBinding.name: 'tf\nbinding',
            TranscriptInitiation.name: 'transcript\ninitiation',
            TranscriptElongation.name: 'transcript\nelongation',
            RnaDegradation.name: 'rna\ndegradation',
            PolypeptideInitiation.name: 'polypeptide\ninitiation',
            PolypeptideElongation.name: 'polypeptide\nelongation',
            Complexation.name: Complexation.name,
            TwoComponentSystem.name: 'two component\nsystem',
            Equilibrium.name: Equilibrium.name,
            ProteinDegradation.name: 'protein\ndegradation',
            Metabolism.name: Metabolism.name,
            ChromosomeReplication.name: 'chromosome\nreplication',
            ChromosomeStructure.name: 'chromosome\nstructure',
            mRNACounts.name: 'mrna\ncounts',
            Division.name: 'division',
            MassListener.name: 'mass',
            # stores
            'unique\nchromosome_domain': 'unique\nchromosome\ndomain',
        },
        'remove_nodes': [
            'aa_enzymes',
            'listeners\nmass\ncell_mass',
            'process_state',
            'listeners\nfba_results',
            'listeners\nenzyme_kinetics',
            'listeners\nmass',
            'listeners\nribosome_data',
            'listeners\nfba_results',
            'listeners\nequilibrium_listener',
            'listeners\nrna_degradation_listener',
            'listeners\ntranscript_elongation_listener',
            'process_state\npolypeptide_elongation',
        ]
    }
    return settings

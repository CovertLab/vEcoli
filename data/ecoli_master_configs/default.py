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
from ecoli.processes.cell_division import Division
from ecoli.processes.listeners.mass_listener import MassListener
from ecoli.processes.listeners.mRNA_counts import mRNACounts
from ecoli.processes.chromosome_structure import ChromosomeStructure

ECOLI_PROCESSES = {
    ChromosomeStructure.name: ChromosomeStructure,
    Metabolism.name: Metabolism,
    TfBinding.name: TfBinding,
    TranscriptInitiation.name: TranscriptInitiation,
    TranscriptElongation.name: TranscriptElongation,
    RnaDegradation.name: RnaDegradation,
    PolypeptideInitiation.name: PolypeptideInitiation,
    PolypeptideElongation.name: PolypeptideElongation,
    Complexation.name: Complexation,
    TwoComponentSystem.name: TwoComponentSystem,
    Equilibrium.name: Equilibrium,
    ProteinDegradation.name: ProteinDegradation,
    ChromosomeReplication.name: ChromosomeReplication,
    mRNACounts.name: mRNACounts,
    MassListener.name: MassListener,
}

ECOLI_TOPOLOGY = {
        TfBinding.name: {
            'promoters': ('unique', 'promoter'),
            'active_tfs': ('bulk',),
            'listeners': ('listeners',),
            # Non-partitioned counts
            'active_tfs_total': ('bulk',),
            'inactive_tfs_total': ('bulk',),},

        TranscriptInitiation.name: {
            'environment': ('environment',),
            'full_chromosomes': ('unique', 'full_chromosome'),
            'RNAs': ('unique', 'RNA'),
            'active_RNAPs': ('unique', 'active_RNAP'),
            'promoters': ('unique', 'promoter'),
            'molecules': ('bulk',),
            'listeners': ('listeners',)},

        TranscriptElongation.name: {
            'environment': ('environment',),
            'RNAs': ('unique', 'RNA'),
            'active_RNAPs': ('unique', 'active_RNAP'),
            'molecules': ('bulk',),
            'bulk_RNAs': ('bulk',),
            'ntps': ('bulk',),
            'listeners': ('listeners',)},

        RnaDegradation.name: {
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

        PolypeptideInitiation.name: {
            'environment': ('environment',),
            'listeners': ('listeners',),
            'active_ribosome': ('unique', 'active_ribosome'),
            'RNA': ('unique', 'RNA'),
            'subunits': ('bulk',)},

        PolypeptideElongation.name: {
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
            'polypeptide_elongation': ('process_state', 'polypeptide_elongation'),
            # Non-partitioned counts
            'molecules_total': ('bulk',),
            'amino_acids_total': ('bulk',),
            'charged_trna_total': ('bulk',),
            'uncharged_trna_total': ('bulk',),},

        Complexation.name: {
            'molecules': ('bulk',),
            'listeners': ('listeners',),
        },

        TwoComponentSystem.name: {
            'listeners': ('listeners',),
            'molecules': ('bulk',)},

        Equilibrium.name: {
            'listeners': ('listeners',),
            'molecules': ('bulk',)},

        ProteinDegradation.name: {
            'metabolites': ('bulk',),
            'proteins': ('bulk',)},

        Metabolism.name: {
            'metabolites': ('bulk',),
            'catalysts': ('bulk',),
            'kinetics_enzymes': ('bulk',),
            'kinetics_substrates': ('bulk',),
            'amino_acids': ('bulk',),
            'listeners': ('listeners',),
            'environment': ('environment',),
            'polypeptide_elongation': ('process_state', 'polypeptide_elongation')},

        ChromosomeReplication.name: {
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

        MassListener.name: {
            'bulk': ('bulk',),
            'unique': ('unique',),
            'listeners': ('listeners',)},

        mRNACounts.name: {
            'listeners': ('listeners',),
            'RNAs': ('unique', 'RNA')},
        
        ChromosomeStructure.name: {
            'fragmentBases': ('bulk',),
            'molecules': ('bulk',),
            'active_tfs': ('bulk',),
            'subunits': ('bulk',),
            'amino_acids': ('bulk',),
            'active_replisomes': ('unique', 'active_replisome',),
            'oriCs': ('unique', 'oriC',),
            'chromosome_domains': ('unique', 'chromosome_domain',),
            'active_RNAPs': ('unique', 'active_RNAP'),
            'RNAs': ('unique', 'RNA'),
            'active_ribosome': ('unique', 'active_ribosome'),
            'full_chromosomes': ('unique', 'full_chromosome',),
            'promoters': ('unique', 'promoter'),
            'DnaA_boxes': ('unique', 'DnaA_box'),
            # TODO Include this only if superhelical density flag is passed
            # 'chromosomal_segments': ('unique', 'chromosomal_segment')
            }
    }

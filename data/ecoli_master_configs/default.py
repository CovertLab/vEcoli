from ecoli.processes.registries import topology_registry

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


ECOLI_DEFAULT_PROCESSES = [
    ChromosomeStructure,
    Metabolism,
    TfBinding,
    TranscriptInitiation,
    TranscriptElongation,
    RnaDegradation,
    PolypeptideInitiation,
    PolypeptideElongation,
    Complexation,
    TwoComponentSystem,
    Equilibrium,
    ProteinDegradation,
    ChromosomeReplication,
    mRNACounts,
    MassListener,
]

ECOLI_PROCESSES = {
    process.name: process
    for process in ECOLI_DEFAULT_PROCESSES}

ECOLI_TOPOLOGY = {
    process_name: topology_registry.access(process_name)
    for process_name in ECOLI_PROCESSES.keys()
}

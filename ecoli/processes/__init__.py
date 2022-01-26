from vivarium import process_registry

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
from ecoli.processes.stubs.exchange_stub import Exchange
from ecoli.processes.listeners.mass_listener import MassListener
from ecoli.processes.listeners.mRNA_counts import mRNACounts
from ecoli.processes.listeners.monomer_counts import MonomerCounts
from ecoli.processes.chromosome_structure import ChromosomeStructure
from ecoli.processes.allocator import Allocator
from ecoli.processes.environment.lysis import Lysis
from ecoli.processes.environment.local_field import LocalField
from ecoli.processes.shape import Shape
from vivarium.processes.timeline import TimelineProcess



# add to registry
process_registry.register(TfBinding.name, TfBinding)
process_registry.register(TranscriptInitiation.name, TranscriptInitiation)
process_registry.register(TranscriptElongation.name, TranscriptElongation)
process_registry.register(RnaDegradation.name, RnaDegradation)
process_registry.register(PolypeptideInitiation.name, PolypeptideInitiation)
process_registry.register(PolypeptideElongation.name, PolypeptideElongation)
process_registry.register(Complexation.name, Complexation)
process_registry.register(TwoComponentSystem.name, TwoComponentSystem)
process_registry.register(Equilibrium.name, Equilibrium)
process_registry.register(ProteinDegradation.name, ProteinDegradation)
process_registry.register(Metabolism.name, Metabolism)
process_registry.register(ChromosomeReplication.name, ChromosomeReplication)
process_registry.register(Mass.name, Mass)
process_registry.register(MassListener.name, MassListener)
process_registry.register(Exchange.name, Exchange)
process_registry.register(mRNACounts.name, mRNACounts)
process_registry.register(MonomerCounts.name, MonomerCounts)
process_registry.register(ChromosomeStructure.name, ChromosomeStructure)
process_registry.register(Allocator.name, Allocator)
process_registry.register(Shape.name, Shape)

# environment processes
process_registry.register(Lysis.name, Lysis)
process_registry.register(LocalField.name, LocalField)

# auxiliary processes
process_registry.register('timeline', TimelineProcess)

# Test marA regulation
process_registry.register('ecoli-tf-binding-marA', TfBinding)

from vivarium import process_registry
from vivarium.core.registry import Registry

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
from ecoli.processes.exchange_stub import Exchange
from ecoli.processes.listeners.mass_listener import MassListener

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

#: Maps process names to topology
topology_registry = Registry()
topology_registry.register(
    TfBinding.name,
    {
            "promoters": ("unique", "promoter"),
            "active_tfs": ("bulk",),
            "inactive_tfs": ("bulk",),
            "listeners": ("listeners",)
    })
from vivarium import process_registry
from ecoli.processes.antibiotics.lysis_initiation import LysisInitiation

from ecoli.processes.tf_unbinding import TfUnbinding
from ecoli.processes.tf_binding import TfBinding
from ecoli.processes.transcript_initiation import TranscriptInitiation
from ecoli.processes.transcript_elongation import TranscriptElongation
from ecoli.processes.rna_degradation import RnaDegradation
from ecoli.processes.rna_maturation import RnaMaturation
from ecoli.processes.polypeptide_initiation import PolypeptideInitiation
from ecoli.processes.polypeptide_elongation import PolypeptideElongation
from ecoli.processes.complexation import Complexation
from ecoli.processes.two_component_system import TwoComponentSystem
from ecoli.processes.equilibrium import Equilibrium
from ecoli.processes.protein_degradation import ProteinDegradation
from ecoli.processes.metabolism import Metabolism
from ecoli.processes.metabolism_redux import MetabolismRedux
from ecoli.processes.metabolism_redux_classic import MetabolismReduxClassic
from ecoli.processes.chromosome_replication import ChromosomeReplication
from ecoli.processes.stubs.exchange_stub import Exchange
from ecoli.processes.listeners.mass_listener import (
    MassListener,
    PostDivisionMassListener,
)
from ecoli.processes.listeners.RNA_counts import RNACounts
from ecoli.processes.listeners.monomer_counts import MonomerCounts
from ecoli.processes.listeners.rna_synth_prob import RnaSynthProb
from ecoli.processes.listeners.dna_supercoiling import DnaSupercoiling
from ecoli.processes.listeners.replication_data import ReplicationData
from ecoli.processes.listeners.rnap_data import RnapData
from ecoli.processes.listeners.unique_molecule_counts import UniqueMoleculeCounts
from ecoli.processes.listeners.ribosome_data import RibosomeData
from ecoli.processes.chromosome_structure import ChromosomeStructure
from ecoli.processes.allocator import Allocator
from ecoli.processes.environment.lysis import Lysis
from ecoli.processes.environment.local_field import LocalField
from ecoli.processes.environment.field_timeline import FieldTimeline
from ecoli.processes.environment.exchange_data import ExchangeData
from ecoli.processes.environment.media_update import MediaUpdate
from ecoli.processes.shape import Shape
from ecoli.processes.antibiotics.cell_wall import CellWall
from ecoli.processes.antibiotics.pbp_binding import PBPBinding
from ecoli.processes.concentrations_deriver import ConcentrationsDeriver
from ecoli.processes.antibiotics.death import DeathFreezeState
from ecoli.processes.antibiotics.antibiotic_transport_steady_state import (
    AntibioticTransportSteadyState,
)
from ecoli.processes.antibiotics.antibiotic_transport_odeint import (
    AntibioticTransportOdeint,
)
from ecoli.processes.antibiotics.permeability import Permeability
from ecoli.processes.antibiotics.tetracycline_ribosome_equilibrium import (
    TetracyclineRibosomeEquilibrium,
)
from ecoli.processes.chemostat import Chemostat
from ecoli.processes.listeners.aggregator import Aggregator
from ecoli.processes.rna_interference import RnaInterference
from ecoli.processes.antibiotics.conc_to_counts import ConcToCounts
from ecoli.processes.global_clock import GlobalClock
from ecoli.processes.antibiotics.murein_division import MureinDivision
from ecoli.processes.bulk_timeline import BulkTimelineProcess


# add to registry
process_registry.register(TfUnbinding.name, TfUnbinding)
process_registry.register(TfBinding.name, TfBinding)
process_registry.register(TranscriptInitiation.name, TranscriptInitiation)
process_registry.register(TranscriptElongation.name, TranscriptElongation)
process_registry.register(RnaDegradation.name, RnaDegradation)
process_registry.register(RnaMaturation.name, RnaMaturation)
process_registry.register(PolypeptideInitiation.name, PolypeptideInitiation)
process_registry.register(PolypeptideElongation.name, PolypeptideElongation)
process_registry.register(Complexation.name, Complexation)
process_registry.register(TwoComponentSystem.name, TwoComponentSystem)
process_registry.register(Equilibrium.name, Equilibrium)
process_registry.register(ProteinDegradation.name, ProteinDegradation)
process_registry.register(Metabolism.name, Metabolism)
process_registry.register(MetabolismRedux.name, MetabolismRedux)
process_registry.register(MetabolismReduxClassic.name, MetabolismReduxClassic)
process_registry.register(ChromosomeReplication.name, ChromosomeReplication)
process_registry.register(MassListener.name, MassListener)
process_registry.register(PostDivisionMassListener.name, PostDivisionMassListener)
process_registry.register(DnaSupercoiling.name, DnaSupercoiling)
process_registry.register(ReplicationData.name, ReplicationData)
process_registry.register(RnapData.name, RnapData)
process_registry.register(UniqueMoleculeCounts.name, UniqueMoleculeCounts)
process_registry.register(RibosomeData.name, RibosomeData)
process_registry.register(Exchange.name, Exchange)
process_registry.register(RNACounts.name, RNACounts)
process_registry.register(MonomerCounts.name, MonomerCounts)
process_registry.register(RnaSynthProb.name, RnaSynthProb)
process_registry.register(ChromosomeStructure.name, ChromosomeStructure)
process_registry.register(Allocator.name, Allocator)
process_registry.register(Shape.name, Shape)
process_registry.register(ConcentrationsDeriver.name, ConcentrationsDeriver)
process_registry.register(Aggregator.name, Aggregator)

# environment processes
process_registry.register(Lysis.name, Lysis)
process_registry.register(LocalField.name, LocalField)
process_registry.register(FieldTimeline.name, FieldTimeline)
process_registry.register(ExchangeData.name, ExchangeData)
process_registry.register(MediaUpdate.name, MediaUpdate)

# auxiliary processes
process_registry.register(Chemostat.name, Chemostat)

# antibiotic processes
process_registry.register(DeathFreezeState.name, DeathFreezeState)
process_registry.register(
    TetracyclineRibosomeEquilibrium.name, TetracyclineRibosomeEquilibrium
)
process_registry.register(
    AntibioticTransportSteadyState.name, AntibioticTransportSteadyState
)
process_registry.register(AntibioticTransportOdeint.name, AntibioticTransportOdeint)
process_registry.register(Permeability.name, Permeability)
process_registry.register(LysisInitiation.name, LysisInitiation)
process_registry.register(CellWall.name, CellWall)
process_registry.register(PBPBinding.name, PBPBinding)
process_registry.register(ConcToCounts.name, ConcToCounts)
process_registry.register(RnaInterference.name, RnaInterference)
process_registry.register(GlobalClock.name, GlobalClock)
process_registry.register(MureinDivision.name, MureinDivision)
process_registry.register(BulkTimelineProcess.name, BulkTimelineProcess)

__all__ = ["processes"]

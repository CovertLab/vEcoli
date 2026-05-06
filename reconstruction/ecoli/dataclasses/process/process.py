"""
SimulationData process associated data
"""

from .antibiotics import CellWall
from .chromosome_structure import ChromosomeStructure
from .complexation import Complexation
from .equilibrium import Equilibrium
from .metabolism import Metabolism
from .replication import Replication
from .rna_decay import RnaDecay
from .transcription import Transcription
from .transcription_regulation import TranscriptionRegulation
from .translation import Translation
from .two_component_system import TwoComponentSystem


class Antibiotics(object):
    """Reference data for conditional antibiotic-related processes.

    Unlike the other ``Process``-attached namespaces, which back universal
    whole-cell processes ParCa fits, this namespace carries static
    reference data that conditional processes (cell wall, PBP binding,
    etc.) consume at simulation time. ParCa does not fit anything here.
    """

    def __init__(self, raw_data, sim_data):
        self.cell_wall = CellWall(raw_data, sim_data)


class Process(object):
    """Process"""

    def __init__(self, raw_data, sim_data):
        self.antibiotics = Antibiotics(raw_data, sim_data)
        self.chromosome_structure = ChromosomeStructure(raw_data, sim_data)
        self.complexation = Complexation(raw_data, sim_data)
        self.equilibrium = Equilibrium(raw_data, sim_data)
        self.metabolism = Metabolism(raw_data, sim_data)
        self.replication = Replication(raw_data, sim_data)
        self.rna_decay = RnaDecay(raw_data, sim_data)
        self.transcription = Transcription(raw_data, sim_data)
        self.transcription_regulation = TranscriptionRegulation(raw_data, sim_data)
        self.translation = Translation(raw_data, sim_data)
        self.two_component_system = TwoComponentSystem(raw_data, sim_data)

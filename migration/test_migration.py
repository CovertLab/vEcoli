from migration.chromosome_replication import (chromosome_replication_actual_update,
                                              chromosome_replication_default,
                                              chromosome_replication_fork_termination,
                                              chromosome_replication_initiate_replication)
from migration.chromosome_structure import chromosome_structure_migration
from migration.complexation import complexation_migration
from migration.composite_mass import run_composite_mass
# from migration.mass_listener import run_mass_listener
from migration.metabolism import (run_metabolism_default,
                                  run_metabolism_aas,
                                  run_metabolism_migration)
from migration.polypeptide_initiation import (run_polypeptide_initiation_default,
                                              run_polypeptide_initiation_migration)
from migration.polypeptide_elongation import run_polypeptide_elongation_migration
from migration.protein_degradation import run_protein_degradation
from migration.rna_degradation import (run_rna_degradation_default,
                                       run_rna_degradation_migration)
from migration.tf_binding import run_tf_binding_migration
from migration.transcript_elongation import run_transcription_elongation
from migration.transcript_initiation import run_transcript_initiation
from migration.two_component_system import run_two_component_system_migration
from migration.equilibrium import run_equilibrium

from ecoli.library.sim_data import LoadSimData
from ecoli.composites.ecoli_nonpartition import SIM_DATA_PATH



class TestMigration:
    sim_data = None

    @classmethod
    def setUpClass(self):
        self.sim_data = LoadSimData(sim_data_path=SIM_DATA_PATH, seed=0)

    @classmethod
    def tearDownClass(self):
        self.sim_data = None  # not really needed

    # CHROMOSOME REPLICATION
    def test_complexation_actual_update(self):
        chromosome_replication_actual_update(self.sim_data)
    def test_chromosome_replication_default(self):
        chromosome_replication_default(self.sim_data)
    def test_chromosome_replication_fork_termination(self):
        chromosome_replication_fork_termination(self.sim_data)
    def test_chromosome_replication_initiate_replication(self):
        chromosome_replication_initiate_replication(self.sim_data)

    # CHROMOSOME STRUCTURE
    def test_chromosome_structure_migration(self):
        chromosome_structure_migration(self.sim_data)

    # COMPLEXATION
    def test_complexation_migration(self):
        complexation_migration(self.sim_data)

    # METABOLISM
    def test_run_metabolism_default(self):
        run_metabolism_default(self.sim_data)
    def test_run_metabolism_aas(self):
        run_metabolism_aas(self.sim_data)
    def test_run_metabolism_migration(self):
        run_metabolism_migration(self.sim_data)

    # MASS
    def test_run_composite_mass(self):
        run_composite_mass(self.sim_data)
    # def test_run_mass_listener():
    #     run_mass_listener()

    # POLYPEPTIDE INITIATION
    def test_run_polypeptide_initiation_default(self):
        run_polypeptide_initiation_default(self.sim_data)
    def test_run_polypeptide_initiation_migration(self):
        run_polypeptide_initiation_migration(self.sim_data)

    # POLYPEPTIDE ELONGATION
    def test_run_polypeptide_elongation_migration(self):
        run_polypeptide_elongation_migration(self.sim_data)

    # PROTEIN DEGRADATION
    def test_run_protein_degradation(self):
        run_protein_degradation(self.sim_data)

    # RNA DEGRADATION
    def test_run_rna_degradation_default(self):
        run_rna_degradation_default(self.sim_data)
    def test_run_rna_degradation_migration(self):
        run_rna_degradation_migration(self.sim_data)

    # TF BINDING
    def test_run_tf_binding_migration(self):
        run_tf_binding_migration(self.sim_data)

    # TRANSCRIPT ELONGATION
    def test_run_transcription_elongation(self):
        run_transcription_elongation(self.sim_data)

    # TRANSCRIPT INITIATION
    def test_run_transcript_initiation(self):
        run_transcript_initiation(self.sim_data)

    # TWO COMPONENT SYSTEM
    def test_run_two_component_system_migration(self):
        run_two_component_system_migration(self.sim_data)

    # EQUILIBRIUM
    def test_run_equilibrium():
        run_equilibrium(self.sim_data)

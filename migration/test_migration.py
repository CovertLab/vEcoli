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

sim_data = LoadSimData(
    sim_data_path=SIM_DATA_PATH,
    seed=0)


class TestMigration:

    # CHROMOSOME REPLICATION
    @staticmethod
    def test_complexation_actual_update():
        chromosome_replication_actual_update(sim_data)
    @staticmethod
    def test_chromosome_replication_default():
        chromosome_replication_default(sim_data)
    @staticmethod
    def test_chromosome_replication_fork_termination():
        chromosome_replication_fork_termination(sim_data)
    @staticmethod
    def test_chromosome_replication_initiate_replication():
        chromosome_replication_initiate_replication(sim_data)

    # CHROMOSOME STRUCTURE
    @staticmethod
    def test_chromosome_structure_migration():
        chromosome_structure_migration(sim_data)

    # COMPLEXATION
    @staticmethod
    def test_complexation_migration():
        complexation_migration(sim_data)

    # METABOLISM
    @staticmethod
    def test_run_metabolism_default():
        run_metabolism_default(sim_data)
    @staticmethod
    def test_run_metabolism_aas():
        run_metabolism_aas(sim_data)
    @staticmethod
    def test_run_metabolism_migration():
        run_metabolism_migration(sim_data)

    # MASS
    @staticmethod
    def test_run_composite_mass():
        run_composite_mass(sim_data)
    # @staticmethod
    # def test_run_mass_listener():
    #     run_mass_listener()

    # POLYPEPTIDE INITIATION
    @staticmethod
    def test_run_polypeptide_initiation_default():
        run_polypeptide_initiation_default(sim_data)
    @staticmethod
    def test_run_polypeptide_initiation_migration():
        run_polypeptide_initiation_migration(sim_data)

    # POLYPEPTIDE ELONGATION
    @staticmethod
    def test_run_polypeptide_elongation_migration():
        run_polypeptide_elongation_migration(sim_data)

    # PROTEIN DEGRADATION
    @staticmethod
    def test_run_protein_degradation():
        run_protein_degradation(sim_data)

    # RNA DEGRADATION
    @staticmethod
    def test_run_rna_degradation_default():
        run_rna_degradation_default(sim_data)
    @staticmethod
    def test_run_rna_degradation_migration():
        run_rna_degradation_migration(sim_data)

    # TF BINDING
    @staticmethod
    def test_run_tf_binding_migration():
        run_tf_binding_migration(sim_data)

    # TRANSCRIPT ELONGATION
    @staticmethod
    def test_run_transcription_elongation():
        run_transcription_elongation(sim_data)

    # TRANSCRIPT INITIATION
    @staticmethod
    def test_run_transcript_initiation():
        run_transcript_initiation(sim_data)

    # TWO COMPONENT SYSTEM
    @staticmethod
    def test_run_two_component_system_migration():
        run_two_component_system_migration(sim_data)

    # EQUILIBRIUM
    @staticmethod
    def test_run_equilibrium():
        run_equilibrium(sim_data)

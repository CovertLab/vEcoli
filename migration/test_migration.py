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
# from migration.protein_degradation import run_protein_degradation
# from migration.rna_degradation import (run_rna_degradation_default,
#                                        run_rna_degradation_migration)
# from migration.tf_binding import run_tf_binding_migration
# from migration.transcript_elongation import run_transcription_elongation
# from migration.transcript_initiation import run_transcript_initiation
# from migration.two_component_system import run_two_component_system_migration

from ecoli.library.sim_data import LoadSimData
from ecoli.composites.ecoli_nonpartition import SIM_DATA_PATH

load_sim_data = LoadSimData(
    sim_data_path=SIM_DATA_PATH,
    seed=0)


class TestMigration:

    # CHROMOSOME REPLICATION
    @staticmethod
    def test_complexation_actual_update():
        chromosome_replication_actual_update(load_sim_data)
    @staticmethod
    def test_chromosome_replication_default():
        chromosome_replication_default(load_sim_data)
    @staticmethod
    def test_chromosome_replication_fork_termination():
        chromosome_replication_fork_termination(load_sim_data)
    @staticmethod
    def test_chromosome_replication_initiate_replication():
        chromosome_replication_initiate_replication(load_sim_data)

    # CHROMOSOME STRUCTURE
    @staticmethod
    def test_chromosome_structure_migration():
        chromosome_structure_migration(load_sim_data)

    # COMPLEXATION
    @staticmethod
    def test_complexation_migration():
        complexation_migration(load_sim_data)

    # METABOLISM
    @staticmethod
    def test_run_metabolism_default():
        run_metabolism_default(load_sim_data)
    @staticmethod
    def test_run_metabolism_aas():
        run_metabolism_aas(load_sim_data)
    @staticmethod
    def test_run_metabolism_migration():
        run_metabolism_migration(load_sim_data)

    # MASS
    @staticmethod
    def test_run_composite_mass():
        run_composite_mass(load_sim_data)
    # @staticmethod
    # def test_run_mass_listener():
    #     run_mass_listener()

    # POLYPEPTIDE INITIATION
    @staticmethod
    def test_run_polypeptide_initiation_default():
        run_polypeptide_initiation_default(load_sim_data)
    @staticmethod
    def test_run_polypeptide_initiation_migration():
        run_polypeptide_initiation_migration(load_sim_data)

    # POLYPEPTIDE ELONGATION
    @staticmethod
    def test_run_polypeptide_elongation_migration():
        run_polypeptide_elongation_migration(load_sim_data)

    # from migration.polypeptide_elongation import run_polypeptide_elongation_migration
    # from migration.protein_degradation import run_protein_degradation
    # from migration.rna_degradation import (run_rna_degradation_default,
    #                                        run_rna_degradation_migration)
    # from migration.tf_binding import run_tf_binding_migration
    # from migration.transcript_elongation import run_transcription_elongation
    # from migration.transcript_initiation import run_transcript_initiation
    # from migration.two_component_system import run_two_component_system_migration
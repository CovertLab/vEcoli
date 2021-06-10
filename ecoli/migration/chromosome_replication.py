from ecoli.library.sim_data import LoadSimData
from ecoli.composites.ecoli_master import SIM_DATA_PATH
from ecoli.processes.chromosome_replication import ChromosomeReplication

from ecoli.migration.migration_utils import run_ecoli_process


load_sim_data = LoadSimData(
            sim_data_path=SIM_DATA_PATH,
            seed=0)

topology = {
    # bulk molecules
    'replisome_trimers': ('bulk',),
    'replisome_monomers': ('bulk',),
    'dntps': ('bulk',),
    'ppi': ('bulk',),

    # unique molecules
    'active_replisomes': ('unique', 'active_replisome',),
    'oriCs': ('unique', 'oriC',),
    'chromosome_domains': ('unique', 'chromosome_domain',),
    'full_chromosomes': ('unique', 'full_chromosome',),

    # other
    'listeners': ('listeners',),
    'environment': ('environment',),
}


def test_chromosome_replication():
    config = load_sim_data.get_chromosome_replication_config()
    chromosome_replication = ChromosomeReplication(config)

    # run the process and get an update
    actual_update = run_ecoli_process(
        chromosome_replication,
        topology,
        total_time=2,
        initial_time=1000,
    )

    import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    test_chromosome_replication()

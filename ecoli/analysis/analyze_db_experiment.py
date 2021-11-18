import argparse
import os
import pickle

from vivarium.core.emitter import (
    data_from_database,
    DatabaseEmitter,
)

from ecoli.composites.ecoli_nonpartition import SIM_DATA_PATH
from ecoli.analysis.compartment_mass_fraction_summary import Plot as CompartmentsMassFraction
from ecoli.analysis.mass_fraction_summary import Plot as MassFraction
from ecoli.analysis.mass_fractions_voronoi import Plot as VoronoiMassFraction
from ecoli.analysis.mrna_counts import Plot as mRNAcounts

OUT_DIR = 'out/analysis/'

ANALYSIS = [
    CompartmentsMassFraction,
    MassFraction,
    VoronoiMassFraction,
    mRNAcounts,
]


def access(
        experiment_id,
        query=None,
):
    # mongo client
    config = {
        'host': '{}:{}'.format('localhost', 27017),
        'database': 'simulations'}
    emitter = DatabaseEmitter(config)
    db = emitter.db

    # access
    data, sim_config = data_from_database(
        experiment_id, db, query)

    return data, experiment_id, sim_config


def make_plots(data, experiment_id='ecoli', sim_config={}):
    out_dir = os.path.join(OUT_DIR, str(experiment_id))

    with open(SIM_DATA_PATH, 'rb') as sim_data_file:
        sim_data = pickle.load(sim_data_file)

    # run plots
    for analysis in ANALYSIS:
        analysis(data, sim_data=sim_data, out_dir=out_dir)


def main():
    # parse
    parser = argparse.ArgumentParser(
        description='access data from db')
    parser.add_argument(
        '--experiment_id', '-e',
        type=str, default=False)
    args = parser.parse_args()
    experiment_id = args.experiment_id

    # get the required data
    query = [
        ('listeners', 'mass'),
        ('listeners', 'mRNA_counts'),
        ('bulk',),
    ]
    data, experiment_id, sim_config = access(experiment_id, query)

    # run plots
    make_plots(data, experiment_id, sim_config)


# python ecoli/analysis/analyze_db_experiment.py -e [experiment_id]
if __name__ == '__main__':
    main()

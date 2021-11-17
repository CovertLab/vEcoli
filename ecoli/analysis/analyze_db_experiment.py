import argparse
import os
from vivarium.core.emitter import (
    data_from_database,
    DatabaseEmitter,
)
from ecoli.analysis.compartment_mass_fraction_summary import Plot as CompartmentsMassFraction
from ecoli.analysis.mass_fraction_summary import Plot as MassFraction
from ecoli.analysis.mass_fractions_voronoi import Plot as VoronoiMassFraction


OUT_DIR = 'out/analysis/'


def access():
    # parse
    parser = argparse.ArgumentParser(description='access data from db')
    parser.add_argument('--experiment_id', '-e',
                        type=str,
                        default=False)
    args = parser.parse_args()
    experiment_id = args.experiment_id

    # mongo client
    config = {
        'host': '{}:{}'.format('localhost', 27017),
        'database': 'simulations'}
    emitter = DatabaseEmitter(config)
    db = emitter.db

    # access
    query = [('listeners', 'mass'), ('bulk',)]
    data, sim_config = data_from_database(
        experiment_id, db, query)

    return data, experiment_id, sim_config


def make_plots(data, experiment_id='ecoli', sim_config={}):
    out_dir = os.path.join(OUT_DIR, str(experiment_id))

    # pull out mass data to improve TableReader runtime
    # TODO -- make this more general
    mass_data = {}
    for t, d in data.items():
        mass_data[t] = {
            'bulk': d['bulk'],
            'listeners': {
                'mass': d['listeners']['mass']}}

    # run plots
    CompartmentsMassFraction(mass_data, out_dir=out_dir)
    MassFraction(mass_data, out_dir=out_dir)
    VoronoiMassFraction(mass_data, out_dir=out_dir)


def main():
    data, experiment_id, sim_config = access()
    make_plots(data, experiment_id, sim_config)


# python ecoli/analysis/analyze_db_experiment.py -e [experiment_id]
if __name__ == '__main__':
    main()

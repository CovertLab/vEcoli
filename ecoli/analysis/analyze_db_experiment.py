import argparse
import os
import pickle

from ecoli.composites.ecoli_master import SIM_DATA_PATH
from ecoli.analysis.db import access
from ecoli.analysis.compartment_mass_fraction_summary import Plot as CompartmentsMassFraction
from ecoli.analysis.mass_fraction_summary import Plot as MassFraction
from ecoli.analysis.mass_fractions_voronoi import Plot as VoronoiMassFraction
from ecoli.analysis.mrna_counts import Plot as mRNAcounts
from ecoli.analysis.protein_counts import Plot as ProteinCounts
from ecoli.analysis.aa_counts import Plot as AACounts

OUT_DIR = 'out/analysis/'

ANALYSIS = [
    AACounts,
    CompartmentsMassFraction,
    MassFraction,
    VoronoiMassFraction,
    mRNAcounts,
    ProteinCounts,
]


def make_plots(data, experiment_id='ecoli', sim_config=None):
    if not sim_config:
        sim_config = {}
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
    parser.add_argument(
        '--agent_id', '-a', type=str, default='',
        help='ID of agent. If unspecified, assume single-cell sim.')
    args = parser.parse_args()
    experiment_id = args.experiment_id

    # get the required data
    query = [
        ('listeners', 'mass'),
        ('listeners', 'mRNA_counts'),
        ('listeners', 'monomer_counts'),
        ('bulk',),
    ]
    if args.agent_id:
        query = [('agents', args.agent_id) + path for path in query]
    data, experiment_id, sim_config = access(experiment_id, query)
    if args.agent_id:
        data = {
            time: timepoint['agents'][args.agent_id]
            for time, timepoint in data.items()
            if args.agent_id in timepoint['agents']
        }

    # run plots
    make_plots(data, experiment_id, sim_config)


# python ecoli/analysis/analyze_db_experiment.py -e [experiment_id]
if __name__ == '__main__':
    main()

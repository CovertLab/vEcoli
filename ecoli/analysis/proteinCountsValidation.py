"""
Compare protein counts to Wisniewski 2014 and Schmidt 2015 data sets
"""

from __future__ import absolute_import, division, print_function

import argparse
import os
from six.moves import cPickle

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import pearsonr

from ecoli.library.sim_data import SIM_DATA_PATH
from wholecell.utils.protein_counts import get_simulated_validation_counts


class Plot():

    def do_plot(self, monomer_counts, simDataFile, validationDataFile, outFile):
        sim_data = cPickle.load(open(simDataFile, "rb"))
        validation_data = cPickle.load(open(validationDataFile, "rb"))

        sim_monomer_ids = sim_data.process.translation.monomer_data["id"]
        wisniewski_ids = validation_data.protein.wisniewski2014Data["monomerId"]
        schmidt_ids = validation_data.protein.schmidt2015Data["monomerId"]
        wisniewski_counts = validation_data.protein.wisniewski2014Data["avgCounts"]
        schmidt_counts = validation_data.protein.schmidt2015Data["glucoseCounts"]

        sim_wisniewski_counts, val_wisniewski_counts = get_simulated_validation_counts(
            wisniewski_counts, monomer_counts, wisniewski_ids, sim_monomer_ids)
        sim_schmidt_counts, val_schmidt_counts = get_simulated_validation_counts(
            schmidt_counts, monomer_counts, schmidt_ids, sim_monomer_ids)

        # noinspection PyTypeChecker
        fig, ax = plt.subplots(2, sharey=True, figsize=(8.5, 11))

        # Wisniewski Counts
        ax[0].scatter(
            np.log10(val_wisniewski_counts + 1),
            np.log10(sim_wisniewski_counts + 1),
            c='w', edgecolor='k', alpha=.7
        )
        ax[0].set_xlabel("log10(Wisniewski 2014 Counts + 1)")
        ax[0].set_ylabel("log10(Simulation Average Counts + 1)")
        ax[0].set_title(
            "Pearson r: %0.2f" %
            pearsonr(
                np.log10(sim_wisniewski_counts + 1),
                np.log10(val_wisniewski_counts + 1)
            )[0]
        )

        # Schmidt Counts
        ax[1].scatter(
            np.log10(val_schmidt_counts + 1),
            np.log10(sim_schmidt_counts + 1),
            c='w', edgecolor='k', alpha=.7
        )
        ax[1].set_xlabel("log10(Schmidt 2015 Counts + 1)")
        ax[1].set_ylabel("log10(Simulation Average Counts + 1)")
        ax[1].set_title(
            "Pearson r: %0.2f" %
            pearsonr(
                np.log10(sim_schmidt_counts + 1), np.log10(val_schmidt_counts + 1)
            )[0]
        )

        # NOTE: This Pearson correlation goes up (at the time of
        # writing) about 0.05 if you only include proteins that you have
        # translational efficiencies for
        plt.xlim(xmin=0)
        plt.ylim(ymin=0)
        plt.savefig(outFile, bbox_inches='tight')
        plt.close("all")


def save_agent_avgs(data):
    agent_monomer_counts = {}
    for time, time_data in data.items():
        for agent_id, agent_data in time_data['agents'].items():
            if len(agent_data) == 0:
                continue
            agent_monomer_counts.setdefault(agent_id, [])
            agent_monomer_counts[agent_id].append(agent_data[
                'listeners']['monomer_counts'])

    agent_avgs = {}
    for agent_id, monomer_counts in agent_monomer_counts.items():
        agent_avgs[agent_id] = np.mean(monomer_counts, axis=0)
    
    with open(f'{args.data.split(".pkl")[0]}_agent_avgs.pkl', 'wb') as f:
        cPickle.dump(agent_avgs, f)
    
    return agent_avgs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--raw_data',
        '-r',
        help='Path to saved pickle from ecoli.analysis.db.get_proteome_data'
    )
    parser.add_argument(
        '--avg_data',
        '-d',
        help='Path to saved pickle of average monomer counts per agent '
            '(ecoli.analysis.db.get_proteome_data output that was processed '
            'by the save_agent_avgs from running this module with raw_data).'
    )
    parser.add_argument(
        '--out_file',
        '-o',
        help='Path to output file.',
        default='out/analysis/proteinCountsValidation.svg'
    )
    parser.add_argument(
        '--sim_data',
        '-s',
        help='Path to sim_data file.',
        default=SIM_DATA_PATH
    )
    parser.add_argument(
        '--validation_data',
        '-v',
        help='Path to validation_data file.',
        default='reconstruction/sim_data/kb/validationData.cPickle'
    )
    args = parser.parse_args()

    if args.raw_data:
        with open(args.data, 'rb') as f:
            data = cPickle.load(f)
        
        avg_data = save_agent_avgs(data)
    
    elif args.avg_data:
        with open(args.avg_data, 'rb') as f:
            avg_data = cPickle.load(f)
    
    agent_monomer_counts = np.array(list(avg_data.values()))

    plot = Plot()
    plot.do_plot(agent_monomer_counts, args.sim_data,
        args.validation_data, args.out_file)
    
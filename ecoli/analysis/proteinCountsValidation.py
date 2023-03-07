"""
Compare protein counts to Wiśniewski 2014 and Schmidt 2015 data sets
"""

from __future__ import absolute_import, division, print_function

import argparse
import os
import json
from six.moves import cPickle

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
import seaborn as sns

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
        fig, ax = plt.subplots(2, figsize=(4, 6))

        # Wisniewski Counts
        val_wisniewski_log = np.log10(val_wisniewski_counts + 1)
        sim_wisniewski_log = np.log10(sim_wisniewski_counts + 1)
        ax[0].scatter(
            val_wisniewski_log,
            sim_wisniewski_log,
            c='k', edgecolor='k', alpha=.05
        )
        ax[0].set_xlabel("Measurement\n$\mathrm{log}_{10}$(protein counts + 1)")
        ax[0].set_ylabel("Simulation\n$\mathrm{log}_{10}$(protein counts + 1)")
        ax[0].set_title('Wiśniewski et al. 2014', pad=22)
        over_30 = val_wisniewski_log>=np.log10(30)
        r_over_30 = pearsonr(
            sim_wisniewski_log[over_30], val_wisniewski_log[over_30]
        )[0]
        r_under_30 = pearsonr(
            sim_wisniewski_log[~over_30], val_wisniewski_log[~over_30]
        )[0]
        ax[0].text(
            x=0.5,
            y=1.1,
            s="count $\geq$ 30: $R^2$ = %0.3f, n = %i" % (
                r_over_30**2, over_30.sum()),
            transform=ax[0].transAxes,
            ha='center',
            va='center'
        )
        ax[0].text(
            x=0.5,
            y=1,
            s="count < 30: $R^2$ = %0.3f, n = %i" % (
                r_under_30**2, len(over_30) - over_30.sum()),
            transform=ax[0].transAxes,
            ha='center',
            va='center'
        )
        reference_line = np.linspace(0, 5)
        ax[0].plot(reference_line, reference_line, c='k')
        sns.despine(ax=ax[0], trim=True, offset=3)

        # Schmidt Counts
        val_schmidt_log = np.log10(val_schmidt_counts + 1)
        sim_schmidt_log = np.log10(sim_schmidt_counts + 1)
        ax[1].scatter(
            val_schmidt_log,
            sim_schmidt_log,
            c='k', edgecolor='k', alpha=.05
        )
        ax[1].set_xlabel("Measurement\n$\mathrm{log}_{10}$(protein counts + 1)")
        ax[1].set_ylabel("Simulation\n$\mathrm{log}_{10}$(protein counts + 1)")
        ax[1].set_title('Schmidt et al. 2016', pad=10)
        over_30 = val_schmidt_log>=np.log10(30)
        r_over_30 = pearsonr(
            sim_schmidt_log[over_30], val_schmidt_log[over_30]
        )[0]
        r_under_30 = pearsonr(
            sim_schmidt_log[~over_30], val_schmidt_log[~over_30]
        )[0]
        ax[1].text(
            x=0.5,
            y=1,
            s="count $\geq$ 30: $R^2$ = %0.3f, n = %i" % (
                r_over_30**2, over_30.sum()),
            transform=ax[1].transAxes,
            ha='center',
            va='center'
        )
        ax[1].text(
            x=0.5,
            y=0.9,
            s="count < 30: $R^2$ = %0.3f, n = %i" % (
                r_under_30**2, len(over_30) - over_30.sum()),
            transform=ax[1].transAxes,
            ha='center',
            va='center'
        )
        reference_line = np.linspace(0, 6)
        ax[1].plot(reference_line, reference_line, c='k')
        sns.despine(ax=ax[1], trim=True, offset=3)

        # NOTE: This Pearson correlation goes up (at the time of
        # writing) about 0.05 if you only include proteins that you have
        # translational efficiencies for
        plt.xlim(xmin=-0.1)
        plt.ylim(ymin=-0.1)
        plt.tight_layout()
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
    agent_avgs = pd.DataFrame(agent_avgs)
    agent_avgs.to_csv(f'{args.data.split(".pkl")[0]}_agent_avgs.csv')
    return agent_avgs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--raw_data',
        '-r',
        help='Path to saved JSON from ecoli.analysis.db.get_proteome_data'
    )
    parser.add_argument(
        '--avg_data',
        '-d',
        help='Path to saved CSV of average monomer counts per agent '
            '(ecoli.analysis.db.get_proteome_data output that was processed '
            'by the save_agent_avgs from running this module with raw_data).'
    )
    parser.add_argument(
        '--out_file',
        '-o',
        help='Path to output file.',
        default='out/analysis/paper_figures/fig_s2a_proteomeValidation.svg'
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
        with open(args.raw_data, 'r') as f:
            data = json.load(f)
        avg_data = save_agent_avgs(data)
    else:
        avg_data = pd.read_csv(args.avg_data, index_col=0)
    
    agent_monomer_counts = avg_data.to_numpy()

    os.makedirs(args.out_file)
    plot = Plot()
    plot.do_plot(agent_monomer_counts, args.sim_data,
        args.validation_data, args.out_file)
    
"""
Plot Protein counts
"""

import pickle
import os

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import pearsonr

from wholecell.utils.fitting import normalize
from wholecell.utils import units

from ecoli.composites.ecoli_master import run_ecoli
from ecoli.composites.ecoli_nonpartition import SIM_DATA_PATH
from ecoli.analysis.tablereader import TableReader


class Plot:

    def __init__(self, data, sim_data=None, out_dir='out/analysis'):
        self.data = data
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

        self.sim_data = sim_data
        if not sim_data:
            with open(SIM_DATA_PATH, 'rb') as sim_data_file:
                self.sim_data = pickle.load(sim_data_file)

        self.do_plot(self.data)

    def do_plot(self, data):
        monomerCounts = TableReader("MonomerCounts", data)
        avgCounts = monomerCounts.readColumn("monomerCounts").mean(axis=1)

        relativeCounts = avgCounts / avgCounts.sum()

        expectedCountsArbitrary = normalize(
            self.sim_data.process.transcription.rna_expression[self.sim_data.condition][
                self.sim_data.relation.RNA_to_monomer_mapping] *
            self.sim_data.process.translation.translation_efficiencies_by_monomer /
            (np.log(2) / self.sim_data.doubling_time.asNumber(units.s) + self.sim_data.process.translation.monomer_data[
                'deg_rate'].asNumber(1 / units.s))
        )

        expectedCountsRelative = expectedCountsArbitrary / expectedCountsArbitrary.sum()

        fig = plt.figure(figsize=(8.5, 11))

        maxLine = 1.1 * max(np.log10(expectedCountsRelative.max() + 1), np.log10(relativeCounts.max() + 1))
        plt.plot([0, maxLine], [0, maxLine], '--r')
        plt.plot(np.log10(expectedCountsRelative + 1), np.log10(relativeCounts + 1), 'o', markeredgecolor='k',
                 markerfacecolor='none')

        plt.xlabel("log10(Expected protein distribution (from fitting))")
        plt.ylabel("log10(Actual protein distribution (average over life cycle))")
        plt.title(
            "PCC (of log values): %0.2f" % pearsonr(np.log10(expectedCountsRelative + 1), np.log10(relativeCounts + 1))[
                0])

        # TODO -- this can go into a general method in an analysis base class
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, 'protein_counts.png'))
        plt.close("all")
        return fig


def run_plot():
    data = run_ecoli(total_time=10)
    Plot(data)


# python ecoli/analysis/protein_counts.py
if __name__ == "__main__":
    run_plot()

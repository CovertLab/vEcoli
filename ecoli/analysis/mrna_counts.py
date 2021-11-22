"""
Plot mRNA counts
"""

import pickle
import os

from matplotlib import pyplot as plt

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

        # Get the names of RNAs from the KB
        is_mrna = self.sim_data.process.transcription.rna_data['is_mRNA']
        # mrna_ids = self.sim_data.process.transcription.rna_data['id'][is_mrna]

        # Get reader for mRNA counts
        mrna_counts_reader = TableReader('mRNACounts', data)

        # # Check that the order of mRNAs in table matches that of KB
        # assert np.all(mrna_ids == mrna_counts_reader.readAttribute('mRNA_ids'))

        # Read final mRNA counts from reader
        counts = mrna_counts_reader.readColumn('mRNA_counts')[:, -1]

        fig = plt.figure(figsize=(8.5, 11))

        expected_counts_arbitrary = self.sim_data.process.transcription.rna_expression[
            self.sim_data.condition][is_mrna]
        expected_counts = expected_counts_arbitrary / expected_counts_arbitrary.sum() * counts.sum()

        max_line = 1.1 * max(expected_counts.max(), counts.max())
        plt.plot([0, max_line], [0, max_line], '--r')
        plt.plot(expected_counts, counts, 'o', markeredgecolor='k', markerfacecolor='none')

        plt.xlabel("Expected RNA count (scaled to total)")
        plt.ylabel("Actual RNA count (at final time step)")

        # TODO -- this can go into a general method in an analysis base class
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, 'mrna_counts.png'))
        plt.close("all")
        return fig


def run_plot():
    data = run_ecoli(total_time=10)
    Plot(data)


# python ecoli/analysis/mrna_counts.py
if __name__ == "__main__":
    run_plot()

"""
Plot amino acid counts
"""

import pickle
import os

from matplotlib import pyplot as plt

from ecoli.composites.ecoli_master import run_ecoli
from ecoli.composites.ecoli_nonpartition import SIM_DATA_PATH
from ecoli.analysis.tablereader import TableReader, read_bulk_molecule_counts


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

        aa_ids = self.sim_data.molecule_groups.amino_acids
        (aaCounts,) = read_bulk_molecule_counts(data, (aa_ids,))

        main_reader = TableReader("Main", data)
        time_vals = main_reader.readColumn('time')
        initialTime = time_vals[0]
        time = main_reader.readColumn("time") - initialTime

        fig = plt.figure(figsize=(8.5, 11))

        for idx in range(21):
            plt.subplot(6, 4, idx + 1)

            plt.plot(time / 60., aaCounts[:, idx], linewidth=2)
            plt.xlabel("Time (min)")
            plt.ylabel("Counts")
            plt.title(aa_ids[idx], fontsize=8)
            plt.tick_params(labelsize=8)

        # TODO -- this can go into a general method in an analysis base class
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, 'aa_counts.png'))
        plt.close("all")
        return fig


def run_plot():
    data = run_ecoli(total_time=20)
    Plot(data)


# python ecoli/analysis/aa_counts.py
if __name__ == "__main__":
    run_plot()

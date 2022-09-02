import os

import numpy as np
import matplotlib.pyplot as plt

from vivarium.core.emitter import DatabaseEmitter, data_from_database
from ecoli.processes.antibiotics.cell_wall import CellWall


class CellWallPlots:
    def __init__(self):
        pass

    def do_plots(self, experiment_id):
        # output directory
        subdir = experiment_id.replace("/", "_").replace(" ", "__").replace(":", "_")
        self.outdir = f"out/analysis/{subdir}/cell_wall/"

        os.makedirs(self.outdir, exist_ok=True)

    def wall_movie(self, experiment_id):
        pass

    def crack_view(self, experiment_id):
        pass

    def strand_length(self, experiment_id):
        pass

    def gap_distribution(self, experiment_id):
        pass


def main():
    # Ids of experiments to plot
    experiment_ids = ["cell_wall_17/08/2022 17:15:07"]

    for exp_id in experiment_ids:
        CellWallPlots().do_plots(exp_id)


if __name__ == "__main__":
    main()

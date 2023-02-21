import os
import pickle

import matplotlib
import numpy as np
import pandas as pd

from ecoli.analysis.antibiotics_colony.plot_utils import HIGHLIGHT_BLUE
from ecoli.analysis.antibiotics_colony.timeseries import plot_timeseries

matplotlib.use("Agg")
import matplotlib.pyplot as plt

GLC_DATA = "data/glc_0/2022-12-08_00-33-56_581605+0000.pkl"
TET_DATA = "data/tet_0/2023-01-05_01-00-44_215314+0000.pkl"
OUTDIR = "out/figure_3"
HIGHLIGHT_LINEAGE = "0011111"


def make_tet_dry_mass_plot(glc_data, tet_data):
    tet_time = tet_data.Time.min()
    max_time = tet_data.Time.max()
    data = pd.concat([glc_data[glc_data.Time <= tet_time], tet_data])

    fig, ax = plt.subplots()
    ax.tick_params("both", which='major', labelsize=8)

    plot_timeseries(
        data,
        axes=[ax],
        columns_to_plot={"Dry mass": HIGHLIGHT_BLUE},
        highlight_lineage=HIGHLIGHT_LINEAGE,
        conc=False,
    )

    ticks = np.array([0, tet_time, max_time]) / 3600
    tick_labels = [f"{tick - tet_time / 3600:.1f}" for tick in ticks]
    tick_labels = [f"{float(t):.0f}" if t[-2:] == ".0" else t for t in tick_labels]
    ax.set_xticks(ticks, tick_labels)

    yticks = ax.get_yticks()
    ax.set_yticks(yticks, [f"{y:.0f} fg" for y in yticks])

    ax.set_xlabel(ax.get_xlabel(), fontsize=9)
    ax.set_ylabel(ax.get_ylabel(), fontsize=9)

    return fig, ax


def main():
    with open(GLC_DATA, "rb") as f:
        glc_data = pickle.load(f)

    with open(TET_DATA, "rb") as f:
        tet_data = pickle.load(f)

    os.makedirs(OUTDIR, exist_ok=True)

    fig, ax = make_tet_dry_mass_plot(glc_data, tet_data)
    ax.set_xlabel("")
    fig.set_size_inches(4, 1)
    fig.tight_layout()
    fig.savefig("out/figure_3/tet_dry_mass.svg")


if __name__ == "__main__":
    main()

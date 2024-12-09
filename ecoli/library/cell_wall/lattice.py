import os
from itertools import groupby

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.transform import resize
from vivarium.library.units import remove_units

from ecoli.library.cell_wall.column_sampler import geom_sampler, sample_lattice


def calculate_lattice_size(
    cell_length,
    inter_strand_distance,
    disaccharide_height,
    disaccharide_width,
    circumference,
    extension_factor,
):
    # Calculate new lattice size
    columns = round(
        remove_units(
            (
                cell_length
                / (extension_factor * (inter_strand_distance + disaccharide_width))
            ).to("dimensionless")
        )
    )
    rows = int(circumference / disaccharide_height)

    return rows, columns


def plot_lattice(lattice, on_cylinder=False, aspect=1):
    if not on_cylinder:
        fig, ax = plt.subplots()
        mappable = ax.imshow(lattice, interpolation="nearest", aspect=aspect)
        fig.colorbar(mappable, ax=ax)
    else:
        print("Downscaling lattice...")
        lattice = resize(
            lattice,
            (lattice.shape[0] // 10, lattice.shape[1] // 10),
            preserve_range=True,
            anti_aliasing=True,
        )
        lattice = lattice.T
        print("Done.\nDrawing on cylinder...")

        h, w = lattice.shape
        theta, z = np.linspace(0, 2 * np.pi, w), np.linspace(0, 1, h)
        THETA, Z = np.meshgrid(theta, z)
        X = np.cos(THETA)
        Y = np.sin(THETA)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        mappable = plt.cm.ScalarMappable(cmap=plt.cm.inferno)
        mappable.set_clim(0, 1)
        mappable.set_array(lattice)
        ax.plot_surface(
            X,
            Y,
            Z,
            rstride=1,
            cstride=1,
            facecolors=mappable.cmap(lattice),
            linewidth=0,
            antialiased=False,
            alpha=0.75,
        )
        fig.colorbar(mappable)

    return fig, ax


def shift_column_to_boundary(column):
    # circular shift the column around
    # until the wrap point represents a gap-strand boundary
    total_shift = 0
    while column[0] == column[-1] and total_shift < column.size:
        column = np.roll(column, 1)
        total_shift += 1

    return column, total_shift


def get_length_distributions(lattice):
    strand_lengths = []
    gap_lengths = []
    lengths = (gap_lengths, strand_lengths)
    for c in range(lattice.shape[1]):
        column = lattice[:, c]
        # circular shift the column around
        # until the wrap point represents a gap-strand boundary
        column, _ = shift_column_to_boundary(column)

        for val, seq in groupby(column):
            seq = list(seq)
            lengths[val].append(len(seq))

    return gap_lengths, strand_lengths


def plot_strand_length_distribution(lengths):
    # Plot experimental data first
    df = pd.read_csv(
        "reconstruction/ecoli/flat/cell_wall/murein_strand_length_distribution.csv"
    )

    fig, ax = plt.subplots()
    X = np.arange(1, 32, 1)
    xlabels = list(map(str, X))
    xlabels[-1] = ">30"

    for strain in set(df["Strain"]):
        strain_data = df[df["Strain"] == strain]
        strain_data.index = strain_data["Length"]
        heights = strain_data.loc[xlabels]["Percent"]
        heights /= heights.sum()
        ax.bar(X, heights, alpha=0.5, label=strain)
    ax.set_xticks(X)
    ax.set_xticklabels(xlabels, rotation=45)

    # Skip plotting simulated data if none was given
    if len(lengths) > 0:
        # Plot simulated data in the same way as experimental data
        # (aggregate strands >30 in length)
        lengths = np.bincount(lengths)
        try:
            lengths[31] = lengths[31:].sum()
            lengths = lengths[:32]
        except IndexError:  # no strands > 30 in length
            pass

        # Normalize as proportions
        lengths = lengths / lengths.sum()

        # Eliminate "0-length" strands
        lengths = lengths[1:]

        ax.scatter(X, lengths, label="Simulation")
        ax.set_xlabel("Strand length")
        ax.set_ylabel("Proportion")

    ax.legend()
    return fig, ax


def plot_distributions_timeseries(distributions, every=1, yscale="linear"):
    fig, ax = plt.subplots()

    ax.set_yscale(yscale)
    ax.boxplot(distributions[::every], vert=True)

    return fig, ax


def plot_length_vs_location(lattice):
    lengths = []
    start_positions = []

    for c in range(lattice.shape[1]):
        column = lattice[:, c]

        # circular shift the column around
        # until the wrap point represents a gap-strand boundary
        column, total_shift = shift_column_to_boundary(column)

        # Get start position and length of each strand
        i = 0
        for val, group in groupby(column):
            run_length = len(list(group))
            if val == 1:
                lengths.append(run_length)
                start_positions.append((i - total_shift) % len(column))
            i += run_length

    # Do plotting
    fig, ax = plt.subplots()
    ax.scatter(lengths, start_positions)
    ax.set_xlabel("Length of strand")
    ax.set_ylabel("Start position (row) of strand")

    return fig, ax


def porosity(lattice):
    # proportion of zeros
    return (lattice.size - lattice.sum()) / lattice.size


def test_strand_length_plots():
    rng = np.random.default_rng(0)
    lattice = sample_lattice(450000 * 4, 3050, 700, geom_sampler(rng, 0.058), rng)

    os.makedirs("out/processes/cell_wall/", exist_ok=True)

    strand_lengths, _ = get_length_distributions(lattice)
    fig, _ = plot_strand_length_distribution(strand_lengths)
    fig.tight_layout()
    fig.savefig("out/processes/cell_wall/test_strand_length_plot.png")

    fig, _ = plot_length_vs_location(lattice)
    fig.tight_layout()
    fig.savefig("out/processes/cell_wall/test_length_vs_location.png")


def main():
    test_strand_length_plots()


if __name__ == "__main__":
    main()

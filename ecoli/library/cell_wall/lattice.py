from itertools import groupby

import mpl_toolkits.mplot3d.axes3d as axes3d
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from skimage.transform import resize

from ecoli.library.cell_wall.column_sampler import geom_sampler, sample_column
from vivarium.library.units import remove_units


def de_novo_lattice(murein_monomers, rows, cols, strand_sampler, rng):
    lattice = np.array(
        [
            sample_column(
                rows,
                murein_monomers / cols,
                strand_sampler,
                rng,
            )
            for _ in range(cols)
        ]
    ).T
    return lattice


def calculate_lattice_size(
    cell_length, crossbridge_length, disaccharide_length, circumference, stretch_factor
):
    # Calculate new lattice size
    columns = round(
        remove_units(
            (
                cell_length
                / (stretch_factor * (crossbridge_length + disaccharide_length))
            ).to("dimensionless")
        )
    )
    rows = int(circumference / disaccharide_length)

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


def get_strand_length_distribution(lattice):
    lengths = []
    for c in range(lattice.shape[1]):
        column = lattice[:, c]
        # circular shift the column around until it starts with a 0
        shift = 1
        while column[0] != 0:
            column = np.roll(column, shift)
            shift += 1

        for _, strand in groupby(column):
            strand = list(strand)
            lengths.append(len(strand))

    return lengths


def plot_strand_length_distribution(lengths):
    # Plot experimental data first
    df = pd.read_csv("data/cell_wall/murein_strand_length_distribution.csv")

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
    if len(lengths) == 0:
        ax.legend()
        return fig, ax

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


def porosity(lattice):
    # proportion of zeros
    return (lattice.size - lattice.sum()) / lattice.size


def test_lattice():
    test_strand_length_plot()


def test_strand_length_plot():
    rng = np.random.default_rng(0)
    lattice = de_novo_lattice(1607480, 3050, 700, geom_sampler(rng, 0.058), rng)

    fig, _ = plot_strand_length_distribution(lattice)
    fig.tight_layout()
    fig.savefig("out/processes/cell_wall/test_strand_length_plot.png")


def main():
    test_strand_length_plot()


if __name__ == "__main__":
    main()

import argparse
import os
import pickle
from itertools import combinations

import matplotlib
import numpy as np
import pandas as pd
from esda.moran import Moran
from libpysal.weights import KNN, DistanceBand
from splot.esda import plot_moran

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ecoli.analysis.antibiotics_colony import COUNTS_PER_FL_TO_NANOMOLAR
from ecoli.analysis.antibiotics_colony.amp_plots import PERIPLASMIC_VOLUME_FRACTION
from ecoli.analysis.antibiotics_colony.plot_utils import (
    BG_GRAY,
    HIGHLIGHT_BLUE,
    HIGHLIGHT_RED,
    prettify_axis,
)


def make_spatial_correlation_plot(glc_data, column, to_conc=False):
    # Filter to just last snapshot
    max_t = glc_data.Time.max()
    data = glc_data[glc_data.Time == max_t][["Boundary", "Volume", column]]

    if to_conc:
        data[column] = data[column] * COUNTS_PER_FL_TO_NANOMOLAR / data["Volume"]

    location = data["Boundary"].apply(lambda b: np.array(b["location"]))
    data["X"] = location.apply(lambda l: l[0])
    data["Y"] = location.apply(lambda l: l[1])

    location = data[["X", "Y"]].values

    # weights = KNN(location, k=2)
    weights = DistanceBand(location, 50, alpha=-2.0, binary=False)
    moran = Moran(data[column], weights, permutations=9999)

    fig, ax = plot_moran(moran)

    return fig, ax, moran


def make_threshold_sweep_plot(glc_data, column, to_conc=False):
    # Filter to just last snapshot
    max_t = glc_data.Time.max()
    data = glc_data[glc_data.Time == max_t][["Boundary", "Volume", column]]

    if to_conc:
        data[column] = data[column] * COUNTS_PER_FL_TO_NANOMOLAR / data["Volume"]

    location = data["Boundary"].apply(lambda b: np.array(b["location"]))
    data["X"] = location.apply(lambda l: l[0])
    data["Y"] = location.apply(lambda l: l[1])

    location = data[["X", "Y"]].values

    thresholds = np.linspace(0, 50, 50)
    i_values = []
    for d in thresholds:
        weights = DistanceBand(location, d, alpha=-2.0, binary=False)
        moran = Moran(data[column], weights, permutations=9999)
        i_values.append(moran.I)

    fig, ax = plt.subplots()
    ax.plot(thresholds, i_values)
    ax.set_ylim(0, 0.06)

    return fig, ax


def make_relatedness_vs_distance_plot(glc_data):
    def relatedness(A, B):
        common_ancestor = os.path.commonprefix([A, B])
        return len(A) + len(B) - 2 * len(common_ancestor)

    max_t = glc_data.Time.max()
    endpoint_data = glc_data[glc_data.Time == max_t]
    endpoint_data.loc[:, "Location"] = endpoint_data["Boundary"].apply(
        lambda b: np.array(b["location"])
    )
    final_agents = {
        agent: location
        for agent, location in zip(endpoint_data["Agent ID"], endpoint_data["Location"])
    }

    relatednesses = []
    distances = []
    for A, B in combinations(final_agents.keys(), 2):
        relatednesses.append(relatedness(A, B))
        distances.append((final_agents[A] @ final_agents[B])**0.5)

    fig, ax = plt.subplots()
    # ax.hist2d(distances, np.array(relatednesses), bins=(50, 16), cmap="Greys")
    ax.scatter(distances, np.array(relatednesses), color=HIGHLIGHT_BLUE, alpha=0.05, linewidths=0)
    ax.set_xlabel("Distance")
    ax.set_ylabel("Relatedness")

    return fig, ax


def load_data(
    glc_data,
    verbose=False,
):
    # Load glc data
    if verbose:
        print("Loading Glucose data...")
    with open(glc_data, "rb") as f:
        glc_data = pickle.load(f)

    # Validate data:
    # - glc_data must be in Glucose condition

    glc_data_condition = glc_data["Condition"].unique()[0]
    assert "Glucose" in glc_data_condition, "glc_data was not from Glucose condition!"

    # Clean data:
    # Add additional columns for periplasmic volume,
    # concentration of AmpC in the periplasm
    glc_data["Periplasmic Volume"] = PERIPLASMIC_VOLUME_FRACTION * glc_data["Volume"]
    glc_data["AmpC conc"] = (
        glc_data["AmpC monomer"] / glc_data["Periplasmic Volume"]
    ) * COUNTS_PER_FL_TO_NANOMOLAR

    return glc_data


def cli():
    parser = argparse.ArgumentParser(
        "Generate analysis plots for ampicillin colony sims."
    )

    parser.add_argument(
        "glc_data",
        type=str,
        help="Locally saved dataframe file for glucose (before addition of ampicillin.)",
    )

    parser.add_argument(
        "--outdir",
        "-d",
        default="out/figure_2",
        help="Directory in which to output the generated figures.",
    )
    parser.add_argument("--svg", "-s", action="store_true", help="Save as svg.")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    return args


def main():
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["svg.fonttype"] = "none"

    options = cli()

    glc_data = load_data(
        options.glc_data,
        options.verbose,
    )

    # Ensure output directory exists
    os.makedirs(options.outdir, exist_ok=True)

    ext = ".svg" if options.svg else ".png"

    # Variable, whether to convert to concentration
    spatial_vars = {
        "OmpF monomer": True,
        "MarR monomer": True,
        "AmpC conc": False,
        "TolC monomer": True,
    }

    # Plot relatedness vs. distance
    if options.verbose:
            print("Plotting distance vs. relatedness:")
    fig, _ = make_relatedness_vs_distance_plot(glc_data)
    fig.set_size_inches(8, 6)
    fig.savefig(os.path.join(options.outdir, f"relatedness_vs_distance{ext}"))

    # Compute and plot spatial autocorrelations
    moran_results = {}
    for col, to_conc in spatial_vars.items():
        if options.verbose:
            print(f"Computing and plotting spatial autocorrelation for {col}:")

        fig, _, moran = make_spatial_correlation_plot(
            glc_data, column=col, to_conc=to_conc
        )
        moran_results[col] = moran

        fig.set_size_inches(6, 4)
        fig.tight_layout()
        fig.savefig(os.path.join(options.outdir, f"{col} Moran plot{ext}"))

    # Plot threshold param sweep
    fig, _ = make_threshold_sweep_plot(glc_data, column="OmpF monomer", to_conc=True)
    fig.set_size_inches(4, 4)
    fig.savefig(os.path.join(options.outdir, f"threshold_sweep{ext}"))

    with open(os.path.join(options.outdir, "Moran tests.txt"), "w") as f:
        for col, moran in moran_results.items():
            f.write(f"\n{col}\n=============\n")
            f.write(f"I = {moran.I}\n")
            f.write(
                f"Simulated p-value from {moran.permutations} permutations: p={moran.p_sim}\n"
            )
            f.write(
                f"After Bonferroni correction with m={len(spatial_vars)} null hypotheses: {len(spatial_vars) * moran.p_sim}"
            )


if __name__ == "__main__":
    main()

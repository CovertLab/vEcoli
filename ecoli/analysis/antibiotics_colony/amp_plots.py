import argparse
import os
import pickle

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

from ecoli.analysis.antibiotics_colony.timeseries import plot_timeseries

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ecoli.analysis.antibiotics_colony import COUNTS_PER_FL_TO_NANOMOLAR
from ecoli.analysis.antibiotics_colony.plot_utils import (
    prettify_axis,
    HIGHLIGHT_BLUE,
    BG_GRAY,
)


def load_data(glc_data, glc_metadata, amp_data, amp_metadata, verbose):
    # Load glc data
    with open(glc_data, "rb") as f:
        glc_data = pickle.load(f)

    # Load glc metadata
    with open(glc_metadata, "rb") as f:
        glc_metadata = pickle.load(f)

    # Load amp data
    with open(amp_data, "rb") as f:
        amp_data = pickle.load(f)

    # Load amp metadata
    with open(amp_metadata, "rb") as f:
        amp_metadata = pickle.load(f)

    # Validate data:
    # - glc_data, glc_metadata must be in Glucose condition,
    #   amp_data, amp_metadata must be in (one of) Ampicillin condition(s)
    # - Condition must match exactly between amp_data and amp_metadata (including dosage)
    # - seed must match across glc_data, glc_metadata, amp_data, amp_metadata
    assert isinstance(glc_data, pd.DataFrame)
    assert isinstance(amp_data, pd.DataFrame)
    assert isinstance(glc_metadata, dict)
    assert isinstance(amp_metadata, dict)
    assert (
        glc_data["Condition"].unique().size
        == amp_data["Condition"].unique().size
        == len(glc_metadata.keys())
        == len(amp_metadata.keys())
        == 1
    ), "One of glc_data, amp_data, glc_metadata, amp_data has data for more than one condition!"

    glc_data_condition = glc_data["Condition"].unique()[0]
    glc_metadata_condition = list(glc_metadata.keys())[0]
    amp_data_condition = amp_data["Condition"].unique()[0]
    amp_metadata_condition = list(amp_metadata.keys())[0]

    # - glc_data, glc_metadata must be in Glucose condition,
    #   amp_data, amp_metadata must be in (one of) Ampicillin condition(s)

    assert "Glucose" in glc_data_condition, "glc_data was not from Glucose condition!"
    assert (
        "Glucose" in glc_metadata_condition
    ), "glc_metadata was not from Glucose condition!"
    assert (
        "Ampicillin" in amp_data_condition
    ), "amp_data was not from Ampicillin condition!"
    assert (
        "Ampicillin" in amp_metadata_condition
    ), "amp_metadata was not from Ampicillin condition!"

    # - Condition must match exactly between amp_data and amp_metadata (including dosage)

    assert (
        amp_data_condition == amp_metadata_condition
    ), "Condition does not match between amp_data and amp_metadata!"

    # - seed must match across glc_data, glc_metadata, amp_data, amp_metadata
    assert (
        glc_data["Seed"].unique().size
        == amp_data["Seed"].unique().size
        == len(glc_metadata[glc_data_condition].keys())
        == len(amp_metadata[amp_data_condition].keys())
        == 1
    ), "One of glc_data, amp_data, glc_metadata, amp_data has data for more than one seed!"

    glc_data_seed = glc_data["Seed"].unique()[0]
    amp_data_seed = amp_data["Seed"].unique()[0]
    glc_metadata_seed = list(glc_metadata[glc_metadata_condition].keys())[0]
    amp_metadata_seed = list(amp_metadata[amp_metadata_condition].keys())[0]

    assert (
        glc_data_seed == amp_data_seed == glc_metadata_seed == amp_metadata_seed
    ), "Seeds do not match across glc_data, glc_metadata, amp_data, amp_metadata!"

    # Merge dataframes from before/after addition of ampicillin
    glc_data = glc_data[glc_data.Time < amp_data.Time.min()]
    data = pd.concat([glc_data, amp_data])

    # Figure out which cells died and when
    def died(lineage, agent_ids):
        d1, d2 = lineage + "0", lineage + "1"
        return (d1 not in agent_ids) and (d2 not in agent_ids)

    # Get all cell ids, set of cells that died, and time of death for those cells
    unique_ids = data["Agent ID"].unique()
    dead_ids = list(filter(lambda id: died(id, unique_ids), unique_ids))
    time_of_death = {id: data["Time"][data["Agent ID"] == id].max() for id in dead_ids}

    # Remove cells still present at sim end from dead
    time_of_death = {id: time for id, time in time_of_death.items() if time != 26002}
    dead_ids = list(time_of_death.keys())

    # Create columns for whether a cell died, and the time of death where applicable
    data["Died"] = data["Agent ID"].isin(dead_ids)
    data["Time of Death"] = data["Agent ID"].map(lambda id: time_of_death.get(id, None))

    return data, {**glc_metadata, **amp_metadata}


def make_figures(data, metadata, verbose):
    fig, axs = plt.subplots(5, 1)

    timeseries_axes = [axs[0]]
    death_factors = [
        "AmpC monomer",
        "Periplasmic ampicillin",
        "PBP1a complex",
        "PBP1b gamma complex",
    ]
    death_factor_axes = axs[1 : (1 + len(death_factors))]

    # snapshots

    # mass vs time with deaths
    timeseries_death_plot(
        data,
        "Dry mass",
        highlight_lineage=data["Agent ID"].iloc[-1],
        ax=timeseries_axes[0],
    )

    # beta-lactamase, AcrAB-TolC... vs. death
    conc_format = ["{:.1f}", "{:.2e}", "{:.1f}", "{:.1f}"]

    for ax, factor, fmt in zip(death_factor_axes, death_factors, conc_format):
        _, _, bins = hist_by_death_plot(data, factor, counts_to_conc=True, ax=ax)

        prettify_axis(
            ax,
            xlim=[bins[0], bins[-1]],
            ylim=np.round(ax.get_ylim()),
            tick_format_x=fmt,
            tick_format_y="{:.0f}",
        )

    # PCA

    # Validation?

    return fig, axs


def timeseries_death_plot(data, var, highlight_lineage=None, ax=None):
    data = data.copy()

    if ax is None:
        _, ax = plt.subplots()

    # data["Highlight"] = (
    #     data["Agent ID"].map(lambda id: highlight_lineage[: len(id)] == id)
    #     if highlight_lineage is not None
    #     else False
    # )

    # sns.lineplot(
    #     x="Time",
    #     y=var,
    #     hue="Highlight",
    #     # palette={False: BG_GRAY, True: HIGHLIGHT_BLUE},
    #     legend=False,
    #     data=data,
    #     ax=ax,
    # )

    # TODO: Check if scaling time messes with background traces? Seems like it...
    plot_timeseries(
        data,
        axes=[ax],
        columns_to_plot={var: HIGHLIGHT_BLUE},
        highlight_lineage=highlight_lineage,
        conc=False,
        mark_death=True,
    )

    return ax


def hist_by_death_plot(
    data,
    var,
    var_summarize=lambda var: var.mean(),
    counts_to_conc=False,
    ax=None,
    xlabel=None,
):
    data = data.copy()

    if ax is None:
        _, ax = plt.subplots()

    # Convert from counts to concentration if requested
    if counts_to_conc:
        data[var] = (
            data[var] / data["Boundary"].map(lambda d: d["volume"])
        ) * COUNTS_PER_FL_TO_NANOMOLAR

    # Prepare data for plotting
    hist_data = pd.concat(
        [
            var_summarize(data.groupby("Agent ID")[var]),
            data.groupby("Agent ID")[["Died"]].max(),
        ],
        axis=1,
    )

    # Replace Died=True with "Lysed", Died=False with "Survived"
    hist_data["Died"] = hist_data["Died"].replace({False: "Survived", True: "Lysed"})

    # Plot histogram
    sns.histplot(
        x=var,
        hue="Died",
        data=hist_data,
        palette={"Lysed": BG_GRAY, "Survived": HIGHLIGHT_BLUE},
        ax=ax,
        edgecolor=None,
    )

    if xlabel is None:
        xlabel = f"{var}{' (nM)' if counts_to_conc else ''}"

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Cells")

    hist, bins = np.histogram(hist_data[var])

    return ax, hist, bins


def save_figures(fig, outdir, as_svg):
    os.makedirs(outdir, exist_ok=True)

    fig.set_size_inches(6, 8)
    fig.tight_layout()
    fig.savefig(
        os.path.join(outdir, f'test.{"svg" if as_svg else "png"}'), bbox_inches="tight"
    )


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
        "glc_metadata",
        type=str,
        help="Locally saved metadata file for glucose (before addition of ampicillin.)",
    )

    parser.add_argument(
        "amp_data",
        type=str,
        help="Locally saved dataframe file for data following addition of ampicillin.",
    )

    parser.add_argument(
        "amp_metadata",
        type=str,
        help="Locally saved metadata file for data following addition of ampicillin.",
    )

    parser.add_argument(
        "--outdir",
        "-d",
        default="out/figure_4",
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

    data, metadata = load_data(
        options.glc_data,
        options.glc_metadata,
        options.amp_data,
        options.amp_metadata,
        options.verbose,
    )

    fig, _ = make_figures(
        data,
        metadata,
        options.verbose,
    )

    save_figures(fig, as_svg=options.svg, outdir=options.outdir)
    plt.close(fig)


if __name__ == "__main__":
    main()

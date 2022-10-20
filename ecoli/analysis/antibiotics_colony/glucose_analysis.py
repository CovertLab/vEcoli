"""
Glucose Figure

[A] [A] [A] [A] [A]
[B   B   B   B] [D]
[C   C   C   C] [E]
[F] [F] [F] [F] [F]
[F] [F] [F] [F] [F]
"""
import argparse
import concurrent.futures
import os
from tabnanny import verbose

import matplotlib
import seaborn
import numpy as np
from scipy.stats import gaussian_kde
from bson import MaxKey, MinKey
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ecoli.analysis.db import access_counts, deserialize_and_remove_units
from ecoli.plots.snapshots import format_snapshot_data, make_tags_figure, plot_tags
from vivarium.library.units import units


MOLECULES = [# ("bulk", "MICF-RNA"),
             ("monomer", "PD00365"),
             ("monomer", "PD00406"),
             ("monomer", "YHIV-MONOMER"),
             ("monomer", "PD00364"),
             ("monomer", "OMPR-MONOMER"),
             ("monomer", "EG10671-MONOMER"),
             ("monomer", "ENVZ-MONOMER"),
             ("monomer", "TEHA-MONOMER"),
             ("monomer", "EG11703-MONOMER"),
             ("monomer", "ACRB-MONOMER"),
             ("monomer", "EG12117-MONOMER"),
             ("monomer", "EG10670-MONOMER"),
             ("monomer", "EG10669-MONOMER"),
             ("monomer", "EG12116-MONOMER"),
             ("monomer", "CMR-MONOMER"),
             ("monomer", "ACRD-MONOMER"),
             ("monomer", "EMRB-MONOMER"),
             ("monomer", "PD04418"),
             ("monomer", "EMRD-MONOMER"),
             ("monomer", "ACRF-MONOMER"),
             ("monomer", "EG11009-MONOMER"),
             ("monomer", "EG10266-MONOMER"),
             ("monomer", "EG11599-MONOMER"),
             ]


def make_snapshot_and_kde_plot(data, bounds, molecule, timepoint=-1):
    time_vec = list(data.keys())

    # Get snapshot figure from plot_tags
    fig = plot_tags(
        data,
        bounds,
        snapshot_times=[time_vec[timepoint]],
        tagged_molecules=[molecule],
        show_timeline=False,
    )

    # Reposition axes, preparing to add kde plot below
    grid = plt.GridSpec(2, 2, wspace=0.2, hspace=0.2)

    # snapshot_ax, conc_ax = fig.get_axes()
    # snapshot_ax.set_position(grid[0, 0].get_position(fig))
    # conc_ax.set_position(grid[0, 1].get_position(fig))
    # snapshot_ax.set_subplotspec(grid[0, 0])
    # conc_ax.set_subplotspec(grid[0, 1])

    # # Add KDE plot
    # kde_ax = fig.add_subplot(grid[1, 0])
    # kde_ax.set(aspect=1)
    # kde_ax.plot(np.arange(10))

    # import ipdb; ipdb.set_trace()
    # d = data[list(data.keys()[-1])]
    # for agent_data in d["agents"].values():
    #     get_value_at_path


    # pdf = gaussian_kde(data)

    return fig, fig.get_axes()


def make_figure_A(fig, axs, data, bounds):
    bounds = bounds[0]

    time_vec = list(data.keys())

    n_snapshots = len(axs)

    # get time data
    time_indices = np.round(np.linspace(0, len(time_vec) - 1, n_snapshots)).astype(int)
    snapshot_times = [time_vec[i] for i in time_indices]

    for i, d in enumerate(data):
        fig = plot_tags(
            d,
            bounds,
            snapshot_times=time_vec[-1],
            tagged_molecules=MOLECULES,
        )

        fig.subplots_adjust(wspace=0.7, hspace=0.1)
        fig.savefig(f"out/figure_2/test_tag_fig_{i}.png")
        plt.close(fig)

    # for tag_ax, ax in zip(tags_fig.axes, axs):
    #     tag_ax.remove()
    #     fig.axes.append(tag_ax)
    #     fig.add_axes(tag_ax)

    #     tag_ax.set_position(ax.get_position())
    #     ax.remove()
    # plt.close(tags_fig)


def make_figure_F(fig, axs, data, bounds, verbose):
    bounds = bounds[0]

    for exp_data in data:
        for molecule in MOLECULES:
            if verbose:
                print(f"Making snapshot and KDE plot for {molecule}...")

            fig, axs = make_snapshot_and_kde_plot(exp_data, bounds, molecule)

            # fig.subplots_adjust(wspace=0.7, hspace=0.1)
            fig.savefig(f"out/figure_2/test_tag_fig_{molecule[-1]}.png", bbox_inches="tight")
            plt.close(fig)


def make_layout(width=8, height=8):
    gs_kw = {"width_ratios": [1, 1, 1, 1, 1], "height_ratios": [1, 1, 1, 1, 1]}
    fig, axs = plt.subplot_mosaic(
        [
            ["A1_", "A2_", "A3_", "A4_", "A5_"],
            ["B__", "B__", "B__", "B__", "D__"],
            ["C__", "C__", "C__", "C__", "E__"],
            ["F1a", "F2a", "F3a", "F4a", "F5a"],
            ["F1b", "F2b", "F3b", "F4b", "F5b"],
        ],
        gridspec_kw=gs_kw,
        figsize=(width, height),
        layout="constrained",
    )

    # Make sure squares are squares
    for i in range(1, 6):
        axs[f"A{i}_"].set_box_aspect(1)

    axs["D__"].set_box_aspect(1)
    axs["E__"].set_box_aspect(1)

    for i in range(1, 6):
        axs[f"F{i}a"].set_box_aspect(1)

    return fig, axs


def get_data(
    experiment_ids, sampling_rate, start_time, end_time, host, port, cpus, verbose
):

    experiment_data = []
    bounds = []

    monomers = [m[-1] for m in MOLECULES if m[-2] == "monomer"]
    mrnas = [m[-1] for m in MOLECULES if m[-2] == "mrna"]
    inner_paths = [
        path for path in MOLECULES if path[-1] not in mrnas and path[-1] not in monomers
    ]
    outer_paths = [("data", "dimensions")]

    for exp_id in experiment_ids:
        if verbose:
            print(f"Accessing data for experiment {exp_id}...")

        data = access_counts(
            experiment_id=exp_id,
            monomer_names=monomers,
            mrna_names=mrnas,
            inner_paths=inner_paths,
            outer_paths=outer_paths,
            host=host,
            port=port,
            sampling_rate=sampling_rate,
            start_time=start_time,
            end_time=end_time,
            cpus=cpus,
        )

        with concurrent.futures.ProcessPoolExecutor(cpus) as executor:
            data_deserialized = list(
                tqdm(
                    executor.map(deserialize_and_remove_units, data.values()),
                    desc="Deserializing data",
                    total=len(data),
                )
            )
        data = dict(zip(data.keys(), data_deserialized))
        first_timepoint = data[min(data)]

        experiment_data.append(data)
        bounds.append(first_timepoint["dimensions"]["bounds"])

    return experiment_data, bounds


def make_figure(data, bounds, verbose):
    fig, axs = make_layout(width=8, height=8)

    os.makedirs("out/figure_2", exist_ok=True)

    if verbose:
        print("Making subfigure A...")

    # make_figure_A(fig, [ax for k, ax in axs.items() if k.startswith("A")], data, bounds)

    if verbose:
        print("Making subfigure F...")

    make_figure_F(fig, [ax for k, ax in axs.items() if k.startswith("F")], data, bounds, verbose)

    fig.savefig("out/glucose_figure.png")


def main():
    parser = argparse.ArgumentParser("Generate glucose figures.")

    parser.add_argument(
        "--experiment_ids",
        "-e",
        nargs="+",
        help="Ids of the experiments to use for the figure",
        required=True,
    )

    parser.add_argument(
        "--sampling_rate",
        "-r",
        type=int,
        default=10,
        help="Number of timepoints to step between frames.",
    )
    parser.add_argument("--start_time", "-s", type=int, default=MinKey())
    parser.add_argument("--end_time", "-f", type=int, default=MaxKey())
    parser.add_argument("--host", "-o", default="localhost", type=str)
    parser.add_argument("--port", "-p", default=27017, type=int)
    parser.add_argument("--cpus", "-c", type=int, default=1)
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    data, bounds = get_data(
        experiment_ids=args.experiment_ids,
        sampling_rate=args.sampling_rate,
        start_time=args.start_time,
        end_time=args.end_time,
        host=args.host,
        port=args.port,
        cpus=args.cpus,
        verbose=args.verbose,
    )
    make_figure(data, bounds, args.verbose)


if __name__ == "__main__":
    main()

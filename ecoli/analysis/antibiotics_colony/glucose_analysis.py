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

import matplotlib
import numpy as np
from bson import MaxKey, MinKey
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ecoli.analysis.db import access_counts, deserialize_and_remove_units
from ecoli.plots.snapshots import format_snapshot_data, make_tags_figure, plot_tags
from vivarium.library.units import units


MOLECULES = [("monomer", "PD00365"), ("monomer", "PD00406")]


def make_figure_A(fig, axs, data, bounds):
    agents, fields = format_snapshot_data(data)
    time_vec = list(agents.keys())

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
        fig.savefig("out/test_tag_fig_{i}.png")

    # for tag_ax, ax in zip(tags_fig.axes, axs):
    #     tag_ax.remove()
    #     fig.axes.append(tag_ax)
    #     fig.add_axes(tag_ax)

    #     tag_ax.set_position(ax.get_position())
    #     ax.remove()
    # plt.close(tags_fig)


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


def get_data(experiment_ids, sampling_rate, start_time, end_time, host, port, cpus):

    experiment_data = []
    bounds = []

    monomers = [m[-1] for m in MOLECULES if m[-2] == "monomer"]
    mrnas = [m[-1] for m in MOLECULES if m[-2] == "mrna"]
    inner_paths = [
        path for path in MOLECULES if path[-1] not in mrnas and path[-1] not in monomers
    ]
    outer_paths = [("data", "dimensions")]

    for exp_id in experiment_ids:
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
                    total=len(data),
                )
            )
        data = dict(zip(data.keys(), data_deserialized))
        first_timepoint = data[min(data)]

        experiment_data.append(data)
        bounds.append(first_timepoint["dimensions"]["bounds"])

    return experiment_data, bounds


def make_figure(data, bounds):
    fig, axs = make_layout(width=8, height=8)

    make_figure_A(fig, [ax for k, ax in axs.items() if k.startswith("A")], data.bounds)

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
    parser.add_argument("--host", "-o", default="localhost", type=str)
    parser.add_argument("--port", "-p", default=27017, type=int)
    parser.add_argument("--cpus", "-c", type=int, default=1)
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    data, bounds = get_data(
        experiment_ids=args.experiment_ids,
        sampling_rate=args.sampling_rate,
        start_time=MinKey(),
        end_time=MaxKey(),
        host=args.host,
        port=args.port,
        cpus=args.cpus,
    )
    make_figure(data, bounds)


if __name__ == "__main__":
    main()

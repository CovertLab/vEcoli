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

from tqdm import tqdm

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vivarium.library.units import units

from ecoli.analysis.db import access_counts, deserialize_and_remove_units
from ecoli.plots.snapshots import format_snapshot_data, make_tags_figure


def make_figure_A(
    fig,
    axs,
    data,
    **kwargs,
):
    agents, fields = format_snapshot_data(data)
    time_vec = list(agents.keys())
    bounds = [30, 30] * units.um

    n_snapshots = len(axs)

    # get time data
    time_indices = np.round(np.linspace(0, len(time_vec) - 1, n_snapshots)).astype(int)
    snapshot_times = [time_vec[i] for i in time_indices]

    tags_fig = make_tags_figure(
        agents=agents,
        bounds=bounds,
        n_snapshots=n_snapshots,
        time_indices=time_indices,
        snapshot_times=snapshot_times,
        **kwargs,
    )
    for tag_ax, ax in zip(tags_fig.axes, axs):
        tag_ax.remove()
        fig.axes.append(tag_ax)
        fig.add_axes(tag_ax)

        tag_ax.set_position(ax.get_position())
        ax.remove()
    plt.close(tags_fig)


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


def get_data(experiment_ids, sampling_rate, start_time, end_time, cpus):

    result = []

    for exp_id in experiment_ids:
        data = access_counts(
            experiment_id=exp_id,
            monomer_names=monomers,
            mrna_names=mrnas,
            inner_paths=inner_paths,
            outer_paths=outer_paths,
            host=args.host,
            port=args.port,
            sampling_rate=sampling_rate,
            start_time=start_time,
            end_time=end_time,
            cpus=cpus)
        
        with concurrent.futures.ProcessPoolExecutor(cpus) as executor:
            data_deserialized = list(tqdm(executor.map(
                deserialize_and_remove_units, data.values()), total=len(data)))
        data = dict(zip(data.keys(), data_deserialized))
        first_timepoint = data[min(data)]

        result.append(data)

    return result


def make_figure(experiment_ids, section=None):
    if section is None:
        section = ["A", "B", "C", "D", "E", "F"]

    data = get_data()

    fig, axs = make_layout(width=8, height=8)

    if "A" in section:
        make_figure_A(fig, [ax for k, ax in axs.items() if k.startswith("A")], data)
    if "B" in section:
        pass
    if "C" in section:
        pass
    if "D" in section:
        pass
    if "E" in section:
        pass
    if "F" in section:
        pass

    fig.savefig("out/glucose_figure.png")


def main():
    parser = argparse.ArgumentParser("Generate glucose figures.")

    parser.add_argument(
        "--section",
        "-s",
        default=None,
        choices=[None, "A", "B", "C", "D", "E", "F"],
        help="Generate a particular sub-figure (defaults to generating all figures)",
        required=False
    )

    parser.add_argument(
        "--experiment_ids",
        "-e",
        nargs="+",
        help="Ids of the experiments to use for the figure",
        required=True
    )

    args = parser.parse_args()

    make_figure(args.experiment_ids, args.section)


if __name__ == "__main__":
    main()

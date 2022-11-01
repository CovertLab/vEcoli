import argparse
import concurrent.futures
import os

import matplotlib
import numpy as np
import seaborn as sns
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from vivarium.core.emitter import DatabaseEmitter
from vivarium.library.dict_utils import get_value_from_path
from vivarium.library.topology import convert_path_style

from ecoli.analysis.db import access_counts, deserialize_and_remove_units
from ecoli.plots.snapshots import plot_tags


def make_snapshot_and_kde_plot(timepoint_data, bounds, molecule, title=None):
    """Generates a figure with a snapshot plot tagging the specified molecule,
    and a smoothed density plot (using KDE) of the distribution of counts for that molecule
    at that time.
    Args:
        timepoint_data: data from one timepoint, in the form {time : {...data...}}
        bounds: physical bounds for the snapshot plot
        molecule: molecule to tag in the snapshot plot / KDE plot.
    Returns:
        fig, axes"""

    time = list(timepoint_data.keys())
    assert len(time) == 1, f"Expected only one timepoint, got {time}"
    time = time[0]

    # Get snapshot figure from plot_tags
    fig = plot_tags(
        timepoint_data,
        bounds,
        snapshot_times=[time],
        tagged_molecules=[molecule],
        show_timeline=False,
        background_color="white",
        default_font_size=plt.rcParams["font.size"],
        scale_bar_length=None,  # TODO: scale bar length looks wrong?
    )
    tag_axes = fig.get_axes()
    snapshot_ax, conc_ax = tag_axes[:2]

    # Prettify axis labels
    snapshot_ax.set(ylabel=None)
    snapshot_ax.set_title(molecule[-1] if title is None else title)

    grid = fig.add_gridspec(2, 2, width_ratios=[2, 1], wspace=0.1, hspace=0.2)

    # Reposition axes, preparing to add kde plot below
    snapshot_ax.set_position(grid[0, 0].get_position(fig))
    conc_ax.set_position(grid[0, 1].get_position(fig))
    snapshot_ax.set_subplotspec(grid[0, 0])
    conc_ax.set_subplotspec(grid[0, 1])

    # Remove and re-create scale bar if present
    for a in conc_ax.get_children():
        if isinstance(a, AnchoredSizeBar):
            a.remove()

            scale_bar_length = 1
            scale_bar = AnchoredSizeBar(
                conc_ax.transData,
                scale_bar_length,
                f"{scale_bar_length} μm",
                "lower left",
                color="black",
                frameon=False,
                sep=scale_bar_length,
                size_vertical=scale_bar_length / 20,
            )
            conc_ax.add_artist(scale_bar)
            break

    # Add KDE plot
    kde_ax = fig.add_subplot(grid[1, 0])

    # Get distribution of concentration across agents
    kde_data = {
        molecule[-1]: [
            (
                get_value_from_path(agent_data, molecule)
                / agent_data.get("boundary", {}).get("volume", 0)
            )
            for agent_data in timepoint_data[time]["agents"].values()
        ]
    }

    # Plot KDE, rugplot
    sns.histplot(data=kde_data, x=molecule[-1], ax=kde_ax)
    # sns.kdeplot(data=kde_data, x=molecule[-1], ax=kde_ax)
    # sns.rugplot(data=kde_data, x=molecule[-1], ax=kde_ax)
    kde_ax.set(xlabel=None)
    kde_ax.set_box_aspect(1)

    return fig, fig.get_axes()


def get_data(experiment_id, time, molecules, host, port, cpus, verbose):

    # Prepare molecule paths for access_counts()
    monomers = [m[-1] for m in molecules if m[-2] == "monomer"]
    mrnas = [m[-1] for m in molecules if m[-2] == "mrna"]
    inner_paths = [
        path for path in molecules if path[-1] not in mrnas and path[-1] not in monomers
    ]
    outer_paths = [("data", "dimensions")]

    if verbose:
        print(f"Accessing data for experiment {experiment_id}...")

    # Query database
    import ipdb; ipdb.set_trace()
    data = access_counts(
        experiment_id=experiment_id,
        monomer_names=monomers,
        mrna_names=mrnas,
        inner_paths=inner_paths,
        outer_paths=outer_paths,
        host=host,
        port=port,
        start_time=time,
        end_time=time,
        cpus=cpus,
    )

    with concurrent.futures.ProcessPoolExecutor(cpus) as executor:
        # Prepare to deserialize data
        data_deserialized = executor.map(deserialize_and_remove_units, data.values())

        # If verbose, add a progress bar
        if verbose:
            data_deserialized = tqdm(
                data_deserialized, desc="Deserializing data", total=len(data)
            )

        # Do the actual deserializing (lazy computation)
        data_deserialized = list(data_deserialized)

    # prep data, physical bounds for returning
    data = dict(zip(data.keys(), data_deserialized))
    bounds = data[time]["dimensions"]["bounds"]

    return data, bounds


def main():
    parser = argparse.ArgumentParser(
        "Generate snapshot and kde figures for specified molecules."
    )

    parser.add_argument(
        "experiment_id",
        help="ID of the experiment for which to make the figure(s).",
    )
    parser.add_argument(
        "--molecule_paths",
        "-m",
        nargs="+",
        required=True,
        help="Paths (in A>B>C form) of the molecule(s) for which to generate figure(s). "
        'Can be preceded by an alias for that molecule, e.g. "OmpF=monomer>EG10671-MONOMER".',
    )
    parser.add_argument(
        "--time",
        "-t",
        type=int,
        default=None,
        help="Timepoint which to plot (defaults to last timepoint).",
    )
    parser.add_argument(
        "--outdir",
        "-d",
        default="out/snapshot_kde_plots",
        help="Directory in which to output the generated figures.",
    )
    parser.add_argument("--host", "-o", default="localhost", type=str)
    parser.add_argument("--port", "-p", default=27017, type=int)
    parser.add_argument("--cpus", "-c", type=int, default=1)
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    # Covert molecule path styles, get names
    molecules = []
    molecule_names = []
    for path in args.molecule_paths:
        p = path.split("=")
        if len(p) == 1:  # no alias given
            p = convert_path_style(p[0])
            molecules.append(p)
            molecule_names.append(p[-1])
        else:
            molecules.append(convert_path_style(p[-1]))
            molecule_names.append(p[0])

    # Get max time if no time specified
    time = args.time
    if time is None:
        if args.verbose:
            print(
                "No timepoint given, trying to infer and use final timepoint from database."
            )
            print(
                "Note that this sometimes fails, in which case, consider specifying an explicit timepoint."
            )

        config = {"host": f"{args.host}:{args.port}", "database": "simulations"}
        emitter = DatabaseEmitter(config)
        db = emitter.db

        time = list(
            db.history.aggregate(
                [
                    {"$match": {"experiment_id": args.experiment_id}},
                    {"$project": {"data.time": 1}},
                    {"$group": {"_id": None, "time": {"$max": "$data.time"}}},
                ]
            )
        )[0]["time"]

    # Get data from database
    data, bounds = get_data(
        experiment_id=args.experiment_id,
        time=2 * (time // 2),  # only even timesteps have the data necessary
        molecules=molecules,
        host=args.host,
        port=args.port,
        cpus=args.cpus,
        verbose=args.verbose,
    )

    os.makedirs(args.outdir, exist_ok=True)

    # Generate one figure per molecule
    for name, molecule in zip(molecule_names, molecules):
        if args.verbose:
            print(f"Plotting snapshot + KDE for {molecule[-1]}...")

        fig, _ = make_snapshot_and_kde_plot(data, bounds, molecule, name)

        fig.set_size_inches(6, 6)
        fig.savefig(
            os.path.join(args.outdir, f"snapshot_and_kde_{molecule[-1]}.png"),
            bbox_inches="tight",
        )
        plt.close(fig)


if __name__ == "__main__":
    main()

import argparse
import concurrent.futures
import os

import matplotlib
import seaborn
import numpy as np
from scipy.stats import gaussian_kde
from bson import MaxKey, MinKey
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vivarium.library.topology import convert_path_style
from vivarium.core.emitter import DatabaseEmitter
from ecoli.analysis.db import access_counts, deserialize_and_remove_units
from ecoli.plots.snapshots import plot_tags


MOLECULES = [
    ("bulk", "MICF-RNA[c]"),
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


def make_snapshot_and_kde_plot(timepoint_data, bounds, molecule):
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

    import ipdb; ipdb.set_trace()

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
        help="Paths (in A>B>C form) of the molecule(s) for which to generate figure(s).",
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

    # Covert molecule path styles
    molecules = [convert_path_style(p) for p in args.molecule_paths]

    # Get max time if no time specified
    time = args.time
    if time is None:
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
        time=time,
        molecules=molecules,
        host=args.host,
        port=args.port,
        cpus=args.cpus,
        verbose=args.verbose,
    )

    # Generate one figure per molecule
    for molecule in molecules:
        fig, axs = make_snapshot_and_kde_plot(data, bounds, molecule)

        os.makedirs(args.outdir, exist_ok=True)
        fig.savefig(os.path.join(args.outdir, f"snapshot_and_kde_{molecule[-1]}.png"))
        plt.close(fig)


if __name__ == "__main__":
    main()

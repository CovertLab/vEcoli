import os
import argparse

import matplotlib
import numpy as np

from ecoli.analysis.db import deserialize_and_remove_units

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from vivarium.core.emitter import DatabaseEmitter
from vivarium.library.units import units


def plot_field_cross_section(
    field_data,
    bounds,
    field,
    x=None,
    y=None,
    ax=None,
    min_color="#aaaaaa",
    max_color="#000000",
    field_units=units.mM,
    bounds_units=units.um,
    time_units_from=units.s,
    time_units_to=units.hr,
    time_digits=2,
):
    """Plot a (usually timeseries) environmental cross-section for a given field molecule.

    Args:
    - field_data: Data for which to do the plot, of the form {time: {field: environmental_lattice}}
    - bounds: Dimensions (width, height) of the environment.
    - field: Which field molecule to plot.
    - x, y: The x- or y-value for which to do the cross section (in same units as bounds).
            One or the other must be supplied, but not both.
    - ax: The axis in which to plot (if provided, otherwise, one is created).
    - min_color, max_color: colors to use for the first and last timepoints.
    """
    if (x is None and y is None) or (x is not None and y is not None):
        raise ValueError(
            f"Either x or y must be supplied (but not both), got {x=} and {y=}."
        )

    if ax is None:
        _, ax = plt.subplots()

    # create colormap
    cm = LinearSegmentedColormap.from_list(
        "timepoint_cmap", [min_color, max_color], N=len(field_data)
    )
    time_vec = sorted(list(field_data.keys()))
    min_time, max_time = time_vec[0], time_vec[-1]

    # create one variable for the x- or y- value, transposing the lattice if appropriate
    # in order to do following calculations in a directionally-agnostic way.
    pos = x if x is not None else y
    side_length = bounds[0 if x is not None else 1]
    do_transpose = x is not None

    for time, fields in field_data.items():
        lattice = fields[field]

        if do_transpose:
            lattice = lattice.T

        # Get lattice row/col index corresponding to snapping pos to nearest row/column
        lattice_i = lattice.shape[0] * (pos / side_length)
        row_data = lattice[int(lattice_i), :]

        ax.plot(
            np.linspace(0, side_length, len(row_data)),
            row_data,
            label=round((time * time_units_from).to(time_units_to), time_digits),
            color=cm((time - min_time) / (max_time - min_time)),
        )
        ax.set_xlabel(
            f"Environment {'Vertical' if x else 'Horizontal'} Axis ({bounds_units})"
        )
        ax.set_ylabel(f"Concentration ({field_units})")

    ax.legend()

    return ax


def get_data(
    experiment_id,
    timepoints,
    molecules,
    host,
    port,
    verbose,
):
    config = {"host": f"{host}:{port}", "database": "simulations"}
    emitter = DatabaseEmitter(config)
    db = emitter.db

    if verbose:
        print("Accessing data...")

    data = db.history.aggregate(
        [
            {
                "$match": {
                    "experiment_id": experiment_id,
                    "data.time": {"$in": timepoints},
                    "data.fields": {"$exists": True},
                }
            },
            {
                "$project": {
                    "data.time": 1,
                    **{f"data.fields.{molecule}": 1 for molecule in molecules},
                }
            },
        ]
    )

    # Make time the primary index,
    # transpose the field arrays into correct orientation
    data = {
        d["data"]["time"]: {
            fieldname: np.array(field).T
            for fieldname, field in d["data"]["fields"].items()
        }
        for d in data
    }

    # Get physical bounds (width, height) of the environment
    bounds = None
    for doc in db.history.find(
        {"experiment_id": experiment_id, "data.time": 0},
        {"data.dimensions.bounds": 1},
    ):
        if doc.get("data", {}).get("dimensions", {}).get("bounds", None) is not None:
            bounds = deserialize_and_remove_units(doc["data"]["dimensions"]["bounds"])

    if bounds is None:
        raise ValueError(
            "Not able to determine physical bounds of environment from database!"
        )

    return data, bounds


def main():
    parser = argparse.ArgumentParser(
        "Generate environmental cross-sections for the specified molecule(s)."
    )

    parser.add_argument(
        "experiment_id",
        help="ID of the experiment for which to make the figure(s).",
    )
    parser.add_argument(
        "--molecules",
        "-m",
        nargs="+",
        required=True,
        help="Identifiers of environmental molecules to plot.",
    )
    parser.add_argument(
        "--timepoints",
        "-t",
        nargs="+",
        required=True,
        type=float,
        help="Timepoints to plot ",
    )
    parser.add_argument(
        "--outdir",
        "-d",
        default="out/field_cross_sections",
        help="Directory in which to output the generated figures.",
    )
    parser.add_argument("--host", "-o", default="localhost", type=str)
    parser.add_argument("--port", "-p", default=27017, type=int)
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    data, bounds = get_data(
        experiment_id=args.experiment_id,
        timepoints=args.timepoints,
        molecules=args.molecules,
        host=args.host,
        port=args.port,
        verbose=args.verbose,
    )

    os.makedirs(args.outdir, exist_ok=True)

    for molecule in args.molecules:
        if args.verbose:
            print(f"Plotting cross section for {molecule}...")

        fig, ax = plt.subplots()
        plot_field_cross_section(data, bounds, molecule, y=bounds[1] / 2, ax=ax)

        fig.savefig(
            os.path.join(args.outdir, f"env_cross_section_{molecule}.png"),
            bbox_inches="tight",
        )
        plt.close(fig)


if __name__ == "__main__":
    main()

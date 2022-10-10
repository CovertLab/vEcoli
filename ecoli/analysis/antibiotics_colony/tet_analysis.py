import argparse
import concurrent.futures
import matplotlib.pyplot as plt

from bson import MaxKey, MinKey
from tqdm import tqdm
from vivarium.library.topology import convert_path_style
from vivarium.core.serialize import deserialize_value
from vivarium.library.units import remove_units

from ecoli.analysis.db import access_counts
from ecoli.analysis.analyze_multigen_db_experiment import plot_agents_multigen


MONGO_IP = "10.138.0.75"
MOLECULES = [
    # "external tet",
    "bulk>tetracycline[c]",
    # "OmpF",
    # "Outer membrance tet(?)",
    # "AcrAB-TolC",
    # "Inhibited Active Ribosomes",
    # "Uninhibited Active Ribosomes",
    # "Inhibited 30S Subunits",
    # "Uninhibited 30S Subunits"
]


def deserialize_and_remove_units(d):
    return remove_units(deserialize_value(d))


def multigen_traces(outfile, data, timeseries_paths, highlight_agents, highlight_color):
    plot_settings = {
        "column_width": 6,
        "row_height": 2,
        "stack_column": True,
        "tick_label_size": 10,
        "linewidth": 2,
        "title_size": 10,
        "include_paths": timeseries_paths,
    }

    fig = plot_agents_multigen(data, dict(plot_settings))
    fig.savefig(outfile, bbox_inches="tight")
    plt.close()


def run_analysis(
    experiment_id,
    outfile,
    tags,
    sampling_rate,
    host,
    port,
    start_time,
    end_time,
    cpus,
    verbose,
):

    # Get the required data
    tags = [convert_path_style(path) for path in tags]

    if verbose:
        print(f"Plotting the following timeseries into {outfile}:")
        for path in tags:
            print(f"> {path}")

    monomers = [path[-1] for path in tags if path[-2] == "monomer"]
    mrnas = [path[-1] for path in tags if path[-2] == "mrna"]
    inner_paths = [
        path for path in tags if path[-1] not in mrnas and path[-1] not in monomers
    ]
    timeseries = [convert_path_style(path) for path in tags]
    outer_paths = [("data", "dimensions")]

    data = access_counts(
        experiment_id,
        monomers,
        mrnas,
        inner_paths,
        outer_paths,
        host,
        port,
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

    # Remove last timestep since data may be empty
    data = dict(list(data.items())[:-1])

    multigen_traces(outfile, data, timeseries, None, None)

    if verbose:
        print("Done.")


def cli():
    parser = argparse.ArgumentParser("")

    # tet experiment: 2022-09-30_16-48-50_780764+0000
    # host: 10.138.0.75

    parser.add_argument("experiment_id", help="Experiment ID")
    parser.add_argument("outfile", help="File in which to save plotted output.")
    parser.add_argument(
        "--tags",
        "-g",
        nargs="*",
        default=[],
        help='Paths (e.g. "a>b>c") to variables to tag.',
    )
    parser.add_argument(
        "--sampling_rate",
        "-r",
        type=int,
        default=1,
        help="Number of timepoints to step between frames.",
    )
    parser.add_argument("--host", "-o", default="localhost", type=str)
    parser.add_argument("--port", "-p", default=27017, type=int)
    parser.add_argument("--start_time", "-s", type=int, default=MinKey())
    parser.add_argument("--end_time", "-e", type=int, default=MaxKey())
    parser.add_argument("--cpus", "-c", type=int, default=1)
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    run_analysis(
        args.experiment_id,
        args.outfile,
        args.tags,
        args.sampling_rate,
        args.host,
        args.port,
        args.start_time,
        args.end_time,
        args.cpus,
        args.verbose,
    )


def main():
    cli()


if __name__ == "__main__":
    main()

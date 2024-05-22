import argparse
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm
from bson import MinKey, MaxKey
from vivarium.plots.agents_multigen import plot_agents_multigen
from vivarium.library.topology import convert_path_style

from ecoli.analysis.db import access, get_agent_ids
from ecoli.analysis.analyze_db_experiment import OUT_DIR
from ecoli.analysis.snapshots_video import deserialize_and_remove_units

AGENTS_PATH = ("agents",)
SKIP_PATHS = [
    ("listeners", "rna_synth_prob"),
    ("listeners", "ribosome_data"),
]


def main():
    # parse
    parser = argparse.ArgumentParser(description="Plot data from multigen experiment.")
    parser.add_argument("experiment_id", type=str)
    parser.add_argument("--host", "-o", default="localhost", type=str)
    parser.add_argument("--port", "-p", default=27017, type=int)
    parser.add_argument("--path", "-t", type=str, nargs="*", default=[])
    parser.add_argument("--agent", "-a", type=str, nargs="*", default=[])
    parser.add_argument("--sampling_rate", "-r", type=int, default=1)
    parser.add_argument("--start_time", "-s", type=int, default=MinKey())
    parser.add_argument("--end_time", "-e", type=int, default=MaxKey())
    parser.add_argument("--cpus", "-c", type=int, default=1)
    args = parser.parse_args()

    agents = get_agent_ids(args.experiment_id, args.host, args.port)
    if args.agent:
        assert set(args.agent) - agents == set()
        agents = args.agent

    paths = [convert_path_style(path) for path in args.path]
    query = []
    for path in paths:
        for agent in agents:
            query.append(("agents", agent) + path)
    if not query:
        query = None

    # Retrieve all simulation data.
    data, experiment_id, sim_config = access(
        args.experiment_id,
        query=query,
        host=args.host,
        port=args.port,
        sampling_rate=args.sampling_rate,
        start_time=args.start_time,
        end_time=args.end_time,
        cpus=args.cpus,
    )

    with ProcessPoolExecutor() as executor:
        data_deserialized = list(
            tqdm(
                executor.map(deserialize_and_remove_units, data.values()),
                total=len(data),
            )
        )
    data = dict(zip(data.keys(), data_deserialized))

    plot_agents_multigen(
        data,
        {
            "agents_key": "agents",
            "skip_paths": SKIP_PATHS,
        },
        out_dir=OUT_DIR,
        filename=f"{experiment_id}_multigen",
    )


if __name__ == "__main__":
    main()

import argparse

from vivarium.core.serialize import deserialize_value
from vivarium.plots.agents_multigen import plot_agents_multigen
from vivarium.library.topology import (
    get_in, assoc_path, convert_path_style)
from vivarium.library.units import remove_units

from ecoli.analysis.db import access, get_agent_ids
from ecoli.analysis.analyze_db_experiment import OUT_DIR

AGENTS_PATH = ('agents',)
SKIP_PATHS = [
    ('listeners', 'rna_synth_prob'),
    ('listeners', 'ribosome_data'),
]


def main():
    # parse
    parser = argparse.ArgumentParser(
        description='Plot data from multigen experiment.')
    parser.add_argument(
        'experiment_id', type=str)
    parser.add_argument(
        '--host', '-o', default='localhost', type=str)
    parser.add_argument(
        '--port', '-p', default=27017, type=int)
    parser.add_argument(
        '--path', '-t', type=str, nargs='*', default=[])
    parser.add_argument(
        '--agent', '-a', type=str, nargs='*', default=[])
    parser.add_argument(
        '--sampling_rate', '-r', type=int, default=1)
    args = parser.parse_args()

    agents = get_agent_ids(
        args.experiment_id, args.host, args.port)
    if args.agent:
        assert set(args.agent) - agents == set()
        agents = args.agent

    paths = [convert_path_style(path) for path in args.path]
    query = []
    for path in paths:
        for agent in agents:
            query.append(('agents', agent) + path)
    if not query:
        query = None

    # Retrieve all simulation data.
    data, experiment_id, sim_config = access(
        args.experiment_id, query=query, host=args.host, port=args.port,
        sampling_rate=args.sampling_rate)
    data = deserialize_value(data)
    data = remove_units(data)

    plot_agents_multigen(
        data,
        {
            'agents_key': 'agents',
            'skip_paths': SKIP_PATHS,
        },
        out_dir=OUT_DIR,
        filename=f'{experiment_id}_multigen',
    )


if __name__ == '__main__':
    main()

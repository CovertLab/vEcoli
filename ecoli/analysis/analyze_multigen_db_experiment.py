import argparse

from vivarium.core.serialize import deserialize_value
from vivarium.plots.agents_multigen import plot_agents_multigen
from vivarium.library.topology import get_in, assoc_path
from vivarium.library.units import remove_units

from ecoli.analysis.analyze_db_experiment import access, OUT_DIR

SERIALIZED_PATHS = (
    ('periplasm', 'global', 'mmol_to_counts'),
    ('periplasm', 'global', 'volume'),
    ('boundary', 'surface_area'),
    ('boundary', 'mmol_to_counts'),
    ('boundary', 'mass'),
    ('permeabilities',),
)
AGENTS_PATH = ('agents',)


def main():
    # parse
    parser = argparse.ArgumentParser(
        description='Plot data from multigen experiment.')
    parser.add_argument(
        '--experiment_id', '-e',
        type=str, default='')
    parser.add_argument(
        '--host', '-o', default='localhost', type=str)
    parser.add_argument(
        '--port', '-p', default=27017, type=int)
    args = parser.parse_args()

    # Retrieve all simulation data.
    data, experiment_id, sim_config = access(
        args.experiment_id, host=args.host, port=args.port)
    data = deserialize_value(data)
    #for time, time_data in data.items():
    #    for agent, agent_data in get_in(time_data, AGENTS_PATH).items():
    #        for path_suffix in SERIALIZED_PATHS:
    #            path = (time,) + AGENTS_PATH + (agent,) + path_suffix
    #            serialized = get_in(data, path)
    #            assert serialized is not None
    #            deserialized = deserialize_value(serialized)
    #            assoc_path(data, path, deserialized)
    data = remove_units(data)

    plot_agents_multigen(
        data,
        {
            'agents_key': 'agents',
        },
        out_dir=OUT_DIR,
        filename=f'{experiment_id}_multigen',
    )


if __name__ == '__main__':
    main()

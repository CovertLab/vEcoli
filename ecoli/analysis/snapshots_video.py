import argparse
import concurrent.futures

from tqdm import tqdm
from bson import MinKey, MaxKey
from vivarium.library.topology import convert_path_style
from vivarium.core.serialize import deserialize_value
from vivarium.library.units import remove_units

from ecoli.analysis.db import access_counts
from ecoli.analysis.analyze_db_experiment import OUT_DIR
from ecoli.plots.snapshots_video import make_video


def deserialize_and_remove_units(d):
    return remove_units(deserialize_value(d))


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Create video of snapshot plots')
    parser.add_argument('experiment_id', help='Experiment ID')
    parser.add_argument(
        '--tags', '-g', nargs='*', default=[],
        help='Paths (e.g. "a>b>c") to variables to tag.')
    parser.add_argument(
        '--timeseries', '-t', nargs='*', default=[],
        help='Paths (e.g. "a>b>c") to variables to plot in timeseries.')
    parser.add_argument(
        '--highlight_agents', '-a', nargs='*',
        help='IDs of agents to highlight')
    parser.add_argument(
        '--sampling_rate', '-r', type=int, default=1,
        help='Number of timepoints to step between frames.')
    parser.add_argument(
        '--fields', '-f', action='store_true',
        help='Generate snapshots video of fields.')
    parser.add_argument(
        '--host', '-o', default='localhost', type=str)
    parser.add_argument(
        '--port', '-p', default=27017, type=int)
    parser.add_argument(
        '--start_time', '-s', type=int, default=MinKey())
    parser.add_argument(
        '--end_time', '-e', type=int, default=MaxKey())
    parser.add_argument(
        '--cpus', '-c', type=int, default=1)
    args = parser.parse_args()

    # Get the required data
    tags = [convert_path_style(path) for path in args.tags]
    monomers = [path[-1] for path in tags if path[-2]=='bulk']
    mrnas = [path[-1] for path in tags if path[-2]=='unique']
    timeseries = [convert_path_style(path) for path in args.timeseries]
    data = access_counts(args.experiment_id, monomers, mrnas, args.host, 
        args.port, sampling_rate=args.step, start_time=args.start_time,
        end_time=args.end_time, cpus=args.cpus)
    
    with concurrent.futures.ProcessPoolExecutor(12) as executor:
        data_deserialized = list(tqdm(executor.map(
            deserialize_and_remove_units, data.values()), total=len(data)))
    data = dict(zip(data.keys(), data_deserialized))
    first_timepoint = data[min(data)]

    # Make the videos
    if args.fields:
        make_video(
            data,
            first_timepoint['dimensions']['bounds'],
            plot_type='fields',
            step=1,
            out_dir=OUT_DIR,
            filename=f'{args.experiment_id}_snapshots',
            highlight_agents=args.highlight_agents,
            show_timeseries=timeseries,
        )
    if args.tags:
        make_video(
            data,
            first_timepoint['dimensions']['bounds'],
            plot_type='tags',
            step=1,
            out_dir=OUT_DIR,
            filename=f'{args.experiment_id}_tags',
            highlight_agents=args.highlight_agents,
            tagged_molecules=tags,
            show_timeseries=timeseries,
            background_color='white',
        )


if __name__ == '__main__':
    main()

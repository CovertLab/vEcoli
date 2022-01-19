import argparse

from vivarium.library.topology import convert_path_style

from ecoli.analysis.analyze_db_experiment import access, OUT_DIR
from ecoli.plots.snapshots_video import make_video


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Create video of snapshot plots')
    parser.add_argument('experiment_id', help='Experiment ID')
    parser.add_argument(
        '--tags', '-g', nargs='*',
        help='Paths (e.g. "a>b>c") to variables to tag.')
    parser.add_argument(
        '--timeseries', '-t', nargs='*',
        help='Paths (e.g. "a>b>c") to variables to plot in timeseries.')
    parser.add_argument(
        '--highlight_agents', '-a', nargs='*',
        help='IDs of agents to highlight')
    parser.add_argument(
        '--step', '-s', type=float, default=1,
        help='Number of timepoints to step between frames.')
    parser.add_argument(
        '--fields', '-f', action='store_true',
        help='Generate snapshots video of fields.')
    args = parser.parse_args()

    # Get the required data
    query = [
        ('fields',),
        ('agents',),
        ('dimensions',),
    ]
    tags = [convert_path_style(path) for path in args.tags]
    timeseries = [convert_path_style(path) for path in args.timeseries]
    data, _, sim_config = access(args.experiment_id, query)
    first_timepoint = data[min(data)]

    # Make the videos
    if args.fields:
        make_video(
            data,
            first_timepoint['dimensions']['bounds'],
            plot_type='fields',
            step=args.step,
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
            step=args.step,
            out_dir=OUT_DIR,
            filename=f'{args.experiment_id}_tags',
            highlight_agents=args.highlight_agents,
            tagged_molecules=tags,
            show_timeseries=timeseries,
            background_color='white',
        )


if __name__ == '__main__':
    main()

import argparse
import importlib
import json
import os
import warnings

import polars as pl

from ecoli.composites.ecoli_configs import CONFIG_DIR_PATH
from ecoli.library.parquet_emitter import get_lazyframes

FILTERS = {
    'experiment_id': (str, 'multi_experiment', 'multi_variant'),
    'variant': (int, 'multi_variant', 'multi_seed'),
    'lineage_seed': (int, 'multi_seed', 'multi_generation'),
    'generation': (int, 'multi_generation', 'multi_daughter'),
    'agent_id': (str, 'multi_daughter', 'single')
}
"""Mapping of data filters to data type, analysis type if more than
one value is given for filter, and analysis type for one value."""

def main():
    parser = argparse.ArgumentParser()
    default_config = os.path.join(CONFIG_DIR_PATH, 'default.json')
    parser.add_argument(
        '--config', '-c',
        default=default_config,
        help=(
            'Path to configuration file for the simulation. '
            'All key-value pairs in this file will be applied on top '
            f'of the options defined in {default_config}.'))
    for data_filter, (data_type, _, _) in FILTERS.items():
        parser.add_argument(
            f'--{data_filter}', nargs='*', type=data_type,
            help=f'Limit data to one or more {data_filter}(s).')
        if data_type is not str:
            parser.add_argument(f'--{data_filter}-range', nargs=2,
                metavar=('START', 'END'), type=data_type,
                help=f'Limit data to range of {data_filter}s not incl. END.')
    parser.add_argument(
        '--sim_data_path', '--sim-data-path', nargs="*", default=None,
        help="Path to the sim_data to use.")
    parser.add_argument(
        '--validation_data_path', '--validation-data-path', default=None,
        help="Path to the validation_data to use.")
    parser.add_argument(
        '--outdir', '-o', default=None,
        help="Directory that all analysis output is saved to.")
    config_file = os.path.join(CONFIG_DIR_PATH, 'default.json')
    args = parser.parse_args()
    with open(config_file, 'r') as f:
        config = json.load(f)
    if args.config is not None:
        config_file = args.config
        with open(os.path.join(args.config), 'r') as f:
            config = {**config, **json.load(f)}
    out_dir = config['emitter']['config'].get('out_dir', None)
    out_uri = config['emitter']['config'].get('out_uri', None)
    config = {**config['analysis_options'], **vars(args)}

    # Changes current working directory so analysis scripts can save
    # plots, etc. without worrying about file paths
    os.chdir(config['outdir'])
    os.makedirs('plots', exist_ok=True)
    os.chdir('plots')

    # Set up Polars filters for data
    analysis_type = None
    pl_filter = None
    last_analysis_level = -1
    filter_types = list(FILTERS.keys())
    for current_analysis_level, (data_filter, (_, analysis_many, analysis_one)
                                 ) in enumerate(FILTERS.items()):
        if config.get(f'{data_filter}_range', None) is not None:
            if config[data_filter] is not None:
                warnings.warn(
                    f"Provided both range and value(s) for {data_filter}. "
                    "Range takes precedence.")
            config[data_filter] = list(range(
                config[f'{data_filter}_range'][0],
                config[f'{data_filter}_range'][1]))
        if config.get(data_filter, None) is not None:
            if last_analysis_level != current_analysis_level - 1:
                skipped_filters = filter_types[
                    last_analysis_level+1:current_analysis_level]
                warnings.warn(f"Filtering by {data_filter} when last filter "
                              f"specified was {filter_types[last_analysis_level]}. "
                              "Will load all applicable data for the skipped "
                              f"filters: {skipped_filters}.")
            if len(config[data_filter]) > 1:
                analysis_type = analysis_many
                if pl_filter is None:
                    pl_filter = pl.col(data_filter).is_in(config[data_filter])
                else:
                    pl_filter = (pl.col(data_filter).is_in(config[data_filter])) & pl_filter
            else:
                analysis_type = analysis_one
                if pl_filter is None:
                    pl_filter = pl.col(data_filter) == config[data_filter][0]
                else:
                    pl_filter = (pl.col(data_filter) == config[data_filter][0]) & pl_filter
            last_analysis_level = current_analysis_level

    # If no filters were provided, assume analyzing ParCa output
    if analysis_type is None:
        analysis_type = 'parca'
        config_lf = pl.DataFrame({})
        history_lf = pl.DataFrame({})
    else:
        # Load Parquet files from output directory / URI specified in config
        config_lf, history_lf = get_lazyframes(out_dir, out_uri)
        config_lf = config_lf.filter(pl_filter)
        history_lf = history_lf.filter(pl_filter)

    # Run the analyses listed under the most specific filter
    analysis_options = config[analysis_type]
    for analysis_name, analysis_params in analysis_options.items():
        analysis_mod = importlib.import_module(f'ecoli.analysis.{analysis_name}')
        analysis_mod.plot(
            analysis_params,
            config_lf,
            history_lf,
            config['sim_data_path'],
            config['validation_data_path'])
    
    # Save copy of config JSON with parameters for plots
    with open(os.path.join(config['outdir'], 'metadata.json'), 'w') as f:
        json.dump(config, f)

if __name__ == '__main__':
    main()

import argparse
import importlib
import os
import warnings

import polars as pl

from ecoli.composites.ecoli_configs import CONFIG_DIR_PATH
from ecoli.experiments.ecoli_master_sim import SimConfig
from ecoli.library.parquet_emitter import get_lazyframes

FILTERS = [
    'experiment_id',
    'variant',
    'seed',
    'generation',
    'cell_id'
]

def main():
    parser = argparse.ArgumentParser()
    default_config = os.path.join(CONFIG_DIR_PATH, 'default.json')
    parser.add_argument(
        '--config',
        default=default_config,
        help=(
            'Path to configuration file for the simulation. '
            'All key-value pairs in this file will be applied on top '
            f'of the options defined in {default_config}.'))
    for data_filter in FILTERS:
        parser.add_argument(
            f'--{data_filter}', nargs='*',
            help=f'Limit data to one or more {data_filter}(s).')
        if data_filter != 'experiment_id':
            parser.add_argument(
                f'--{data_filter}-range', nargs=2, metavar=('START', 'END'),
                help=f'Limit data to range of {data_filter}s not incl. END.')
    parser.add_argument(
        '--sim_data_paths', nargs="*", default=None,
        help="Path to the sim_data to use.")
    parser.add_argument(
        '--validation_data_path', default=None,
        help="Path to the validation_data to use.")
    parser.add_argument(
        '--outdir', '-o', default=None,
        help="Change directory to this path. A folder named plot will be "
            "created there and all files saved to that folder will be "
            "saved by Nextflow.")
    config = SimConfig(parser=parser)
    config.update_from_cli()

    # Changes current working directory so analysis scripts just need to
    # save any plots, etc. into the plot folder as a relative path
    os.chdir(config['outdir'])
    os.makedirs('plot', exist_ok=True)

    # Load Parquet files from output directory / URI specified in config
    emitter_config = config['emitter']['config']
    config_lf, history_lf = get_lazyframes(
        emitter_config.get('out_dir', None),
        emitter_config.get('out_uri', None))

    # Filters data
    analysis_type = None
    last_analysis_level = -1
    for current_analysis_level, data_filter in enumerate(FILTERS):
        if config[f'{data_filter}_range'] is not None:
            if config[data_filter] is not None:
                warnings.warn(
                    f"Provided both range and value(s) for {data_filter}. "
                    "Range takes precedence.")
            config[data_filter] = list(range(
                config[f'{data_filter}_range'][0],
                config[f'{data_filter}_range'][1]))
        if config[data_filter] is not None:
            if last_analysis_level != current_analysis_level - 1:
                skipped_filters = FILTERS[
                    last_analysis_level+1:current_analysis_level]
                warnings.warn(f"Filtering by {data_filter} when last filter "
                              f"specified was {FILTERS[last_analysis_level]}. "
                              "Will load all applicable data for the skipped "
                              f"filters: {skipped_filters}.")
            if len(config[data_filter]) > 1:
                analysis_type = f'multi{data_filter}'
                pl_filter = pl.col(data_filter).is_in(config['filter'])
            else:
                if current_analysis_level + 1 < len(FILTERS):
                    analysis_type = f'multi{FILTERS[current_analysis_level+1]}'
                else:
                    analysis_type = 'single'
                pl_filter = pl.col(data_filter) == config['filter']
            config_lf = config_lf.filter(pl_filter)
            history_lf = history_lf.filter(pl_filter)
            last_analysis_level = current_analysis_level
    # If no filters were provided, assume analyzing ParCa output
    if analysis_type is None:
        analysis_type = 'parca'

    # Run the analyses listed under the most specific filter
    analysis_options = config['analysis_options'][analysis_type]
    for analysis_name, analysis_params in analysis_options.items():
        analysis_mod = importlib.import_module(f'ecoli.analysis.{analysis_name}')
        analysis_mod.plot(
            analysis_params,
            history_lf,
            config_lf,
            config['sim_data_path'],
            config['validation_data_path'])

if __name__ == '__main__':
    main()

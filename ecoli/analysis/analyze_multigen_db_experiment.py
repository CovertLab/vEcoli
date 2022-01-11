import argparse
import os
import pickle

from vivarium.core.emitter import (
    data_from_database,
    DatabaseEmitter,
)
from vivarium.plots.agents_multigen import plot_agents_multigen

from ecoli.composites.ecoli_nonpartition import SIM_DATA_PATH
from ecoli.analysis.analyze_db_experiment import access, OUT_DIR

OUT_DIR = 'out/analysis/'


def main():
    # parse
    parser = argparse.ArgumentParser(
        description='Plot data from multigen experiment.')
    parser.add_argument(
        '--experiment_id', '-e',
        type=str, default='')
    args = parser.parse_args()

    # Retrieve all simulation data.
    data, experiment_id, sim_config = access(args.experiment_id)

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

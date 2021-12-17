import os

import numpy as np
from vivarium.core.composition import simulate_composite, BASE_OUT_DIR
from vivarium.core.control import run_library_cli
from vivarium.plots.agents_multigen import plot_agents_multigen
from ecoli.composites.environment.lattice import (
    Lattice, make_lattice_config)
from ecoli.plots.snapshots import (
    plot_snapshots, format_snapshot_data)
from ecoli.plots.snapshots_video import make_video

from ecoli.composites.antibiotics_grow import GrowDivideAntibioticsCell


OUT_DIR = os.path.join(BASE_OUT_DIR, 'experiments', 'antibiotics_colony')


def get_antibiotics_grow_lattice_composite(
        diffusion_rate=0.001,
        initial_antibiotic_concentration=1e-3,
        bins=[5, 5],
        bounds=[10, 10],
        depth=10,
        growth_rate=np.log(2) / (48 * 60),
        growth_noise=0,
        num_agents=1):
    env_config = make_lattice_config(
        concentrations={
            'antibiotic': initial_antibiotic_concentration,
        },
        n_bins=bins,
        bounds=bounds,
        depth=depth,
        diffusion=diffusion_rate,
    )
    env_composer = Lattice(env_config)
    env_composite = env_composer.generate()

    agent_composer = GrowDivideAntibioticsCell({
        'growth': {
            'default_growth_rate': growth_rate,
            'default_growth_noise': growth_noise,
            'time_step': 1,
        },
        'local_field': {
            'time_step': 1,
        },
    })
    for i in range(num_agents):
        agent_id = f'{i}_agent'
        agent = agent_composer.generate({'agent_id': agent_id})
        env_composite.merge(composite=agent, path=('agents', agent_id))

    return env_composite


def main():
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    bounds = [10, 10]
    composite = get_antibiotics_grow_lattice_composite(
        bounds=bounds)
    sim_settings = {
        'total_time': 48 * 60 * 4 + 10,
        'return_raw_data': True,
        'progress_bar': True,
    }
    data = simulate_composite(composite, sim_settings)

    multigen_plot_settings = {
        'include_paths': [
            ('periplasm', 'concs', 'antibiotic'),
            ('periplasm', 'concs', 'antibiotic_hydrolyzed'),
            ('boundary', 'surface_area'),
            ('boundary', 'length'),
            ('boundary', 'external', 'antibiotic'),
            ('boundary', 'angle'),
        ],
    }
    multigen_fig = plot_agents_multigen(data, multigen_plot_settings)
    multigen_fig.savefig(os.path.join(OUT_DIR, 'multigen.png'))

    # format the data for plot_snapshots
    agents, fields = format_snapshot_data(data)

    # save snapshots figure
    plot_snapshots(
        bounds,
        agents=agents,
        fields=fields,
        n_snapshots=4,
        out_dir=OUT_DIR,
        filename='snapshots'
    )

    # make snapshot video
    make_video(
        data,
        bounds,
        plot_type='fields',
        step=40,  # render every nth snapshot
        out_dir=OUT_DIR,
        filename='snapshots_video',
    )


test_library = {
    '0': main,
}

# run experiments in test_library from the command line with:
# python ecoli/experiments/lattice/antibiotics_colony.py -n [experiment id]
if __name__ == '__main__':
    run_library_cli(test_library)

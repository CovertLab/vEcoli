"""
==================
Colony Simulations
==================
"""
import os

from vivarium.core.composition import simulate_composite
from vivarium.core.control import run_library_cli
from vivarium.library.units import units

# vivarium-multibody imports
from vivarium_multibody.composites.lattice import (
    Lattice, make_lattice_config)
from vivarium_multibody.composites.grow_divide import GrowDivide
from vivarium_multibody.plots.snapshots import (
    plot_snapshots, format_snapshot_data)
from vivarium_multibody.plots.snapshots_video import make_video


OUTDIR = os.path.join('out', 'experiments', 'colony')
if not os.path.exists(OUTDIR):
    os.makedirs(OUTDIR)


def get_lattice_composite(
        diffusion_rate=0.001,
        initial_concentration=1.0,
        bins=[10, 10],
        bounds=[10, 10],
        growth_rate=0.0005,
        growth_noise=10**-6,
        depth=10
):
    """Get a composite with Lattice and GrowDivide agent"""

    lattice_config = make_lattice_config(
            bounds=bounds,
            n_bins=bins,
            concentrations={
                'glc': initial_concentration},
            diffusion=diffusion_rate,
            depth=depth)

    # make the composite
    lattice_composer = Lattice(lattice_config)
    lattice_composite = lattice_composer.generate()

    growth_config = {
        'default_growth_rate': growth_rate,
        'default_growth_noise': growth_noise,
    }
    grow_divide_composer = GrowDivide({
        'agent_id': '0',
        'growth': growth_config,
    })

    agent_id = '0'
    grow_divide_composite = grow_divide_composer.generate(path=('agents', agent_id))
    lattice_composite.merge(composite=grow_divide_composite)

    return lattice_composite


def simulate_grow_divide_lattice(
        lattice_composite,
        total_time=100,
        initial_state=None,
):
    """Run a simulation"""

    agent_id = '0'
    if initial_state is None:
        initial_state = {
            'agents': {
                agent_id: {
                    'global': {
                        'mass': 1000 * units.femtogram}
                }}}

    sim_settings = {
        'total_time': total_time,
        'initial_state': initial_state,
        'return_raw_data': True,
    }
    lattice_grow_divide_data = simulate_composite(
        lattice_composite, sim_settings)

    return lattice_grow_divide_data


def main():
    bounds = [10, 10]

    lattice_composite = get_lattice_composite(
        bounds=bounds
    )

    data = simulate_grow_divide_lattice(
        lattice_composite,
        total_time=4000,
    )

    # format the data for plot_snapshots
    agents, fields = format_snapshot_data(data)

    # save snapshots figure
    plot_snapshots(
        bounds,
        agents=agents,
        fields=fields,
        n_snapshots=4,
        out_dir=OUTDIR,
        filename=f"lattice_snapshots")

    # make snapshot video
    make_video(
        data,
        bounds,
        plot_type='fields',
        step=40,  # render every nth snapshot
        out_dir=OUTDIR,
        filename=f"snapshots_vid",
    )


test_library = {
    '0': main,
}

# run experiments in test_library from the command line with:
# python ecoli/experiments/colony.py -n [experiment id]
if __name__ == '__main__':
    run_library_cli(test_library)

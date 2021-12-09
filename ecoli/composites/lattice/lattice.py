import os
import argparse
import numpy as np

from vivarium.core.composer import Composer
from vivarium.core.engine import Engine
from vivarium.core.composition import (
    COMPOSITE_OUT_DIR,
)
from vivarium.library.dict_utils import deep_merge
from vivarium.library.units import units

# processes
from ecoli.processes.lattice.multibody_physics import (
    Multibody, make_random_position)
from ecoli.processes.lattice.diffusion_field import (
    DiffusionField)
from ecoli.composites.lattice.grow_divide import (
    GrowDivideExchange, GrowDivide)

# plots
from vivarium_multibody.plots.snapshots import (
    format_snapshot_data, get_agent_ids, plot_snapshots,
    DEFAULT_SV)


NAME = 'lattice_environment'


# make a configuration dictionary for the Lattice compartment
def make_lattice_config(
        time_step=None,
        jitter_force=None,
        bounds=None,
        n_bins=None,
        depth=None,
        concentrations=None,
        random_fields=None,
        molecules=None,
        diffusion=None,
        keep_fields_emit=None,
        set_config=None,
        parallel=None,
):
    config = {'multibody': {}, 'diffusion': {}}

    if time_step:
        config['multibody']['time_step'] = time_step
        config['diffusion']['time_step'] = time_step
    if bounds:
        config['multibody']['bounds'] = bounds
        config['diffusion']['bounds'] = bounds
        config['diffusion']['n_bins'] = bounds
    if n_bins:
        config['diffusion']['n_bins'] = n_bins
    if jitter_force:
        config['multibody']['jitter_force'] = jitter_force
    if depth:
        config['diffusion']['depth'] = depth
    if diffusion:
        config['diffusion']['diffusion'] = diffusion
    if concentrations:
        config['diffusion']['gradient'] = {
            'type': 'uniform',
            'molecules': concentrations}
        if random_fields:
            config['diffusion']['gradient']['type'] = 'random'
        molecules = list(concentrations.keys())
        config['diffusion']['molecules'] = molecules
    elif molecules:
        # molecules are a list, assume uniform concentrations of 1
        config['diffusion']['molecules'] = molecules
    if keep_fields_emit:
        # by default no fields are emitted
        config['diffusion']['_schema'] = {
            'fields': {
                field_id: {
                    '_emit': False}
                for field_id in molecules
                if field_id not in keep_fields_emit}}
    if parallel:
        config['diffusion']['_parallel'] = True
        config['multibody']['_parallel'] = True
    if set_config:
        config = deep_merge(config, set_config)

    return config


class Lattice(Composer):
    """
    Lattice:  A two-dimensional lattice environmental model with multibody physics and diffusing molecular fields.
    """

    name = NAME
    defaults = {
        # To exclude a process, from the compartment, set its
        # configuration dictionary to None, e.g. colony_mass_deriver
        'multibody': {
            'bounds': [10, 10],
            'size': [10, 10],
        },
        'diffusion': {
            'molecules': ['glc'],
            'n_bins': [10, 10],
            'size': [10, 10],
            'depth': 3000.0,
            'diffusion': 1e-2,
        },
    }

    def generate_processes(self, config):
        processes = {
            'multibody': Multibody(config['multibody']),
            'diffusion': DiffusionField(config['diffusion'])
        }
        return processes

    def generate_topology(self, config):
        return {
            'multibody': {
                'agents': ('agents',),
            },
            'diffusion': {
                'agents': ('agents',),
                'fields': ('fields',),
                'dimensions': ('dimensions',),
            },
        }


def test_lattice(
        n_agents=1,
        total_time=1000,
        exchange=False,
        external_molecule='X',
        bounds=[25, 25],
        n_bins=None,
        initial_field=None,
        growth_rate=0.05,  # fast growth
        growth_noise=5e-4,
):
    # lattice configuration
    lattice_config_kwargs = {
        'bounds': bounds,
        'n_bins': n_bins or bounds,
        'depth': 2,
        'diffusion': 1e-3,
        'time_step': 60,
        'jitter_force': 1e-5,
        'concentrations': {
            external_molecule: 1.0}
    }
    if initial_field is not None:
        lattice_config_kwargs['concentrations'] = {
            external_molecule: initial_field}
    lattice_config = make_lattice_config(**lattice_config_kwargs)

    # agent configuration
    agent_config = {
        'growth': {
            'growth_rate': growth_rate,
            'default_growth_noise': growth_noise},
        'divide_condition': {
            'threshold': 2500 * units.fg}}
    exchange_config = {
        'exchange': {
            'molecules': [external_molecule]}}

    # lattice composer
    lattice_composer = Lattice(lattice_config)
    # agent composer
    if exchange:
        agent_composer = GrowDivideExchange({
            **agent_config, **exchange_config})
    else:
        agent_composer = GrowDivide(agent_config)

    # make the composite
    lattice_agent_composite = lattice_composer.generate()

    # add agents
    agent_ids = [str(agent_id) for agent_id in range(n_agents)]
    for agent_id in agent_ids:
        agent = agent_composer.generate({'agent_id': agent_id})
        lattice_agent_composite.merge(composite=agent, path=('agents', agent_id))

    # initial state
    initial_state = {
        # 'fields': {
        #     external_molecule: initial_field if (initial_field is not None) else np.ones((n_bins[0], n_bins[1]))},
        'agents': {
            agent_id: {
                'boundary': {
                    'location': make_random_position(bounds),
                    'mass': 1500 * units.fg}
            } for agent_id in agent_ids}}

    # make the experiment
    experiment_config = {
        'processes': lattice_agent_composite.processes,
        'topology': lattice_agent_composite.topology,
        'initial_state': initial_state,
        'progress_bar': True}
    spatial_experiment = Engine(**experiment_config)

    # run the simulation
    spatial_experiment.update(total_time)
    data = spatial_experiment.emitter.get_data_unitless()
    return data


def main():
    out_dir = os.path.join(COMPOSITE_OUT_DIR, NAME)
    os.makedirs(out_dir, exist_ok=True)
    parser = argparse.ArgumentParser(description='lattice composite')
    parser.add_argument('-exchange', '-e', action='store_true', default=False, help='simulate agents with exchange')
    args = parser.parse_args()

    bounds = [25, 25]

    if args.exchange:
        # GrowDivide agents with Exchange
        data = test_lattice(
            exchange=True,
            n_agents=3,
            total_time=4000,
            bounds=bounds)
    else:
        # GrowDivide agents
        n_bins = [20, 20]
        initial_field = np.zeros((n_bins[0], n_bins[1]))
        initial_field[:, -1] = 100
        data = test_lattice(
            n_agents=3,
            total_time=4000,
            bounds=bounds,
            n_bins=n_bins,
            initial_field=initial_field)

    # format the data for plot_snapshots
    agents, fields = format_snapshot_data(data)
    initial_ids = list(data[0]['agents'].keys())
    agent_ids = get_agent_ids(agents)

    # make colors based on initial agents
    agent_colors = {}
    hues = [n/360 for n in [120, 270, 300, 240, 360, 30, 60]]
    for idx, initial_id in enumerate(initial_ids):
        hue = hues[idx]
        color = [hue] + DEFAULT_SV
        for agent_id in agent_ids:
            if agent_id.startswith(initial_id, 0, len(initial_id)):
                agent_colors[agent_id] = color

    plot_snapshots(
        bounds,
        agents=agents,
        fields=fields,
        n_snapshots=4,
        agent_colors=agent_colors,
        out_dir=out_dir,
        filename=f"lattice_snapshots{'_exchange' if args.exchange else ''}")


if __name__ == '__main__':
    main()

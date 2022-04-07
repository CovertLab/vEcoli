import copy

import numpy as np

from vivarium.core.composition import (
    composite_in_experiment, simulate_experiment)
from vivarium.library.units import units, Quantity
from vivarium.library.dict_utils import deep_merge
from vivarium.library.topology import get_in
from vivarium.processes.growth_rate import GrowthRate
from vivarium.processes.divide_condition import DivideCondition
from vivarium.processes.meta_division import MetaDivision
from vivarium.plots.topology import plot_topology
from vivarium.plots.agents_multigen import plot_agents_multigen

from ecoli.composites.environment.lattice import (
    Lattice, make_lattice_config)
from ecoli.processes.environment.derive_globals import DeriveGlobals
from ecoli.plots.snapshots import plot_snapshots

from ecoli.composites.antibiotics_simple import SimpleAntibioticsCell


class GrowDivideAntibioticsCell(SimpleAntibioticsCell):

    defaults = copy.deepcopy(SimpleAntibioticsCell.defaults)
    defaults = deep_merge(defaults, {
        'agent_id': 'agent',
        'boundary_path': ('boundary',),
        'agents_path': ('..', '..', 'agents'),
        'fields_path': ('..', '..', 'fields'),
        'dimensions_path': ('..', '..', 'dimensions'),
        'growth': {
            'variables': ['mass'],
            'default_growth_rate': np.log(2) / 10,
            'time_step': 0.1,
        },
        'divide_condition': {
            'threshold': 1339 * 2 * units.fg,
        },
        'daughter_path': tuple(),
    })

    def generate_processes(self, config):
        processes = super().generate_processes(config)

        division_config = {
            'daughter_path': config['daughter_path'],
            'agent_id': config['agent_id'],
            'composer': self,
            **config.get('division', {})
        }

        added_processes = {
            'growth': GrowthRate(config['growth']),
            'globals_deriver': DeriveGlobals(),
            'divide_condition': DivideCondition(config['divide_condition']),
            'division': MetaDivision(division_config),
        }
        assert set(processes) & set(added_processes) == set()
        processes.update(added_processes)
        return processes

    def generate_topology(self, config):
        topology = super().generate_topology(config)

        boundary_path = config['boundary_path']
        added_topology = {
            'growth': {
                'variables': boundary_path,
                'rates': ('rates',),
            },
            'globals_deriver': {
                'global': boundary_path
            },
            'divide_condition': {
                'variable': boundary_path + ('mass',),
                'divide': boundary_path + ('divide',)
            },
            'division': {
                'global': boundary_path,
                'agents': config['agents_path'],
            },
        }
        assert set(topology) & set(added_topology) == set()
        topology.update(added_topology)
        return topology


def demo():
    env_config = make_lattice_config(
        concentrations={
            'antibiotic': 1e-3,
        },
        diffusion=1,
    )
    env_config['diffusion']['time_step'] = 0.1
    env_config['multibody']['time_step'] = 0.1
    env_composer = Lattice(env_config)
    env_composite = env_composer.generate()

    agent_composer = GrowDivideAntibioticsCell()
    for i in range(2):
        agent_id = f'{i}_agent'
        agent = agent_composer.generate({'agent_id': agent_id})
        env_composite.merge(composite=agent, path=('agents', agent_id))

    plot_settings = {
        'dashed_edges': True,
        'graph_format': 'hierarchy',
        'node_distance': 5,
        'font_size': 10,
    }

    topology_fig = plot_topology(env_composite, plot_settings)

    exp = composite_in_experiment(
        env_composite, initial_state=env_composite.initial_state())
    data = simulate_experiment(
        exp, {'total_time': 25, 'return_raw_data': True})

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

    snapshots_fig = plot_snapshots(
        bounds=get_in(data, (max(data), 'dimensions', 'bounds')),
        agents={
            time: d['agents']
            for time, d in data.items()
        },
        fields={
            time: d['fields']
            for time, d in data.items()
        },
    )

    return topology_fig, multigen_fig, snapshots_fig


# python ecoli/composites/antibiotics_grow.py
if __name__ == '__main__':
    demo()

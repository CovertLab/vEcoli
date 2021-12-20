"""
=====
Lysis
=====
"""
import os
import numpy as np
from scipy import constants

from vivarium.core.process import Step
from vivarium.core.composer import Composer
from vivarium.core.engine import Engine, pf
from vivarium.library.units import units
from vivarium.composites.toys import ToyTransport
from vivarium.processes.timeline import TimelineProcess
from ecoli.composites.lattice.lattice import Lattice
from ecoli.processes.lattice.local_field import LocalField
from ecoli.plots.snapshots import plot_snapshots, format_snapshot_data, get_agent_ids
from ecoli.plots.snapshots_video import make_video


AVOGADRO = constants.N_A


class Lysis(Step):
    defaults = {
        'secreted_molecules': ['GLC'],
    }

    def __init__(self, parameters=None):
        super().__init__(parameters)
        self.agent_id = self.parameters['agent_id']

    def ports_schema(self):
        return {
            'trigger': {
                '_default': False
            },
            'agents': {},
            'internal': {
                mol_id: {
                    '_default': 1
                } for mol_id in self.parameters['secreted_molecules']
            },
            'fields': {
                mol_id: {
                    '_default': np.ones(1),
                } for mol_id in self.parameters['secreted_molecules']
            }
        }

    def next_update(self, timestep, states):
        if states['trigger']:
            # import ipdb; ipdb.set_trace()
            internal = states['internal']

            return {
                'trigger': {
                    '_updater': 'set',
                    '_value': False},
                'agents': {
                    # remove self
                    '_delete': [self.agent_id]
                },
                'fields': {}  # TODO place internal states in fields
            }
        return {}


def mass_from_count(count, mw):
    mol = count / AVOGADRO
    return mw * mol


class MassStep(Step):
    defaults = {
        'molecular_weights': {},
    }

    def __init__(self, parameters=None):
        super().__init__(parameters)
        self.molecular_weights = self.parameters['molecular_weights']

    def ports_schema(self):

        return {
            'molecules': {
                mol_id: {
                    '_default': 0.0,
                } for mol_id in self.molecular_weights.keys()},
            'mass': {
                '_default': 0.0 * units.fg,
                '_updater': 'set'
            },
        }

    def next_update(self, timestep, states):
        # calculate bulk molecule mass
        total_mass = 0.0
        for molecule_id, count in states['molecules'].items():
            if count > 0:
                added_mass = mass_from_count(count, self.molecular_weights.get(molecule_id))
                total_mass += added_mass
        return {
            'mass': total_mass,
        }


class LysisAgent(Composer):
    defaults = {
        'lysis': {},
        'transport': {},
        'mass_step': {
            'molecular_weights': {
                'GLC': 100 * units.fg
            }
        },
        'local_field': {},
        'boundary_path': ('boundary',),
        'fields_path': ('..', '..', 'fields'),
        'dimensions_path': ('..', '..', 'dimensions',),
        'agents_path': ('..', '..', 'agents',),
    }

    def __init__(self, config=None):
        super().__init__(config)

    def generate_processes(self, config):
        return {
            'transport': ToyTransport(config['transport'])
        }

    def generate_steps(self, config):
        assert config['agent_id']
        lysis_config = {
            'agent_id': config['agent_id'],
            **config['lysis']}

        return {
            'local_field': LocalField(config['local_field']),
            'mass_step': MassStep(config['mass_step']),
            'lysis': Lysis(lysis_config),
        }

    def generate_flow(self, config):
        return {
            'local_field': [],
            'mass_step': [],
            'lysis': [('mass_step',),],
        }

    def generate_topology(self, config):
        boundary_path = config['boundary_path']
        fields_path = config['fields_path']
        dimensions_path = config['dimensions_path']
        agents_path = config['agents_path']

        return {
            'transport': {
                'internal': ('internal',),
                'external': boundary_path + ('external',),
            },
            'local_field': {
                'exchanges': boundary_path + ('exchange',),
                'location': boundary_path + ('location',),
                'fields': fields_path,
                'dimensions': dimensions_path,
            },
            'mass_step': {
                'molecules': ('internal',),
                'mass': ('boundary', 'mass')
            },
            'lysis': {
                'trigger': ('boundary', 'death',),
                'internal': ('internal',),
                'agents': agents_path,
                'fields': fields_path,
            },
        }


def test_lysis():

    bounds = [25, 25]
    agent_id = '1'
    agent_path = ('agents', agent_id)
    lattice_composer = Lattice({
        'diffusion': {
            'molecules': ['GLC'],
            'bounds': bounds,
        },
        'multibody': {
            'bounds': bounds,
        }})
    agent_composer = LysisAgent()

    full_composite = lattice_composer.generate()
    agent_composite = agent_composer.generate({'agent_id': agent_id})
    full_composite.merge(composite=agent_composite, path=agent_path)

    # add a timeline process to trigger lysis
    timeline_config = {
        'timeline': [
            (5, {('death',): True}),
        ]
    }
    timeline_process = TimelineProcess(timeline_config)
    full_composite.merge(
        processes={
            'timeline': timeline_process},
        topology={
            'timeline': {
                'death': ('boundary', 'death',),
                'global': ('..', '..', 'global',),
        }},
        path=agent_path
    )

    initial_state = full_composite.initial_state()

    experiment = Engine(
        processes=full_composite.processes,
        steps=full_composite.steps,
        topology=full_composite.topology,
        flow=full_composite.flow,
        initial_state=initial_state,
    )

    experiment.update(10)
    data = experiment.emitter.get_data_unitless()

    # print(pf(data['agents']))
    # format the data for plot_snapshots
    agents, fields = format_snapshot_data(data)
    initial_ids = list(data[0]['agents'].keys())
    agent_ids = get_agent_ids(agents)

    out_dir = os.path.join('out', 'experiments', 'lysis')
    os.makedirs(out_dir, exist_ok=True)
    plot_snapshots(
        bounds,
        agents=agents,
        fields=fields,
        n_snapshots=4,
        # agent_colors=agent_colors,
        out_dir=out_dir,
        filename=f"lysis_snapshots")

    # make snapshot video
    make_video(
        data,
        bounds,
        plot_type='fields',
        # step=40,  # render every nth snapshot
        out_dir=out_dir,
        filename='lysis_video',
    )


# python ecoli/processes/antibiotics/lysis.py
if __name__ == "__main__":
    test_lysis()

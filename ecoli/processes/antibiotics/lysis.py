"""
=====
Lysis
=====
"""
import numpy as np

from vivarium.core.process import Step
from vivarium.core.composer import Composer
from vivarium.core.engine import Engine, pf
from vivarium.composites.toys import ToyTransport
from vivarium.processes.timeline import TimelineProcess
from ecoli.composites.lattice.lattice import Lattice
from ecoli.processes.lattice.local_field import LocalField


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
            import ipdb; ipdb.set_trace()
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



def test_lysis():

    class LysisAgent(Composer):
        defaults = {
            'lysis': {},
            'transport': {},
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
                'lysis': Lysis(lysis_config),
            }

        def generate_flow(self, config):
            return {
                'local_field': (),
                'lysis': (),
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
                'lysis': {
                    'trigger': ('boundary', 'death',),
                    'internal': ('internal',),
                    'agents': agents_path,
                    'fields': fields_path,
                },
            }

    agent_id = '1'
    agent_path = ('agents', agent_id)
    lattice_composer = Lattice({
        'diffusion': {
            'molecules': ['GLC']}
        }
    )
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
    data = experiment.emitter.get_timeseries()

    print(pf(data['agents']))

    import ipdb; ipdb.set_trace()


# python ecoli/processes/antibiotics/lysis.py
if __name__ == "__main__":
    test_lysis()

"""
=====
Lysis
=====
"""

from vivarium.core.process import Step
from vivarium.core.composer import Composer
from vivarium.core.engine import Engine, pf
from vivarium.composites.toys import ToyTransport
from ecoli.composites.lattice.lattice import Lattice
from ecoli.processes.lattice.local_field import LocalField


class Lysis(Step):
    defaults = {}

    def __init__(self, parameters=None):
        super().__init__(parameters)

    def ports_schema(self):
        return {
            'trigger': {},
            'agents': {},
            'fields': {}
        }

    def next_update(self, timestep, states):
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

        def generate_processes(self, config):
            return {
                'transport': ToyTransport(config['transport'])
            }

        def generate_steps(self, config):
            return {
                'local_field': LocalField(config['local_field']),
                'lysis': Lysis(config['lysis']),
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
                    'trigger': ('internal', 'GLC',),
                    'agents': agents_path,
                    'fields': fields_path,
                },
            }

    agent_id = '1'
    lattice_composer = Lattice({
        'diffusion': {
            'molecules': ['GLC']}
        }
    )
    agent_composer = LysisAgent()

    full_composite = lattice_composer.generate()
    agent = agent_composer.generate({'agent_id': agent_id})
    full_composite.merge(composite=agent, path=('agents', agent_id))

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

    # import ipdb; ipdb.set_trace()


# python ecoli/processes/antibiotics/lysis.py
if __name__ == "__main__":
    test_lysis()

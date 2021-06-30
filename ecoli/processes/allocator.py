import random

from vivarium.core.process import Deriver, Process
from vivarium.core.composer import Composer
from vivarium.core.engine import Engine, pf


class Allocate(Deriver):

    defaults = {
        'process_ports': []
    }

    def __init__(self, config):
        super().__init__(config)

    def ports_schema(self):
        return {
            process_port: {}
            for process_port in self.parameters['process_ports']
        }

    def next_update(self, timestep, states):
        return {}



class ToyProcess(Process):
    defaults = {
        'molecules': [],
        'update': {}}
    def __init__(self, config):
        super().__init__(config)
    def ports_schema(self):
        return {
            'molecules': {
                mol_id: {
                    '_default': 0,
                    '_emit': True}
                for mol_id in self.parameters['molecules']}}
    def next_update(self, timestep, states):
        molecules_update = {}
        for mol_id, value in states['molecules'].items():
            molecules_update[mol_id] = max(
                self.parameters['update'].get(mol_id, 0),
                -value)
        return {'molecules': molecules_update}


class ToyComposer(Composer):
    defaults = {
        'molecules': [],
        'supply_update': {},
        'use_update': {},
        'partitioned_update': {},
    }
    def generate_processes(self, config):
        molecules = {'molecules': config['molecules']}
        supply_update = {'update': config['supply_update']}
        use_update = {'update': config['use_update']}
        partitioned_update = {'update': config['partitioned_update']}
        return {
            'toy_supply': ToyProcess({**molecules, **supply_update}),
            'toy_use': ToyProcess({**molecules, **use_update}),
            'toy_partitioned': ToyProcess({**molecules, **partitioned_update}),
            'allocate': Allocate(molecules)
        }
    def generate_topology(self, config):
        return {
            'toy_supply': {
                'molecules': ('bulk',)
            },
            'toy_use': {
                'molecules': ('bulk',)
            },
            'toy_partitioned': {
                'molecules': ('partitioned_bulk',)
            },
            'allocate': {
                'supply': ('bulk',),
                'target': ('partitioned_bulk',),
            }
        }


def test_allocate():

    config = {
        'molecules': [
            'A',
            # 'B',
        ],
        'supply_update': {
            'A': 0,
            # 'B': 3,
        },
        'use_update': {
            'A': -2,
            # 'B': -3,
        },
        'partitioned_update': {
            'A': -3,
            # 'B': -1,
        },
    }
    allocate_composer = ToyComposer(config)
    allocate_composite = allocate_composer.generate()
    initial_state = {
        'bulk': {
            'A': 10,
            'B': 10,
            'C': 10},
        'partitioned_bulk': {
            'A': 0,
            'B': 0,
            'C': 0}
    }

    experiment = Engine({
        'processes': allocate_composite['processes'],
        'topology': allocate_composite['topology'],
        'initial_state': initial_state,
    })

    experiment.update(10)

    timeseries = experiment.emitter.get_timeseries()
    print(pf(timeseries))



if __name__ == "__main__":
    test_allocate()
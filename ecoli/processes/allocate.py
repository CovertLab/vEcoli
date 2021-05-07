import random

from vivarium.core.process import Deriver, Process, Composer
from vivarium.core.experiment import Experiment, pf


class PartitionInt(int):
    """integer that tracks allocated counts

    subclassing int reference: https://stackoverflow.com/questions/3238350/subclassing-int-in-python
    """
    initial = 0
    demand = 0
    added = 0

    def get_remaining(self):
        return self.initial - self.demand

    def reset(self):
        self.initial = int(self)
        self.demand = 0
        self.added = 0

    def make_new(self, value):
        new = self.__class__(value)
        new.initial = self.initial
        new.demand = self.demand
        new.added = self.added
        return new

    def print_info(self):
        return f"{int(self)} " \
               f"(initial: {self.initial} " \
               f"demand: {self.demand} " \
               f"added: {self.added})"

    def __new__(cls, value, *args, **kwargs):
        return super(cls, cls).__new__(cls, value)

    def __add__(self, other):
        value = super().__add__(other)
        if other > 0:
            self.added = other
        else:
            self.demand -= other
        return self.make_new(value)

    def __sub__(self, other):
        value = super().__sub__(other)
        if other < 0:
            self.added -= other
        else:
            self.demand = other
        return self.make_new(value)

    def __mul__(self, other):
        # raise ValueError("PartitionInt can not be multiplied")
        return super().__mul__(other)

    def __truediv__(self, other):
        return super().__truediv__(other)
        # raise ValueError("PartitionInt can not be divided")

    # def __repr__(self):
    #     return f"{int(self)}"


def partition_updater(current_value, update):
    if isinstance(current_value, int) and \
            not isinstance(current_value, PartitionInt):
        value = PartitionInt(current_value)
    else:
        value = current_value

    if isinstance(update, dict):
        update_value = update['value']
        partition = update['partition']
        if partition == 'reset':
            value.reset()
    else:
        update_value = update

    return value + update_value


class Allocate(Deriver):

    defaults = {
        'molecules': []
    }

    def __init__(self, config):
        super().__init__(config)

    def ports_schema(self):
        molecules_schema = {
                mol_id: {
                    '_default': PartitionInt(0),
                    '_updater': partition_updater
                }
                for mol_id in self.parameters['molecules']}
        return {
            'supply': molecules_schema,
            'target': molecules_schema}

    def next_update(self, timestep, states):
        supply = states['supply']
        target = states['target']

        target_update = {}
        supply_update = {}
        for mol_id, value in supply.items():
            supply_remaining = value.get_remaining()
            target_stock = target[mol_id]
            target_demand = target[mol_id].demand

            # allocate to make expected target_stock = 0
            allocate = target_demand - target_stock

            # don't allocate more than is available
            allocate = min(allocate, supply_remaining)

            target_update[mol_id] = {
                'value': allocate,
                'partition': 'reset'}
            supply_update[mol_id] = {
                'value': -allocate,
                'partition': 'reset'}

        import ipdb; ipdb.set_trace()

        return {
            'supply': supply_update,
            'target': target_update}



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
                    '_default': PartitionInt(0),
                    '_updater': partition_updater,
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
            # 'C'
        ],
        'supply_update': {
            'A': 3,
            # 'B': 3,
            # 'C': 4
        },
        'use_update': {
            'A': -2,
            # 'B': -3,
            # 'C': -4
        },
        'partitioned_update': {
            'A': -1,
            # 'B': -1,
            # 'C': -1,
        },
    }
    allocate_composer = ToyComposer(config)
    allocate_composite = allocate_composer.generate()
    initial_state = {
        'bulk': {
            'A': PartitionInt(10),
            'B': PartitionInt(10),
            'C': PartitionInt(10)},
        'partitioned_bulk': {
            'A': PartitionInt(0),
            'B': PartitionInt(0),
            'C': PartitionInt(0)}
    }

    experiment = Experiment({
        'processes': allocate_composite['processes'],
        'topology': allocate_composite['topology'],
        'initial_state': initial_state,
    })

    experiment.update(10)

    timeseries = experiment.emitter.get_timeseries()

    print(pf(timeseries))

    import ipdb; ipdb.set_trace()



if __name__ == "__main__":
    test_allocate()
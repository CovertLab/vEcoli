import random

from vivarium.core.process import Deriver, Process, Composer
from vivarium.core.experiment import Experiment


class PartitionInt(int):
    """integer that tracks allocated counts

    subclassing int reference: https://stackoverflow.com/questions/3238350/subclassing-int-in-python
    """
    allocated = 0

    def get_remaining(self):
        return max(self - self.allocated, 0)

    def get_allocated(self):
        return self.allocated

    def add_to_allocated(self, value):
        self.allocated += value

    def reset_allocated(self):
        self.allocated = 0

    def make_new(self, value):
        new = self.__class__(value)
        new.add_to_allocated(self.allocated)
        return new

    def __new__(cls, value, *args, **kwargs):
        return super(cls, cls).__new__(cls, value)

    def __add__(self, other):
        value = super().__add__(other)
        if other < 0:
            self.allocated -= other
        return self.make_new(value)

    def __sub__(self, other):
        value = super().__sub__(other)
        return self.make_new(value)

    def __mul__(self, other):
        value = super().__mul__(other)
        return self.make_new(value)

    def __div__(self, other):
        value = super().__div__(other)
        return self.make_new(value)


def partition_updater(current_value, update):
    if isinstance(current_value, int) and \
            not isinstance(current_value, PartitionInt):
        value = PartitionInt(current_value)
    else:
        value = current_value

    if update == 'reset_partition':
        value.reset_allocated()
        return value

    return value + update


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
            'allocated': molecules_schema,
        }

    def next_update(self, timestep, states):

        remaining_state = {
                state: value.get_remaining()
                for state, value in states['supply'].items()}

        return {
            'supply': {
                state: 'reset_partition'
                for state in states['supply'].keys()},
            'allocated': remaining_state}



class ToyProcess(Process):
    defaults = {'molecules': []}
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
        return {
            'molecules': {
                mol_id: random.randint(-value, value)
                for mol_id, value in states['molecules'].items()}}


class ToyComposer(Composer):
    defaults = {'molecules': []}
    def generate_processes(self, config):
        molecules = config['molecules']
        return {
            'toy1': ToyProcess({'molecules': molecules}),
            'toy2': ToyProcess({'molecules': molecules}),
            'allocate': Allocate({'molecules': molecules})
        }
    def generate_topology(self, config):
        return {
            'toy1': {
                'molecules': ('bulk',)
            },
            'toy2': {
                'molecules': ('partitioned_bulk',)
            },
            'allocate': {
                'supply': ('bulk',),
                'allocated': ('partitioned_bulk',),
            }
        }


def test_allocate():

    config = {
        'molecules': ['A', 'B', 'C']
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

    data = experiment.emitter.get_data()

    import ipdb; ipdb.set_trace()



if __name__ == "__main__":
    test_allocate()
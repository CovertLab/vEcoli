import random

from vivarium.core.process import Deriver, Process, Composer
from vivarium.core.experiment import Experiment



class PartitionedCount(int):
    partition = 0


def partition_updater(current_value, update):
    if isinstance(current_value, int) and \
            not isinstance(current_value, PartitionedCount):
        value = PartitionedCount(current_value)
    else:
        value = current_value

    if update == 'reset_partition':
        value.partition = 0
        return value
    elif update < 0:
        value.partition += update

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
                    '_default': PartitionedCount(0),
                    '_updater': partition_updater
                }
                for mol_id in self.parameters['molecules']}
        return {
            'supply': molecules_schema,
            'allocated': molecules_schema,
        }

    def next_update(self, timestep, states):

        allocated_update = {
                state: value + value.partition
                for state, value in states['supply'].items()}


        partition_state = {
                state: value.partition
                for state, value in states['supply'].items()}
        print(f'STATES: {states}')
        print(f'PARTITION: {partition_state}')
        print(f'ALLOCATE: {allocated_update}')


        import ipdb;
        ipdb.set_trace()


        return {
            'supply': {
                state: 'reset_partition'
                for state in states['supply'].keys()},
            'allocated': allocated_update
        }



class ToyProcess(Process):
    defaults = {'molecules': []}
    def __init__(self, config):
        super().__init__(config)
    def ports_schema(self):
        return {
            'molecules': {
                mol_id: {
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
            'A': PartitionedCount(10),
            'B': PartitionedCount(10),
            'C': PartitionedCount(10)},
        'partitioned_bulk': {
            'A': PartitionedCount(0),
            'B': PartitionedCount(0),
            'C': PartitionedCount(0)}
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
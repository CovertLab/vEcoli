import random

from vivarium.core.process import Deriver, Process, Composer
from vivarium.core.experiment import Experiment



class PartitionedCount(int):
    partition = 0


def partition_updater(current_value, update):
    if isinstance(current_value, int) and \
            not isinstance(current_value, PartitionedCount):
        value = PartitionedCount(current_value)
    elif isinstance(current_value, PartitionedCount):
        value = current_value


    import ipdb; ipdb.set_trace()


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
            'demand': molecules_schema,
            'allocated': molecules_schema,
        }

    def next_update(self, timestep, states):

        # meet last request with all that is available in supply.
        # TODO -- don't give all of the demand, just what is available
        update = {
            'supply': {
                state: -value
                for state, value in states['demand'].items()},
            'demand': {
                state: {
                    '_updater': 'set',
                    '_value': 0}
                for state in self.parameters['molecules']},
            'allocated': states['demand'],
        }

        import ipdb;
        ipdb.set_trace()

        return update


class ToyProcess(Process):
    defaults = {'molecules': []}
    def __init__(self, config):
        super().__init__(config)
    def ports_schema(self):
        return {
            'molecules': {
                mol_id: {
                    '_default': 0}
                for mol_id in self.parameters['molecules']}}
    def next_update(self, timestep, states):
        return {
            'molecules': {
                mol_id: random.randint(-10, 10)
                for mol_id in self.parameters['molecules']}}


class ToyComposer(Composer):
    defaults = {'molecules': []}
    def generate_processes(self, config):
        molecules = config['molecules']
        return {
            'toy': ToyProcess({'molecules': molecules}),
            'allocate': Allocate({'molecules': molecules})
        }
    def generate_topology(self, config):
        return {
            'toy': {
                'molecules': ('bulk',)
            },
            'allocate': {
                'supply': ('bulk',),
                'demand': ('requested_bulk',),
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
            'A': PartitionedCount(1),
            'B': PartitionedCount(2),
            'C': PartitionedCount(3)}
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
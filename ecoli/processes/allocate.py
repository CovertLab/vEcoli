from vivarium.core.process import Deriver

class Allocate(Deriver):

    defaults = {
        'molecules': []
    }

    def __init__(self, config):
        super().__init__(config)

    def ports_schema(self):
        molecules_schema = {
                mol_id: {}
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
                    '_update': 'set',
                    '_value': 0}
                for state in states['request'].keys()},
            'allocated': states['demand'],
        }

        import ipdb;
        ipdb.set_trace()

        return update

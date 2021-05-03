from vivarium.core.process import Deriver

class Allocate(Deriver):

    defaults = {
        'molecules': []
    }

    def __init__(self, config):
        super().__init__(config)

    def ports_schema(self):
        return {
            'supply': {
                mol_id: {}
                for mol_id in self.parameters['molecules']},
            'request': {
                mol_id: {}
                for mol_id in self.parameters['molecules']}}

    def next_update(self, timestep, states):

        # meet last request with all that is available in supply.
        update = {
            'request': {
                state: -value
                for state, value in states['request'].items()},
            'supply': {
                state: -value
                for state, value in states['request'].items()}}

        return update

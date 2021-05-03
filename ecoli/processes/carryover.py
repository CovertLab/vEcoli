from vivarium.core.process import Deriver

class CarryOver(Deriver):
    defaults = {
        'molecules': []
    }
    def __init__(self, config):
        super.__init__(config)
    def ports_schema(self):
        return {
            'source': {
                mol_id: {}
                for mol_id in self.parameters['molecules']},
            'target': {
                mol_id: {}
                for mol_id in self.parameters['molecules']}
        }
    def next_update(self, timestep, states):
        return {
            'target': states['source'],
            'source': {
                state: -value
                for state, value in states['source'].items()}
        }
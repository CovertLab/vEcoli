from vivarium.core.process import Process


class Chemostat(Process):

    defaults = {
        # Map from variable names to the values (must support
        # subtraction) those variables should be held at.
        'targets': {},
    }
    name = 'chemostat'

    def ports_schema(self):
        schema = {
            variable: {
                '_default': target * 0,
            }
            for variable, target in self.parameters['targets'].items()
        }
        return schema

    def next_update(self, timestep, state):
        targets = self.parameters['targets']
        update = {
            variable: {
                '_value': targets[variable] - current,
                '_updater': 'accumulate',
            }
            for variable, current in state.items()
        }
        return update

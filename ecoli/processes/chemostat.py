from vivarium.core.process import Process


class Chemostat(Process):
    defaults = {
        # Map from variable names to the values (must support
        # subtraction) those variables should be held at.
        "targets": {},
        "delay": 0,
    }
    name = "chemostat"

    def __init__(self, parameters=None):
        super().__init__(parameters)
        self.seconds_to_wait = self.parameters["delay"]

    def ports_schema(self):
        schema = {
            variable: {
                "_default": target * 0,
            }
            for variable, target in self.parameters["targets"].items()
        }
        return schema

    def next_update(self, timestep, state):
        if self.seconds_to_wait > 0:
            self.seconds_to_wait -= timestep
            return {}

        targets = self.parameters["targets"]
        update = {
            variable: {
                "_value": targets[variable] - current,
                "_updater": "accumulate",
            }
            for variable, current in state.items()
        }
        return update

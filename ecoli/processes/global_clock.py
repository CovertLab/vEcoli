from vivarium.core.process import Process

class GlobalClock(Process):
    """
    A process for tracking global time
    """
    name = 'global_clock'
    defaults = {
        'time_step': 1.0}

    def ports_schema(self):
        return {
            'global_time': {
                '_default': 0.0,
                '_updater': 'accumulate'},
            'timestep': {'_default': self.parameters['time_step']}}
    
    def calculate_timestep(self, states):
        return states['timestep']

    def next_update(self, timestep, states):
        return {'global_time': timestep}

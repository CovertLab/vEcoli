import abc

from vivarium.core.process import Process
from vivarium.library.dict_utils import deep_merge
from ecoli.processes.registries import topology_registry


class EcoliProcess(Process):
    name = None
    topology = None
    defaults = {
        'evolve_only': False,
        'request_only': False,
    }

    def __init__(self, parameters=None):
        super().__init__(parameters)
        assert self.name
        assert self.topology

        topology_registry.register(self.name, self.topology)

    @abc.abstractmethod
    def ports_schema(self):
        return {}

    @abc.abstractmethod
    def calculate_request(self, timestep, states):
        return {}

    @abc.abstractmethod
    def evolve_state(self, timestep, states):
        return {}

    def next_update(self, timestep, states):

        if self.parameters['request_only']:
            return self.calculate_request(timestep, states)
        elif self.parameters['evolve_only']:
            return self.evolve_state(timestep, states)

        requests = self.calculate_request(timestep, states)
        states = deep_merge(states, requests)
        update = self.evolve_state(timestep, states)

        return update

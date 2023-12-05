"""
======================
Partitioning Processes
======================

This bundle of processes includes Requester, Evolver, and PartitionedProcess.
PartitionedProcess is the inherited base class for all Processes that can be
partitioned; these processes provide calculate_request or evolve_state methods,
rather than the usual Process next_update.

A PartitionedProcess can be passed into a Requester and Evolver, which call its
calculate_request and evolve_state methods in coordination with an Allocator process,
which reads the requests and allocates molecular counts for the evolve_state.

"""
import abc

from vivarium.core.process import Step, Process
from vivarium.library.dict_utils import deep_merge

from ecoli.processes.registries import topology_registry

class Requester(Step):
    """ Requester Step

    Accepts a PartitionedProcess as an input, and runs in coordination with an
    Evolver that uses the same PartitionedProcess.
    """
    defaults = {'process': None}

    def __init__(self, parameters=None):
        assert isinstance(parameters["process"], PartitionedProcess)
        if parameters["process"].parallel:
            raise RuntimeError(
                'PartitionedProcess objects cannot be parallelized.')
        parameters['name'] = f'{parameters["process"].name}_requester'
        self.last_update_time = 0
        super().__init__(parameters)

    def update_condition(self, timestep, states):
        return (self.last_update_time + states['timestep']
                ) == states['global_time']

    def ports_schema(self):
        process = self.parameters.get('process')
        ports = process.get_schema()
        ports['request'] = {
            'bulk': {
                '_updater': 'set',
                '_divider': 'null',
                '_emit': False,
            }
        }
        ports['process'] = {
            '_default': tuple(),
            '_updater': 'set',
            '_divider': 'null',
            '_emit': False
        }
        ports['global_time'] = {'_default': 0}
        ports['timestep'] = {'_default': process.parameters['time_step']}
        ports['first_update'] = {
            '_default': True,
            '_updater': 'set',
            '_divider': {'divider': 'set_value',
                'config': {'value': True}}}
        self.cached_bulk_ports = list(ports['request'].keys())
        return ports

    def next_update(self, timestep, states):
        self.last_update_time = states['global_time']

        process = states['process'][0]
        request = process.calculate_request(
            self.parameters['time_step'], states)
        process.request_set = True

        request['request'] = {}
        # Send bulk requests through request port
        for bulk_port in self.cached_bulk_ports:
            bulk_request = request.pop(bulk_port, None)
            if bulk_request != None:
                request['request'][bulk_port] = bulk_request

        # Ensure listeners are updated if present
        listeners = request.pop('listeners', None)
        if listeners != None:
            request['listeners'] = listeners

        # Update shared process instance
        request['process'] = (process,)
        # Leave rest of update untouched
        return request


class Evolver(Step):
    """ Evolver Step

    Accepts a PartitionedProcess as an input, and runs in coordination with an
    Requester that uses the same PartitionedProcess.
    """
    defaults = {'process': None}

    def __init__(self, parameters=None):
        assert isinstance(parameters["process"], PartitionedProcess)
        parameters['name'] = f'{parameters["process"].name}_evolver'
        self.last_update_time = 0
        super().__init__(parameters)
    
    def update_condition(self, timestep, states):
        return (self.last_update_time + states['timestep']
                ) == states['global_time']

    def ports_schema(self):
        process = self.parameters.get('process')
        ports = process.get_schema()
        ports['allocate'] = {
            'bulk': {
                '_updater': 'set',
                '_divider': 'null',
                '_emit': False,
            }
        }
        ports['process'] = {
            '_default': tuple(),
            '_updater': 'set',
            '_divider': 'null',
            '_emit': False
        }
        ports['global_time'] = {'_default': 0}
        ports['timestep'] = {'_default': process.parameters['timestep']}
        ports['first_update'] = {
            '_default': True,
            '_updater': 'set',
            '_divider': {'divider': 'set_value',
                'config': {'value': True}}}
        return ports

    # TODO(Matt): Have evolvers calculate timestep, returning zero if the requester hasn't run.
    # def calculate_timestep(self, states):
    #     if not self.process.request_set:
    #         return 0
    #     else:
    #         return self.process.calculate_timestep(states)

    def next_update(self, timestep, states):
        self.last_update_time = states['global_time']

        allocations = states.pop('allocate')
        states = deep_merge(states, allocations)
        process = states['process'][0]

        # If the Requester has not run yet, skip the Evolver's update to
        # let the Requester run in the next time step. This problem
        # often arises after division because after the step divider
        # runs, Vivarium wants to run the Evolvers instead of re-running
        # the Requesters. Skipping the Evolvers in this case means our
        # timesteps are slightly off. However, the alternative is to run
        # self.process.calculate_request and discard the result before
        # running the Evolver this timestep, which means we skip the
        # Allocator. Skipping the Allocator can cause the simulation to
        # crash, so having a slightly off timestep is preferable.
        if not process.request_set:
            return {}

        update = process.evolve_state(timestep, states)
        update['process'] = (process,)
        return update


class PartitionedProcess(Process):
    """ Partitioned Process Base Class

    This is the base class for all processes whose updates can be partitioned.
    """

    def __init__(self, parameters=None):
        super().__init__(parameters)

        # set partition mode
        self.evolve_only = self.parameters.get('evolve_only', False)
        self.request_only = self.parameters.get('request_only', False)
        self.request_set = False

        # register topology
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
        if self.request_only:
            return self.calculate_request(timestep, states)
        if self.evolve_only:
            return self.evolve_state(timestep, states)

        requests = self.calculate_request(timestep, states)
        bulk_requests = requests.pop('bulk', [])
        if bulk_requests:
            bulk_copy = states['bulk'].copy()
            for bulk_idx, request in bulk_requests:
                bulk_copy[bulk_idx] = request
            states['bulk'] = bulk_copy
        states = deep_merge(states, requests)
        update = self.evolve_state(timestep, states)
        if 'listeners' in requests:
            update['listeners'] = deep_merge(update['listeners'], requests['listeners'])
        return update

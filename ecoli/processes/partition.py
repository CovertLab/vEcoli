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

def filter_bulk_ports(schema, update=None):
    """Retrieve only ports for bulk molecules and modify if desired.

    Args:
        schema (Dict): The ports schema to change
        update (Dict): Dictionary of new attributes to apply
            to all bulk ports (e.g. {'_updater': 'set'}).

    Returns:
        Dict: Ports schema that only includes bulk molecules
        with the new schemas.
    """
    # All bulk ports will have {'bulk': True} somewhere in schema
    if schema.get('bulk', False) == True:
        # Do not modify input schema
        schema = dict(schema)
        if update:
            schema.update(update)
        return schema
    filtered = {}
    for k, v in schema.items():
        if isinstance(v, dict):
            sub_filtered = filter_bulk_ports(v)
            if sub_filtered:
                filtered[k] = sub_filtered
    return filtered


def filter_bulk_topology(topology):
    """Retrieve only topology for partitioned bulk molecules.
    Assumes all ports with '_total' in name are not partitioned.

    Args:
        topology (Dict): The topology to filter

    Returns:
        Dict: Topology that only includes bulk molecules
        with the new schemas.
    """
    filtered = {}
    for k, v in topology.items():
        if '_total' in k:
            continue
        if isinstance(v, dict):
            sub_filtered = filter_bulk_topology(v)
            if sub_filtered:
                filtered[k] = sub_filtered
        elif isinstance(v, tuple) and 'bulk' in v:
            filtered[k] = v
    return filtered


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
        super().__init__(parameters)

    def ports_schema(self):
        ports = self.parameters.pop('process').get_schema()
        ports['request'] = filter_bulk_ports(ports,
            {'_updater': 'set', '_divider': 'null', '_emit': False})
        ports['evolvers_ran'] = {'_default': True}
        ports['process'] = {
            '_default': tuple(),
            '_updater': 'set',
            '_divider': 'null',
            '_emit': False
        }
        return ports

    def next_update(self, timestep, states):
        process = states['process'][0]
        request = process.calculate_request(
            self.parameters['time_step'], states)
        process.request_set = True

        # Ensure listeners are updated if passed by calculate_request
        listeners = request.pop('listeners', None)
        update = {
            'request': request,
            'process': (process,)
        }
        if listeners != None:
            update['listeners'] = listeners

        return update

    def update_condition(self, timestep, states):
        return states['evolvers_ran']


class Evolver(Step):
    """ Evolver Step

    Accepts a PartitionedProcess as an input, and runs in coordination with an
    Requester that uses the same PartitionedProcess.
    """
    defaults = {'process': None}

    def __init__(self, parameters=None):
        assert isinstance(parameters["process"], PartitionedProcess)
        parameters['name'] = f'{parameters["process"].name}_evolver'
        super().__init__(parameters)

    def ports_schema(self):
        ports = self.parameters.pop('process').get_schema()
        ports['allocate'] = filter_bulk_ports(ports,
            {'_updater': 'set', '_divider': 'null', '_emit': False})
        ports['evolvers_ran'] = {
            '_default': True,
            '_updater': 'set',
            '_divider': {
                'divider': 'set_value',
                'config': {
                    'value': True,
                },
            },
            '_emit': False,
        }
        ports['process'] = {
            '_default': tuple(),
            '_updater': 'set',
            '_divider': 'null',
            '_emit': False
        }
        return ports

    # TODO(Matt): Have evolvers calculate timestep, returning zero if the requester hasn't run.
    # def calculate_timestep(self, states):
    #     if not self.process.request_set:
    #         return 0
    #     else:
    #         return self.process.calculate_timestep(states)

    def next_update(self, timestep, states):
        allocations = states.pop('allocate')
        states = deep_merge(states, allocations)
        process = states['process'][0]

        # If the Requester has not run yet, skip the Evolver's update to
        # let the Requester run in the next time step. This problem
        # often arises fater division because after the step divider
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
        update['evolvers_ran'] = True
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

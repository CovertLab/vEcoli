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

from vivarium.core.process import Deriver, Process
from vivarium.library.dict_utils import deep_merge
from ecoli.processes.registries import topology_registry


def change_bulk_updater(schema, new_updater):
    """Retrieve port schemas for all bulk molecules
    and modify their updater

    Args:
        schema (Dict): The ports schema to change
        new_updater (String): The new updater to use

    Returns:
        Dict: Ports schema that only includes bulk molecules
        with the new updater
    """
    bulk_schema = {}
    if '_properties' in schema:
        if schema['_properties']['bulk']:
            topo_copy = schema.copy()
            topo_copy.update({'_updater': new_updater, '_emit': False})
            return topo_copy
    for port, value in schema.items():
        if has_bulk_property(value):
            bulk_schema[port] = change_bulk_updater(value, new_updater)
    return bulk_schema


def has_bulk_property(schema):
    """Check to see if a subset of the ports schema contains
    a bulk molecule using {'_property': {'bulk': True}}

    Args:
        schema (Dict): Subset of ports schema to check for bulk

    Returns:
        Bool: Whether the subset contains a bulk molecule
    """
    if isinstance(schema, dict):
        if '_properties' in schema:
            if schema['_properties']['bulk']:
                return True

        for value in schema.values():
            if isinstance(value, dict):
                if has_bulk_property(value):
                    return True
    return False


def get_bulk_topo(topo):
    """Return topology of only bulk molecules, excluding stores with
    '_total' in name (for non-partitioned counts)
    NOTE: Does not work with '_path' key

    Args:
        topo (Dict): Experiment topology

    Returns:
        Dict: Experiment topology with non-bulk stores excluded
    """
    if 'bulk' in topo:
        return topo
    if isinstance(topo, dict):
        bulk_topo = {}
        for port, value in topo.items():
            if path_in_bulk(value) and '_total' not in port:
                bulk_topo[port] = get_bulk_topo(value)
    return bulk_topo


def path_in_bulk(topo):
    """Check whether a subset of the topology is contained within
    the bulk store

    Args:
        topo (Dict): Subset of experiment topology

    Returns:
        Bool: Whether subset contains stores listed under 'bulk'
    """
    if 'bulk' in topo:
        return True
    if isinstance(topo, dict):
        for value in topo.values():
            if path_in_bulk(value):
                return True
    return False


class Requester(Deriver):
    """ Requester Deriver

    Accepts a PartitionedProcess as an input, and runs in coordination with an
    Evolver that uses the same PartitionedProcess.
    """
    defaults = {'process': None}

    def __init__(self, parameters=None):
        super().__init__(parameters)
        self.process = self.parameters['process']
        assert isinstance(self.process, PartitionedProcess)

    def ports_schema(self):
        ports = self.process.get_schema()
        ports_copy = ports.copy()
        ports['request'] = change_bulk_updater(ports_copy, 'set')
        return ports

    def next_update(self, timestep, states):
        update = self.process.calculate_request(
            self.parameters['time_step'], states)
        if not self.process.request_set:
            self.process.request_set = True

        # Ensure listeners are updated if passed by calculate_request
        listeners = update.pop('listeners', None)
        if listeners != None:
            return {'request': update, 'listeners': listeners}

        return {'request': update}


class Evolver(Process):
    """ Evolver Process

    Accepts a PartitionedProcess as an input, and runs in coordination with an
    Requester that uses the same PartitionedProcess.
    """
    defaults = {'process': None}

    def __init__(self, parameters=None):
        super().__init__(parameters)
        self.process = self.parameters['process']
        assert isinstance(self.process, PartitionedProcess)

    def ports_schema(self):
        ports = self.process.get_schema()
        ports_copy = ports.copy()
        ports['allocate'] = change_bulk_updater(ports_copy, 'set')
        return ports

    # try later
    # def calculate_timestep(self, states):
    #     if not self.process.request_set:
    #         return 0
    #     else:
    #         return self.process.calculate_timestep(states)

    def next_update(self, timestep, states):
        states = deep_merge(states, states.pop('allocate'))

        # run request if it has not yet run
        if not self.process.request_set:
            return {}
        else:
            return self.process.evolve_state(timestep, states)


class PartitionedProcess(Process):
    """ Partitioned Process Base Class

    This is the base class for all processes whose updates can be partitioned.
    """
    name = None
    topology = None
    request_set = False

    def __init__(self, parameters=None):
        super().__init__(parameters)

        # set partition mode
        self.evolve_only = self.parameters.get('evolve_only', False)
        self.request_only = self.parameters.get('request_only', False)

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
        elif self.evolve_only:
            return self.evolve_state(timestep, states)

        requests = self.calculate_request(timestep, states)
        states = deep_merge(states, requests)
        update = self.evolve_state(timestep, states)
        if 'listeners' in requests:
            update['listeners'] = deep_merge(update['listeners'], requests['listeners'])
        return update

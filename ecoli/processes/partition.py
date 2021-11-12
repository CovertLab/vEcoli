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
import os
import pickle

from vivarium.core.process import Deriver, Process
from vivarium.core.registry import divider_registry
from vivarium.library.dict_utils import deep_merge
from ecoli.processes.registries import topology_registry


def divide_set_none(_):
    return [None, None]


divider_registry.register('set_none', divide_set_none)


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
        self.process = parameters['process']
        assert isinstance(self.process, PartitionedProcess)
        if self.process.parallel:
            parameters['_parallel'] = True
        parameters['name'] = f'{self.process.name}_requester'
        super().__init__(parameters)

    def ports_schema(self):
        ports = self.process.get_schema()
        ports_copy = ports.copy()
        ports['request'] = change_bulk_updater(ports_copy, 'set')
        ports['hidden_state'] = {
            self.process.name: {
                '_default': None,
                '_updater': 'set',
                '_emit': False,
                '_divider': 'set_none',
            },
        }
        return ports

    def next_update(self, timestep, states):
        if self.process.parallel:
            hidden_state = states.pop('hidden_state')
            serialized = hidden_state[self.process.name]
            # If the simulation is brand-new, there might not be a
            # serialized process in the store yet.
            if serialized:
                partitioning_hidden_state = pickle.loads(serialized)
                self.process.set_partitioning_hidden_state(
                    partitioning_hidden_state)

        request = self.process.calculate_request(
            self.parameters['time_step'], states)
        self.process.request_set = True

        # Ensure listeners are updated if passed by calculate_request
        listeners = request.pop('listeners', None)
        update = {
            'request': request,
        }
        if listeners != None:
            update['listeners'] = listeners

        if self.process.parallel:
            update['hidden_state'] = {
                self.process.name: pickle.dumps(
                    self.process.get_partitioning_hidden_state())
            }

        return update


class Evolver(Process):
    """ Evolver Process

    Accepts a PartitionedProcess as an input, and runs in coordination with an
    Requester that uses the same PartitionedProcess.
    """
    defaults = {'process': None}

    def __init__(self, parameters=None):
        self.process = parameters['process']
        assert isinstance(self.process, PartitionedProcess)
        if self.process.parallel:
            parameters['_parallel'] = True
        parameters['name'] = f'{self.process.name}_evolver'
        super().__init__(parameters)

    def ports_schema(self):
        ports = self.process.get_schema()
        ports_copy = ports.copy()
        ports['allocate'] = change_bulk_updater(ports_copy, 'set')
        ports['hidden_state'] = {
            self.process.name: {
                '_default': None,
                '_updater': 'set',
                '_emit': False,
                '_divider': 'set_none',
            },
        }
        return ports

    def next_update(self, timestep, states):
        if self.process.parallel:
            hidden_state = states.pop('hidden_state')
            partitioning_hidden_state = pickle.loads(
                hidden_state[self.process.name])
            self.process.set_partitioning_hidden_state(
                partitioning_hidden_state)

        states = deep_merge(states, states.pop('allocate'))

        # run request if it has not yet run
        if not self.process.request_set:
            _ = self.process.calculate_request(timestep, states)
            self.process.request_set = True

        update = self.process.evolve_state(timestep, states)
        if self.process.parallel:
            update['hidden_state'] = {
                self.process.name: pickle.dumps(
                    self.process.get_partitioning_hidden_state())
            }
        return update


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

    def get_partitioning_hidden_state(self):
        '''Returns a dictionary with the hidden state for partitioning.

        The returned dictionary should be as small as possible and contain
        only those variables that need to be passed between
        :py:class:`Evolver` and :py:class:`Requester` instances since
        serializing this data is expensive.

        Returns:
            The hidden state, as a dictionary. Each key-value pair should in
            general store a single instance variable's value as the value
            and the variable's name as the key. By default, this format is
            used, with the variable names coming from
            ``self.parameters['hidden_state_instance_variables']``. However,
            the only real requirement is that the class's
            :py:meth:`set_partitioning_hidden_state` method know how to
            correctly apply the state.
        '''
        variables = self.parameters.get(
            'partitioning_hidden_state_instance_variables', [])
        variables.append('request_set')
        return {var: getattr(self, var) for var in variables}

    def set_partitioning_hidden_state(self, state):
        '''Set the hidden state for partitioning.

        This method simply updates ``self.__dict__`` with the contents of
        ``state``, which should work for many processes. However, subclasses
        can also override this method if needed.

        Args:
            state: The state dictionary from
                :py:meth:`get_partitioning_hidden_state`.
        '''
        self.__dict__.update(state)

    def next_update(self, timestep, states):
        if self.request_only:
            return self.calculate_request(timestep, states)
        if self.evolve_only:
            return self.evolve_state(timestep, states)

        requests = self.calculate_request(timestep, states)
        states = deep_merge(states, requests)
        update = self.evolve_state(timestep, states)
        if 'listeners' in requests:
            update['listeners'] = deep_merge(update['listeners'], requests['listeners'])
        return update

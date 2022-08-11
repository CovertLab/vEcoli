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
import copy
import os
import pickle

import numpy as np
from vivarium.core.process import Step, Process
from vivarium.library.dict_utils import deep_merge, make_path_dict

from ecoli.processes.registries import topology_registry
from ecoli.library.convert_update import convert_numpy_to_builtins


def check_whether_evolvers_have_run(evolvers_ran, proc_name):
    return evolvers_ran


def change_bulk_schema(
        schema, new_updater='', new_divider='', new_emit=False):
    """Retrieve and modify port schemas for all bulk molecules.

    Args:
        schema (Dict): The ports schema to change
        new_updater (String): The new updater to use. Updater is
            unchanged if this is an empty string.
        new_divider (String): The new divider to use. Divider is
            unchanged if this is an empty string.
        new_emit (String): The new emitter to use. False by default.

    Returns:
        Dict: Ports schema that only includes bulk molecules
        with the new schemas.
    """
    bulk_schema = {}
    schema_updates = {
        '_emit': new_emit,
    }
    if new_updater:
        schema_updates['_updater'] = new_updater
    if new_divider:
        schema_updates['_divider'] = new_divider
    if '_properties' in schema:
        if schema['_properties']['bulk']:
            topo_copy = schema.copy()
            topo_copy.update(schema_updates)
            return topo_copy
    for port, value in schema.items():
        if has_bulk_property(value):
            bulk_schema[port] = change_bulk_schema(
                value, new_updater, new_divider, new_emit)
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


class Requester(Step):
    """ Requester Step

    Accepts a PartitionedProcess as an input, and runs in coordination with an
    Evolver that uses the same PartitionedProcess.
    """
    defaults = {'process': None}

    def __init__(self, parameters=None):
        self.process = parameters['process']
        assert isinstance(self.process, PartitionedProcess)
        if self.process.parallel:
            raise RuntimeError(
                'PartitionedProcess objects cannot be parallelized.')
        parameters['name'] = f'{self.process.name}_requester'
        super().__init__(parameters)

    def ports_schema(self):
        ports = self.process.get_schema()
        ports_copy = ports.copy()
        ports['request'] = change_bulk_schema(
            ports_copy, new_updater='set', new_divider='null')
        ports['evolvers_ran'] = {'_default': True}
        return ports

    def next_update(self, timestep, states):
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

        return convert_numpy_to_builtins(update)

    def update_condition(self, timestep, states):
        return check_whether_evolvers_have_run(
            states['evolvers_ran'], self.name)

    def __getstate__(self) -> dict:
        """Return parameters

        Unlike in vivarium.core.process.Process, here we return a
        parameters dict that includes ``self.process`` instead of a
        copy. This ensures that pickle will notice that the Requester
        and Evolver in a pair share a process instance and will preserve
        that shared object upon deserialization.
        """
        # Shallow copy since we just want to avoid changing
        # `super()._original_parameters['process'].
        state = copy.copy(super().__getstate__())
        state['process'] = self.process
        return state


class Evolver(Process):
    """ Evolver Process

    Accepts a PartitionedProcess as an input, and runs in coordination with an
    Requester that uses the same PartitionedProcess.
    """
    defaults = {'process': None}

    def __init__(self, parameters=None):
        self.process = parameters['process']
        assert isinstance(self.process, PartitionedProcess)
        parameters['name'] = f'{self.process.name}_evolver'
        super().__init__(parameters)

    def ports_schema(self):
        ports = self.process.get_schema()
        ports_copy = ports.copy()
        ports['allocate'] = change_bulk_schema(
            ports_copy, new_updater='set', new_divider='null')
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
        return ports

    # TODO(Matt): Have evolvers calculate timestep, returning zero if the requester hasn't run.
    # def calculate_timestep(self, states):
    #     if not self.process.request_set:
    #         return 0
    #     else:
    #         return self.process.calculate_timestep(states)

    def next_update(self, timestep, states):
        allocations = states.pop('allocate')
        allocated_molecules = list(allocations.keys())
        states = deep_merge(states, allocations)

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
        if not self.process.request_set:
            return {}

        update = copy.deepcopy(
            self.process.evolve_state(timestep, states))
        update['evolvers_ran'] = True
        return convert_numpy_to_builtins(update)

    def __getstate__(self) -> dict:
        """Return parameters

        Unlike in vivarium.core.process.Process, here we return a
        parameters dict that includes ``self.process`` instead of a
        copy. This ensures that pickle will notice that the Requester
        and Evolver in a pair share a process instance and will preserve
        that shared object upon deserialization.
        """
        # Shallow copy since we just want to avoid changing
        # `super()._original_parameters['process'].
        state = copy.copy(super().__getstate__())
        state['process'] = self.process
        return state


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
        states = deep_merge(states, requests)
        update = self.evolve_state(timestep, states)
        if 'listeners' in requests:
            update['listeners'] = deep_merge(update['listeners'], requests['listeners'])
        return update

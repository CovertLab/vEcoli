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
import warnings

import numpy as np

from ecoli.library.bigraph_bridge import BigraphStep as Step, BigraphProcess as Process
from ecoli.library.schema_types import UNIQUE_TYPES
from vivarium.library.dict_utils import deep_merge

from ecoli.processes.registries import topology_registry


def is_partitioned_process(obj):
    """Return True if ``obj`` (a class or instance) is a PartitionedProcess.

    Robust to module-identity mismatches. When an external harness purges and
    re-imports the ``ecoli`` package (e.g. to switch between checkouts), two
    distinct ``PartitionedProcess`` class objects can coexist: one held by a
    stale process registry and one freshly imported. ``isinstance`` /
    ``issubclass`` against the freshly-imported base then returns False even
    though the class genuinely IS a PartitionedProcess — which would silently
    skip the Requester/Evolver/Allocator split and let processes run unified
    (over-consuming shared bulk counts → negative counts → downstream NaNs).
    Comparing by MRO class name instead keeps the split intact regardless of
    which module object a given class was built against.
    """
    cls = obj if isinstance(obj, type) else type(obj)
    return any(base.__name__ == "PartitionedProcess" for base in cls.__mro__)


class Requester(Step):
    """Requester Step

    Accepts a PartitionedProcess as an input, and runs in coordination with an
    Evolver that uses the same PartitionedProcess.

    process-bigraph interface: Requesters read from all ports but only
    write to request, process, next_update_time, and optionally listeners.
    """

    config_schema = {
        'process': 'shared_process_ref',
        'time_step': 'float{1.0}',
        '_parallel': 'boolean{false}',
        'name': 'string',
    }

    defaults = {"process": None}

    def inputs(self):
        process = self.config.get("process")
        ports = process.inputs()
        timestep = process.parameters.get('timestep', 1) if process else 1
        # Requester also reads these control ports
        ports['global_time'] = 'float{0.0}'
        ports['timestep'] = f'integer{{{timestep}}}'
        # Plain ``overwrite[float]`` shares mother's value on divide
        # (matches v1's ``_divider: "set"`` semantics). The daughter
        # inherits mother's last-written ``next_update_time``
        # (= current_time + interval) so the evolver gate
        # ``next_update_time <= global_time`` is False on the daughter's
        # divide tick — preventing it from firing with stale state.
        ports['next_update_time'] = f'overwrite[float]{{{float(timestep)}}}'
        ports['process'] = 'shared_process'
        return ports

    def outputs(self):
        process = self.config.get("process")
        timestep = process.parameters.get('timestep', 1) if process else 1
        result = {
            'request': {'_type': 'overwrite[map[list[integer]]]', '_default': {}},
            'process': 'shared_process',
            # Default next_update_time to the wrapped process's
            # timestep so the partition gate correctly fires on the
            # first tick (next_update_time <= global_time=0 only when
            # the default is non-positive — using timestep matches the
            # cell_state.setdefault that lived in build_ecoli_document).
            'next_update_time': {
                '_type': 'overwrite[float]',
                '_default': float(timestep),
            },
        }
        # Pull the actual listener schema from the wrapped process so
        # per-field types are preserved (not flattened to map[...]).
        if process is not None:
            proc_outputs = process.outputs()
            listeners = proc_outputs.get('listeners')
            if listeners:
                result['listeners'] = listeners
            for key in proc_outputs:
                if key not in result and key not in ('bulk', 'bulk_total', 'listeners'):
                    result[key] = proc_outputs[key]
        return result

    def __init__(self, parameters=None, core=None):
        assert is_partitioned_process(parameters["process"])
        if parameters["process"].parallel:
            raise RuntimeError("PartitionedProcess objects cannot be parallelized.")
        parameters["name"] = f"{parameters['process'].name}_requester"
        super().__init__(parameters, core=core)
        # Cache the request port keys — always just ['bulk'] for
        # the standard partition setup. Previously set as a side
        # effect of ports_schema(); now initialized eagerly.
        self.cached_bulk_ports = ['bulk']

    def perform_update(self, states):
        """v2 gate: only run when next_update_time <= global_time."""
        next_t = states.get("next_update_time")
        global_t = states.get("global_time")
        if next_t is None or global_t is None:
            return True  # missing ports — run by default
        return next_t <= global_t

    def update_condition(self, timestep, states):
        """v1 gate: same logic as perform_update, with warning."""
        if states["next_update_time"] <= states["global_time"]:
            if states["next_update_time"] < states["global_time"]:
                warnings.warn(
                    f"{self.name} updated at t="
                    f"{states['global_time']} instead of t="
                    f"{states['next_update_time']}. Decrease the "
                    "timestep of the global_clock process for more "
                    "accurate timekeeping."
                )
            return True
        return False

    def ports_schema(self):
        process = self.parameters.get("process")
        ports = process.get_schema()
        ports["request"] = {
            "bulk": {
                "_updater": "set",
                "_divider": "null",
                "_emit": False,
            }
        }
        ports["process"] = {
            "_default": tuple(),
            "_updater": "set",
            "_divider": "null",
            "_emit": False,
        }
        ports["global_time"] = {"_default": 0.0}
        ports["timestep"] = {"_default": process.parameters["timestep"]}
        ports["next_update_time"] = {
            "_default": process.parameters["timestep"],
            "_updater": "set",
            "_divider": "set",
        }
        self.cached_bulk_ports = list(ports["request"].keys())
        return ports

    def next_update(self, timestep, states):
        proc_state = states["process"]
        # v1 wires "process" as ``(instance,)`` tuple; v2's
        # SharedProcessRef.realize returns the bare instance. Tolerate
        # both so the same code runs on either engine.
        process = proc_state[0] if isinstance(proc_state, (list, tuple)) else proc_state
        request = process.calculate_request(states["timestep"], states)
        process.request_set = True

        request["request"] = {}
        # Send bulk requests through request port
        for bulk_port in self.cached_bulk_ports:
            bulk_request = request.pop(bulk_port, None)
            if bulk_request is not None:
                request["request"][bulk_port] = bulk_request

        # Ensure listeners are updated if present
        listeners = request.pop("listeners", None)
        if listeners is not None:
            request["listeners"] = listeners

        # Update shared process instance
        request["process"] = (process,)
        return request


class Evolver(Step):
    """Evolver Step

    Accepts a PartitionedProcess as an input, and runs in coordination with an
    Requester that uses the same PartitionedProcess.

    process-bigraph interface: Evolvers read from all ports but only
    write to everything except allocate, global_time, and timestep.
    """

    _input_only_ports = {'allocate', 'global_time', 'timestep'}

    config_schema = {
        'process': 'shared_process_ref',
        'time_step': 'float{1.0}',
        '_parallel': 'boolean{false}',
        'name': 'string',
    }

    defaults = {"process": None}

    def inputs(self):
        process = self.config.get("process")
        ports = process.inputs()
        timestep = process.parameters.get('timestep', 1) if process else 1
        # Evolver also reads these control ports
        ports['allocate'] = {'bulk': 'array[integer[64]]'}
        ports['global_time'] = 'float{0.0}'
        ports['timestep'] = f'integer{{{timestep}}}'
        # divide_reset matches v1's ``_divider: "set"`` (see Requester).
        ports['next_update_time'] = f'overwrite[float]{{{float(timestep)}}}'
        ports['process'] = 'shared_process'
        return ports

    def outputs(self):
        process = self.parameters.get("process")
        timestep = process.parameters.get('timestep', 1) if process else 1
        ports = process.outputs()
        # Evolver writes next_update_time and process in addition to
        # whatever the wrapped process declares.
        ports['next_update_time'] = {
            '_type': 'overwrite[float]',
            '_default': float(timestep),
        }
        ports['process'] = 'shared_process'
        # Evolver doesn't write to allocate, global_time, timestep
        for k in ('allocate', 'global_time', 'timestep'):
            ports.pop(k, None)
        return ports

    def __init__(self, parameters=None, core=None):
        assert is_partitioned_process(parameters["process"])
        parameters["name"] = f"{parameters['process'].name}_evolver"
        super().__init__(parameters, core=core)

    def perform_update(self, states):
        """v2 gate: only run when next_update_time <= global_time."""
        next_t = states.get("next_update_time")
        global_t = states.get("global_time")
        if next_t is None or global_t is None:
            return True  # missing ports — run by default
        return next_t <= global_t

    def update_condition(self, timestep, states):
        """v1 gate: same logic as perform_update, with warning."""
        if states["next_update_time"] <= states["global_time"]:
            if states["next_update_time"] < states["global_time"]:
                warnings.warn(
                    f"{self.name} updated at t="
                    f"{states['global_time']} instead of t="
                    f"{states['next_update_time']}. Decrease the "
                    "timestep for the global clock process for more "
                    "accurate timekeeping."
                )
            return True
        return False

    def ports_schema(self):
        process = self.parameters.get("process")
        ports = process.get_schema()
        ports["allocate"] = {
            "bulk": {
                "_updater": "set",
                "_divider": "null",
                "_emit": False,
            }
        }
        ports["process"] = {
            "_default": tuple(),
            "_updater": "set",
            "_divider": "null",
            "_emit": False,
        }
        ports["global_time"] = {"_default": 0.0}
        ports["timestep"] = {"_default": process.parameters["timestep"]}
        ports["next_update_time"] = {
            "_default": process.parameters["timestep"],
            "_updater": "set",
            "_divider": "set",
        }
        return ports

    def next_update(self, timestep, states):
        allocations = states.pop("allocate")
        for key, value in allocations.items():
            if isinstance(value, list):
                value = np.array(value)
            states[key] = value
        proc_state = states["process"]
        process = proc_state[0] if isinstance(proc_state, (list, tuple)) else proc_state

        # If the Requester has not run yet, skip the Evolver's update to
        # let the Requester run in the next time step.
        if not process.request_set:
            return {}

        update = process.evolve_state(states["timestep"], states)
        update["process"] = (process,)
        update["next_update_time"] = states["global_time"] + states["timestep"]
        return update


class PartitionedProcess(Process):
    """Partitioned Process Base Class

    This is the base class for all processes whose updates can be partitioned.

    Subclasses must implement:
      - ``ports_schema()``: v1 bidirectional port schema
      - ``calculate_request(timestep, states)``: compute resource requests
      - ``evolve_state(timestep, states)``: compute state updates

    Subclasses may define ``_output_ports`` as a set of port names that
    appear in the delta returned by ``evolve_state()``.  All other ports
    are treated as input-only for the dependency graph.

    For v2, subclasses must override ``inputs()`` and ``outputs()`` to
    declare typed ports; the view is projected through those schemas so
    processes see only the declared fields.
    """

    _output_ports = None
    _input_only_ports = None

    def __init__(self, parameters=None):
        super().__init__(parameters)

        # set partition mode
        self.evolve_only = self.parameters.get("evolve_only", False)
        self.request_only = self.parameters.get("request_only", False)
        self.request_set = False

        # register topology
        assert self.name
        assert self.topology
        topology_registry.register(self.name, self.topology)

    @abc.abstractmethod
    def ports_schema(self):
        return {}

    @abc.abstractmethod
    def inputs(self):
        """Declare the nested schema of state this process reads."""
        return {}

    @abc.abstractmethod
    def outputs(self):
        """Declare the nested schema of state this process writes."""
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
        bulk_requests = requests.pop("bulk", [])
        if bulk_requests:
            bulk_copy = states["bulk"].copy()
            for bulk_idx, request in bulk_requests:
                bulk_copy[bulk_idx] = request
            states["bulk"] = bulk_copy
        states = deep_merge(states, requests)
        update = self.evolve_state(timestep, states)
        if "listeners" in requests:
            update["listeners"] = deep_merge(update["listeners"], requests["listeners"])
        return update

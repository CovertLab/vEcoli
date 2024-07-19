r"""
=============
EngineProcess
=============

Tunnel Ports
============

Sometimes, a process inside the EngineProcess might need to be wired to
a store outside the EngineProcess, or an outside process might need to
be wired to an inside store. We handle this with _tunnels_.

Here is a state hierarchy showing how a tunnel connects an outside
process (``A``) to an inside store (``store1``). We call this a "tunnel
in" because the exterior process is tunneling into EngineProcess to see
an internal store.

.. code-block:: text

         /\
        /  \         EngineProcess
    +---+   +-------------------------+
    | A |   |                         |
    +---+   |        /\               |
      :     |       /  \              |
      :     |  +---+    store1        |
   ...:     |  | B |       ^          |
   :        |  +---+       |          |
   :        |              |          |
   :..tunnel_outer <----- next_update |
            |                         |
            +-------------------------+

Here is another example where a tunnel connects an inside process
(``B``) to an outside store (``store2``). We call this a "tunnel out"
because the interior process is tunneling outside of EngineProcess to
see an external store.

.. code-block:: text

         /\
        /  \         EngineProcess
       /    +-------------------------+
    store2  |                         |
      :     |        /\               |
      :     |       /  \              |
      :     |  +---+    tunnel_inner  |
   ...:     |  | B |.......:    ^     |
   :        |  +---+            |     |
   :        |                   |     |
   :..tunnel_outer <----- next_update |
            |                         |
            +-------------------------+

In these diagrams, processes are boxes, stores are labeled nodes in the
tree, solid lines show the state hierarchy, and dotted lines show the
topology wiring.

These tunnels are the only way that the EngineProcess exchanges
information with the outside simulation.
"""

import copy
import warnings
from typing import Any

import numpy as np
from vivarium.core.composer import Composer
from vivarium.core.emitter import get_emitter, SharedRamEmitter
from vivarium.core.engine import Engine
from vivarium.core.process import Process, Step
from vivarium.core.registry import updater_registry, divider_registry
from vivarium.core.store import DEFAULT_SCHEMA
from vivarium.library.topology import get_in

from ecoli.library.sim_data import RAND_MAX
from ecoli.library.schema import remove_properties, empty_dict_divider, not_a_process
from ecoli.library.updaters import inverse_updater_registry
from ecoli.processes.cell_division import daughter_phylogeny_id


def _get_path_net_depth(path):
    depth = 0
    for node in path:
        if node == "..":
            depth -= 1
        else:
            depth += 1
    return depth


def cap_tunneling_paths(topology, outer=tuple()):
    tunnels = {}
    caps = []
    for key, val in topology.items():
        if isinstance(val, dict):
            tunnels.update(cap_tunneling_paths(val, outer + (key,)))
        elif isinstance(val, tuple) and val:  # Never cap empty paths
            path_depth = _get_path_net_depth(val)
            # Note that the last node in ``outer`` is the process name,
            # which doesn't count as part of the path depth.
            outer_path = outer[:-1]
            # Paths are relative to the process's parent node.
            if -path_depth > len(outer_path) - 1:
                # Path extends ouside EngineProcess, so cap it.
                assert val[-1] != ".."
                tunnel_inner = f"{val[-1]}_tunnel"
                # Capped path is to tunnel_inner, which is at the top
                # level of the hierarchy.
                capped_path = tuple([".."] * len(outer_path)) + (tunnel_inner,)
                tunnels[outer + (key,)] = tunnel_inner
                caps.append((key, capped_path))
    for cap_key, cap_path in caps:
        topology[cap_key] = cap_path
    return tunnels


class SchemaStub(Step):
    """Stub process for providing schemas to an inner simulation. Run as
    a Step otherwise its timestep could influence when other Steps run.
    For example, if a SchemaStub was a process with timestep 1 while all
    other processes had timestep 2, Steps like mass listener would run
    after every second instead of every 2 seconds, wasting time.

    When using :py:class:`ecoli.processes.engine_process.EngineProcess`,
    there may be processes in the outer simulation whose schemas you are
    expecting to affect variables in the inner simulation. You can
    include this stub process in your inner simulation to provide those
    schemas from the outer simulation.

    The process takes a single parameter, ``ports_schema``, whose value
    is the process's ports schema to be provided to the inner
    simulation.
    """

    defaults: dict[str, Any] = {
        "ports_schema": {},
    }

    def ports_schema(self):
        return self.parameters["ports_schema"]

    def next_update(self, timestep, states):
        return {}


class EngineProcess(Process):
    defaults = {
        # Map from tunnel name to path to internal store.
        "tunnels_in": {},
        # Map from tunnel name to schema. Schemas are optional.
        "tunnel_out_schemas": {},
        "emit_paths": tuple(),
        # Map from process name to a map from path in the inner
        # simulation to the schema that should be stubbed in at that
        # path. A stub process will be added with a port for each
        # schema, and the topology will wire each port to the specified
        # path.
        "stub_schemas": {},
        "agent_id": "0",
        "inner_composer": None,
        "outer_composer": None,
        "inner_composer_config": {},
        "outer_composer_config": {},
        "seed": 0,
        "inner_emitter": "null",
        "divide": False,
        "division_threshold": None,
        "division_variable": None,
        "start_time": 0,
        "experiment_id": "",
        "inner_same_timestep": False,
    }
    # TODO: Handle name clashes between tunnels.

    def __init__(self, parameters=None):
        parameters = parameters or {}
        super().__init__(parameters)
        # Pass config to generate() to avoid deep copy
        inner_composite = self.parameters["inner_composer"]().generate(
            self.parameters["inner_composer_config"]
        )
        inner_initial_state = inner_composite.initial_state(
            self.parameters["inner_composer_config"]
        )

        self.tunnels_out = cap_tunneling_paths(inner_composite["topology"])
        self.tunnels_in = self.parameters["tunnels_in"]

        processes = inner_composite["processes"]
        topology = inner_composite["topology"]
        steps = inner_composite.get("steps")
        for process, d in self.parameters["stub_schemas"].items():
            stub_ports_schema = {}
            stub_process_name = f"{process}_stub"
            topology[stub_process_name] = {}
            for path, schema in d.items():
                port = ">".join(path)
                stub_ports_schema[port] = schema
                topology[stub_process_name][port] = path
            stub = SchemaStub({"ports_schema": stub_ports_schema})
            steps[stub_process_name] = stub

        self.emitter = None

        # Since unique numpy updater is an class method, internal
        # deepcopying in vivarium-core causes this warning to appear
        warnings.filterwarnings(
            "ignore",
            message="Incompatible schema "
            "assignment at .+ Trying to assign the value <bound method "
            "UniqueNumpyUpdater\.updater .+ to key updater, which already "
            "has the value <bound method UniqueNumpyUpdater\.updater",
        )
        self.sim = Engine(
            processes=processes,
            steps=steps,
            flow=inner_composite.get("flow"),
            topology=topology,
            initial_state=inner_initial_state,
            experiment_id=self.parameters["experiment_id"],
            emitter="null",
            display_info=False,
            progress_bar=False,
            initial_global_time=self.parameters["start_time"],
        )
        # Unnecessary references to initial_state
        self.sim.initial_state = None
        self.parameters["inner_composer_config"].pop("initial_state", None)

        # Only apply overrides to first cell in simulation
        self.parameters["inner_composer_config"].pop("initial_state_overrides", None)

        if self.parameters["emit_paths"]:
            self.sim.state.set_emit_values([tuple()], False)
            self.sim.state.set_emit_values(
                self.parameters["emit_paths"],
                True,
            )
        self.random_state = np.random.RandomState(seed=self.parameters["seed"])

        self.updater_registry_reverse = {
            updater_registry.access(key): key for key in updater_registry.main_keys
        }

    def create_emitter(self):
        if isinstance(self.parameters["inner_emitter"], str):
            self.emitter_config = {"type": self.parameters["inner_emitter"]}
        else:
            self.emitter_config = self.parameters["inner_emitter"]
        self.emitter_config["experiment_id"] = self.parameters["experiment_id"]
        self.emitter = get_emitter(self.emitter_config)

    def ports_schema(self):
        schema = {
            "agents": {},
        }
        for port_path, tunnel in self.tunnels_out.items():
            process_path = port_path[:-1]
            port = port_path[-1]
            process = get_in(
                self.sim.processes, process_path, get_in(self.sim.steps, process_path)
            )
            tunnel_schema = process.get_schema()[port]
            schema[tunnel] = copy.deepcopy(tunnel_schema)
        for tunnel, path in self.tunnels_in.items():
            tunnel_schema = self.sim.state.get_path(path).get_config()
            # Don't waste time dividing outer sim state since it will be
            # overwritten by inner daughter states (also removes need to
            # emit all unique molecules required by certain dividers like
            # that for active_ribosome)
            tunnel_schema["_divider"] = empty_dict_divider
            # Internal sim state is fully defined, making subschemas
            # redundant (also not properly parsed during store generation)
            # This also avoids duplicated emits from the outer sim.
            tunnel_schema = remove_properties(
                tunnel_schema, ["_subschema", "_emit", "_value"]
            )
            schema[tunnel] = tunnel_schema
        for tunnel, tunnel_schema in self.parameters["tunnel_out_schemas"].items():
            schema[tunnel] = tunnel_schema
        return schema

    def initial_state(self, config=None):
        state = {}
        # We ignore tunnels out because those are to stores like fields
        # or dimensions that are outside the cell and therefore don't
        # get divided.
        for tunnel, path in self.tunnels_in.items():
            state[tunnel] = self.sim.state.get_path(path).get_value()
        return state

    def calculate_timestep(self, states):
        timestep = np.inf
        for proc_path, process in self.sim.process_paths.items():
            _, proc_state = self.sim._process_state(proc_path)
            if self.parameters["inner_same_timestep"]:
                # Warn user if inner process has different timestep from rest
                if (
                    timestep != process.calculate_timestep(proc_state)
                    and timestep != np.inf
                ):
                    warnings.warn(
                        "Time step mismatch for process "
                        f"{process.name}: {timestep} != "
                        + str(process.calculate_timestep({}))
                    )
            timestep = min(timestep, process.calculate_timestep(proc_state))
        return timestep

    def send_command(self, command, args=None, kwargs=None, run_pre_check=True) -> None:
        """Override to handle special command 'get_inner_state' which
        lets engine process pull out a dictionary containing the entire
        inner simulation state."""
        if run_pre_check:
            self.pre_send_command(command, args, kwargs)
        args = args or tuple()
        kwargs = kwargs or {}

        if command == "get_inner_state":
            self._command_result = self.sim.state.get_value(condition=not_a_process)
        else:
            self._pending_command = None
            super().send_command(command, args, kwargs)

    def next_update(self, timestep, states):
        # Create emitter only after all pickling/unpickling/forking
        if not self.emitter:
            self.create_emitter()

        # Check whether we are being forced to finish early. This check
        # should happen before we mutate the inner simulation state to
        # make sure that self.calculate_timestep() returns the same
        # value as it did to the Engine. However, this is just
        # precautionary for now because currently,
        # self.calculate_timestep() does not depend on the inner state.
        # This only works because self.calculate_timestep() returns the
        # same timestep that the inner Engine would normally use. If
        # self.calculate_timestep() returned a timestep smaller than
        # what the inner Engine would normally use, the outer simulation
        # could be ending and forcing this process to complete, but
        # since the timestep could by chance equal
        # self.calculate_timestep(), we would not know to force the
        # inner simulation to complete.
        force_complete = timestep != self.calculate_timestep({})

        # Assert that nothing got wired into `null`.
        try:
            miswired_vars = self.sim.state.get_path(("null",)).inner.keys()
        except Exception as e:
            # There might not be a ('null',) store, which is okay.
            if str(e) == "('null',) is not a valid path from ()":
                miswired_vars = tuple()
            else:
                raise e
        if miswired_vars:
            raise RuntimeError(
                f'Variables mistakenly wired to ("null",): {miswired_vars}'
            )

        # Update the internal state with tunnel data.
        # TODO: Check whether we need to deepcopy. Even if there are mutable
        # states, mutating both the outer and inner states seems like it would
        # save time because we end up crafting an update at the end to make
        # the outer state match the inner state anyways
        for tunnel, path in self.tunnels_in.items():
            self.sim.state.get_path(path).set_value(states[tunnel])
        for tunnel in self.tunnels_out.values():
            self.sim.state.get_path((tunnel,)).set_value(states[tunnel])

        # Emit data from inner simulation. We emit at the start of
        # next_update() because the inner simulation is in-sync with the
        # outer simulation here after the internal state has been
        # synchronized with the tunnels from the outer simulation. In
        # other words, since we rely on the outer Engine to apply the
        # updates, we have to wait for those updates from the previous
        # timestep to be applied before we emit data.
        data = self.sim.state.emit_data()
        data["time"] = self.sim.global_time
        emit_config = {
            "table": "history",
            "data": data,
        }
        self.emitter.emit(emit_config)

        # Run inner simulation for timestep.
        self.sim.run_for(timestep)
        if force_complete:
            self.sim.complete()

        update = {}

        # Check for division and perform if needed.
        division_threshold = self.parameters["division_threshold"]
        division_variable = self.sim.state.get_path(
            self.parameters["division_variable"]
        ).get_value()
        if self.parameters["divide"] and division_variable >= division_threshold:
            # Perform division.
            daughters = []
            daughter_states = self.sim.state.divide_value()
            daughter_ids = daughter_phylogeny_id(self.parameters["agent_id"])
            for daughter_id, inner_state in zip(daughter_ids, daughter_states):
                emitter_config = dict(self.emitter_config)
                emitter_config["embed_path"] = ("agents", daughter_id)
                new_seed = self.random_state.randint(RAND_MAX)
                inner_composer_config = {
                    **self.parameters["inner_composer_config"],
                    "seed": new_seed,
                    "agent_id": daughter_id,
                    "initial_state": inner_state,
                }
                # Pass config to generate() to avoid deep copy
                outer_composite = self.parameters["outer_composer"]().generate(
                    {
                        **self.parameters["outer_composer_config"],
                        "agent_id": daughter_id,
                        "seed": new_seed,
                        "start_time": self.sim.global_time,
                        "inner_emitter": emitter_config,
                        "inner_composer_config": inner_composer_config,
                    }
                )
                daughter = {
                    "key": daughter_id,
                    "processes": outer_composite.processes,
                    "steps": outer_composite.steps,
                    "flow": outer_composite.flow,
                    "topology": outer_composite.topology,
                    "initial_state": outer_composite.initial_state(),
                }
                daughters.append(daughter)
            update["agents"] = {
                "_divide": {
                    "mother": self.parameters["agent_id"],
                    "daughters": daughters,
                },
            }

        # Craft an update to pass data back out through the tunnels.
        for tunnel, path in self.tunnels_in.items():
            store = self.sim.state.get_path(path)
            inverted_update = _inverse_update(
                states[tunnel],
                store.get_value(),
                store,
                self.updater_registry_reverse,
            )
            if not (isinstance(inverted_update, dict) and inverted_update == {}):
                update[tunnel] = inverted_update
        for tunnel in self.tunnels_out.values():
            store = self.sim.state.get_path((tunnel,))
            inverted_update = _inverse_update(
                states[tunnel],
                store.get_value(),
                store,
                self.updater_registry_reverse,
            )
            if not (isinstance(inverted_update, dict) and inverted_update == {}):
                update[tunnel] = inverted_update
        return update


def _inverse_update(initial_state, final_state, store, updater_registry_reverse):
    # Handle the base case where we are on a leaf node.
    if not store.inner:
        if isinstance(store.updater, str):
            updater_name = store.updater
        else:
            # If the updater is not a string, look up its name using the
            # reverse lookup table.
            updater_name = updater_registry_reverse[store.updater]
        if updater_name == DEFAULT_SCHEMA:
            updater_name = "accumulate"

        inverse_updater = inverse_updater_registry.access(updater_name)
        assert inverse_updater
        return inverse_updater(initial_state, final_state)

    # Loop over the keys in the initial state (to be updated) and recurse.
    # TODO: Handle edge case where we add/remove stores in final_state
    update = {}
    for key in initial_state.keys():
        sub_update = _inverse_update(
            initial_state[key],
            final_state[key],
            store.inner[key],
            updater_registry_reverse,
        )
        if sub_update:
            update[key] = sub_update
    return update


class _ProcA(Process):
    def ports_schema(self):
        return {
            "port_a": {
                "_default": 0,
                "_updater": "accumulate",
                "_emit": True,
                "_divider": "split",
            },
            "port_c": {
                "_default": 0,
                "_updater": "accumulate",
                "_emit": True,
                "_divider": "set",
            },
            "port_d": {
                "_default": 0,
                "_updater": "accumulate",
                "_emit": True,
                "_divider": "zero",
            },
        }

    def next_update(self, timestep, states):
        """Each timestep, ``port_a += port_c``."""
        return {
            "port_a": states["port_c"],
            "port_d": 1,
        }


class _ProcB(Process):
    def ports_schema(self):
        return {
            "port_b": {
                "_default": 0,
                "_updater": "accumulate",
                "_emit": True,
                "_divider": "set",
            },
        }

    def next_update(self, timestep, states):
        """Each timestep, ``port_b += 1``."""
        return {
            "port_b": 1,
        }


class _ProcC(Process):
    def ports_schema(self):
        return {
            "port_c": {
                "_default": 0,
                "_updater": "accumulate",
                "_emit": True,
            },
            "port_b": {
                "_default": 0,
                "_updater": "accumulate",
                "_emit": True,
            },
        }

    def next_update(self, timestep, states):
        """Each timestep, ``port_c += port_b``."""
        return {
            "port_c": states["port_b"],
        }


class _InnerComposer(Composer):
    def generate_processes(self, config):
        return {
            "procA": _ProcA(),
            "procB": _ProcB(),
        }

    def generate_topology(self, config):
        return {
            "procA": {
                "port_a": ("a",),
                "port_c": ("c",),
                "port_d": ("d",),
            },
            "procB": {
                "port_b": ("..", "b"),
            },
        }


class _OuterComposer(Composer):
    def generate_processes(self, config):
        inner_composer_config = config.pop("inner_composer_config", None)
        proc = EngineProcess(
            {
                "inner_composer": _InnerComposer,
                "inner_composer_config": inner_composer_config,
                "outer_composer": _OuterComposer,
                "outer_composer_config": config,
                "agent_id": config["agent_id"],
                "tunnels_in": {
                    "c_tunnel": ("c",),
                },
                "time_step": 1,
                "divide": True,
                "division_threshold": 4,
                "division_variable": ("d",),
                "inner_emitter": config["inner_emitter"],
                "start_time": config["start_time"],
                "experiment_id": config["experiment_id"],
            }
        )
        return {
            "engine": proc,
        }

    def generate_topology(self, config):
        return {
            "engine": {
                "b_tunnel": (
                    "..",
                    "..",
                    "b",
                ),
                "c_tunnel": (
                    "..",
                    "..",
                    "c",
                ),
                "agents": ("..", "..", "agents"),
            },
        }


def test_engine_process():
    """
    Here's a schematic diagram of the hierarchy created in this test:

    .. code-block:: text

            +-------------+------------+------------------+
            |             |            |                  |
            |           +-+-+          |                  |
            b...........| C |..........c    +-------------+-----------+
            :           +---+          :    |       EngineProcess     |
            :                          :    | +---+----+-----+-----+  |
            :                          :    | |   |    |     |     |  |
            :                          :    | | +-+-+..c     |     |  |
            :                          :    | | | A |        |     |  |
            :                          :    | | +---+........a     |  |
            :                          :    | |                    |  |
            :                          :    | +---+                |  |
            :                          :    | | B |..........b_tunnel |
            :                          :    | +---+                   |
            :                          :    |                         |
            :                          :    +---c_tunnel---b_tunnel---+
            :                          :...........:           :
            :                                                  :
            :..................................................:

    Notice that ``c_tunnel`` is a tunnel in from outer process ``C`` to
    inner store `c`, and ``b_tunnel`` is a tunnel out from inner process
    ``B`` to outer store ``b``.
    """
    experiment_id = "test_experiment_id"

    # Clear the emitter's data in case it has been filled by another
    # test.
    SharedRamEmitter.saved_data.clear()

    agent_path = ("agents", "0")
    outer_composer = _OuterComposer(
        {
            "experiment_id": experiment_id,
            "agent_id": agent_path[-1],
            "inner_composer_config": {},
            "start_time": 0,
            "inner_emitter": {
                "type": "shared_ram",
                "embed_path": agent_path,
            },
        }
    )
    outer_composite = outer_composer.generate(path=agent_path)

    schema = get_in(
        outer_composite.processes,
        ("agents", "0", "engine"),
    ).get_schema()
    expected_schema = {
        "agents": {},
        "b_tunnel": {
            "_default": 0,
            "_updater": "accumulate",
            "_emit": True,
            "_divider": "set",
        },
        # The schema for c_tunnel is complete, even though we only
        # specified a partial schema, because this schema is pulled from
        # the filled inner simulation hierarchy.
        "c_tunnel": {
            "_default": 0,
            "_updater": updater_registry.access("accumulate"),
            "_divider": divider_registry.access("empty_dict"),
        },
    }
    assert schema == expected_schema

    enviro_composite = {
        "processes": {
            "procC": _ProcC(),
        },
        "steps": {},
        "flow": {},
        "topology": {
            "procC": {
                "port_b": ("b",),
                "port_c": ("c",),
            },
        },
    }
    outer_composite.merge(enviro_composite)
    engine = Engine(
        composite=outer_composite,
        experiment_id=experiment_id,
        emitter={
            "type": "shared_ram",
        },
    )
    engine.update(8)

    data = engine.emitter.get_data()
    expected_data = {
        0: {
            "agents": {
                "0": {"a": 0, "b_tunnel": 0, "c": 0, "d": 0},
            },
            "b": 0,
            "c": 0,
        },
        1.0: {
            "agents": {
                "0": {"a": 0, "b_tunnel": 1, "c": 0, "d": 1},
            },
            "b": 1,
            "c": 0,
        },
        2.0: {
            "agents": {
                "0": {"a": 0, "b_tunnel": 2, "c": 1, "d": 2},
            },
            "b": 2,
            "c": 1,
        },
        3.0: {
            "agents": {
                "0": {"a": 1, "b_tunnel": 3, "c": 3, "d": 3},
            },
            "b": 3,
            "c": 3,
        },
        4.0: {
            "agents": {
                "00": {"a": 2, "b_tunnel": 4, "c": 6, "d": 0},
                "01": {"a": 2, "b_tunnel": 4, "c": 6, "d": 0},
            },
            "b": 4,
            "c": 6,
        },
        # Note that now b is incrementing by 2 because it's getting +1
        # updates from both cells.
        5.0: {
            "agents": {
                "00": {"a": 8, "b_tunnel": 6, "c": 10, "d": 1},
                "01": {"a": 8, "b_tunnel": 6, "c": 10, "d": 1},
            },
            "b": 6,
            "c": 10,
        },
        6.0: {
            "agents": {
                "00": {"a": 18, "b_tunnel": 8, "c": 16, "d": 2},
                "01": {"a": 18, "b_tunnel": 8, "c": 16, "d": 2},
            },
            "b": 8,
            "c": 16,
        },
        7.0: {
            "agents": {
                "00": {"a": 34, "b_tunnel": 10, "c": 24, "d": 3},
                "01": {"a": 34, "b_tunnel": 10, "c": 24, "d": 3},
            },
            "b": 10,
            "c": 24,
        },
        8.0: {
            "agents": {
                "000": {},
                "001": {},
                "010": {},
                "011": {},
            },
            "b": 12,
            "c": 34,
        },
    }
    assert data == expected_data


def test_cap_tunneling_paths():
    topology = {
        "procA": {
            "port_a": ("a",),
        },
        "procB": {
            "port_b": ("..", "b"),
        },
    }
    capped = {
        "procA": {
            "port_a": ("a",),
        },
        "procB": {
            "port_b": ("b_tunnel",),
        },
    }
    expected_tunnels = {
        ("procB", "port_b"): "b_tunnel",
    }
    tunnels = cap_tunneling_paths(topology)
    assert topology == capped
    assert tunnels == expected_tunnels


if __name__ == "__main__":
    test_engine_process()

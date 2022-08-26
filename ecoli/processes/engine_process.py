'''
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
'''
import copy

import numpy as np
from vivarium.core.engine import Engine
from vivarium.core.process import Process
from vivarium.core.registry import updater_registry
from vivarium.core.store import DEFAULT_SCHEMA, Store
from vivarium.library.topology import get_in

from ecoli.library.sim_data import RAND_MAX
from ecoli.library.updaters import inverse_updater_registry
from ecoli.processes.cell_division import daughter_phylogeny_id


def _get_path_net_depth(path):
    depth = 0
    for node in path:
        if node == '..':
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
                assert val[-1] != '..'
                tunnel_inner = f'{val[-1]}_tunnel'
                # Capped path is to tunnel_inner, which is at the top
                # level of the hierarchy.
                capped_path = tuple(
                    ['..'] * len(outer_path)) + (tunnel_inner,)
                tunnels[outer + (key,)] = tunnel_inner
                caps.append((key, capped_path))
    for cap_key, cap_path in caps:
        topology[cap_key] = cap_path
    return tunnels


class SchemaStub(Process):
    '''Stub process for providing schemas to an inner simulation.

    When using :py:class:`ecoli.processes.engine_process.EngineProcess`,
    there may be processes in the outer simulation whose schemas you are
    expecting to affect variables in the inner simulation. You can
    include this stub process in your inner simulation to provide those
    schemas from the outer simulation.

    The process takes a single parameter, ``ports_schema``, whose value
    is the process's ports schema to be provided to the inner
    simulation.
    '''

    defaults = {
        'ports_schema': {},
    }

    def ports_schema(self):
        return self.parameters['ports_schema']

    def next_update(self, timestep, states):
        return {}


class EngineProcess(Process):
    defaults = {
        'composite': {},
        # Map from tunnel name to path to internal store.
        'tunnels_in': {},
        # Map from tunnel name to schema. Schemas are optional.
        'tunnel_out_schemas': {},
        # Map from process name to a map from path in the inner
        # simulation to the schema that should be stubbed in at that
        # path. A stub process will be added with a port for each
        # schema, and the topology will wire each port to the specified
        # path.
        'stub_schemas': {},
        'initial_inner_state': {},
        'agent_id': '0',
        'composer': None,
        'seed': 0,
        'inner_emitter': 'null',
        'divide': False,
        'division_threshold': 0,
        'division_variable': tuple(),
    }
    # TODO: Handle name clashes between tunnels.

    def __init__(self, parameters=None):
        parameters = parameters or {}
        parameters['_no_original_parameters'] = True
        super().__init__(parameters)
        composite = self.parameters['composite']
        self.tunnels_out = cap_tunneling_paths(
            composite['topology'])
        self.tunnels_in = self.parameters['tunnels_in']

        processes = composite['processes']
        topology = composite['topology']
        for process, d in self.parameters['stub_schemas'].items():
            stub_ports_schema = {}
            stub_process_name = f'{process}_stub'
            topology[stub_process_name] = {}
            for path, schema in d.items():
                port = '>'.join(path)
                stub_ports_schema[port] = schema
                topology[stub_process_name][port] = path
            stub = SchemaStub({'ports_schema': stub_ports_schema})
            processes[stub_process_name] = stub

        self.sim = Engine(
            processes=processes,
            steps=composite.get('steps'),
            flow=composite.get('flow'),
            topology=topology,
            initial_state=self.parameters['initial_inner_state'],
            emitter=self.parameters['inner_emitter'],
            display_info=False,
            progress_bar=False,
        )
        self.random_state = np.random.RandomState(
            seed=self.parameters['seed'])

        self.updater_registry_reverse = {
            updater_registry.access(key): key
            for key in updater_registry.main_keys
        }

    def ports_schema(self):
        schema = {
            'agents': {},
        }
        for port_path, tunnel in self.tunnels_out.items():
            process_path = port_path[:-1]
            port = port_path[-1]
            process = get_in(
                self.sim.processes,
                process_path,
                get_in(self.sim.steps, process_path))
            tunnel_schema = process.get_schema()[port]
            schema[tunnel] = copy.deepcopy(tunnel_schema)
        for tunnel, path in self.tunnels_in.items():
            tunnel_schema = self.sim.state.get_path(path).get_config()
            schema[tunnel] = tunnel_schema
        for tunnel, tunnel_schema in self.parameters[
                'tunnel_out_schemas'].items():
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
        for process in self.sim.processes.values():
            timestep = min(timestep, process.calculate_timestep({}))
        return timestep

    def get_inner_state(self) -> None:
        def not_a_process(value):
            return not (isinstance(value, Store) and value.topology)
        return self.sim.state.get_value(condition=not_a_process)

    def send_command(self, command, args = None, kwargs = None,
            run_pre_check = True) -> None:
        if run_pre_check:
            self.pre_send_command(command, args, kwargs)
        args = args or tuple()
        kwargs = kwargs or {}

        if command == 'get_inner_state':
            self._command_result = self.get_inner_state()
        else:
            self._pending_command = None
            super().send_command(command, args, kwargs)

    def next_update(self, timestep, states):
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
            miswired_vars = self.sim.state.get_path(('null',)).inner.keys()
        except Exception as e:
            # There might not be a ('null',) store, which is okay.
            if str(e) == "('null',) is not a valid path from ()":
                miswired_vars = tuple()
            else:
                raise e
        if miswired_vars:
            raise RuntimeError(
                f'Variables mistakenly wired to ("null",): {miswired_vars}')

        # Update the internal state with tunnel data.
        for tunnel, path in self.tunnels_in.items():
            incoming_state = states[tunnel]
            self.sim.state.get_path(path).set_value(incoming_state)
        for tunnel in self.tunnels_out.values():
            incoming_state = states[tunnel]
            self.sim.state.get_path((tunnel,)).set_value(incoming_state)

        # Run inner simulation for timestep.
        self.sim.run_for(timestep)
        if force_complete:
            self.sim.complete()

        # Check for division and perform if needed.
        division_threshold = self.parameters['division_threshold']
        division_variable = self.sim.state.get_path(
            self.parameters['division_variable']).get_value()
        if (
                self.parameters['divide']
                and division_variable >= division_threshold):
            # Perform division.
            daughters = []
            composer = self.parameters['composer']
            daughter_states = self.sim.state.divide_value()
            daughter_ids = daughter_phylogeny_id(
                self.parameters['agent_id'])
            for daughter_id, inner_state in zip(
                    daughter_ids, daughter_states):
                composite = composer.generate({
                    'agent_id': daughter_id,
                    'initial_cell_state': inner_state,
                    'seed': self.random_state.randint(RAND_MAX),
                })
                daughter = {
                    'key': daughter_id,
                    'processes': composite.processes,
                    'steps': composite.steps,
                    'flow': composite.flow,
                    'topology': composite.topology,
                    'initial_state': composite.initial_state(),
                }
                daughters.append(daughter)
            return {
                'agents': {
                    '_divide': {
                        'mother': self.parameters['agent_id'],
                        'daughters': daughters,
                    },
                },
            }

        # Craft an update to pass data back out through the tunnels.
        update = {}
        for tunnel, path in self.tunnels_in.items():
            store = self.sim.state.get_path(path)
            inverted_update = _inverse_update(
                states[tunnel],
                store.get_value(),
                store,
                self.updater_registry_reverse,
            )
            if not (isinstance(inverted_update, dict)
                    and inverted_update == {}):
                update[tunnel] = inverted_update
        for tunnel in self.tunnels_out.values():
            store = self.sim.state.get_path((tunnel,))
            inverted_update = _inverse_update(
                states[tunnel],
                store.get_value(),
                store,
                self.updater_registry_reverse,
            )
            if not (isinstance(inverted_update, dict)
                    and inverted_update == {}):
                update[tunnel] = inverted_update
        return update


def _inverse_update(
        initial_state, final_state, store, updater_registry_reverse):
    if not store.inner:
        # Handle the base case where we are on a leaf node.
        if isinstance(store.updater, str):
            updater_name = store.updater
        else:
            # If the updater is not a string, look up its name using the
            # reverse lookup table.
            updater_name = updater_registry_reverse[store.updater]
        if updater_name == DEFAULT_SCHEMA:
            updater_name = 'accumulate'

        inverse_updater = inverse_updater_registry.access(updater_name)
        assert inverse_updater
        return inverse_updater(initial_state, final_state)

    # Loop over the keys in the store and recurse.
    update = {}
    for key in store.inner:
        # TODO: What if key is missing from initial or final?
        sub_update = _inverse_update(
            initial_state[key], final_state[key], store.inner[key],
            updater_registry_reverse)
        if not (isinstance(sub_update, dict) and sub_update == {}):
            update[key] = sub_update
    return update


class ProcA(Process):

    def ports_schema(self):
        return {
            'port_a': {
                '_default': 0,
                '_updater': 'accumulate',
                '_emit': True,
            },
            'port_c': {
                '_default': 0,
                '_updater': 'accumulate',
                '_emit': True,
            },
        }

    def next_update(self, timestep, states):
        '''Each timestep, ``port_a += port_c``.'''
        return {
            'port_a': states['port_c'],
        }


class ProcB(Process):

    def ports_schema(self):
        return {
            'port_b': {
                '_default': 0,
                '_updater': 'accumulate',
                '_emit': True,
            },
            'agents': {
                '_default': {},
                '_emit': False,
            }
        }

    def next_update(self, timestep, states):
        '''Each timestep, ``port_b += 1``.'''
        return {
            'port_b': 1,
        }


class ProcC(Process):

    def ports_schema(self):
        return {
            'port_c': {
                '_default': 0,
                '_updater': 'accumulate',
                '_emit': True,
            },
            'port_b': {
                '_default': 0,
                '_updater': 'accumulate',
                '_emit': True,
            },
        }

    def next_update(self, timestep, states):
        '''Each timestep, ``port_c += port_b``.'''
        return {
            'port_c': states['port_b'],
        }


def test_engine_process():
    '''
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
    '''
    inner_composite = {
        'processes': {
            'procA': ProcA(),
            'procB': ProcB(),
        },
        'topology': {
            'procA': {
                'port_a': ('a',),
                'port_c': ('c',),
            },
            'procB': {
                'port_b': ('..', 'b'),
                'agents': ('agents',),
            },
        },
    }
    proc = EngineProcess({
        'composite': inner_composite,
        'tunnels_in': {
            'c_tunnel': ('c',),
        },
        'time_step': 1,
        'inner_emitter': 'timeseries',
    })
    schema = proc.get_schema()
    expected_schema = {
        'agents': {},
        'b_tunnel': {
            '_default': 0,
            '_updater': 'accumulate',
            '_emit': True,
        },
        # The schema for c_tunnel is complete, even though we only
        # specified a partial schema, because this schema is pulled from
        # the filled inner simulation hierarchy.
        'c_tunnel': {
            '_default': 0,
            '_emit': True,
            '_updater': updater_registry.access('accumulate'),
            '_value': 0,
        },
    }
    assert schema == expected_schema

    outer_composite = {
        'processes': {
            'procC': ProcC(),
            'engine': proc,
        },
        'topology': {
            'procC': {
                'port_b': ('b',),
                'port_c': ('c',),
            },
            'engine': {
                'b_tunnel': ('b',),
                'c_tunnel': ('c',),
                'agents': ('agents',),
            },
        },
    }
    engine = Engine(**outer_composite)
    engine.update(4)

    outer_data = engine.emitter.get_timeseries()
    inner_data = proc.sim.emitter.get_timeseries()
    expected_outer_data = {
        'b': [0, 1, 2, 3, 4],
        'c': [0, 0, 1, 3, 6],
        'time': [0.0, 1.0, 2.0, 3.0, 4.0],
    }
    # Note that these outputs appear "behind" for stores a and c because
    # the EngineProcess doesn't see the impact of its updates until the
    # start of the following timestep. We update the internal state at
    # the beginning of the timestep before running the processes, so the
    # simulation is still functionally correct.
    expected_inner_data = {
        'a': [0, 0, 0, 1, 4],
        'b_tunnel': [0, 1, 2, 3, 4],
        'c': [0, 0, 0, 1, 3],
        'time': [0.0, 1.0, 2.0, 3.0, 4.0],
    }
    assert outer_data == expected_outer_data
    assert inner_data == expected_inner_data


def test_cap_tunneling_paths():
    topology = {
        'procA': {
            'port_a': ('a',),
        },
        'procB': {
            'port_b': ('..', 'b'),
        },
    }
    capped = {
        'procA': {
            'port_a': ('a',),
        },
        'procB': {
            'port_b': ('b_tunnel',),
        },
    }
    expected_tunnels = {
        ('procB', 'port_b'): 'b_tunnel',
    }
    tunnels = cap_tunneling_paths(topology)
    assert topology == capped
    assert tunnels == expected_tunnels


if __name__ == '__main__':
    test_engine_process()

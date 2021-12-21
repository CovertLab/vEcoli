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

from vivarium.core.engine import Engine
from vivarium.core.process import Process, Step
from vivarium.library.topology import get_in, assoc_path
from vivarium.core.registry import updater_registry

from ecoli.library.updaters import inverse_updater_registry


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
        elif isinstance(val, tuple):
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


class EngineProcess(Process):
    defaults = {
        'composite': {},
        # Map from tunnel name to (path to internal store, schema)
        'tunnels_in': {},
        'initial_inner_state': {},
        'agent_id': '0',
        'composer': None,
    }
    # TODO: Handle name clashes between tunnels.

    def __init__(self, parameters=None):
        super().__init__(parameters)
        composite = self.parameters['composite']
        self.tunnels_out = cap_tunneling_paths(
            composite['topology'])
        self.tunnels_in = self.parameters['tunnels_in']
        self.sim = Engine(
            processes=composite['processes'],
            steps=composite.get('steps'),
            flow=composite.get('flow'),
            topology=composite['topology'],
            initial_state=self.parameters['initial_inner_state'],
            emitter='null',
            display_info=False,
            progress_bar=False,
        )

    def ports_schema(self):
        schema = {}
        for port_path, tunnel in self.tunnels_out.items():
            process_path = port_path[:-1]
            port = port_path[-1]
            tunnel_schema = get_in(
                self.sim.processes, process_path).get_schema()[port]
            schema[tunnel] = copy.deepcopy(tunnel_schema)
        for tunnel, (_, tunnel_schema) in self.tunnels_in.items():
            schema[tunnel] = tunnel_schema
        return schema


    def next_update(self, timestep, states):
        # Update the internal state with tunnel data.
        for tunnel, (path, _) in self.tunnels_in.items():
            incoming_state = states[tunnel]
            self.sim.state.get_path(path).set_value(incoming_state)
        for tunnel in self.tunnels_out.values():
            incoming_state = states[tunnel]
            self.sim.state.get_path((tunnel,)).set_value(incoming_state)

        # Run inner simulation for timestep.
        # TODO: What if internal processes have a longer timestep than
        # this process?
        self.sim.update(timestep)

        agents = self.sim.state.get_path(('agents',)).inner
        if len(agents) > 1:
            # Division has occurred.
            daughters = []
            for daughter_id in agents:
                composite = self.parameters['composer'].generate({
                    'agent_id': daughter_id,
                    'initial_cell_state': agents[daughter_id].get_value(
                        condition=lambda x: not isinstance(
                            x.value, Process)
                    ),
                })
                daughter = {
                    'daughter': daughter_id,
                    'processes': composite.processes,
                    'steps': composite.steps,
                    'flow': composite.flow,
                    'topology': composite.topology,
                }
                daughters.append(daughter)
            return {
                '_divide': {
                    'mother': self.parameters['agent_id'],
                    'daughters': daughters,
                }

            }

        # Craft an update to pass data back out through the tunnels.
        update = {}
        for tunnel, (path, _) in self.tunnels_in.items():
            store = self.sim.state.get_path(path)
            update[tunnel] = _inverse_update(
                states[tunnel],
                store.get_value(),
                store,
            )
        for tunnel in self.tunnels_out.values():
            store = self.sim.state.get_path((tunnel,))
            update[tunnel] = _inverse_update(
                states[tunnel],
                store.get_value(),
                store,
            )
        return update


def _inverse_update(initial_state, final_state, store):
    if store.updater:
        # Handle the base case where we have an updater. Note that this
        # could still be at a branch if we put an updater on a branch
        # node.
        # TODO: Handle non-string updaters.
        assert isinstance(store.updater, str)
        inverse_updater = inverse_updater_registry.access(store.updater)
        assert inverse_updater
        return inverse_updater(initial_state, final_state)

    # Loop over the keys in the store and recurse.
    update = {}
    for key in store.inner:
        # TODO: What if key is missing from initial or final?
        sub_update = _inverse_update(
            initial_state[key], final_state[key], store.inner[key])
        if sub_update != {}:
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
            },
        },
    }
    proc = EngineProcess({
        'composite': inner_composite,
        'tunnels_in': {
            'c_tunnel': (
                ('c',),
                {
                    '_default': 0,
                    '_emit': True,
                },
            ),
        },
    })
    schema = proc.get_schema()
    expected_schema = {
        'b_tunnel': {
            '_default': 0,
            '_updater': 'accumulate',
            '_emit': True,
        },
        'c_tunnel': {
            '_default': 0,
            '_emit': True,
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

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
from vivarium.core.process import Process
from vivarium.library.topology import get_in, assoc_path


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
        update = {}
        # Update the internal state.
        for tunnel, (path, _) in self.tunnels_in.items():
            incoming_state = states[tunnel]
            self.sim.state.get_path(path).set_value(incoming_state)

        # update the sim state
        self.sim.state.set_value(states)

        # run the sim
        self.sim.update(timestep)

        # get the values to update
        current_state = self.store.get_value(self.parameters['ports'])

        # TODO -- convert to update
        update = current_state

        return update


class ProcA(Process):

    def ports_schema(self):
        return {
            'port_a': {
                '_default': 0,
                '_emit': True,
            },
        }

    def next_update(self, timestep, states):
        return {}


class ProcB(Process):

    def ports_schema(self):
        return {
            'port_b': {
                '_default': 0,
                '_emit': True,
            },
        }

    def next_update(self, timestep, states):
        return {}


def test_engine_process():
    inner_composite = {
        'processes': {
            'procA': ProcA(),
            'procB': ProcB(),
        },
        'topology': {
            'procA': {
                'port_a': ('a',),
            },
            'procB': {
                'port_b': ('..', 'b'),
            },
        },
    }
    proc = EngineProcess({
        'composite': inner_composite,
        'tunnels_in': {
            'c': {
                '_default': 5,
                '_emit': True,
            },
        },
    })
    schema = proc.get_schema()
    expected_schema = {
        'b_tunnel': {
            '_default': 0,
            '_emit': True,
        },
        'c_tunnel': {
            '_default': 5,
            '_emit': True,
        },
    }
    assert schema == expected_schema


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

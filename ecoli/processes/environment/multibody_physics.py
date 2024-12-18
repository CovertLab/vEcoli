"""
==========================
Multibody physics process
==========================
"""

import os

import random
import math

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches

# vivarium imports
from vivarium.library.units import units, remove_units
from vivarium.core.process import Process
from vivarium.core.composition import (
    process_in_experiment,
    simulate_experiment,
    PROCESS_OUT_DIR,
)

# vivarium-cell imports
from ecoli.processes.environment.derive_globals import volume_from_length
from ecoli.library.pymunk_multibody import PymunkMultibody
from ecoli.analysis.colony.snapshots import (
    plot_snapshots,
    format_snapshot_data,
)


NAME = "multibody"

DEFAULT_BOUNDS = [40, 40] * units.um
DEFAULT_LENGTH_UNIT = units.um
DEFAULT_MASS_UNIT = units.fg
DEFAULT_VELOCITY_UNIT = units.um / units.s
DEFAULT_VOLUME_UNIT = DEFAULT_LENGTH_UNIT**3

# constants
PI = math.pi


def random_body_position(body):
    # pick a random point along the boundary
    width, length = body.dimensions
    if random.randint(0, 1) == 0:
        # force along ends
        if random.randint(0, 1) == 0:
            # force on the left end
            location = (random.uniform(0, width), 0)
        else:
            # force on the right end
            location = (random.uniform(0, width), length)
    else:
        # force along length
        if random.randint(0, 1) == 0:
            # force on the bottom end
            location = (0, random.uniform(0, length))
        else:
            # force on the top end
            location = (width, random.uniform(0, length))
    return location


def daughter_locations(value, state):
    parent_length = state["length"]
    parent_angle = state["angle"]
    pos_ratios = [-0.25, 0.25]
    daughter_locations = []
    for daughter in range(2):
        dx = parent_length * pos_ratios[daughter] * math.cos(parent_angle)
        dy = parent_length * pos_ratios[daughter] * math.sin(parent_angle)
        location = [value[0] + dx, value[1] + dy]
        daughter_locations.append(location)
    return daughter_locations


class Multibody(Process):
    """Simulates collisions and forces between agent bodies with a multi-body physics engine.

    :term:`Ports`:
        * ``agents``: The store containing all agent sub-compartments. Each agent in
            this store has values for location, angle, length, width, mass, thrust, and torque.

    Arguments:
        initial_parameters(dict): Accepts the following configuration keys:

        jitter_force: force applied to random positions along agent
          bodies to mimic thermal fluctuations. Produces Brownian motion.
        agent_shape (:py:class:`str`): agents can take the shapes
          ``rectangle``, ``segment``, or ``circle``.
        bounds (:py:class:`list`): size of the environment in
          micrometers, with ``[x, y]``.
        mother_machine (:py:class:`bool`): if set to ``True``, mother
          machine barriers are introduced.
        animate (:py:class:`bool`): interactive matplotlib option to
          animate multibody. To run with animation turned on set True, and use
          the TKAgg matplotlib backend:

          .. code-block:: console

              $ MPLBACKEND=TKAgg uv run vivarium/processes/snapshots.py

    Notes:
        * rotational diffusion in liquid medium with viscosity = 1 mPa.s: :math:`Dr = 3.5 \\pm0.3 rad^{2}/s`
          (Saragosti, et al. 2012. Modeling E. coli tumbles by rotational diffusion.)
        * translational diffusion in liquid medium with viscosity = 1 mPa.s: :math:`Dt = 100 um^{2}/s`
          (Saragosti, et al. 2012. Modeling E. coli tumbles by rotational diffusion.)
    """

    name = NAME
    defaults = {
        "jitter_force": 1e-4,  # pN
        "agent_shape": "segment",
        "bounds": DEFAULT_BOUNDS,
        "length_unit": DEFAULT_LENGTH_UNIT,
        "mass_unit": DEFAULT_MASS_UNIT,
        "velocity_unit": DEFAULT_VELOCITY_UNIT,
        "boundary_key": "boundary",
        "mother_machine": False,
        "animate": False,
        "seed": 0,
    }

    def __init__(self, parameters=None):
        super().__init__(parameters)

        # multibody parameters
        jitter_force = self.parameters["jitter_force"]
        self.agent_shape = self.parameters["agent_shape"]
        self.bounds = self.parameters["bounds"]
        self.mother_machine = self.parameters["mother_machine"]

        # units
        self.length_unit = self.parameters["length_unit"]
        self.mass_unit = self.parameters["mass_unit"]
        self.velocity_unit = self.parameters["velocity_unit"]

        # make the multibody object
        if self.mother_machine:
            assert isinstance(self.mother_machine, dict), (
                "mother_machine must be a dictionary with keys "
                "spacer_thickness, channel_height, channel_space"
            )
        multibody_config = {
            "agent_shape": self.agent_shape,
            "jitter_force": jitter_force,
            "bounds": remove_units(self.bounds),
            "barriers": self.mother_machine,
            "physics_dt": self.parameters["timestep"] / 10,
            "seed": self.parameters["seed"],
        }
        self.physics = PymunkMultibody(multibody_config)

        # interactive plot for visualization
        self.animate = self.parameters["animate"]
        if self.animate:
            plt.ion()
            self.ax = plt.gca()
            self.ax.set_aspect("equal")

    def ports_schema(self):
        glob_schema = {
            "*": {
                self.parameters["boundary_key"]: {
                    "location": {
                        "_emit": True,
                        "_default": [0.5 * bound for bound in self.bounds],
                        "_updater": "set",
                        "_divider": {
                            "divider": daughter_locations,
                            "topology": {
                                "length": (
                                    "..",
                                    "length",
                                ),
                                "angle": (
                                    "..",
                                    "angle",
                                ),
                            },
                        },
                    },
                    "length": {"_emit": True, "_default": 2.0 * units.um},
                    "width": {"_emit": True, "_default": 1.0 * units.um},
                    "angle": {
                        "_emit": True,
                        "_default": 0.0,
                        "_updater": "set",
                    },
                    "mass": {
                        "_emit": True,
                        "_default": 1339 * units.fg,
                    },
                    "thrust": {
                        "_default": 0.0,
                        "_updater": "set",
                    },
                    "torque": {
                        "_default": 0.0,
                        "_updater": "set",
                    },
                }
            }
        }
        schema = {"agents": glob_schema}

        return schema

    def next_update(self, timestep, states):
        agents = states["agents"]

        # animate before update
        agents = remove_units(agents)
        if self.animate:
            self.animate_frame(agents)

        # update multibody with new agents
        self.physics.update_bodies(agents)

        # run simulation
        self.physics.run(timestep)

        # get new agent positions
        agent_positions = self.physics.get_body_positions()
        update = {"agents": agent_positions}

        # for mother machine configurations, remove agents above the channel height
        if self.mother_machine:
            channel_height = self.mother_machine["channel_height"]
            delete_agents = []
            for agent_id, position in agent_positions.items():
                location = position["boundary"]["location"]
                y_loc = location[1]
                if y_loc > channel_height:
                    # cell has moved past the channels
                    delete_agents.append(agent_id)
            if delete_agents:
                update["agents"] = {
                    agent_id: position
                    for agent_id, position in agent_positions.items()
                    if agent_id not in delete_agents
                }

                update["agents"]["_delete"] = [agent_id for agent_id in delete_agents]

        for agent in update["agents"].values():
            agent["boundary"]["location"] *= units.um

        return update

    ## matplotlib interactive plot
    def animate_frame(self, agents):
        plt.cla()
        for agent_id, data in agents.items():
            # location, orientation, length
            data = data["boundary"]
            x_center = data["location"][0]
            y_center = data["location"][1]
            angle = data["angle"] / PI * 180 + 90  # rotate 90 degrees to match field
            length = data["length"]
            width = data["width"]

            # get bottom left position
            x_offset = width / 2
            y_offset = length / 2
            theta_rad = math.radians(angle)
            dx = x_offset * math.cos(theta_rad) - y_offset * math.sin(theta_rad)
            dy = x_offset * math.sin(theta_rad) + y_offset * math.cos(theta_rad)

            x = x_center - dx
            y = y_center - dy

            if self.agent_shape == "rectangle" or self.agent_shape == "segment":
                # Create a rectangle
                rect = patches.Rectangle(
                    (x, y), width, length, angle=angle, linewidth=1, edgecolor="b"
                )
                self.ax.add_patch(rect)

            elif self.agent_shape == "circle":
                # Create a circle
                circle = patches.Circle((x, y), width, linewidth=1, edgecolor="b")
                self.ax.add_patch(circle)

        bounds = remove_units(self.bounds)

        plt.xlim([0, bounds[0]])
        plt.ylim([0, bounds[1]])
        plt.draw()
        plt.pause(0.01)


# configs
def make_random_position(bounds):
    unitless_bounds = [bound.to(units.um).magnitude for bound in bounds]
    return [
        np.random.uniform(0, unitless_bounds[0]),
        np.random.uniform(0, unitless_bounds[1]),
    ] * units.um


def single_agent_config(config):
    # cell dimensions
    width = 1.0 * units.um
    length = 2.0 * units.um
    volume = volume_from_length(length, width)
    bounds = config.get("bounds", DEFAULT_BOUNDS)
    location = config.get("location")
    if location:
        location = [loc * bounds[n] for n, loc in enumerate(location)]
    else:
        location = make_random_position(bounds)

    return {
        "boundary": {
            "location": location,
            "angle": np.random.uniform(0, 2 * PI),
            "volume": volume,
            "length": length,
            "width": width,
            "mass": 1339 * units.fg,
            "thrust": 0,
            "torque": 0,
        }
    }


def agent_body_config(config):
    agent_ids = config["agent_ids"]
    agent_config = {agent_id: single_agent_config(config) for agent_id in agent_ids}
    return {"agents": agent_config}


default_gd_config = {"bounds": DEFAULT_BOUNDS}
default_gd_config.update(
    agent_body_config({"bounds": DEFAULT_BOUNDS, "agent_ids": ["1", "2"]})
)


class InvokeUpdate(object):
    def __init__(self, update):
        self.update = update

    def get(self, timeout=0):
        return self.update


# tests and simulations
def test_multibody(n_agents=1, time=10, return_data=False):
    agent_ids = [str(agent_id) for agent_id in range(n_agents)]
    multibody_config = {
        "agents": agent_body_config({"bounds": DEFAULT_BOUNDS, "agent_ids": agent_ids})
    }

    multibody = Multibody(multibody_config)

    # initialize agent's boundary state
    initial_agents_state = multibody_config["agents"]
    initial_state = {"agents": initial_agents_state}
    experiment = process_in_experiment(multibody, initial_state=initial_state)

    # run experiment
    settings = {"timestep": 1, "total_time": time, "return_raw_data": True}
    data = simulate_experiment(experiment, settings)
    if return_data:
        return data


def test_growth_division(
    config=default_gd_config,
    growth_rate=0.05,
    growth_rate_noise=0.001,
    division_volume=0.4**3 * units.fL,
    total_time=10,
    timestep=1,
    experiment_settings=None,
    return_data=False,
):
    if not experiment_settings:
        experiment_settings = {}
    initial_agents_state = config["agents"]

    # make the process
    multibody = Multibody(config)
    experiment = process_in_experiment(multibody, experiment_settings)
    experiment.state._update_subschema(
        ("agents",),
        {
            "boundary": {
                "mass": {"_updater": "set", "_divider": "split"},
                "length": {"_updater": "set", "_divider": "split"},
                "volume": {"_updater": "set", "_divider": "split"},
            }
        },
    )
    experiment.state._apply_subschemas()

    # make initial agent state
    experiment.state.set_value({"agents": initial_agents_state})
    agents_store = experiment.state.get_path(["agents"])

    # emit initial state
    experiment._emit_store_data()

    # run simulation
    time = 0
    while time < total_time:
        experiment.update(timestep)
        time += timestep
        agents_state = agents_store.get_value()

        invoked_update = []
        for agent_id, state in agents_state.items():
            state = state["boundary"]
            length = state["length"]
            width = state["width"]
            mass = state["mass"]  # .magnitude

            # update
            growth_rate2 = (
                growth_rate + np.random.normal(0.0, growth_rate_noise)
            ) * timestep
            new_mass = mass + mass * growth_rate2
            new_length = length + length * growth_rate2
            new_volume = volume_from_length(new_length, width)

            if new_volume > division_volume:
                daughter_ids = [str(agent_id) + "0", str(agent_id) + "1"]
                daughter_updates = []
                for daughter_id in daughter_ids:
                    daughter_updates.append(
                        {
                            "key": daughter_id,
                            "processes": {},
                            "topology": {},
                            "initial_state": {},
                        }
                    )
                update = {
                    "_divide": {"mother": agent_id, "daughters": daughter_updates}
                }
            else:
                update = {
                    agent_id: {
                        "boundary": {
                            "volume": new_volume,
                            "length": new_length,
                            "mass": new_mass,
                        }
                    }
                }  # * units.fg

            invoked_update.append((InvokeUpdate({"agents": update}), None))

        # update experiment
        experiment._send_updates(invoked_update)

    experiment.end()
    if return_data:
        return experiment.emitter.get_data_unitless()


def run_growth_division(
    out_dir="out",
    animate=True,
):
    n_agents = 2
    agent_ids = [str(agent_id) for agent_id in range(n_agents)]

    # configure the multibody process
    bounds = DEFAULT_BOUNDS
    multibody_config = {
        "animate": animate,
        # 'jitter_force': 1e0,
        "bounds": bounds,
    }
    body_config = {"bounds": bounds, "agent_ids": agent_ids}
    multibody_config.update(agent_body_config(body_config))

    # experiment settings
    experiment_settings = {"progress_bar": False, "display_info": False}

    # run the test
    gd_data = test_growth_division(
        config=multibody_config,
        growth_rate=0.05,
        growth_rate_noise=0.001,
        division_volume=volume_from_length(4, 1) * units.fL,
        total_time=100,
        experiment_settings=experiment_settings,
        return_data=True,
    )

    agents, fields = format_snapshot_data(gd_data)
    return plot_snapshots(bounds, agents=agents, fields=fields, out_dir=out_dir)


def test_daughter_locations():
    locations = daughter_locations(
        [2, 2],
        {"length": 2 * math.sqrt(2), "angle": -math.pi / 4},
    )
    assert locations == [[1.5, 2.5], [2.5, 1.5]]


def main():
    out_dir = os.path.join(PROCESS_OUT_DIR, NAME)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    run_growth_division(out_dir)


if __name__ == "__main__":
    main()

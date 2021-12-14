import random
import math
import numpy as np

import pymunk


PI = math.pi
DEBUG_SIZE = 600  # size of the pygame debug screen


def get_force_with_angle(force, angle):
    x = force * math.cos(angle)
    y = force * math.sin(angle)
    return [x, y]


def front_from_corner(width, length, corner_position, angle):
    half_width = width/2
    dx = length * math.cos(angle) + half_width * math.cos(angle + PI/2)  # PI/2 gives a half-rotation for the width component
    dy = length * math.sin(angle) + half_width * math.sin(angle + PI/2)
    front_position = [corner_position[0] + dx, corner_position[1] + dy]
    return np.array([front_position[0], front_position[1], angle])


def corner_from_center(width, length, center_position, angle):
    half_length = length/2
    half_width = width/2
    dx = half_length * math.cos(angle) + half_width * math.cos(angle + PI/2)
    dy = half_length * math.sin(angle) + half_width * math.sin(angle + PI/2)
    corner_position = [center_position[0] - dx, center_position[1] - dy]
    return np.array([corner_position[0], corner_position[1], angle])


def random_body_position(body):
    ''' pick a random point along the boundary'''
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


class NullScreen(object):
    def update_screen(self):
        pass
    def configure(self, config):
        pass


class PymunkMultibody(object):
    """
    Multibody object for interfacing with pymunk
    """

    defaults = {
        'agent_shape': 'segment',
        # hardcoded parameters
        'elasticity': 0.9,
        'damping': 0.5,  # 1 is no damping, 0 is full damping
        'angular_damping': 0.8,
        'friction': 0.9,  # does this do anything?
        'physics_dt': 0.001,
        'force_scaling': 5e4,  # scales from pN
        # configured parameters
        'jitter_force': 1e-3,  # pN
        'bounds': [20, 20],
        'barriers': False,
        'initial_agents': {},
        # for debugging
        'screen': None,
    }

    def __init__(self, config):
        # hardcoded parameters
        self.elasticity = self.defaults['elasticity']
        self.friction = self.defaults['friction']
        self.damping = self.defaults['damping']
        self.angular_damping = self.defaults['angular_damping']
        self.physics_dt = config.get('physics_dt', self.defaults['physics_dt'])
        self.force_scaling = self.defaults['force_scaling']

        # configured parameters
        self.agent_shape = config.get('agent_shape', self.defaults['agent_shape'])
        self.jitter_force = config.get('jitter_force', self.defaults['jitter_force'])
        self.bounds = config.get('bounds', self.defaults['bounds'])
        barriers = config.get('barriers', self.defaults['barriers'])

        # initialize pymunk space
        self.space = pymunk.Space()

        # debug screen
        self.screen = config.get('screen')
        if self.screen is None:
            self.screen = NullScreen()
        self.screen.configure({
            'space': self.space,
            'bounds': self.bounds})

        # add static barriers
        self.add_barriers(self.bounds, barriers)

        # initialize agents
        initial_agents = config.get('initial_agents', self.defaults['initial_agents'])
        self.bodies = {}
        for agent_id, specs in initial_agents.items():
            self.add_body_from_center(agent_id, specs)

    def run(self, timestep):
        if self.physics_dt > timestep:
            print('timestep skipped by pymunk_multibody: {}'.format(timestep))
            return

        time = 0
        while time < timestep:
            time += self.physics_dt

            # apply forces
            for body in self.space.bodies:
                self.apply_jitter_force(body)
                self.apply_motile_force(body)
                self.apply_viscous_force(body)

            # run for a physics timestep
            self.space.step(self.physics_dt)

        self.screen.update_screen()

    def apply_motile_force(self, body):
        width, length = body.dimensions
        motile_location = (width / 2, 0)  # apply force at back end of body
        thrust = 0.0
        torque = 0.0
        motile_force = [thrust, torque]

        if hasattr(body, 'thrust'):
            thrust = body.thrust
            torque = body.torque
            motile_force = [thrust, 0.0]

            # add to angular velocity
            body.angular_velocity += torque

        scaled_motile_force = [force * self.force_scaling for force in motile_force]
        body.apply_impulse_at_local_point(scaled_motile_force, motile_location)

    def apply_jitter_force(self, body):
        jitter_location = random_body_position(body)
        jitter_force = [
            random.normalvariate(0, self.jitter_force),
            random.normalvariate(0, self.jitter_force)]
        scaled_jitter_force = [
            force * self.force_scaling
            for force in jitter_force]
        body.apply_impulse_at_local_point(
            scaled_jitter_force,
            jitter_location)

    def apply_viscous_force(self, body):
        # dampen velocity
        body.velocity = body.velocity * self.damping + (body.force / body.mass) * self.physics_dt

        # dampen angular velocity
        body.angular_velocity = body.angular_velocity * self.angular_damping + (body.torque / body.moment) * self.physics_dt

    def add_barriers(self, bounds, barriers):
        """ Create static barriers """
        thickness = 50.0
        offset = thickness
        x_bound = bounds[0]
        y_bound = bounds[1]

        static_body = self.space.static_body
        static_lines = [
            pymunk.Segment(
                static_body,
                (0.0-offset, 0.0-offset),
                (x_bound+offset, 0.0-offset),
                thickness),
            pymunk.Segment(
                static_body,
                (x_bound+offset, 0.0-offset),
                (x_bound+offset, y_bound+offset),
                thickness),
            pymunk.Segment(
                static_body,
                (x_bound+offset, y_bound+offset),
                (0.0-offset, y_bound+offset),
                thickness),
            pymunk.Segment(
                static_body,
                (0.0-offset, y_bound+offset),
                (0.0-offset, 0.0-offset),
                thickness),
        ]

        if barriers:
            assert isinstance(barriers, dict)
            spacer_thickness = barriers.get('spacer_thickness', 0.1)
            channel_height = barriers.get('channel_height', 0.7 * bounds[1])
            channel_space = barriers.get('channel_space', 1.5)
            n_lines = math.floor(x_bound/channel_space)

            machine_lines = [
                pymunk.Segment(
                    static_body,
                    (channel_space * line, 0),
                    (channel_space * line, channel_height), spacer_thickness)
                for line in range(n_lines)]
            static_lines += machine_lines

        for line in static_lines:
            line.elasticity = 0.0  # bounce
            line.friction = 0.8
            self.space.add(line)

    def get_shape(self, boundary):
        '''
        shape documentation at: https://pymunk-tutorial.readthedocs.io/en/latest/shape/shape.html
        '''

        if self.agent_shape == 'segment':
            width = boundary['width']
            length = boundary['length']

            half_width = width / 2
            half_length = length / 2 - half_width
            shape = pymunk.Segment(
                None,
                (-half_length, 0),
                (half_length, 0),
                radius=half_width)

        elif self.agent_shape == 'circle':
            length = boundary['length']
            half_length = length / 2
            shape = pymunk.Circle(None, radius=half_length, offset=(0, 0))

        elif self.agent_shape == 'rectangle':
            width = boundary['width']
            length = boundary['length']
            half_length = length / 2
            half_width = width / 2
            shape = pymunk.Poly(None,
                ((-half_length, -half_width),
                 (half_length, -half_width),
                 (half_length, half_width),
                 (-half_length, half_width)))

        return shape

    def get_inertia(self, shape, mass):
        if self.agent_shape == 'rectangle':
            inertia = pymunk.moment_for_poly(mass, shape.get_vertices())
        elif self.agent_shape == 'circle':
            radius = shape.radius
            inertia = pymunk.moment_for_circle(mass, radius, radius)
        elif self.agent_shape == 'segment':
            a = shape.a
            b = shape.b
            radius = shape.radius
            inertia = pymunk.moment_for_segment(mass, a, b, radius)

        return inertia

    def add_body_from_center(self, body_id, specs):
        boundary = specs['boundary']
        mass = boundary['mass']
        center_position = boundary['location']
        angle = boundary['angle']
        angular_velocity = boundary.get('angular_velocity', 0.0)
        width = boundary['width']
        length = boundary['length']

        # get shape, inertia, make body, assign body to shape
        shape = self.get_shape(boundary)
        inertia = self.get_inertia(shape, mass)
        body = pymunk.Body(mass, inertia)
        shape.body = body

        body.position = (
            center_position[0],
            center_position[1])
        body.angle = angle
        body.dimensions = (width, length)
        body.angular_velocity = angular_velocity

        shape.elasticity = self.elasticity
        shape.friction = self.friction

        # add body and shape to space
        self.space.add(body, shape)

        # add body to agents dictionary
        self.bodies[body_id] = (body, shape)

    def update_body(self, body_id, specs):
        boundary = specs['boundary']
        length = boundary['length']
        width = boundary['width']
        mass = boundary['mass']
        thrust = boundary['thrust']
        torque = boundary['torque']

        body, shape = self.bodies[body_id]
        position = body.position
        angle = body.angle

        # get shape, inertia, make body, assign body to shape
        new_shape = self.get_shape(boundary)
        inertia = self.get_inertia(new_shape, mass)
        new_body = pymunk.Body(mass, inertia)
        new_shape.body = new_body

        new_body.position = position
        new_body.angle = angle
        new_body.velocity = body.velocity
        new_body.angular_velocity = body.angular_velocity
        new_body.dimensions = (width, length)
        new_body.thrust = thrust
        new_body.torque = torque

        new_shape.elasticity = shape.elasticity
        new_shape.friction = shape.friction

        # swap bodies
        self.space.remove(body, shape)
        self.space.add(new_body, new_shape)

        # update body
        self.bodies[body_id] = (new_body, new_shape)

    def update_bodies(self, bodies):
        # if an agent has been removed from the agents store,
        # remove it from space and bodies
        removed_bodies = [
            body_id for body_id in self.bodies.keys()
            if body_id not in bodies.keys()]
        for body_id in removed_bodies:
            body, shape = self.bodies[body_id]
            self.space.remove(body, shape)
            del self.bodies[body_id]

        # update agents, add new agents
        for body_id, specs in bodies.items():
            if body_id in self.bodies:
                self.update_body(body_id, specs)
            else:
                self.add_body_from_center(body_id, specs)

    def get_body_position(self, agent_id):
        body, shape = self.bodies[agent_id]
        return {
            'location': [pos for pos in body.position],
            'angle': body.angle,
        }

    def get_body_positions(self):
        return {
            body_id: {
                'boundary': self.get_body_position(body_id)}
            for body_id in self.bodies.keys()}



def test_multibody(
        total_time=2,
        agent_shape='rectangle',
        n_agents=1,
        jitter_force=1e1,
        screen=None):

    bounds = [500, 500]
    center_location = [0.5*loc for loc in bounds]
    agents = {
        str(agent_idx): {
            'boundary': {
                'location': center_location,
                'angle': random.uniform(0,2*PI),
                'volume': 15,
                'length': 30,
                'width': 10,
                'mass': 1,
                'thrust': 1e3,
                'torque': 0.0}}
        for agent_idx in range(n_agents)
    }
    config = {
        'agent_shape': agent_shape,
        'jitter_force': jitter_force,
        'bounds': bounds,
        'barriers': False,
        'initial_agents': agents,
        'screen': screen
    }
    multibody = PymunkMultibody(config)

    # run simulation
    time = 0
    time_step = 1
    while time < total_time:
        time += time_step
        multibody.run(time_step)


if __name__ == '__main__':
    test_multibody(10)

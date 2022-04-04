'''
===============
Diffusion Field
===============
'''

import sys
import os
import argparse
import copy

import numpy as np
from scipy import constants
from scipy.ndimage import convolve

from vivarium.core.process import Process, assoc_path
from vivarium.core.engine import Engine
from vivarium.core.composition import (
    PROCESS_OUT_DIR
)
from vivarium.library.units import units, remove_units
from vivarium.library.topology import get_in

from ecoli.library.lattice_utils import (
    count_to_concentration,
    get_bin_site,
    get_bin_volume,
    make_gradient,
    apply_exchanges,
    ExchangeAgent,
)
from ecoli.plots.snapshots import plot_snapshots

NAME = 'diffusion_field'

# laplacian kernel for diffusion
LAPLACIAN_2D = np.array([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
AVOGADRO = constants.N_A


class DiffusionField(Process):
    '''
    Diffusion in 2-dimensional fields of molecules with agent exchange

    Agent uptake and secretion occurs at agent locations.

    Notes:

    * Diffusion constant of glucose in 0.5 and 1.5 percent agarose gel
      is around :math:`6 * 10^{-10} \\frac{m^2}{s}` (Weng et al. 2005.
      Transport of glucose and poly(ethylene glycol)s in agarose gels).
    * Conversion to micrometers:
      :math:`6 * 10^{-10} \\frac{m^2}{s}=600 \\frac{micrometers^2}{s}`.
    '''

    name = NAME
    defaults = {
        'time_step': 1,
        'molecules': ['glc'],
        'initial_state': {},
        'n_bins': [10, 10],
        'bounds': [10 * units.um, 10 * units.um],
        'depth': 3000.0 * units.um,  # um
        'diffusion': 5e-1 * units.um**2 / units.sec,
        'gradient': {},
        'boundary_path': ('boundary',)
    }

    def __init__(self, parameters=None):
        super().__init__(parameters)

        # initial state
        self.molecule_ids = self.parameters['molecules']
        self.initial = self.parameters['initial_state']

        # parameters
        self.n_bins = self.parameters['n_bins']
        self.bounds = self.parameters['bounds']
        depth = self.parameters['depth']

        # diffusion
        diffusion = self.parameters['diffusion']
        bins_x = self.n_bins[0]
        bins_y = self.n_bins[1]
        length_x = self.bounds[0]
        length_y = self.bounds[1]
        dx = length_x / bins_x
        dy = length_y / bins_y
        dx2 = dx * dy
        self.diffusion = diffusion / dx2
        self.diffusion_dt = 0.01 * units.sec
        # self.diffusion_dt = 0.5 * dx ** 2 * dy ** 2 / (2 * self.diffusion * (dx ** 2 + dy ** 2))

        # volume, to convert between counts and concentration
        self.bin_volume = get_bin_volume(self.n_bins, self.bounds, depth)

        # initialize gradient fields
        gradient = self.parameters['gradient']
        if gradient:
            unitless_bounds = [
                bound.to(units.um).magnitude for bound in self.bounds]
            gradient_fields = make_gradient(
                gradient, self.n_bins, unitless_bounds)
            self.initial.update(gradient_fields)

    def initial_state(self, config=None):
        return {
            'fields': {
                field: self.initial.get(field, self.ones_field())
                for field in self.molecule_ids
            },
        }

    def ports_schema(self):

        # place the agent boundary schema at the configured boundary path
        boundary_path = self.parameters['boundary_path']
        boundary_schema = {
            'location': {
                '_default': [
                    0.5 * bound
                    for bound in self.bounds],
                '_emit': True,
            },
            'external': {
                molecule: {
                    '_default': 0.0 * units.mM}
                for molecule in self.molecule_ids
            },
            'exchanges': {
                molecule: {
                    '_default': 0.0}
                for molecule in self.molecule_ids
            },
        }
        agent_schema = assoc_path({}, boundary_path, boundary_schema)

        # make the full schema
        schema = {
            'agents': {
                '*': agent_schema
            },
            'fields': {
                field: {
                    '_default': self.ones_field(),
                    '_updater': 'nonnegative_accumulate',
                    '_emit': True,
                }
                for field in self.molecule_ids
            },
            'dimensions': {
                'bounds': {
                    '_default': self.parameters['bounds'],
                    '_updater': 'set',
                    '_emit': True,
                },
                'n_bins': {
                    '_default': self.parameters['n_bins'],
                    '_updater': 'set',
                    '_emit': True,
                },
                'depth': {
                    '_default': self.parameters['depth'],
                    '_updater': 'set',
                    '_emit': True,
                }
            },
        }
        return schema

    def next_update(self, timestep, states):
        fields = states['fields']
        agents = states['agents']

        # make new fields for the updated state
        new_fields = copy.deepcopy(fields)

        ###################
        # apply exchanges #
        ###################
        new_fields, agent_updates = apply_exchanges(
            agents, new_fields,
            self.parameters['boundary_path'],
            self.n_bins, self.bounds, self.bin_volume)

        # diffuse field
        new_fields = self.diffuse(new_fields, timestep)

        # get total delta from exchange, diffusion, reaction
        delta_fields = {
            mol_id: new_fields[mol_id] - field
            for mol_id, field in fields.items()}

        # get each agent's new local environment
        local_environments = self.get_local_environments(agents, new_fields)

        update = {'fields': delta_fields}
        if local_environments:
            update.update({'agents': local_environments})

        return update

    def count_to_concentration(self, count):
        return count_to_concentration(
            count, self.bin_volume
        ).to(units.mmol / units.L)

    def get_bin_site(self, location):
        return get_bin_site(location, self.n_bins, self.bounds)

    def get_single_local_environments(self, specs, fields):
        bin_site = self.get_bin_site(specs['location'])
        local_environment = {}
        for mol_id, field in fields.items():
            local_environment[mol_id] = {
                '_value': field[bin_site] * units.mM,
                '_updater': 'set'}
        return local_environment

    def get_local_environments(self, agents, fields):
        local_environments = {}
        if agents:
            for agent_id, specs in agents.items():
                local_environments[agent_id] = {'boundary': {}}
                local_environments[agent_id]['boundary']['external'] = \
                    self.get_single_local_environments(specs['boundary'], fields)
        return local_environments

    def ones_field(self):
        return np.ones((self.n_bins[0], self.n_bins[1]), dtype=np.float64)

    # diffusion functions
    def diffusion_delta(self, field, timestep):
        ''' calculate concentration changes cause by diffusion'''
        field_new = field.copy()
        t = 0.0
        dt = min(timestep, self.diffusion_dt.to(units.sec).magnitude)
        diffusion = self.diffusion.to(1 / units.sec).magnitude
        while t < timestep:
            field_new += diffusion * dt * convolve(field_new, LAPLACIAN_2D, mode='reflect')
            t += dt

        return field_new - field, field_new

    def diffuse(self, fields, timestep):
        new_fields = {}
        for mol_id, field in fields.items():

            # run diffusion if molecule field is not uniform
            if len(set(field.flatten())) != 1:
                delta, new_field = self.diffusion_delta(field, timestep)
            else:
                delta = np.zeros_like(field)
                new_field = field
            new_fields[mol_id] = new_field

        return new_fields


# testing
def get_random_field_config():
    n_bins = (20, 20)
    return {
        'molecules': ['glc'],
        'initial_state': {
            'glc': 2.0 * np.random.rand(n_bins[0], n_bins[1])},
        'n_bins': n_bins,
        'bounds': (20, 20) * units.um,
        'depth': 5 * units.um,
        'diffusion': 1e-2 * units.um**2 / units.sec,  # slow diffusion
    }


def get_gaussian_config():
    return {
        'molecules': ['glc'],
        'n_bins': (20, 20),
        'bounds': (20, 20) * units.um,
        'depth': 100 * units.um,
        'gradient': {
            'type': 'gaussian',
            'molecules': {
                'glc': {
                    'center': [0.5, 0.5],
                    'deviation': 1}}}}


def get_exponential_config():
    return {
        'molecules': ['glc'],
        'n_bins': (20, 20),
        'bounds': (20, 20) * units.um,
        'depth': 100 * units.um,
        'gradient': {
            'type': 'exponential',
            'molecules': {
                'glc': {
                    'center': [1.0, 1.0],
                    'base': 1 + 1e-3,
                    'scale': 10.0}}}}

def test_all():
    run_diffusion_field(
        config=get_random_field_config(), total_time=60, filename='random')
    run_diffusion_field(
        config=get_gaussian_config(), total_time=60, filename='gaussian')
    run_diffusion_field(
        config=get_exponential_config(), total_time=60, filename='exponential')


def plot_fields(data, config, out_dir='out', filename='fields'):
    unitless_config = remove_units(config)
    fields = {time: time_data['fields'] for time, time_data in data.items()}
    plot_snapshots(
        unitless_config['bounds'],
        fields=fields,
        out_dir=out_dir,
        filename=filename)


def run_diffusion_field(config=None, total_time=100, filename='snapshots'):
    config = config or {}
    diff_process = DiffusionField(config)

    # make the toy exchange agent
    agent_id = '0'
    agent_params = {
        'mol_ids': ['glc'],
        'default_exchange': 100,
        'max_move': 1.0,
    }
    agent_process = ExchangeAgent(agent_params)

    # get initial fields
    initial_fields = diff_process.initial_state()

    # put them together in a simulation
    sim = Engine(
        processes={
            'diff': diff_process,
            'agents': {
                agent_id: {
                    'exchange': agent_process
                }
            }
        },
        topology={
            'diff': {
                port: (port,)
                for port in diff_process.ports_schema().keys()
            },
            'agents': {
                agent_id: {
                    'exchange': {'boundary': ('boundary',)}
                }
            }
        },
        initial_state=initial_fields
    )
    sim.update(total_time)

    # plot
    data = sim.emitter.get_data_unitless()

    # add empty angle back in for the plot (this is undesirable)
    for t in data.keys():
        data[t]['agents'][agent_id]['boundary']['angle'] = 0.0
        data[t]['agents'][agent_id]['boundary']['length'] = 1.0
        data[t]['agents'][agent_id]['boundary']['width'] = 1.0

    out_dir = os.path.join(PROCESS_OUT_DIR, 'environment', NAME)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    snapshots_fig = plot_snapshots(
        n_snapshots=6,
        bounds=get_in(data, (max(data), 'dimensions', 'bounds')),
        agents={
            time: d['agents']
            for time, d in data.items()
        },
        fields={
            time: d['fields']
            for time, d in data.items()
        },
        out_dir=out_dir,
        filename=filename,
    )


# python ecoli/processes/environment/diffusion_field.py
if __name__ == '__main__':
    # test_all()

    parser = argparse.ArgumentParser(description='diffusion_field')
    parser.add_argument('--random', '-r', action='store_true', default=False)
    parser.add_argument('--gaussian', '-g', action='store_true', default=False)
    parser.add_argument('--exponential', '-e', action='store_true', default=False)
    args = parser.parse_args()
    no_args = (len(sys.argv) == 1)

    if no_args:
        test_all()
    if args.random:
        run_diffusion_field(
            config=get_random_field_config(), total_time=60, filename='random')
    if args.gaussian:
        run_diffusion_field(
            config=get_gaussian_config(), total_time=60, filename='gaussian')
    if args.exponential:
        run_diffusion_field(
            config=get_exponential_config(), total_time=60, filename='exponential')
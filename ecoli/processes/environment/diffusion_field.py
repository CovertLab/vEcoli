'''
===============
Diffusion Field
===============
'''

import sys
import os
import argparse

import numpy as np
from scipy import constants
from scipy.ndimage import convolve

from vivarium.core.process import Process
from vivarium.core.composition import (
    simulate_process,
    PROCESS_OUT_DIR
)
from vivarium.library.units import units

from ecoli.library.lattice_utils import (
    count_to_concentration,
    get_bin_site,
    get_bin_volume,
    make_gradient,
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

    def initial_state(self, config):
        return {
            'fields': {
                field: self.initial.get(field, self.ones_field())
                for field in self.molecule_ids
            },
        }

    def ports_schema(self):
        local_concentration_schema = {
            molecule: {
                '_default': 0.0 * units.mM
            }
            for molecule in self.molecule_ids}

        # agents glob schema
        schema = {
            'agents': {
                '*': {
                    'boundary': {
                        'location': {
                            '_default': [
                                0.5 * bound for bound in self.bounds],
                            '_updater': 'set'},
                        'external': local_concentration_schema,
                    },
                },
            },
        }

        # fields
        fields_schema = {
            'fields': {
                field: {
                    '_default': self.ones_field(),
                    '_updater': 'nonnegative_accumulate',
                    '_emit': True,
                }
                for field in self.molecule_ids
            },
        }
        schema.update(fields_schema)

        # dimensions
        dimensions_schema = {
            'dimensions': {
                'bounds': {
                    '_value': self.parameters['bounds'],
                    '_updater': 'set',
                    '_emit': True,
                },
                'n_bins': {
                    '_value': self.parameters['n_bins'],
                    '_updater': 'set',
                    '_emit': True,
                },
                'depth': {
                    '_value': self.parameters['depth'],
                    '_updater': 'set',
                    '_emit': True,
                }
            },
        }
        schema.update(dimensions_schema)
        return schema

    def next_update(self, timestep, states):
        fields = states['fields']
        agents = states['agents']

        # diffuse field
        delta_fields, new_fields = self.diffuse(fields, timestep)

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
        delta_fields = {}
        new_fields = {}
        for mol_id, field in fields.items():

            # run diffusion if molecule field is not uniform
            if len(set(field.flatten())) != 1:
                delta, new_field = self.diffusion_delta(field, timestep)
            else:
                delta = np.zeros_like(field)
                new_field = field
            delta_fields[mol_id] = delta
            new_fields[mol_id] = new_field

        return delta_fields, new_fields


# testing
def get_random_field_config(config={}):
    bounds = config.get('bounds', (20, 20) * units.um)
    n_bins = config.get('n_bins', (10, 10))
    return {
        'molecules': ['glc'],
        'initial_state': {
            'glc': np.random.rand(n_bins[0], n_bins[1])},
        'n_bins': n_bins,
        'bounds': bounds}


def get_gaussian_config(config={}):
    molecules = config.get('molecules', ['glc'])
    bounds = config.get('bounds', (50, 50) * units.um)
    n_bins = config.get('n_bins', (20, 20))
    center = config.get('center', [0.5, 0.5])
    deviation = config.get('deviation', 5)
    diffusion = config.get('diffusion', 5e-1 * units.um**2 / units.sec)

    return {
        'molecules': molecules,
        'n_bins': n_bins,
        'bounds': bounds,
        'diffusion': diffusion,
        'gradient': {
            'type': 'gaussian',
            'molecules': {
                'glc': {
                    'center': center,
                    'deviation': deviation}}}}


def get_exponential_config(config={}):
    molecules = config.get('molecules', ['glc'])
    bounds = config.get('bounds', (40, 40) * units.um)
    n_bins = config.get('n_bins', (20, 20))
    center = config.get('center', [1.0, 1.0])
    base = config.get('base', 1 + 2e-4)
    scale = config.get('scale', 0.1)
    diffusion = config.get('diffusion', 1e1 * units.um**2 / units.sec)

    return {
        'molecules': molecules,
        'n_bins': n_bins,
        'bounds': bounds,
        'diffusion': diffusion,
        'gradient': {
            'type': 'exponential',
            'molecules': {
                'glc': {
                    'center': center,
                    'base': base,
                    'scale': scale}}}}


def test_diffusion_field(
        config={},
        initial_state={},
        time=10,
):
    diffusion = DiffusionField(config)
    settings = {
        'return_raw_data': True,
        'initial_state': initial_state,
        'total_time': time,
        'timestep': 1}
    return simulate_process(diffusion, settings)


def test_all():
    test_diffusion_field(
        config=get_random_field_config(), initial_state={}, time=60)
    test_diffusion_field(
        config=get_gaussian_config(), initial_state={}, time=60)
    test_diffusion_field(
        config=get_exponential_config(), initial_state={}, time=60)


def plot_fields(data, config, out_dir='out', filename='fields'):
    fields = {time: time_data['fields'] for time, time_data in data.items()}
    snapshots_data = {
        'fields': fields,
        'config': config}
    plot_config = {
        'out_dir': out_dir,
        'filename': filename}
    plot_snapshots(snapshots_data, plot_config)


if __name__ == '__main__':
    out_dir = os.path.join(PROCESS_OUT_DIR, NAME)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    parser = argparse.ArgumentParser(description='diffusion_field')
    parser.add_argument('--random', '-r', action='store_true', default=False)
    parser.add_argument('--gaussian', '-g', action='store_true', default=False)
    parser.add_argument('--exponential', '-e', action='store_true', default=False)
    args = parser.parse_args()
    no_args = (len(sys.argv) == 1)

    if args.random or no_args:
        config = get_random_field_config()
        data = test_diffusion_field(
            config=config,
            initial_state={},
            time=60)
        plot_fields(data, config, out_dir, 'random_field')

    if args.gaussian or no_args:
        config = get_gaussian_config()
        data = test_diffusion_field(
            config=config,
            initial_state={},
            time=60)
        plot_fields(data, config, out_dir, 'gaussian_field')

    if args.exponential or no_args:
        config = get_exponential_config()
        data = test_diffusion_field(
            config=config,
            initial_state={},
            time=60)
        plot_fields(data, config, out_dir, 'exponential_field')

'''
========================
Reaction Diffusion Field
========================
'''

import os
import numpy as np
from scipy import constants
from scipy.ndimage import convolve

from vivarium.core.process import Process
from vivarium.core.composition import PROCESS_OUT_DIR
from vivarium.core.composer import Composer
from vivarium.core.engine import Engine
from vivarium.library.units import units

from ecoli.library.lattice_utils import (
    count_to_concentration,
    get_bin_site,
    get_bin_volume, make_gradient,
)
from vivarium.library.topology import get_in
from ecoli.plots.snapshots import plot_snapshots


NAME = 'reaction_diffusion_field'

# laplacian kernel for diffusion
LAPLACIAN_2D = np.array([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
AVOGADRO = constants.N_A


class ReactionDiffusionField(Process):
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
        'bounds': [10, 10],
        'depth': 3000.0,  # um
        'diffusion': 5e-1,
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
        self.diffusion_dt = 0.01
        # self.diffusion_dt = 0.5 * dx ** 2 * dy ** 2 / (2 * self.diffusion * (dx ** 2 + dy ** 2))

        # volume, to convert between counts and concentration
        self.bin_volume = get_bin_volume(self.n_bins, self.bounds, depth)

        # initialize gradient fields
        gradient = self.parameters['gradient']
        if gradient:
            gradient_fields = make_gradient(gradient, self.n_bins, self.bounds)
            self.initial.update(gradient_fields)

    def initial_state(self, config):
        return {
            'fields': {
                field: self.initial.get(field, self.ones_field())
                for field in self.molecule_ids
            },
        }

    def ports_schema(self):
        schema = {
            'agents': {
                '*': {
                    'boundary': {
                        'location': {
                            '_default': [
                                0.5 * bound
                                for bound in self.bounds],
                            '_emit': True,
                        },
                        'external': {
                            molecule: {
                                '_default': 0.0}
                            for molecule in self.molecule_ids
                        },
                        'exchanges': {
                            molecule: {
                                '_default': 0.0}
                            for molecule in self.molecule_ids
                        },
                        'angle': {'_default': 0.0, '_emit': True},  # TODO -- remove this, should not be required
                        'length': {'_default': 0.0, '_emit': True},  # TODO -- remove this, should not be required
                        'width': {'_default': 0.0, '_emit': True},  # TODO -- remove this, should not be required
                    }
                }
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
        return schema

    def next_update(self, timestep, states):
        fields = states['fields']
        agents = states['agents']

        ###################
        # apply exchanges #
        ###################

        # track changes to the field from exchange
        exchange_delta_fields = {}
        for field_id, field in fields.items():
            exchange_delta_fields[field_id] = np.zeros_like(field)

        # apply exchanges from each agent
        agent_updates = {}
        for agent_id, agent_state in agents.items():
            boundary = agent_state['boundary']
            exchanges = boundary['exchanges']
            location = boundary['location']
            bin_site = get_bin_site(location, self.n_bins, self.bounds)

            reset_exchanges = {}
            for mol_id, value in exchanges.items():

                # delta concentration
                exchange = value * units.count
                bin_volume = self.bin_volume * units.L
                concentration = count_to_concentration(exchange, bin_volume).to(
                    units.mmol / units.L).magnitude

                delta_field = np.zeros((self.n_bins[0], self.n_bins[1]), dtype=np.float64)
                delta_field[bin_site[0], bin_site[1]] += concentration
                exchange_delta_fields[mol_id] += delta_field

                # reset the exchange value
                reset_exchanges[mol_id] = {
                    '_value': -value,
                    '_updater': 'accumulate'}

            agent_updates[agent_id] = {'exchanges': reset_exchanges}

        # the updated fields, which get passed to diffusion
        updated_fields = {
            mol_id: field + exchange_delta_fields[mol_id]
            for mol_id, field in fields.items()}

        #################
        # diffuse field #
        #################

        diffusion_delta_fields, new_fields = self.diffuse(updated_fields, timestep)

        # get total delta from exchange, diffusion, reaction
        delta_fields = {
            mol_id: exchange_delta_fields[mol_id] + diffusion_delta_fields[mol_id]
            for mol_id, field in fields.items()}

        # get each agent's new local environment
        local_environments = self.get_local_environments(agents, new_fields)

        update = {
            'fields': delta_fields,
            'agents': local_environments}

        return update

    def count_to_concentration(self, count):
        return count_to_concentration(
            count * units.count, self.bin_volume * units.L
        ).to(units.mmol / units.L).magnitude

    def get_bin_site(self, location):
        return get_bin_site(location, self.n_bins, self.bounds)

    def get_single_local_environments(self, specs, fields):
        bin_site = self.get_bin_site(specs['location'])
        local_environment = {}
        for mol_id, field in fields.items():
            local_environment[mol_id] = {
                '_value': field[bin_site],
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
        dt = min(timestep, self.diffusion_dt)
        while t < timestep:
            field_new += self.diffusion * dt * convolve(field_new, LAPLACIAN_2D, mode='reflect')
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


class ExchangeAgent(Process):
    defaults = {
        'mol_ids': ['glc'],
        'default_exchange': 100,
    }
    def __init__(self, parameters=None):
        super().__init__(parameters)
    def ports_schema(self):
        return {
            'boundary': {
                'external': {
                    '*': {'_default': 0.0}},
                'exchanges': {
                    '*': {'_default': self.parameters['default_exchange']}},
            }
        }
    def next_update(self, timestep, states):
        max_ex = self.parameters['default_exchange']
        max_move = 1.0
        return {
            'boundary': {
                'exchanges': {
                    mol_id: np.random.uniform(-max_ex, max_ex)
                    for mol_id in self.parameters['mol_ids']},
                'location': [
                    np.random.uniform(-max_move, max_move),
                    np.random.uniform(-max_move, max_move)
                ],
            }
        }


def main():
    total_time = 100

    # make the reaction diffusion process
    params = {'depth': 10}
    rxn_diff_process = ReactionDiffusionField(params)

    # make the toy exchange agent
    agent_id = '0'
    agent_params = {}
    agent_process = ExchangeAgent(agent_params)

    # put them together in a simulation
    sim = Engine(
        processes={
            'rxn_diff': rxn_diff_process,
            'agents': {
                agent_id: {
                    'exchange': agent_process
                }
            }
        },
        topology={
            'rxn_diff': {
                port: (port,)
                for port in rxn_diff_process.ports_schema().keys()
            },
            'agents': {
                agent_id: {
                    'exchange': {'boundary': ('boundary',)}
                }
            }
        }
    )
    sim.update(total_time)

    # plot
    data = sim.emitter.get_data()
    out_dir = os.path.join(PROCESS_OUT_DIR, 'environment', NAME)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    snapshots_fig = plot_snapshots(
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
        filename='snapshots',
    )


# python ecoli/processes/environment/reaction_diffusion_field.py
if __name__ == '__main__':
    main()
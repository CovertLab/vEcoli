'''
========================
Reaction Diffusion Field
========================
'''

import os
import numpy as np
from scipy import constants
from scipy.ndimage import convolve

from vivarium.core.process import Process, assoc_path
from vivarium.core.composition import PROCESS_OUT_DIR
from vivarium.core.engine import Engine
from vivarium.library.units import units

from ecoli.library.lattice_utils import (
    count_to_concentration,
    get_bin_site,
    get_bin_volume,
)
from vivarium.library.topology import get_in
from ecoli.plots.snapshots import plot_snapshots


NAME = 'reaction_diffusion'

# laplacian kernel for diffusion
LAPLACIAN_2D = np.array([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
AVOGADRO = constants.N_A


class ReactionDiffusion(Process):
    '''
    Reaction and Diffusion in 2-dimensional fields of molecules, with agent
    uptake and secretion at agent locations.

    Parameters: TODO

    '''

    name = NAME
    defaults = {
        'time_step': 1,
        'molecules': [],
        'n_bins': [10, 10],
        'bounds': [10, 10],
        'depth': 3000.0,  # um
        'diffusion': 5e-1,
        'reactions': {},
        'kinetic_parameters': {},
        'boundary_path': ('boundary',)
    }

    def __init__(self, parameters=None):
        super().__init__(parameters)

        # initial state
        self.molecule_ids = self.parameters['molecules']

        # parameters
        self.n_bins = self.parameters['n_bins']
        self.bounds = self.parameters['bounds']
        depth = self.parameters['depth']
        self.bin_volume = get_bin_volume(self.n_bins, self.bounds, depth)

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

    def initial_state(self, config):
        """
        sets uniform initial state at the concentration provided for each the molecule_id in `config`
        """
        return {
            'fields': {
                mol_id: config.get(mol_id, 0.0) * self.ones_field()
                for mol_id in self.molecule_ids
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
        dimensions = states['dimensions']

        # volume, to convert between counts and concentration
        self.n_bins = dimensions['n_bins']
        self.bounds = dimensions['bounds']
        self.bin_volume = get_bin_volume(
            self.n_bins,
            self.bounds,
            dimensions['depth'])

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
            boundary = get_in(agent_state, self.parameters['boundary_path'])
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
        new_fields = {
            mol_id: field + exchange_delta_fields[mol_id]
            for mol_id, field in fields.items()}

        #####################
        # react and diffuse #
        #####################

        reaction_delta_fields, new_fields = self.react(new_fields, timestep)
        diffusion_delta_fields, new_fields = self.diffuse(new_fields, timestep)

        # get total delta from exchange, diffusion, reaction
        delta_fields = {
            mol_id: exchange_delta_fields[mol_id] + diffusion_delta_fields[mol_id] + reaction_delta_fields[mol_id]
            for mol_id, field in fields.items()}

        # get each agent's new local environment
        local_environments = self.get_local_environments(agents, new_fields)

        update = {
            'fields': delta_fields,
            'agents': local_environments}

        return update

    def get_single_local_environments(self, specs, fields):
        bin_site = get_bin_site(specs['location'], self.n_bins, self.bounds)
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
                    self.get_single_local_environments(
                        get_in(specs, self.parameters['boundary_path']),
                        fields)
        return local_environments

    def zeros_field(self):
        return np.zeros((self.n_bins[0], self.n_bins[1]), dtype=np.float64)

    def ones_field(self):
        return np.ones((self.n_bins[0], self.n_bins[1]), dtype=np.float64)

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

    def react(self, fields, timestep):
        delta_fields = {field_id: self.zeros_field() for field_id in self.molecule_ids}
        new_fields = fields.copy()
        reactions = self.parameters['reactions']
        kinetic_parameters = self.parameters['kinetic_parameters']
        for rxn_id, rxn in reactions.items():

            # get the parameters
            kinetics = kinetic_parameters[rxn_id]
            stoich = rxn['stoichiometry']
            assert len(stoich.keys()) == 1, 'reactions can only do one substrate'
            substrate_id = list(stoich.keys())[0]

            catalyst_id = rxn['catalyzed by']
            kcat = kinetics[catalyst_id]['kcat_f']
            km = kinetics[catalyst_id][substrate_id]

            # get the state
            catalyst_field = fields[catalyst_id]
            substrate_field = fields[substrate_id]

            if np.sum(catalyst_field) > 0.0 and np.sum(substrate_field) > 0.0:
                # calculate flux and delta
                denominator = substrate_field + km
                flux = kcat * catalyst_field * substrate_field / denominator
                delta = stoich[substrate_id] * flux * timestep

                # updates
                delta_fields[substrate_id] = delta
                new_fields[substrate_id] += delta

        return delta_fields, new_fields


class ExchangeAgent(Process):
    defaults = {
        'mol_ids': [],
        'default_exchange': 100,
    }

    def __init__(self, parameters=None):
        super().__init__(parameters)

    def ports_schema(self):
        mol_ids = self.parameters['mol_ids']

        def accumulate_location(value, update):
            return [
                max(value[0] + update[0], 0),
                max(value[1] + update[1], 0)
            ]

        return {
            'boundary': {
                'external': {
                    mol_id: {'_default': 0.0} for mol_id in mol_ids},
                'exchanges': {
                    mol_id: {'_default': self.parameters['default_exchange']}
                    for mol_id in mol_ids},
                'location': {
                    '_default': [0.5, 0.5],
                    '_updater': accumulate_location,
                }
            }
        }

    def next_update(self, timestep, states):
        max_move = 5.0
        mol_ids = self.parameters['mol_ids']
        return {
            'boundary': {
                'exchanges': {
                    mol_id: self.parameters['default_exchange']
                    for mol_id in mol_ids},
                'location': [
                    np.random.uniform(-max_move, max_move),
                    np.random.uniform(-max_move, max_move)
                ],
            }
        }


def main():
    total_time = 100
    depth = 2
    n_bins = [50, 50]
    bounds = [50, 50]

    # make the reaction diffusion process
    params = {
        # 'boundary_path': ('a', 'b', 'c'), # needs to be 'boundary' for snapshots plot
        'molecules': [
            'beta-lactam',
            'beta-lactamase',
        ],
        'n_bins': n_bins,
        'bounds': bounds,
        'depth': depth,
        'reactions': {
            'antibiotic_hydrolysis': {
                'stoichiometry': {
                    'beta-lactam': -1,
                },
                'catalyzed by': 'beta-lactamase'
            }
        },
        'kinetic_parameters': {
            'antibiotic_hydrolysis': {
                'beta-lactamase': {
                    'kcat_f': 10.0,  # kcat for forward reaction
                    'beta-lactam': 0.1,  # Km
                },
            },
        },
    }
    rxn_diff_process = ReactionDiffusion(params)

    # make the toy exchange agent
    agent_id = '0'
    agent_params = {'mol_ids': ['beta-lactamase']}
    agent_process = ExchangeAgent(agent_params)

    # get initial fields
    initial_fields = rxn_diff_process.initial_state({
        'beta-lactam': 1.0})

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
        },
        initial_state=initial_fields
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

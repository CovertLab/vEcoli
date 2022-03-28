"""
=====
Lysis
=====
"""
import os
import random
import numpy as np
from scipy import constants

from vivarium.core.process import Step, Process
from vivarium.core.composer import Composer
from vivarium.core.engine import Engine, pf
from vivarium.library.units import units, remove_units
from ecoli.processes.environment.multibody_physics import PI
from ecoli.processes.environment.local_field import LocalField
from ecoli.library.lattice_utils import (
    get_bin_site,
    get_bin_volume,
    count_to_concentration,
)



AVOGADRO = constants.N_A


class Lysis(Step):
    name = 'lysis'
    defaults = {
        'secreted_molecules': [],
        'nonspatial': False,
        'bin_volume': 1e-6 * units.L,
    }

    def __init__(self, parameters=None):
        super().__init__(parameters)
        self.agent_id = self.parameters['agent_id']
        self.nonspatial = self.parameters['nonspatial']
        self.bin_volume = self.parameters['bin_volume']

    def ports_schema(self):
        fields_schema = {
                mol_id: {
                    '_default': np.ones(1),
                } for mol_id in self.parameters['secreted_molecules']
            }

        return {
            'trigger': {
                '_default': False
            },
            'agents': {},
            'internal': {
                mol_id: {
                    '_default': 1,
                    '_emit': True,
                } for mol_id in self.parameters['secreted_molecules']
            },
            'fields': {
                **fields_schema,
                '_output': True,
            },
            'location': {
                '_default': [0.5, 0.5]
            },
            'dimensions': {
                'bounds': {
                    '_default': [1, 1],
                },
                'n_bins': {
                    '_default': [1, 1],
                },
                'depth': {
                    '_default': 1,
                },
            }
        }

    def next_update(self, timestep, states):
        if states['trigger']:
            location = remove_units(states['location'])
            n_bins = states['dimensions']['n_bins']
            bounds = states['dimensions']['bounds']
            depth = states['dimensions']['depth']

            # get bin volume
            if self.nonspatial:
                bin_volume = self.bin_volume
            else:
                bin_site = get_bin_site(location, n_bins, bounds)
                bin_volume = get_bin_volume(n_bins, bounds, depth) * units.L

            # apply internal states to fields
            internal = states['internal']
            delta_fields = {}
            for mol_id, value in internal.items():

                # delta concentration
                exchange = value * units.count
                concentration = count_to_concentration(exchange, bin_volume).to(
                    units.mmol / units.L).magnitude

                if self.nonspatial:
                    delta_fields[mol_id] = {
                        '_value': concentration,
                        '_updater': 'accumulate'}
                else:
                    delta_field = np.zeros((n_bins[0], n_bins[1]), dtype=np.float64)
                    delta_field[bin_site[0], bin_site[1]] += concentration
                    delta_fields[mol_id] = {
                        '_value': delta_field,
                        '_updater': 'accumulate'}

            # remove agent and apply delta to field
            return {
                'agents': {'_delete': [self.agent_id]},
                'fields': delta_fields
            }
        return {}


def mass_from_count(count, mw):
    mol = count / AVOGADRO
    return mw * mol


class ToyTransportBurst(Process):
    """
    Toy process for testing Lysis.
    Uptakes a molecule from a field, and triggers lysis.
    """

    defaults = {
        'uptake_rate': {'GLC': 1},
        'molecular_weights': {'GLC': 1},  # * units.fg
        'burst_mass': 2000,  # * units.fg,
    }

    def __init__(self, parameters=None):
        super().__init__(parameters)
        self.molecules = list(self.parameters['uptake_rate'].keys())

    def ports_schema(self):
        return {
            'external': {
                key: {
                    '_default': 0.0,
                    '_emit': True,
                } for key in self.molecules
            },
            'exchanges': {
                key: {
                    '_default': 0.0,
                    '_emit': True,
                } for key in self.molecules
            },
            'internal': {
                key: {
                    '_default': 0.0,
                    '_emit': True,
                } for key in self.molecules
            },
            'mass': {
                '_default': 0.0,  # * units.fg
            },
            'length': {
                '_default': 0.0,
            },
            'burst_trigger': {
                '_default': False,
                '_updater': 'set',
                '_emit': True,
            },
        }

    def next_update(self, timestep, states):
        added = {}
        exchanged = {}
        added_mass = 0.0
        for mol_id, e_state in states['external'].items():
            exchange_concs = e_state * self.parameters['uptake_rate'][mol_id]
            exchange_counts = exchange_concs

            added[mol_id] = exchange_counts
            exchanged[mol_id] = -1 * exchange_counts
            added_mass += mass_from_count(
                exchange_counts, self.parameters['molecular_weights'][mol_id])

        if states['mass'] + added_mass >= self.parameters['burst_mass']:
            return {'burst_trigger': True}

        # extend length relative to mass
        added_length = added_mass * states['length'] / states['mass']

        return {
            'internal': added,
            'exchanges': exchanged,
            'mass': added_mass,
            'length': added_length,
        }


class LysisAgent(Composer):
    """
    Agent that uptakes a molecule from a lattice environment,
    bursts upon reaching a set mass, and spills hte molecules
    back into the environment
    """

    defaults = {
        'lysis': {
            'secreted_molecules': ['GLC']
        },
        'transport_burst': {
            'uptake_rate': {
                'GLC': 2,
            },
            'molecular_weights': {
                'GLC': 1e22,  # * units.fg
            },
            'burst_mass': 2000,  # * units.fg
        },
        'local_field': {},
        'boundary_path': ('boundary',),
        'fields_path': ('..', '..', 'fields'),
        'dimensions_path': ('..', '..', 'dimensions',),
        'agents_path': ('..', '..', 'agents',),
    }

    def generate_processes(self, config):
        return {
            'transport_burst': ToyTransportBurst(config['transport_burst'])
        }

    def generate_steps(self, config):
        assert config['agent_id']
        lysis_config = {
            'agent_id': config['agent_id'],
            **config['lysis']}
        return {
            'local_field': LocalField(config['local_field']),
            'lysis': Lysis(lysis_config),
        }

    def generate_flow(self, config):
        return {
            'local_field': [],
            'lysis': [],
        }

    def generate_topology(self, config):
        boundary_path = config['boundary_path']
        fields_path = config['fields_path']
        dimensions_path = config['dimensions_path']
        agents_path = config['agents_path']

        return {
            'transport_burst': {
                'internal': ('internal',),
                'exchanges': boundary_path + ('exchanges',),
                'external': boundary_path + ('external',),
                'mass': boundary_path + ('mass',),
                'length': boundary_path + ('length',),
                'burst_trigger': boundary_path + ('burst',),
            },
            'local_field': {
                'exchanges': boundary_path + ('exchanges',),
                'location': boundary_path + ('location',),
                'fields': fields_path,
                'dimensions': dimensions_path,
            },
            'lysis': {
                'trigger': boundary_path + ('burst',),
                'internal': ('internal',),
                'agents': agents_path,
                'fields': fields_path,
                'location': boundary_path + ('location',),
                'dimensions': dimensions_path,
            },
        }


def test_lysis(
        n_cells=1,
        molecule_name='GLC',
        total_time=60,
        emit_step=1,
        bounds=[25, 25],
        n_bins=[5, 5],
        uptake_rate_max=25
):
    from ecoli.composites.environment.lattice import Lattice

    lattice_composer = Lattice({
        'diffusion': {
            'molecules': [molecule_name],
            'bounds': bounds,
            'n_bins': n_bins,
            'gradient': {
                'type': 'uniform',
                'molecules': {
                    molecule_name: 10.0,
                }
            },
        },
        'multibody': {
            'bounds': bounds,
        }})

    # initialize the composite with a lattice
    full_composite = lattice_composer.generate()

    # configure the agent composer
    agent_composer = LysisAgent({
        'lysis': {
            'secreted_molecules': [molecule_name]
        },
        'transport_burst': {
            'molecular_weights': {
                molecule_name: 1e22,  # * units.fg
            },
            'burst_mass': 2000,  # * units.fg
        },
    })

    # make individual agents, with unique uptake rates
    agent_ids = [str(idx) for idx in range(n_cells)]
    for agent_id in agent_ids:
        uptake_rate = random.randrange(uptake_rate_max)
        agent_composite = agent_composer.generate({
            'agent_id': agent_id,
            'transport_burst': {
                'uptake_rate': {
                    molecule_name: uptake_rate,
                }}})
        agent_path = ('agents', agent_id)
        full_composite.merge(composite=agent_composite, path=agent_path)

    # get initial state
    initial_state = full_composite.initial_state()
    initial_state['agents'] = {}
    for agent_id in agent_ids:
        agent_angle = random.uniform(0, 2*PI)
        initial_state['agents'][agent_id] = {
            'boundary': {
                'angle': agent_angle}}

    # run the simulation and return the data
    sim = Engine(
        processes=full_composite.processes,
        steps=full_composite.steps,
        topology=full_composite.topology,
        flow=full_composite.flow,
        initial_state=initial_state,
        emit_step=emit_step)
    sim.update(total_time)
    data = sim.emitter.get_data_unitless()
    return data


def main():
    from ecoli.plots.snapshots import plot_snapshots, format_snapshot_data
    from ecoli.plots.snapshots_video import make_video

    bounds = [15, 15]
    molecule_name = 'beta-lactam'

    data = test_lysis(
        n_cells=8,
        molecule_name=molecule_name,
        total_time=1000,
        emit_step=10,
        bounds=bounds,
        n_bins=[11, 11],
    )

    # format the data for plot_snapshots
    agents, fields = format_snapshot_data(data)

    out_dir = os.path.join('out', 'experiments', 'lysis')
    os.makedirs(out_dir, exist_ok=True)
    plot_snapshots(
        bounds,
        agents=agents,
        fields=fields,
        n_snapshots=5,
        out_dir=out_dir,
        filename=f"lysis_snapshots")

    # make snapshot video
    make_video(
        data,
        bounds,
        plot_type='fields',
        out_dir=out_dir,
        filename='lysis_video',
    )


# python ecoli/processes/environment/lysis.py
if __name__ == "__main__":
    main()

"""
=====
Lysis
=====
"""
import os
import numpy as np
from scipy import constants

from vivarium.core.process import Step, Process
from vivarium.core.composer import Composer
from vivarium.core.engine import Engine, pf
from vivarium.library.units import units, remove_units
from vivarium.processes.timeline import TimelineProcess
from ecoli.composites.environment.lattice import Lattice
from ecoli.processes.lattice.local_field import LocalField
from ecoli.library.lattice_utils import (
    get_bin_site,
    get_bin_volume,
    count_to_concentration,
)

from ecoli.plots.snapshots import plot_snapshots, format_snapshot_data, get_agent_ids
from ecoli.plots.snapshots_video import make_video


AVOGADRO = constants.N_A


class Lysis(Step):
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
    defaults = {
        'uptake_rate': {'GLC': 1},
        'molecular_weights': {'GLC': 1 * units.fg},
        'burst_mass': 2000 * units.fg,
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
                '_default': 0.0 * units.fg,
                '_emit': True,
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

        return {
            'internal': added,
            'exchanges': exchanged,
            'mass': added_mass,
        }


class LysisAgent(Composer):
    defaults = {
        'lysis': {
            'secreted_molecules': ['GLC']
        },
        'transport_burst': {
            'uptake_rate': {
                'GLC': 2,
            },
            'molecular_weights': {
                'GLC': 1e22 * units.fg
            },
            'burst_mass': 2000 * units.fg,
        },
        'local_field': {},
        'boundary_path': ('boundary',),
        'fields_path': ('..', '..', 'fields'),
        'dimensions_path': ('..', '..', 'dimensions',),
        'agents_path': ('..', '..', 'agents',),
    }

    def __init__(self, config=None):
        super().__init__(config)

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
        total_time=60,
        emit_step=1,
        death_trigger_time=50,
        bounds=[25, 25],
        n_bins=[5, 5],
):

    agent_id = '1'
    agent_path = ('agents', agent_id)
    lattice_composer = Lattice({
        'diffusion': {
            'molecules': ['GLC'],
            'bounds': bounds,
            'n_bins': n_bins,
            'gradient': {
                'type': 'uniform',
                'molecules': {
                    'GLC': 10.0,
                }
            },
        },
        'multibody': {
            'bounds': bounds,
        }})

    # configure the agent
    agent_composer = LysisAgent({
        'transport_burst': {
            'uptake_rate': {
                'GLC': 5,
            },
            # 'molecular_weights': {
            #     'GLC': 1e22 * units.fg
            # },
            # 'burst_mass': 2000 * units.fg,
        }})

    # combine composites
    full_composite = lattice_composer.generate()
    agent_composite = agent_composer.generate({'agent_id': agent_id})
    full_composite.merge(composite=agent_composite, path=agent_path)

    # add a timeline process to trigger lysis
    timeline_config = {
        'timeline': [
            (death_trigger_time, {('death',): True}),
        ]}
    timeline_process = TimelineProcess(timeline_config)
    full_composite.merge(
        processes={
            'timeline': timeline_process},
        topology={
            'timeline': {
                'death': ('boundary', 'burst',),
                'global': ('..', '..', 'global',),
        }},
        path=agent_path
    )

    initial_state = full_composite.initial_state()

    experiment = Engine(
        processes=full_composite.processes,
        steps=full_composite.steps,
        topology=full_composite.topology,
        flow=full_composite.flow,
        initial_state=initial_state,
        emit_step=emit_step,
    )

    experiment.update(total_time)
    data = experiment.emitter.get_data_unitless()
    return data


def main():
    bounds = [15, 15]
    n_bins = [11, 11]
    data = test_lysis(
        total_time=700,
        emit_step=10,
        death_trigger_time=500,
        bounds=bounds,
        n_bins=n_bins,
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



# python ecoli/processes/antibiotics/lysis.py
if __name__ == "__main__":
    main()

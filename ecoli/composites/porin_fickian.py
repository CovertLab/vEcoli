from numpy import array
from vivarium.core.composer import Composer
from vivarium.core.emitter import timeseries_from_data
from vivarium.core.engine import Engine
from vivarium.library.units import units
from vivarium.plots.simulation_output import plot_variables
from ecoli.processes.antibiotics.porin_permeability import PorinPermeability, CEPH_OMPC_CON_PERM,\
    CEPH_OMPF_CON_PERM, SA_AVERAGE
from ecoli.processes.antibiotics.fickian_diffusion import FickianDiffusion
from vivarium.processes.timeline import TimelineProcess
from ecoli.processes.antibiotics.nonspatial_environment import NonSpatialEnvironment
from ecoli.states.wcecoli_state import get_state_from_file


class PorinFickian(Composer):
    defaults = {
        'nonspatial': {},
        'fickian': {},
        'timeline': {},
        'porin_permeability': {},
    }

    def generate_processes(self, config):
        fick_diffusion = FickianDiffusion(config['fickian'])
        timeline = TimelineProcess(config['timeline'])
        return {'fickian': fick_diffusion,
                'timeline': timeline,
                }

    def generate_topology(self, config):
        return {
            'nonspatial': {
                'external': ('environment', 'external',),
                'exchanges': ('boundary', 'exchanges',),
                'fields': ('environment', 'fields'),
                'dimensions': ('environment', 'dimensions'),
                'global': ('listeners', 'mass',),  # To get location, volume, and mmol_to_counts
            },
            'fickian': {
                'internal': ('boundary', 'internal',),
                'external': ('environment', 'external',),
                'fluxes': ('fluxes',),
                'exchanges': ('boundary', 'exchanges',),
                'volume_global': ('listeners', 'mass',),
                'mass_global': ('listeners', 'mass',),
                'permeabilities': ('boundary', 'permeabilities',)
            },
            'timeline': {
                'global': ('global',),  # The global time is read here
                'porins': ('bulk',),  # This port is based on the declared timeline
            },
            'porin_permeability': {
                'porins': ('bulk',),
                'permeabilities': ('boundary', 'permeabilities',),
                'surface_area': ('boundary', 'surface_area')
            },
        }

    def generate_steps(self, config):
        nonspatial = NonSpatialEnvironment(config['nonspatial'])
        porin_permeability = PorinPermeability(config['porin_permeability'])
        return {'nonspatial': nonspatial,
                'porin_permeability': porin_permeability}


def main():
    sim_time = 10

    initial_state = get_state_from_file(path='data/vivecoli_t1000.json')
    initial_state['boundary'] = {}
    initial_state['boundary']['surface_area'] = SA_AVERAGE
    initial_state['listeners']['mass']['dry_mass'] = initial_state['listeners']['mass']['dry_mass'] * units.fg
    initial_state['environment']['fields'] = {}
    initial_state['environment']['fields']['cephaloridine'] = array([[1e-3]])
    initial_state['bulk']['CPLX0-7533[o]'] = 500
    initial_state['bulk']['CPLX0-7534[o]'] = 500

    timeline = []
    for i in range(10):
        timeline.append(
            (i, {
                ('porins', 'CPLX0-7533[o]'): initial_state['bulk']['CPLX0-7533[o]'] + ((i + 1) * 500),
                ('porins', 'CPLX0-7534[o]'): initial_state['bulk']['CPLX0-7534[o]'] + ((i + 1) * 500),
            })
        )

    config = {
        'nonspatial': {
            'env_volume': 3000.0 * units.fL
        },
        'fickian': {
            'time_step': 0.1,
            'molecules_to_diffuse': ['cephaloridine'],
            'initial_state': {
                'internal': {
                    'cephaloridine': 0,  # mM
                },
                'external': {
                    'cephaloridine': 1e-3,  # mM
                },
            }
        },
        'timeline': {
            'time_step': 1.0,
            'timeline': timeline,
        },
        'porin_permeability': {
            'porin_ids': ['CPLX0-7533[o]', 'CPLX0-7534[o]'],
            'diffusing_molecules': {
                'cephaloridine': {
                    'CPLX0-7533[o]': CEPH_OMPC_CON_PERM,
                    'CPLX0-7534[o]': CEPH_OMPF_CON_PERM
                }
            },
        },
    }
    composer = PorinFickian(config)
    composite = composer.generate()

    sim = Engine(composite=composite, initial_state=initial_state)
    sim.update(sim_time)
    timeseries_data = timeseries_from_data(sim.emitter.get_data())
    plot_variables(timeseries_data, [('environment', 'external', 'cephaloridine'),
                                     ('boundary', 'internal', 'cephaloridine'),
                                     ('bulk', 'CPLX0-7533[o]'), ('bulk', 'CPLX0-7534[o]')],
                   out_dir='out', filename='porin_fickian_counts')


if __name__ == '__main__':
    main()

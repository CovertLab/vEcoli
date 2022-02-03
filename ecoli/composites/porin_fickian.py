from vivarium.core.composer import Composer
from vivarium.core.emitter import timeseries_from_data
from vivarium.core.engine import Engine
from vivarium.library.units import units
from vivarium.plots.simulation_output import plot_variables
from ecoli.processes.antibiotics.porin_permeability import PorinPermeability, CEPH_OMPC_CON_PERM,\
    CEPH_OMPF_CON_PERM, SA_AVERAGE
from ecoli.processes.antibiotics.fickian_diffusion import FickianDiffusion
from ecoli.states.wcecoli_state import get_state_from_file


class PorinFickian(Composer):
    defaults = {
        'fickian': {},
        'porin_permeability': {},
    }

    def __init__(self, config=None):
        super().__init__(config)

    def generate_processes(self, config):
        fick_diffusion = FickianDiffusion(config['fickian'])
        return {'fickian': fick_diffusion}

    def generate_topology(self, config):
        return {
            'fickian': {
                'internal': ('bulk',),  # This is where the antibiotic will diffuse into
                'external': ('environment',),
                'fluxes': ('fluxes',),
                'exchanges': ('exchanges',),
                'volume_global': ('listeners', 'mass',),
                'mass_global': ('listeners', 'mass',),
                'permeabilities': ('boundary', 'permeabilities')
            },
            'porin_permeability': {
                'porins': ('bulk',),
                'permeabilities': ('boundary', 'permeabilities',),
                'surface_area': ('boundary', 'surface_area')
            }
        }

    def generate_steps(self, config):
        porin_permeability = PorinPermeability(config['porin_permeability'])
        return {'porin_permeability': porin_permeability}


def main():
    sim_time = 10
    config = {
        'porin_permeability': {
            'porin_ids': ['CPLX0-7533[o]', 'CPLX0-7534[o]'],
            'diffusing_molecules': {
                'cephaloridine': {
                    'CPLX0-7533[o]': CEPH_OMPC_CON_PERM,
                    'CPLX0-7534[o]': CEPH_OMPF_CON_PERM
                }
            },
        },
        'fickian': {
            'molecules_to_diffuse': ['cephaloridine'],
            'initial_state': {
                'internal': {
                    'cephaloridine': 0,  # mM
                },
                'external': {
                    'cephaloridine': 1e-3,  # mM
                },
            }
        }
    }
    composer = PorinFickian(config)
    composite = composer.generate()

    initial_state = get_state_from_file(path='data/vivecoli_t1000.json')
    initial_state['boundary'] = {}
    initial_state['boundary']['surface_area'] = SA_AVERAGE
    initial_state['listeners']['mass']['dry_mass'] = initial_state['listeners']['mass']['dry_mass'] * units.fg

    sim = Engine(composite=composite, initial_state=initial_state)
    sim.update(sim_time)
    timeseries_data = timeseries_from_data(sim.emitter.get_data())
    plot_variables(timeseries_data, [('environment', 'cephaloridine'), ('bulk', 'cephaloridine'),
                                     ('bulk', 'CPLX0-7533[o]'), ('bulk', 'CPLX0-7534[o]')],
                   out_dir='data', filename='porin_fickian_counts')


if __name__ == '__main__':
    main()

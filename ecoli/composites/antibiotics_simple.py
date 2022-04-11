"""
=======================================
Composite Model of Antibiotics Response
=======================================
"""

from vivarium.core.composer import Composer
from vivarium.core.engine import Engine
from vivarium.core.emitter import timeseries_from_data
from vivarium.core.serialize import deserialize_value
from vivarium.library.units import units, remove_units
from vivarium.plots.simulation_output import plot_variables
from vivarium.processes.timeline import TimelineProcess
from vivarium.core.control import run_library_cli

from ecoli.processes.antibiotics.exchange_aware_bioscrape import ExchangeAwareBioscrape
from ecoli.processes.antibiotics.permeability import (
    Permeability, CEPH_OMPC_CON_PERM, CEPH_OMPF_CON_PERM, OUTER_BILAYER_CEPH_PERM, TET_OMPF_CON_PERM,
    OUTER_BILAYER_TET_PERM, INNER_BILAYER_TET_PERM, SA_AVERAGE
)
from ecoli.processes.antibiotics.nonspatial_environment import (
    NonSpatialEnvironment
)
from ecoli.processes.shape import Shape


INITIAL_ENVIRONMENT_CEPH = 0.1239 * units.mM
INITIAL_ENVIRONMENT_TET = 0.1239 * units.mM
BETA_LACTAM_KEY = 'cephaloridine'
TET_KEY = 'tetracycline'


class SimpleAntibioticsCell(Composer):
    '''
    This composite includes the minimum amount of steps/processes needed to
    simulate the diffusion of a beta-lactam and tetracycline into E. coli.
    '''

    default = {
        'ceph_tet_bioscrape': {},
        'shape': {},
        'nonspatial_environment': {},
        'permeability': {},
    }

    def generate_processes(self, config):
        ceph_tet_bioscrape = ExchangeAwareBioscrape(config['ceph_tet_bioscrape'])
        return {
            'ceph_tet_bioscrape': ceph_tet_bioscrape,
        }

    def generate_steps(self, config):
        nonspatial_environment = NonSpatialEnvironment(config['nonspatial_environment'])
        shape = Shape(config['shape'])
        permeability = Permeability(config['permeability'])
        return {
            'nonspatial_environment': nonspatial_environment,
            'shape': shape,
            'permeability': permeability,
        }

    def generate_topology(self, config=None):
        boundary_path = config['boundary_path']
        topology = {
            'ceph_tet_bioscrape': {
                'delta_species': ('delta_concs',),
                'exchanges': boundary_path + ('exchanges',),
                'external': boundary_path + ('external',),
                'globals': ('global',),
                'rates': {
                    '_path': ('kinetic_parameters',),
                    'mass': ('..',) + boundary_path + ('mass',),
                    'volume_p': ('..', 'periplasm', 'volume',),
                    'volume_c': ('..', 'cytoplasm', 'volume',),
                },
                'species': ('concs',),
            },
            'shape': {
                'cell_global': boundary_path,
                'periplasm_global': ('periplasm',),
                'cytoplasm_global': ('cytoplasm',),
                'listener_cell_mass': ('mass_listener', 'dry_mass'),
            },
            'nonspatial_environment': {
                'external': boundary_path + ('external',),
                'exchanges': boundary_path + ('exchanges',),
                'fields': ('environment', 'fields',),
                'dimensions': ('environment', 'dimensions'),
                'global': ('global',),
            },
            'permeability': {
                'porins': ('bulk',),
                'permeabilities': ('kinetic_parameters',),
                'surface_area': boundary_path + ('surface_area',)
            },
        }
        return topology


def get_increasing_porins_timeline():
    timeline = []
    for i in range(0, 120, 10):
        timeline.append(
            (i, {
                ('porins', 'CPLX0-7533[o]'):  (i * 10),
                ('porins', 'CPLX0-7534[o]'):  (i * 10),
            },
             )
        )
    return timeline


def demo(
    timeline=get_increasing_porins_timeline(),
    filename='antibiotics_simple'
):
    sim_time = timeline[-1][0]  # end at the last timeline entry

    config = {
        'boundary_path': ('boundary',),
        'ceph_tet_bioscrape': {
            'external_species': ('cephaloridine_environment', 'tetracycline_environment',),
            'name_map': (
                (('external', 'cephaloridine',), 'cephaloridine_environment',),
                (('external', 'tetracycline',), 'tetracycline_environment',),
            ),
            'sbml_file': 'data/ceph_tet_sbml.xml',
            'time_step': 0.5,
        },
        'shape': {},
        'nonspatial_environment': {
            'concentrations': {
                BETA_LACTAM_KEY: INITIAL_ENVIRONMENT_CEPH,
                TET_KEY: INITIAL_ENVIRONMENT_TET
            },
            'internal_volume': 1.2 * units.fL,
            'env_volume': 1 * units.mL,
        },
        'permeability': {
            'porin_ids': ['CPLX0-7533[o]', 'CPLX0-7534[o]'],
            'diffusing_molecules': {
                'outer_cephaloridine_permeability': {
                    'concentration_perm': {
                        'CPLX0-7533[o]': CEPH_OMPC_CON_PERM,
                        'CPLX0-7534[o]': CEPH_OMPF_CON_PERM
                    },
                    'bilayer_perm': OUTER_BILAYER_CEPH_PERM
                },
                'outer_tetracycline_permeability': {
                    'concentration_perm': {
                        'CPLX0-7534[o]': TET_OMPF_CON_PERM,
                    },
                    'bilayer_perm': OUTER_BILAYER_TET_PERM
                },
                'inner_tetracycline_permeability': {
                    'concentration_perm': {},
                    'bilayer_perm': INNER_BILAYER_TET_PERM
                }
            },
        },
    }
    composite = SimpleAntibioticsCell(config).generate()

    # add a timeline process to the composite
    timeline_config = {
        'time_step': 1.0,
        'timeline': timeline,
    }
    timeline_process = {
        'timeline': TimelineProcess(timeline_config)
    }
    timeline_topology = {
        'timeline': {
            'global': ('global',),  # The global time is read here
            'porins': ('bulk',),  # This port is based on the declared timeline
        }
    }
    composite.merge(processes=timeline_process, topology=timeline_topology)

    # initial state
    initial_state = composite.initial_state()
    initial_state['boundary']['surface_area'] = SA_AVERAGE
    initial_state['bulk'] = {}

    sim = Engine(composite=composite, initial_state=initial_state)
    sim.update(sim_time)
    data = sim.emitter.get_data()
    data = deserialize_value(data)
    data = remove_units(data)
    timeseries_data = timeseries_from_data(data)
    plot_variables(
        timeseries_data,
        variables=[
            ('bulk', 'CPLX0-7533[o]'),
            ('bulk', 'CPLX0-7534[o]'),
            ('boundary', 'external', 'cephaloridine'),
            ('boundary', 'external', 'tetracycline'),
            ('concs', 'cephaloridine_periplasm'),
            ('concs', 'cephaloridine_hydrolyzed'),
            ('concs', 'tetracycline_periplasm'),
            ('concs', 'tetracycline_cytoplasm')
        ],
        out_dir='out/composites/antibiotics_simple',
        filename=filename
    )


demo_library = {
    '0': demo
}


# python ecoli/composites/antibiotics_simple.py [-n function_id]
if __name__ == '__main__':
    run_library_cli(demo_library)

from numpy import array
from vivarium.core.composer import Composer
from vivarium.core.composition import (
    composite_in_experiment, simulate_experiment)
from vivarium.core.engine import Engine
from vivarium.core.emitter import timeseries_from_data
from vivarium.library.units import units
from vivarium.plots.simulation_output import plot_variables
from vivarium.plots.topology import plot_topology
from vivarium.processes.timeline import TimelineProcess
from vivarium_convenience.processes.convenience_kinetics import ConvenienceKinetics

from ecoli.states.wcecoli_state import get_state_from_file
from ecoli.processes.enzyme_kinetics import EnzymeKinetics
from ecoli.processes.antibiotics.fickian_diffusion import (
    FickianDiffusion,
)
from ecoli.processes.antibiotics.nonspatial_environment import (
    NonSpatialEnvironment)
from ecoli.processes.antibiotics.porin_permeability import PorinPermeability, CEPH_OMPC_CON_PERM,\
    CEPH_OMPF_CON_PERM, CEPH_PH_PERM, SA_AVERAGE
from ecoli.processes.antibiotics.shape import ShapeDeriver


INITIAL_INTERNAL_ANTIBIOTIC = 0
INITIAL_EXTERNAL_ANTIBIOTIC = 1e-3
ANTIBIOTIC_KEY = 'cephaloridine'
PUMP_KEY = 'TRANS-CPLX-201'
# Source: (Wülfing & Plückthun, 1994)
PERIPLASM_FRACTION = 0.3
BETA_LACTAMASE_KEY = 'beta-lactamase'


class PARAMETERS:
    # TODO(MATT): no units maybe?
    # Reported in (Nagano & Nikaido, 2009)
    CEPH_PUMP_KCAT = 1e1  # / units.sec  # TODO(Matt): Placeholder
    # Reported in (Nagano & Nikaido, 2009)
    CEPH_PUMP_KM = 4.95e-3  # * units.millimolar  # TODO(Matt): Placeholder.
    # Reported in (Galleni et al., 1988)
    CEPH_BETA_LACTAMASE_KCAT = 130  # / units.sec
    # Reported in (Galleni et al., 1988)
    CEPH_BETA_LACTAMASE_KM = 170  # * units.micromolar


class SimpleAntibioticsCell(Composer):
    '''Integrate antibiotic resistance and susceptibility with wcEcoli

    Integrates the WcEcoli process, which wraps the wcEcoli model, with
    processes to model antibiotic susceptibility (diffusion-based
    import and death) and resistance (hydrolysis and transport-based
    efflux). Also includes derivers.
    '''

    defaults = {
        'efflux': {},
        'fickian_diffusion': {},
        'shape_deriver': {},
        'timeline': {},
        'porin_permeability': {},
        'nonspatial_environment': {},
    }

    def generate_processes(self, config):
        # efflux = EnzymeKinetics(config['efflux'])
        efflux = ConvenienceKinetics(config['efflux'])
        fickian_diffusion = FickianDiffusion(
            config['fickian_diffusion'])
        timeline = TimelineProcess(config['timeline'])
        return {
            'efflux': efflux,
            'fickian_diffusion': fickian_diffusion,
            'timeline': timeline,
        }

    def generate_topology(self, config=None):
        topology = {
            # 'efflux': {
            #     'internal': ('periplasm', 'concs'),
            #     'external': ('boundary', 'external',),
            #     'exchanges': ('boundary', 'exchanges',),
            #     'fluxes': ('fluxes',),
            #     'global': ('periplasm', 'global'),
            # },
            # 'fickian_diffusion': {
            #     'internal': ('periplasm', 'concs'),
            #     'external': ('boundary', 'external',),
            #     'exchanges': ('boundary', 'exchanges',),
            #     'fluxes': ('fluxes',),
            #     'volume_global': ('periplasm', 'global'),
            #     'mass_global': ('boundary',),
            #     'permeabilities': ('boundary', 'permeabilities',)
            # },
            # 'shape_deriver': {
            #     'cell_global': ('boundary',),
            #     'periplasm_global': ('periplasm', 'global')
            # },
            # 'timeline': {
            #     'global': ('global',),  # The global time is read here
            #     'porins': ('bulk',),  # This port is based on the declared timeline
            # },
            # 'porin_permeability': {
            #     'porins': ('bulk',),
            #     'permeabilities': ('boundary', 'permeabilities',),
            #     'surface_area': ('boundary', 'surface_area',)
            # },
            # 'nonspatial_environment': {
            #     'external': ('boundary', 'external'),
            #     'exchanges': ('boundary', 'exchanges'),
            #     'fields': ('environment', 'fields'),
            #     'dimensions': ('environment', 'dimensions'),
            #     'global': ('boundary',),
            'efflux': {
                'internal': ('periplasm', 'concs'),
                'external': ('boundary', 'external',),
                'exchanges': ('boundary', 'exchanges',),
                'fluxes': ('fluxes',),
                'global': ('global',),
            },
            'fickian_diffusion': {
                'internal': ('periplasm', 'concs'),
                'external': ('boundary', 'external',),
                'exchanges': ('boundary', 'exchanges',),
                'fluxes': ('fluxes',),
                'volume_global': ('global',),
                'mass_global': ('global',),
                'permeabilities': ('boundary', 'permeabilities',)
            },
            'shape_deriver': {
                'cell_global': ('global',),
                'periplasm_global': ('global',)
            },
            'timeline': {
                'global': ('global',),  # The global time is read here
                'porins': ('bulk',),  # This port is based on the declared timeline
            },
            'porin_permeability': {
                'porins': ('bulk',),
                'permeabilities': ('boundary', 'permeabilities',),
                'surface_area': ('boundary', 'surface_area',)
            },
            'nonspatial_environment': {
                'external': ('boundary', 'external'),
                'exchanges': ('boundary', 'exchanges'),
                'fields': ('environment', 'fields'),
                'dimensions': ('environment', 'dimensions'),
                'global': ('global',),
            },
        }
        return topology

    def generate_steps(self, config):
        shape_deriver = ShapeDeriver(config['shape_deriver'])
        porin_permeability = PorinPermeability(config['porin_permeability'])
        nonspatial_environment = NonSpatialEnvironment(config['nonspatial_environment'])
        return {
            'shape_deriver': shape_deriver,
            'porin_permeability': porin_permeability,
            'nonspatial_environment': nonspatial_environment,
        }


def demo():
    sim_time = 10

    initial_state = get_state_from_file(path='data/vivecoli_t1000.json')
    initial_state['boundary'] = {}
    initial_state['boundary']['surface_area'] = SA_AVERAGE
    initial_state['listeners']['mass']['dry_mass'] = initial_state['listeners']['mass']['dry_mass'] * units.fg
    # initial_state['boundary']['external'] = {}
    # initial_state['boundary']['external']['cephaloridine'] = array([[INITIAL_EXTERNAL_ANTIBIOTIC]])
    initial_state['periplasm'] = {}
    initial_state['periplasm']['concs'] = {}
    initial_state['periplasm']['concs']['beta-lactamase'] = 1e-3
    initial_state['bulk']['CPLX0-7533[o]'] = 500
    initial_state['bulk']['CPLX0-7534[o]'] = 500

    initial_state['periplasm']['concs']['TRANS-CPLX-201'] = 1e-3  # Is this right?

    timeline = []
    for i in range(10):
        timeline.append(
            (i, {
                ('porins', 'CPLX0-7533[o]'): initial_state['bulk']['CPLX0-7533[o]'] + ((i + 1) * 500),
                ('porins', 'CPLX0-7534[o]'): initial_state['bulk']['CPLX0-7534[o]'] + ((i + 1) * 500),
            })
        )
    # Maybe use antibiotic_transport and hydrolysis instead of efflux just to get it working?
    # If that works, copy over the efflux config from those two python files and add in second antibiotic
    config = {
        'efflux': {
            'reactions': {
                'cephaloridine_tolc': {
                    'stoichiometry': {
                        ('internal', 'cephaloridine'): -1,
                        ('external', 'cephaloridine'): 1
                    },
                    'is reversible': False,
                    'catalyzed by': [('internal', 'TRANS-CPLX-201')]
                },
                'cephaloridine_beta-lactamase': {
                    'stoichiometry': {
                        ('internal', 'cephaloridine'): -1,
                        ('internal', 'cephaloridine_hydrolyzed'): 1
                    },
                    'is reversible': False,
                    'catalyzed by': [('internal', 'beta-lactamase')]
                }
            },
            'kinetic_parameters': {
                'cephaloridine_tolc': {
                    ('internal', 'TRANS-CPLX-201'): {
                        ('internal', 'cephaloridine'): PARAMETERS.CEPH_PUMP_KM,
                        'kcat_f': PARAMETERS.CEPH_PUMP_KCAT,
                    }
                },
                'cephaloridine_beta-lactamase': {
                    ('internal', 'beta-lactamase'): {
                        ('internal', 'cephaloridine'): PARAMETERS.CEPH_BETA_LACTAMASE_KM,
                        'kcat_f': PARAMETERS.CEPH_BETA_LACTAMASE_KCAT
                    }
                }
            },
            'ports': {
                'internal': ['TRANS-CPLX-201', 'cephaloridine', 'cephaloridine_hydrolyzed', 'beta-lactamase'],
                'external': ['cephaloridine']
            },
            'initial_state': {
                'internal': {
                    'TRANS-CPLX-201': 1e-3,
                    'cephaloridine': INITIAL_INTERNAL_ANTIBIOTIC,
                    'cephaloridine_hydrolyzed': 0,
                    'beta-lactamase': 1e-3,
                },
                'external': {
                    'cephaloridine': INITIAL_EXTERNAL_ANTIBIOTIC
                }
            },
            'time_step': 0.1,
        },
        'fickian_diffusion': {
            'time_step': 0.1,
            'molecules_to_diffuse': ['cephaloridine'],
            'initial_state': {
                'internal': {
                    'cephaloridine': INITIAL_INTERNAL_ANTIBIOTIC,  # mM
                },
                'external': {
                    'cephaloridine': INITIAL_EXTERNAL_ANTIBIOTIC,  # mM
                },
                # 'global': {
                #     'periplasm_volume': (
                #         1.2 * units.fL * PERIPLASM_FRACTION),
                # }
            },
            'surface_area_mass_ratio': 132 * units.cm ** 2 / units.mg,
        },
        'shape_deriver': {},
        'timeline': {
            'time_step': 1.0,
            'timeline': timeline,
        },
        'porin_permeability': {
            'porin_ids': ['CPLX0-7533[o]', 'CPLX0-7534[o]'],
            'diffusing_molecules': {
                'cephaloridine': {
                    'per_porin_perm': {
                        'CPLX0-7533[o]': CEPH_OMPC_CON_PERM,
                        'CPLX0-7534[o]': CEPH_OMPF_CON_PERM
                    },
                    'ph_perm': CEPH_PH_PERM
                },
            },
        },
        'nonspatial_environment': {
            'concentrations': {
                'cephaloridine': INITIAL_EXTERNAL_ANTIBIOTIC,
            },
            'internal_volume': 1.2 * units.fL,
            'env_volume': 1 * units.mL,
        },
    }

    composer = SimpleAntibioticsCell(config)
    composite = composer.generate()

    sim = Engine(composite=composite, initial_state=initial_state)
    sim.update(sim_time)
    timeseries_data = timeseries_from_data(sim.emitter.get_data())
    # plot_variables(
    #     timeseries_data,
    #     variables=[
    #         ('periplasm', 'concs', 'cephaloridine'),
    #         ('periplasm', 'concs', 'cephaloridine_hydrolyzed'),
    #         ('boundary', 'external', 'cephaloridine'),
    #     ],
    # )
    import ipdb; ipdb.set_trace()
    plot_variables(
        timeseries_data,
        variables=[
            ('periplasm', 'concs', 'cephaloridine_hydrolyzed'),
        ],
    )


if __name__ == '__main__':
    demo()

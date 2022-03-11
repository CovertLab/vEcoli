from vivarium.core.composer import Composer
from vivarium.core.composition import (
    composite_in_experiment, simulate_experiment)
from vivarium.core.engine import Engine
from vivarium.core.emitter import timeseries_from_data
from vivarium.library.units import units
from vivarium.plots.simulation_output import plot_variables
from vivarium.plots.topology import plot_topology
from vivarium_convenience.processes.convenience_kinetics import ConvenienceKinetics
from vivarium.processes.timeline import TimelineProcess

from ecoli.processes.antibiotics.antibiotic_transport import AntibioticTransport
from ecoli.processes.antibiotics.antibiotic_hydrolysis import AntibioticHydrolysis
from ecoli.processes.antibiotics.fickian_diffusion import (
    FickianDiffusion,
)
from ecoli.processes.antibiotics.permeability import Permeability, CEPH_OMPC_CON_PERM,\
    CEPH_OMPF_CON_PERM, OUTER_CEPH_PH_PERM, TET_OMPF_CON_PERM, OUTER_TET_PH_PERM, INNER_TET_PH_PERM, SA_AVERAGE
from ecoli.processes.antibiotics.nonspatial_environment import (
    NonSpatialEnvironment)
from ecoli.processes.antibiotics.shape import ShapeDeriver

from numpy import array

INITIAL_EXTERNAL_BETA_LACTAM = 1e-3  # * units.mM
INITIAL_PERIPLASM_BETA_LACTAM = 0  # * units.mM
INITIAL_CYTOSOL_BETA_LACTAM = 0  # * units.mM
INITIAL_EXTERNAL_TET = 1e-3   # * units.mM
INITIAL_PERIPLASM_TET = 0   # * units.mM
INITIAL_CYTOSOL_TET = 0  # * units.mM
INITIAL_PUMP = 1e-3  # * units.mM
BETA_LACTAM_KEY = 'cephaloridine'
TET_KEY = 'tetracycline'
PUMP_KEY = 'TRANS-CPLX-201'

# Source: (Wülfing & Plückthun, 1994)
PERIPLASM_FRACTION = 0.3

CYTOSOL_FRACTION = 1 - PERIPLASM_FRACTION
BETA_LACTAMASE_KEY = 'beta-lactamase'


class PARAMETERS:
    # Calculated from V_max reported in (Nagano & Nikaido, 2009)
    # TODO: CEPH_PUMP parameters are placeholders as we're eventually going to use the Hill Equation for cephaloridine
    # TODO: instead of Michaelis-Menten
    # CEPH_PUMP_KCAT = 0.0956090147363198  # / units.sec  # TODO: Placeholder
    # Reported in (Nagano & Nikaido, 2009)
    CEPH_PUMP_KM = 4.95e-3  # * units.millimolar  # TODO: Placeholder
    # Reported in (Galleni et al., 1988)
    CEPH_BETA_LACTAMASE_KCAT = 130 / units.sec
    # Reported in (Galleni et al., 1988)
    CEPH_BETA_LACTAMASE_KM = 170 * units.micromolar

    # Calculated from V_max reported in (Nikaido, 2012)
    # TET_PUMP_KCAT = 0.00015759727703788977  # / units.sec
    # Reported in (Nikaido, 2012)
    TET_PUMP_KM = 200e-3  # * units.millimolar

    TOLC_KCAT = 1e1  # / units.sec  # TODO: Placeholder. Not supposed to be constant regardless of substrate.

class SimpleAntibioticsCell(Composer):
    '''Integrate antibiotic resistance and susceptibility with wcEcoli
    Integrates the WcEcoli process, which wraps the wcEcoli model, with
    processes to model antibiotic susceptibility (diffusion-based
    import and death) and resistance (hydrolysis and transport-based
    efflux). Also includes derivers.
    '''

    default = {
        'efflux': {},
        'hydrolysis': {},
        'ext_periplasm_diffusion': {},
        'periplasm_cytosol_diffusion': {},
        'shape_deriver': {},
        'nonspatial_environment': {},
        'outer_permeability': {},
        'inner_permeability': {}
    }

    def generate_processes(self, config):
        efflux = ConvenienceKinetics(config['efflux'])
        hydrolysis = AntibioticHydrolysis(config['hydrolysis'])
        ext_periplasm_diffusion = FickianDiffusion(config['ext_periplasm_diffusion'])
        periplasm_cytosol_diffusion = FickianDiffusion(config['periplasm_cytosol_diffusion'])
        timeline = TimelineProcess(config['timeline'])
        return {
            'efflux': efflux,
            'hydrolysis': hydrolysis,
            'ext_periplasm_diffusion': ext_periplasm_diffusion,
            'periplasm_cytosol_diffusion': periplasm_cytosol_diffusion,
            'timeline': timeline
        }

    def generate_steps(self, config):
        nonspatial_environment = NonSpatialEnvironment(config['nonspatial_environment'])
        shape_deriver = ShapeDeriver(config['shape_deriver'])
        outer_permeability = Permeability(config['outer_permeability'])
        inner_permeability = Permeability(config['inner_permeability'])
        return {
            'nonspatial_environment': nonspatial_environment,
            'shape_deriver': shape_deriver,
            'outer_permeability': outer_permeability,
            'inner_permeability': inner_permeability
        }

    def generate_topology(self, config=None):
        boundary_path = config['boundary_path']
        topology = {
            'efflux': {
                'internal': ('periplasm', 'concs'),
                'external': boundary_path + ('external',),
                'exchanges': boundary_path + ('exchanges',),
                'pump_port': ('periplasm', 'concs'),
                'fluxes': ('fluxes',),
                'global': ('periplasm', 'global',),
            },
            'hydrolysis': {
                'internal': ('periplasm', 'concs'),
                'catalyst_port': ('periplasm', 'concs',),
                'fluxes': ('fluxes',),
                'global': ('periplasm', 'global',),
            },
            'ext_periplasm_diffusion': {
                'internal': ('periplasm', 'concs',),
                'external': boundary_path + ('external',),
                'exchanges': boundary_path + ('exchanges',),
                'fluxes': ('fluxes',),
                'volume_global': ('periplasm', 'global',),
                'mass_global': boundary_path,
                'permeabilities': boundary_path + ('outer_permeabilities',)
            },
            'periplasm_cytosol_diffusion': {
                'internal': ('cytosol', 'concs',),
                'external': ('periplasm', 'concs',),
                'exchanges': boundary_path + ('exchanges',),
                'fluxes': ('fluxes',),
                'volume_global': ('cytosol', 'global',),
                'mass_global': boundary_path,
                'permeabilities': boundary_path + ('inner_permeabilities',)
            },
            'shape_deriver': {
                'cell_global': boundary_path,
                'periplasm_global': ('periplasm', 'global',),
                'cytosol_global': ('cytosol', 'global',),
            },
            'nonspatial_environment': {
                'external': boundary_path + ('external',),
                'exchanges': boundary_path + ('exchanges',),
                'fields': ('environment', 'fields',),
                'dimensions': ('environment', 'dimensions'),
                'global': boundary_path,
            },
            'outer_permeability': {
                'porins': ('bulk',),
                'permeabilities': boundary_path + ('outer_permeabilities',),
                'surface_area': boundary_path + ('surface_area',)
            },
            'inner_permeability': {
                'porins': ('bulk',),
                'permeabilities': boundary_path + ('inner_permeabilities',),
                'surface_area': boundary_path + ('surface_area',)
            },
            'timeline': {
                'global': ('global',),  # The global time is read here
                'porins': ('bulk',),  # This port is based on the declared timeline
            },
        }
        return topology


def demo():
    sim_time = 2000

    timeline = []
    for i in range(10):
        timeline.append(
            (i, {
                ('porins', 'CPLX0-7533[o]'):  ((i + 2) * 500),
                ('porins', 'CPLX0-7534[o]'):  ((i + 2) * 500),
            })
        )

    config = {
        'boundary_path': ('boundary',),
        'efflux': {
            'reactions': {
                'export': {
                    'stoichiometry': {
                        ('internal', BETA_LACTAM_KEY): -1,
                        ('external', BETA_LACTAM_KEY): 1,
                        ('internal', TET_KEY): -1,
                        ('external', TET_KEY): 1,
                    },
                    'is_reversible': False,
                    'catalyzed by': [
                        ('pump_port', PUMP_KEY)],
                },
            },
            'kinetic_parameters': {
                'export': {
                    ('pump_port', PUMP_KEY): {
                        'kcat_f': PARAMETERS.TOLC_KCAT,
                        ('internal', BETA_LACTAM_KEY): PARAMETERS.CEPH_PUMP_KM,
                        ('internal', TET_KEY): PARAMETERS.TET_PUMP_KM
                    },
                },
            },
            'initial_state': {
                'fluxes': {
                    'export': 0.0,
                },
                'internal': {
                    BETA_LACTAM_KEY: INITIAL_PERIPLASM_BETA_LACTAM,
                    TET_KEY: INITIAL_PERIPLASM_TET
                },
                'external': {
                    BETA_LACTAM_KEY: INITIAL_EXTERNAL_BETA_LACTAM,
                    TET_KEY: INITIAL_EXTERNAL_TET
                },
                'pump_port': {
                    PUMP_KEY: INITIAL_PUMP,
                },
            },
            'port_ids': ['internal', 'external', 'pump_port'],
            'time_step': 0.05,
        },
        'hydrolysis': {
            'initial_catalyst': 1e-3,
            'catalyst': BETA_LACTAMASE_KEY,
            'initial_target_internal': INITIAL_PERIPLASM_BETA_LACTAM,
            'target': BETA_LACTAM_KEY,
            'kcat': PARAMETERS.CEPH_BETA_LACTAMASE_KCAT,
            'Km': PARAMETERS.CEPH_BETA_LACTAMASE_KM,
            'time_step': 0.05,
        },
        'ext_periplasm_diffusion': {
            'initial_state': {
                'external': {
                    BETA_LACTAM_KEY: INITIAL_EXTERNAL_BETA_LACTAM,
                    TET_KEY: INITIAL_EXTERNAL_TET
                },
                'internal': {
                    BETA_LACTAM_KEY: INITIAL_PERIPLASM_BETA_LACTAM,
                    TET_KEY: INITIAL_PERIPLASM_TET
                },
                'mass_global': {
                    'dry_mass': 300 * units.fg,
                },
                'volume_global': {
                    'volume': 1.2 * units.fL * PERIPLASM_FRACTION,
                },
            },
            'molecules_to_diffuse': [BETA_LACTAM_KEY, TET_KEY],
            # From (Nagano & Nikaido, 2009)
            'surface_area_mass_ratio': 132 * units.cm ** 2 / units.mg,
            'time_step': 0.05,
        },
        'periplasm_cytosol_diffusion': {
            'initial_state': {
                'external': {
                    BETA_LACTAM_KEY: INITIAL_PERIPLASM_BETA_LACTAM,
                    TET_KEY: INITIAL_PERIPLASM_TET
                },
                'internal': {
                    BETA_LACTAM_KEY: INITIAL_CYTOSOL_BETA_LACTAM,
                    TET_KEY: INITIAL_CYTOSOL_TET
                },
                'mass_global': {
                    'dry_mass': 300 * units.fg,
                },
                'volume_global': {
                    'volume': 1.2 * units.fL * CYTOSOL_FRACTION,
                },
            },
            'molecules_to_diffuse': [TET_KEY],
            # From (Nagano & Nikaido, 2009)
            'surface_area_mass_ratio': 132 / CYTOSOL_FRACTION * units.cm ** 2 / units.mg,  # Dividng by 0.7 as cytosol has 70% of mass
            'time_step': 0.05,
        },
        'shape_deriver': {},
        'nonspatial_environment': {
            'concentrations': {
                BETA_LACTAM_KEY: INITIAL_EXTERNAL_BETA_LACTAM,
                TET_KEY: INITIAL_EXTERNAL_TET
            },
            'internal_volume': 1.2 * units.fL,
            'env_volume': 1 * units.mL,
        },
        'timeline': {
            'time_step': 1.0,
            'timeline': timeline,
        },
        'outer_permeability': {
            'porin_ids': ['CPLX0-7533[o]', 'CPLX0-7534[o]'],
            'diffusing_molecules': {
                'cephaloridine': {
                    'per_porin_perm': {
                        'CPLX0-7533[o]': CEPH_OMPC_CON_PERM,
                        'CPLX0-7534[o]': CEPH_OMPF_CON_PERM
                    },
                    'ph_perm': OUTER_CEPH_PH_PERM
                },
                'tetracycline': {
                    'per_porin_perm': {
                        'CPLX0-7534[o]': TET_OMPF_CON_PERM,
                    },
                    'ph_perm': OUTER_TET_PH_PERM
                }
            },
        },
        'inner_permeability': {
            'porin_ids': [],
            'diffusing_molecules': {
                'tetracycline': {
                    'per_porin_perm': {},
                    'ph_perm': INNER_TET_PH_PERM
                }
            },
        },
    }

    composite = SimpleAntibioticsCell(config).generate()
    initial_state = composite.initial_state()
    initial_state['boundary']['surface_area'] = SA_AVERAGE
    initial_state['bulk'] = {}
    initial_state['bulk']['CPLX0-7533[o]'] = 500
    initial_state['bulk']['CPLX0-7534[o]'] = 500
    initial_state['environment'] = {}
    initial_state['environment']['fields'] = {}
    initial_state['environment']['fields']['cephaloridine'] = array([[INITIAL_EXTERNAL_BETA_LACTAM]])
    initial_state['environment']['fields']['tetracycline'] = array([[INITIAL_EXTERNAL_TET]])

    sim = Engine(composite=composite, initial_state=initial_state)
    sim.update(sim_time)
    timeseries_data = timeseries_from_data(sim.emitter.get_data())
    plot_variables(
        timeseries_data,
        variables=[
            ('boundary', 'external', 'cephaloridine'),
            ('boundary', 'external', 'tetracycline'),
            ('periplasm', 'concs', 'cephaloridine'),
            ('periplasm', 'concs', 'tetracycline'),
            ('periplasm', 'concs', 'cephaloridine_hydrolyzed'),
            ('cytosol', 'concs', 'tetracycline'),
        ],
        out_dir='out',
        filename='antibiotics_simple'
    )


if __name__ == '__main__':
    demo()

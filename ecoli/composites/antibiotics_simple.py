from numpy import array
from vivarium.core.composer import Composer
from vivarium.core.composition import (
    composite_in_experiment, simulate_experiment)
from vivarium.library.units import units
from vivarium.plots.simulation_output import plot_variables
from vivarium.plots.topology import plot_topology
from vivarium.processes.timeline import TimelineProcess

from ecoli.states.wcecoli_state import get_state_from_file
from ecoli.processes.enzyme_kinetics import EnzymeKinetics
from ecoli.processes.antibiotics.antibiotic_hydrolysis import AntibioticHydrolysis
from ecoli.processes.antibiotics.fickian_diffusion import (
    FickianDiffusion,
)
from ecoli.processes.antibiotics.nonspatial_environment import (
    NonSpatialEnvironment)
from ecoli.processes.antibiotics.porin_permeability import PorinPermeability, CEPH_OMPC_CON_PERM,\
    CEPH_OMPF_CON_PERM, SA_AVERAGE
from ecoli.processes.antibiotics.shape import ShapeDeriver


INITIAL_INTERNAL_ANTIBIOTIC = 0
INITIAL_EXTERNAL_ANTIBIOTIC = 1e-3
ANTIBIOTIC_KEY = 'cephaloridine'
PUMP_KEY = 'TRANS-CPLX-201'
# Source: (Wülfing & Plückthun, 1994)
PERIPLASM_FRACTION = 0.3
BETA_LACTAMASE_KEY = 'beta-lactamase'


class PARAMETERS:
    # Reported in (Nagano & Nikaido, 2009)
    CEPH_PUMP_KCAT = 1e1 / units.sec  # TODO(Matt): Placeholder
    # Reported in (Nagano & Nikaido, 2009)
    CEPH_PUMP_KM = 4.95e-3 * units.millimolar  # TODO(Matt): Placeholder
    # Reported in (Galleni et al., 1988)
    CEPH_BETA_LACTAMASE_KCAT = 130 / units.sec
    # Reported in (Galleni et al., 1988)
    CEPH_BETA_LACTAMASE_KM = 170 * units.micromolar


class SimpleAntibioticsCell(Composer):
    '''Integrate antibiotic resistance and susceptibility with wcEcoli

    Integrates the WcEcoli process, which wraps the wcEcoli model, with
    processes to model antibiotic susceptibility (diffusion-based
    import and death) and resistance (hydrolysis and transport-based
    efflux). Also includes derivers.
    '''

    defaults = {
        'boundary_path': ('boundary',),
        'efflux': {
            'reactions': {
                'cephaloridine_tolc': {
                    'stoichiometry': {
                        ('periplasm', 'concs', 'cephaloridine'): -1,
                        ('boundary', 'external', 'cephaloridine'): 1
                    },
                    'is reversible': False,
                    'catalyzed by': [('bulk', 'TRANS-CPLX-201')]
                },
                'cephaloridine_beta-lactamse': {
                    'stoichiometry': {
                        ('periplasm', 'concs', 'cephaloridine'): -1,
                        ('periplasm', 'concs', 'cephaloridine_hydrolyzed'): 1
                    },
                    'is reversible': False,
                    'catalyzed by': [('periplasm', 'concs', 'beta-lactamase')]
                }
            },
            'kinetic_parameters': {
                'cephaloridine_tolc': {
                    ('bulk', 'TRANS-CPLX-201'): {
                        ('periplasm', 'concs', 'cephaloridine'): PARAMETERS.CEPH_PUMP_KM,
                        'kcat_f': PARAMETERS.CEPH_PUMP_KCAT,
                    }
                },
                'cephaloridine_beta-lactamase': {
                    ('periplasm', 'concs', 'beta-lactamase'): {
                        ('periplasm', 'concs', 'cephaloridine'): PARAMETERS.CEPH_BETA_LACTAMASE_KM,
                        'kcat_f': PARAMETERS.CEPH_BETA_LACTAMASE_KCAT
                    }
                }
            },
            'time_step': 0.1,
        },
        'fickian_diffusion': {
            'default_state': {
                'external': {
                    ANTIBIOTIC_KEY: INITIAL_EXTERNAL_ANTIBIOTIC,
                },
                'internal': {
                    ANTIBIOTIC_KEY: INITIAL_INTERNAL_ANTIBIOTIC,
                },
                'global': {
                    'periplasm_volume': (
                        1.2 * units.fL * PERIPLASM_FRACTION),
                },
            },
            'molecules_to_diffuse': [ANTIBIOTIC_KEY],
            # (Nagano & Nikaido, 2009) reports that their mutant strain,
            # RAM121, has 10-fold faster influx of nitrocefin with a
            # permeability of 0.2e-5 cm/s, so wildtype has a
            # permeability of 0.2e-6 cm/s.
            'permeability': 0.2e-6 * units.cm / units.sec,
            # From (Nagano & Nikaido, 2009)
            'surface_area_mass_ratio': 132 * units.cm**2 / units.mg,
            'time_step': 0.1,
        },
        'shape_deriver': {},
        'porin_permeability': {
            'porin_ids': ['CPLX0-7533[o]', 'CPLX0-7534[o]'],
            'diffusing_molecules': {
                'cephaloridine': {
                    'CPLX0-7533[o]': CEPH_OMPC_CON_PERM,
                    'CPLX0-7534[o]': CEPH_OMPF_CON_PERM
                },
            },
        },
        'timeline': {},
    }

    def generate_processes(self, config):
        efflux = EnzymeKinetics(config['efflux'])
        hydrolysis = AntibioticHydrolysis(config['hydrolysis'])
        fickian_diffusion = FickianDiffusion(
            config['fickian_diffusion'])
        timeline = TimelineProcess(config['timeline'])
        return {
            'efflux': efflux,
            'hydrolysis': hydrolysis,
            'fickian_diffusion': fickian_diffusion,
            'timeline': timeline,
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
                'global': ('periplasm', 'global'),
            },
            'hydrolysis': {
                'internal': ('periplasm', 'concs'),
                'catalyst_port': ('periplasm', 'concs'),
                'fluxes': ('fluxes',),
                'global': ('periplasm', 'global'),
            },
            'fickian_diffusion': {
                'internal': ('periplasm', 'concs'),
                'external': boundary_path + ('external',),
                'exchanges': boundary_path + ('exchanges',),
                'fluxes': ('fluxes',),
                'volume_global': ('periplasm', 'global'),
                'mass_global': boundary_path,
                'permeabilities': boundary_path + ('permeabilities',)
            },
            'shape_deriver': {
                'cell_global': boundary_path,
                'periplasm_global': ('periplasm', 'global')
            },
            'timeline': {
                'global': ('global',),  # The global time is read here
                'porins': ('bulk',),  # This port is based on the declared timeline
            },
            'porin_permeability': {
                'porins': ('bulk',),
                'permeabilities': boundary_path + ('permeabilities',),
                'surface_area': boundary_path + ('surface_area',)
            },
        }
        return topology

    def generate_steps(self, config):
        shape_deriver = ShapeDeriver(config['shape_deriver'])
        porin_permeability = PorinPermeability(config['porin_permeability'])
        return {
            'shape_deriver': shape_deriver,
            'porin_permeability': porin_permeability
        }


def demo():
    composite = SimpleAntibioticsCell().generate()
    env = NonSpatialEnvironment({
        'concentrations': {
            'cephaloridine': INITIAL_EXTERNAL_ANTIBIOTIC,
        },
        'internal_volume': 1.2 * units.fL,
        'env_volume': 1 * units.mL,
    })
    composite.merge(
        composite=env.generate(),
        topology={
            'nonspatial_environment': {
                'external': ('boundary', 'external'),
                'exchanges': ('boundary', 'exchanges'),
                'fields': ('environment', 'fields'),
                'dimensions': ('environment', 'dimensions'),
                'global': ('boundary',),
            }
        }
    )

    initial_state = get_state_from_file(path='data/vivecoli_t1000.json')
    initial_state['boundary'] = {}
    initial_state['boundary']['surface_area'] = SA_AVERAGE
    initial_state['listeners']['mass']['dry_mass'] = initial_state['listeners']['mass']['dry_mass'] * units.fg
    # initial_state['boundary']['external'] = {}
    # initial_state['boundary']['external']['cephaloridine'] = array([[INITIAL_EXTERNAL_ANTIBIOTIC]])
    initial_state['periplasm'] = {}
    initial_state['periplasm']['concs'] = {}
    initial_state['periplasm']['concs']['beta-lactamse'] = array([[1e-3]])
    initial_state['bulk']['CPLX0-7533[o]'] = 500
    initial_state['bulk']['CPLX0-7534[o]'] = 500

    import ipdb; ipdb.set_trace()

    exp = composite_in_experiment(
        composite,
        initial_state=initial_state,
    )
    data = simulate_experiment(
        exp,
        {'total_time': 10})
    fig = plot_variables(
        data,
        variables=[
            ('periplasm', 'concs', 'cephaloridine'),
            ('periplasm', 'concs', 'cephaloridine_hydrolyzed'),
            ('boundary', 'external', 'cephaloridine'),
        ],
    )
    return fig, data


if __name__ == '__main__':
    demo()

from numpy import array
from vivarium.core.composer import Composer
from vivarium.core.composition import (
    composite_in_experiment, simulate_experiment)
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
                'cephaloridine_beta-lactamase': {
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
        'timeline': {},
        'porin_permeability': {
            'porin_ids': ['CPLX0-7533[o]', 'CPLX0-7534[o]'],
            'diffusing_molecules': {
                'cephaloridine': {
                    'CPLX0-7533[o]': CEPH_OMPC_CON_PERM,
                    'CPLX0-7534[o]': CEPH_OMPF_CON_PERM
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
            'efflux': {
                'internal': ('periplasm', 'concs'),
                'external': ('boundary', 'external',),
                'exchanges': ('boundary', 'exchanges',),
                'fluxes': ('fluxes',),
                'global': ('periplasm', 'global'),
            },
            'fickian_diffusion': {
                'internal': ('periplasm', 'concs'),
                'external': ('boundary', 'external',),
                'exchanges': ('boundary', 'exchanges',),
                'fluxes': ('fluxes',),
                'volume_global': ('periplasm', 'global'),
                'mass_global': ('boundary',),
                'permeabilities': ('boundary', 'permeabilities',)
            },
            'shape_deriver': {
                'cell_global': ('boundary',),
                'periplasm_global': ('periplasm', 'global')
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
                'global': ('boundary',),
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
    composite = SimpleAntibioticsCell().generate()

    initial_state = get_state_from_file(path='data/vivecoli_t1000.json')
    initial_state['boundary'] = {}
    initial_state['boundary']['surface_area'] = SA_AVERAGE
    initial_state['listeners']['mass']['dry_mass'] = initial_state['listeners']['mass']['dry_mass'] * units.fg
    # initial_state['boundary']['external'] = {}
    # initial_state['boundary']['external']['cephaloridine'] = array([[INITIAL_EXTERNAL_ANTIBIOTIC]])
    initial_state['periplasm'] = {}
    initial_state['periplasm']['concs'] = {}
    initial_state['periplasm']['concs']['beta-lactamase'] = array([[1e-3]])
    initial_state['bulk']['CPLX0-7533[o]'] = 500
    initial_state['bulk']['CPLX0-7534[o]'] = 500


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

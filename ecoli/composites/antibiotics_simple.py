import os

from vivarium.core.composer import Composer
from vivarium.core.composition import (
    composite_in_experiment, simulate_experiment, BASE_OUT_DIR)
from vivarium.library.units import units
from vivarium.plots.simulation_output import plot_variables
from vivarium.plots.topology import plot_topology

from ecoli.processes.antibiotics.antibiotic_transport import AntibioticTransport
from ecoli.processes.antibiotics.antibiotic_hydrolysis import AntibioticHydrolysis
from ecoli.processes.antibiotics.fickian_diffusion import (
    FickianDiffusion,
)
from ecoli.processes.antibiotics.nonspatial_environment import (
    NonSpatialEnvironment)
from ecoli.processes.shape import Shape


OUT_DIR = os.path.join(BASE_OUT_DIR, 'experiments', 'antibiotics_colony')
INITIAL_INTERNAL_ANTIBIOTIC = 0 * units.mM
INITIAL_EXTERNAL_ANTIBIOTIC = 0.1239 * units.mM
ANTIBIOTIC_KEY = 'antibiotic'
PUMP_KEY = 'pump'
# Source: (Wülfing & Plückthun, 1994)
PERIPLASM_FRACTION = 0.3
BETA_LACTAMASE_KEY = 'beta-lactamase'


class PARAMETERS:
    # Reported in (Nagano & Nikaido, 2009)
    PUMP_KCAT = 1e1 / units.sec
    # Reported in (Nagano & Nikaido, 2009)
    PUMP_KM = 4.95e-3 * units.millimolar
    # Reported in (Galleni et al., 1988)
    BETA_LACTAMASE_KCAT = 490 / units.sec
    # Reported in (Galleni et al., 1988)
    BETA_LACTAMASE_KM = 500 * units.micromolar


class SimpleAntibioticsCell(Composer):
    '''Integrate antibiotic resistance and susceptibility with wcEcoli

    Integrates the WcEcoli process, which wraps the wcEcoli model, with
    processes to model antibiotic susceptibility (diffusion-based
    import and death) and resistance (hydrolysis and transport-based
    efflux). Also includes derivers.
    '''

    name = 'simple_antibiotics_cell'
    defaults = {
        'boundary_path': ('boundary',),
        'efflux': {
            'initial_pump': 0.45e-3 * units.mM,
            'initial_internal_antibiotic': INITIAL_INTERNAL_ANTIBIOTIC,
            'intial_external_antibiotic': INITIAL_EXTERNAL_ANTIBIOTIC,
            'kcat': PARAMETERS.PUMP_KCAT,
            'Km': PARAMETERS.PUMP_KM,
            'pump_key': PUMP_KEY,
            'antibiotic_key': ANTIBIOTIC_KEY,
            'time_step': 0.1,
        },
        'hydrolysis': {
            'initial_catalyst': 0.5e-3 * units.mM,
            'catalyst': BETA_LACTAMASE_KEY,
            'initial_target_internal': INITIAL_INTERNAL_ANTIBIOTIC,
            'target': ANTIBIOTIC_KEY,
            'kcat': PARAMETERS.BETA_LACTAMASE_KCAT,
            'Km': PARAMETERS.BETA_LACTAMASE_KM,
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
                        1 * units.fL * PERIPLASM_FRACTION),
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
        'shape': {}
    }

    def generate_processes(self, config):
        efflux = AntibioticTransport(config['efflux'])
        hydrolysis = AntibioticHydrolysis(config['hydrolysis'])
        fickian_diffusion = FickianDiffusion(
            config['fickian_diffusion'])
        shape = Shape(config['shape'])
        return {
            'efflux': efflux,
            'hydrolysis': hydrolysis,
            'fickian_diffusion': fickian_diffusion,
            'shape': shape,
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
                'mass_global': ('mass_listener',),
            },
            'shape': {
                'cell_global': boundary_path,
                'periplasm_global': ('periplasm', 'global'),
                'listener_cell_mass': ('mass_listener', 'dry_mass'),
            },
        }
        return topology


def demo():
    composite = SimpleAntibioticsCell().generate()
    env = NonSpatialEnvironment({
        'concentrations': {
            'antibiotic': INITIAL_EXTERNAL_ANTIBIOTIC.magnitude,
        },
        'internal_volume': 1 * units.fL,
        'env_volume': 1 * units.mL,
    })
    composite.merge(
        composite=env.generate(),
        topology={
            'nonspatial_environment': {
                'external': ('boundary', 'external'),
                'exchanges': ('boundary','exchanges'),
                'fields': ('environment', 'fields'),
                'dimensions': ('environment', 'dimensions'),
                'global': ('boundary',),
            }
        }
    )

    exp = composite_in_experiment(
        composite,
        initial_state=composite.initial_state(),
    )
    data = simulate_experiment(
        exp,
        {'total_time': 10})
    fig = plot_variables(
        data,
        variables=[
            ('periplasm', 'concs', 'antibiotic'),
            ('periplasm', 'concs', 'antibiotic_hydrolyzed'),
            ('boundary', 'external', 'antibiotic'),
        ],
    )
    return fig, data


def main():
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    fig, _ = demo()
    fig.savefig(os.path.join(OUT_DIR, SimpleAntibioticsCell.name))


if __name__ == '__main__':
    main()

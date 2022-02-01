from matplotlib import pyplot as plt
import numpy as np
from scipy import constants

from vivarium.library.units import units, Quantity, remove_units
from vivarium.core.process import Process
from vivarium.core.composer import Composite
from vivarium.core.composition import (
    composite_in_experiment, simulate_experiment)
from vivarium.plots.simulation_output import plot_variables

from ecoli.processes.antibiotics.nonspatial_environment import (
    NonSpatialEnvironment)


AVOGADRO = constants.N_A * 1 / units.mol


class FickianDiffusion(Process):

    name = "fickian_diffusion"
    defaults = {
        'molecules_to_diffuse': ['antibiotic'],
        'initial_state': {
            'internal': {
                'antibiotic': 0,  # mM
            },
            'external': {
                'antibiotic': 1e-3,  # mM
            },
            'mass_global': {
                'dry_mass': 300 * units.fg,
            },
            'volume_global': {
                'volume': 1.2 * units.fL,
            },
        },
        'default_default': 0,
        'permeability': 1e-5 * units.cm / units.sec,
        'surface_area_mass_ratio': 132 * units.cm**2 / units.mg,
    }

    def ports_schema(self):

        schema = {
            'internal': {
                # Molecule concentration in mmol/L
                molecule: {
                    '_default': self.parameters['default_default'],
                    '_divider': 'set',
                    '_emit': True,
                }
                for molecule in self.parameters['molecules_to_diffuse']
            },
            'external': {
                # Molecule concentration in mmol/L
                molecule: {
                    '_default': self.parameters['default_default'],
                    '_divider': 'set',
                    '_emit': True,
                }
                for molecule in self.parameters['molecules_to_diffuse']
            },
            'fluxes': {
                # Molecule counts
                molecule: {
                    '_default': self.parameters['default_default'],
                    '_divider': 'set',
                }
                for molecule in self.parameters['molecules_to_diffuse']
            },
            'exchanges': {
                # Molecule counts
                molecule: {
                    '_default': self.parameters['default_default'],
                    '_divider': 'split',
                }
                for molecule in self.parameters['molecules_to_diffuse']
            },
            'volume_global': {
                'volume': {
                    '_default': self.parameters['default_default'],
                    '_divider': 'split',
                },
            },
            'mass_global': {
                'dry_mass': {
                    '_default': (
                        self.parameters['default_default'] * units.fg),
                    '_divider': 'split',
                },
            },
            'permeabilities': {
                mol_id: {
                    '_default': 1e-5 * units.cm / units.sec,
                    '_emit': True,
                    '_updater': 'set'
                } for mol_id in self.parameters['molecules_to_diffuse']
            }
        }

        for port, port_conf in self.parameters['initial_state'].items():
            for variable, default in port_conf.items():
                if variable in schema[port]:
                    schema[port][variable]['_default'] = default

        return schema

    def initial_state(self, config=None):
        config = config or {}
        parameters = self.parameters
        parameters.update(config)

        initial_state = {
            'internal': {
                molecule: parameters['default_default']
                for molecule in self.parameters['molecules_to_diffuse']
            },
            'external': {
                molecule: parameters['default_default']
                for molecule in self.parameters['molecules_to_diffuse']
            },
            'fluxes': {
                molecule: parameters['default_default']
                for molecule in self.parameters['molecules_to_diffuse']
            },
            'exchanges': {
                molecule: parameters['default_default']
                for molecule in self.parameters['molecules_to_diffuse']
            },
            'volume_global': {
                'volume': parameters['default_default'],
            },
            'mass_global': {
                'dry_mass': parameters['default_default'] * units.fg,
            },
        }
        initial_state.update(parameters['initial_state'])
        return initial_state

    def next_update(self, timestep, states):
        permeability = states['permeabilities']['antibiotic']
        area_mass = self.parameters['surface_area_mass_ratio']
        mass = states['mass_global']['dry_mass']
        flux_mmol = {}
        for molecule in self.parameters['molecules_to_diffuse']:
            # Flux is positive when leaving the cell
            delta_concentration = (
                states['internal'][molecule]
                - states['external'][molecule]
            ) * units.mmol / units.L
            # Fick's first law of diffusion:
            rate = permeability * area_mass * delta_concentration
            flux = rate * mass * timestep * units.sec
            flux_mmol[molecule] = flux  #.to('millimole/milligram')
        flux_counts = {
            molecule: flux * AVOGADRO  #.to('1/milligram')
            for molecule, flux in flux_mmol.items()
        }
        if not isinstance(states['volume_global']['volume'], Quantity):
            volume = states['volume_global']['volume'] * units.fL
        else:
            volume = states['volume_global']['volume']
        update = {
            'fluxes': {
                molecule: mol_flux.to(units.count).magnitude
                for molecule, mol_flux in flux_counts.items()
            },
            'exchanges': {
                molecule: mol_flux.to(units.count).magnitude
                for molecule, mol_flux in flux_counts.items()
            },
            'internal': {
                molecule: - (
                    mol_flux / volume
                ).to(units.mmol / units.L).magnitude
                for molecule, mol_flux in flux_mmol.items()
            },
        }
        return update


def demo():
    proc = FickianDiffusion()
    env = NonSpatialEnvironment({
        'concentrations': {
            molecule: FickianDiffusion.defaults[
                'initial_state']['external'][molecule]
            for molecule in FickianDiffusion.defaults[
                'molecules_to_diffuse']
        },
        'internal_volume': 1.2 * units.fL,
        'env_volume': 1 * units.fL,
    })
    composite = Composite({
        'processes': {
            proc.name: proc,
            env.name: env,
        },
        'topology': {
            **proc.generate_topology(),
            **env.generate_topology(),
        },
    })
    exp = composite_in_experiment(
        composite,
        initial_state=composite.initial_state(),
    )
    data = simulate_experiment(exp, {'total_time': 10})
    fig = plot_variables(
        data,
        variables=[
            ('internal', 'antibiotic'),
            ('external', 'antibiotic'),
        ],
    )
    return fig, data


def get_expected_demo_data():
    p = FickianDiffusion.defaults['permeability']
    x_am = FickianDiffusion.defaults['surface_area_mass_ratio']

    def rate(internal, external, dry_mass):
        # Note: delta_t = 1 sec
        return p * x_am * (internal - external) * dry_mass

    state = {
        'internal': 0 * units.millimolar,
        'external': 1e-3 * units.millimolar,
        'dry_mass': 300 * units.fg,
    }
    data = {key: [val] for key, val in state.items()}
    data['time'] = [0]
    for i in range(10):
        flux = rate(
            state['internal'], state['external'], state['dry_mass']
        ) * AVOGADRO * units.sec  # dt = 1 sec

        state['internal'] -= flux / (1.2 * units.fL) / AVOGADRO
        state['external'] += flux / (1 * units.fL) / AVOGADRO

        for key, val in state.items():
            data[key].append(val)
        data['time'].append(i + 1)

    return remove_units(data)


def test_fickian_diffusion():
    _, simulated_data = demo()
    expected_data = get_expected_demo_data()
    assert simulated_data['time'] == expected_data['time']
    np.testing.assert_allclose(
        simulated_data['internal']['antibiotic'],
        expected_data['internal'],
        rtol=0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        simulated_data['external']['antibiotic'],
        expected_data['external'],
        rtol=0,
        atol=1e-15,
    )


def get_demo_vs_expected_plot(demo_data, expected_data):
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(10, 5))
    ax1.plot(
        demo_data['time'], demo_data['internal']['antibiotic'],
        label='simulated', alpha=0.5,
    )
    ax1.plot(
        expected_data['time'], expected_data['internal'],
        label='expected', alpha=0.5,
    )
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Concentration (mM)')
    ax1.set_title('Internal Antibiotic')
    ax1.legend()

    ax2.plot(
        demo_data['time'],
        demo_data['external']['antibiotic'],
        label='simulated', alpha=0.5,
    )
    ax2.plot(
        expected_data['time'], expected_data['external'],
        label='expected', alpha=0.5,
    )
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Concentration (mM)')
    ax2.set_title('External Antibiotic')

    fig.tight_layout()

    return fig


def main():
    _, data = demo()
    expected = get_expected_demo_data()
    fig = get_demo_vs_expected_plot(data, expected)
    fig.savefig('test2.png')


if __name__ == '__main__':
    main()

'''
==========================
Simulate Antibiotic Export
==========================
'''


import copy

from matplotlib import pyplot as plt
import numpy as np
from scipy.constants import N_A

from vivarium.core.composition import (
    composite_in_experiment, simulate_experiment)
from vivarium.core.composer import Composite
from vivarium.library.units import units, Quantity
from vivarium.library.dict_utils import deep_merge, deep_merge_check
from vivarium.plots.simulation_output import plot_variables
from vivarium_convenience.processes.convenience_kinetics import ConvenienceKinetics

from ecoli.processes.antibiotics.nonspatial_environment import (
    NonSpatialEnvironment)

AVOGADRO = N_A / units.mol


class AntibioticTransport(ConvenienceKinetics):

    name = 'antibiotic_transport'
    defaults = {
        'kcat': 1 / units.sec,
        'Km': 1e-3 * units.millimolar,
        'pump_key': 'pump',
        'antibiotic_key': 'antibiotic',
        'initial_internal_antibiotic': 1e-3 * units.millimolar,
        'initial_external_antibiotic': 0 * units.millimolar,
        'initial_pump': 1e-3 * units.millimolar,
        'time_step': 1,
    }

    def __init__(self, parameters=None):
        initial_parameters = parameters or {}
        super_defaults = super().defaults
        deep_merge_check(self.defaults, super_defaults)
        parameters = copy.deepcopy(self.defaults)
        deep_merge(parameters, initial_parameters)

        kcat = parameters['kcat'].to(1 / units.sec).magnitude
        km = parameters['Km'].to(units.millimolar).magnitude
        initial_internal = parameters['initial_internal_antibiotic'].to(
            units.millimolar).magnitude
        initial_external = parameters['initial_external_antibiotic'].to(
            units.millimolar).magnitude
        initial_pump = parameters['initial_pump'].to(
            units.millimolar).magnitude


        kinetics_parameters = {
            'reactions': {
                'export': {
                    'stoichiometry': {
                        ('internal', parameters['antibiotic_key']): -1,
                        ('external', parameters['antibiotic_key']): 1,
                    },
                    'is_reversible': False,
                    'catalyzed by': [
                        ('pump_port', parameters['pump_key'])],
                },
            },
            'kinetic_parameters': {
                'export': {
                    ('pump_port', parameters['pump_key']): {
                        'kcat_f': kcat,
                        ('internal', parameters['antibiotic_key']): km,
                    },
                },
            },
            'initial_state': {
                'fluxes': {
                    'export': 0.0,
                },
                'internal': {
                    parameters['antibiotic_key']: initial_internal,
                },
                'external': {
                    parameters['antibiotic_key']: initial_external,
                },
                'pump_port': {
                    parameters['pump_key']: initial_pump,
                },
            },
            'port_ids': ['internal', 'external', 'pump_port'],
            'time_step': parameters['time_step'],
            '_original_parameters': parameters,
        }

        super().__init__(kinetics_parameters)


def demo():
    proc = AntibioticTransport()
    env = NonSpatialEnvironment({
        'concentrations': {
            'antibiotic': AntibioticTransport.defaults[
                'initial_external_antibiotic'],
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
    kcat = AntibioticTransport.defaults['kcat'].to(
        1 / units.sec).magnitude
    km = AntibioticTransport.defaults['Km'].to(
        units.millimolar).magnitude

    def rate(substrate, enzyme):
        return kcat * enzyme * substrate / (km + substrate)

    state = {
        'internal': 1e-3,  # mM
        'external': 0,  # mM
        'pump': 1e-3,  # mM
    }
    data = {key: [val] for key, val in state.items()}
    data['time'] = [0]
    for i in range(10):
        v = rate(state['internal'], state['pump'])
        state['internal'] -= v  # dt = 1 sec
        flux = v * units.millimolar * 1.2 * units.fL * AVOGADRO
        state['external'] += (flux / (1 * units.fL) / AVOGADRO).to(
            units.millimolar).magnitude

        for key, val in state.items():
            data[key].append(val)
        data['time'].append(i + 1)

    return data


def test_antibiotic_transport():
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

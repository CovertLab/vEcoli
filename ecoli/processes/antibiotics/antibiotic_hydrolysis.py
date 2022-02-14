import copy

from matplotlib import pyplot as plt
import numpy as np

from vivarium.library.units import units, Quantity
from vivarium.library.dict_utils import deep_merge, deep_merge_check
from vivarium.core.composition import simulate_process
from vivarium.plots.simulation_output import plot_variables
from vivarium_convenience.processes.convenience_kinetics import ConvenienceKinetics


class AntibioticHydrolysis(ConvenienceKinetics):

    name = 'antibiotic_hydrolysis'
    defaults = {
        'kcat': 1 / units.sec,
        'Km': 1e-3 * units.millimolar,
        'target': 'antibiotic',
        'initial_target_internal': 1e-3 * units.millimolar,
        'initial_hydrolyzed_internal': 0 * units.millimolar,
        'catalyst': 'catalyst',
        'initial_catalyst': 1e-3 * units.millimolar,
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
        initial_target = parameters['initial_target_internal'].to(
            units.millimolar).magnitude
        initial_hydrolyzed = parameters[
            'initial_hydrolyzed_internal'].to(
                units.millimolar).magnitude
        initial_catalyst = parameters['initial_catalyst'].to(
            units.millimolar).magnitude
        hydrolyzed_key = f'{parameters["target"]}_hydrolyzed'

        kinetics_parameters = {
            'reactions': {
                'hydrolysis': {
                    'stoichiometry': {
                        ('internal', parameters['target']): -1,
                        ('internal', hydrolyzed_key): 1,
                    },
                    'is_reversible': False,
                    'catalyzed by': [
                        ('catalyst_port', parameters['catalyst'])],
                },
            },
            'kinetic_parameters': {
                'hydrolysis': {
                    ('catalyst_port', parameters['catalyst']): {
                        'kcat_f': kcat,
                        ('internal', parameters['target']): km,
                    },
                },
            },
            'initial_state': {
                'fluxes': {
                    'hydrolysis': 0.0,
                },
                'internal': {
                    parameters['target']: initial_target,
                    hydrolyzed_key: initial_hydrolyzed,
                },
                'catalyst_port': {
                    parameters['catalyst']: initial_catalyst,
                },
            },
            'port_ids': ['internal', 'catalyst_port'],
            'added_port_ids': ['fluxes', 'global'],
            'time_step': parameters['time_step'],
            '_original_parameters': parameters,
        }

        super().__init__(kinetics_parameters)


def demo():
    proc = AntibioticHydrolysis()
    data = simulate_process(proc, {'total_time': 10})
    fig = plot_variables(
        data,
        variables=[
            ('internal', 'antibiotic'),
            ('internal', 'antibiotic_hydrolyzed'),
        ],
    )
    return fig, data


def get_expected_demo_data():
    kcat = AntibioticHydrolysis.defaults['kcat'].to(
        1 / units.sec).magnitude
    km = AntibioticHydrolysis.defaults['Km'].to(
        units.millimolar).magnitude

    def rate(substrate, enzyme):
        return kcat * enzyme * substrate / (km + substrate)

    state = {
        'antibiotic': 1e-3,  # mM
        'antibiotic_hydrolyzed': 0,  # mM
        'catalyst': 1e-3,  # mM
    }
    data = {key: [val] for key, val in state.items()}
    data['time'] = [0]
    for i in range(10):
        v = rate(state['antibiotic'], state['catalyst'])
        state['antibiotic'] -= v  # dt = 1 sec
        state['antibiotic_hydrolyzed'] += v  # dt = 1 sec

        for key, val in state.items():
            data[key].append(val)
        data['time'].append(i + 1)

    return data


def test_antibiotic_hydrolysis():
    _, simulated_data = demo()
    expected_data = get_expected_demo_data()
    assert simulated_data['time'] == expected_data['time']
    np.testing.assert_allclose(
        simulated_data['internal']['antibiotic'],
        expected_data['antibiotic'],
        rtol=0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        simulated_data['internal']['antibiotic_hydrolyzed'],
        expected_data['antibiotic_hydrolyzed'],
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
        expected_data['time'], expected_data['antibiotic'],
        label='expected', alpha=0.5,
    )
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Concentration (mM)')
    ax1.set_title('Antibiotic')
    ax1.legend()

    ax2.plot(
        demo_data['time'],
        demo_data['internal']['antibiotic_hydrolyzed'],
        label='simulated', alpha=0.5,
    )
    ax2.plot(
        expected_data['time'], expected_data['antibiotic_hydrolyzed'],
        label='expected', alpha=0.5,
    )
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Concentration (mM)')
    ax2.set_title('Hydrolyzed Antibiotic')

    fig.tight_layout()

    return fig

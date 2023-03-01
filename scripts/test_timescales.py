import os

from matplotlib import pyplot as plt
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import root_scalar, root
from vivarium.library.units import units

from ecoli.library.parameters import param_store


CHANGE_RELATIVE_THRESHOLD = 0.001  # 0.1% change acceptable
MAX_TIMESCALE = 2  # Seconds
OUT_DIR = os.path.join(
    os.path.dirname(__file__), '..', 'out')


def _assert_timescale(derivative_func, y0, dt):
    result = odeint(derivative_func, y0, np.linspace(0, dt, 100))
    delta = result[-1] - result[-2]
    assert np.all(abs(delta) <= result[-1] * CHANGE_RELATIVE_THRESHOLD)


def _get_michaelis_menten_derivative(
        rate_constant, michaelis_constant, hill_coefficient,
        enzyme_conc):
    def derivative(y0, _):
        return -rate_constant * enzyme_conc * y0**hill_coefficient / (
            y0**hill_coefficient + michaelis_constant)
    return derivative


def _get_fickian_derivative(
        diffusivity, external_conc, volume, inner_bias=1):
    def derivative(y0, _):
        return diffusivity * (inner_bias * external_conc - y0) / volume
    return derivative


ODES = {
    'cephaloridine': {
        # Map from name to (derivative_func, y0)
        'Efflux': (
            _get_michaelis_menten_derivative(
                param_store.get(('cephaloridine', 'efflux', 'kcat')).to(
                    1 / units.sec).magnitude,
                param_store.get(('cephaloridine', 'efflux', 'km')).to(
                    units.mM).magnitude,
                param_store.get(('cephaloridine', 'efflux', 'n')),
                param_store.get(('concs', 'initial_pump')).to(
                    units.mM).magnitude,
            ),
            # Assume diffusion is fast.
            param_store.get(('cephaloridine', 'mic')).to(
                units.mM).magnitude,
        ),
        'Hydrolysis': (
            _get_michaelis_menten_derivative(
                param_store.get(
                    ('cephaloridine', 'hydrolysis', 'kcat')
                ).to(1 / units.sec).magnitude,
                param_store.get(('cephaloridine', 'hydrolysis', 'km')).to(
                    units.mM).magnitude,
                param_store.get(('cephaloridine', 'hydrolysis', 'n')),
                param_store.get(('concs', 'initial_hydrolase')).to(
                    units.mM).magnitude,
            ),
            # Assume diffusion is fast.
            param_store.get(('cephaloridine', 'mic')).to(
                units.mM).magnitude,
        ),
        'Diffusion': (
            _get_fickian_derivative(
                (
                    param_store.get(('shape', 'initial_outer_area'))
                    * param_store.get(
                        ('cephaloridine', 'permeability', 'outer'))
                    ).to(units.L / units.sec).magnitude,
                param_store.get(('cephaloridine', 'mic')).to(
                    units.mM).magnitude,
                (
                    param_store.get(('shape', 'initial_cell_volume'))
                    * param_store.get(('shape', 'periplasm_fraction'))
                ).to(units.L).magnitude,
            ),
            0,
        ),
    },
    'ampicillin': {
        # Map from name to (derivative_func, y0)
        'Efflux': (
            _get_michaelis_menten_derivative(
                param_store.get(('ampicillin', 'efflux', 'kcat')).to(
                    1 / units.sec).magnitude,
                param_store.get(('ampicillin', 'efflux', 'km')).to(
                    units.mM).magnitude,
                param_store.get(('ampicillin', 'efflux', 'n')),
                param_store.get(('concs', 'initial_pump')).to(
                    units.mM).magnitude,
            ),
            # Assume diffusion is fast.
            param_store.get(('ampicillin', 'mic')).to(
                units.mM).magnitude,
        ),
        'Hydrolysis': (
            _get_michaelis_menten_derivative(
                param_store.get(
                    ('ampicillin', 'hydrolysis', 'kcat')
                ).to(1 / units.sec).magnitude,
                param_store.get(('ampicillin', 'hydrolysis', 'km')).to(
                    units.mM).magnitude,
                param_store.get(('ampicillin', 'hydrolysis', 'n')),
                param_store.get(('concs', 'initial_hydrolase')).to(
                    units.mM).magnitude,
            ),
            # Assume diffusion is fast.
            param_store.get(('ampicillin', 'mic')).to(
                units.mM).magnitude,
        ),
        'Diffusion': (
            _get_fickian_derivative(
                (
                    param_store.get(('shape', 'initial_outer_area'))
                    * param_store.get(
                        ('ampicillin', 'permeability', 'outer'))
                    ).to(units.L / units.sec).magnitude,
                param_store.get(('ampicillin', 'mic')).to(
                    units.mM).magnitude,
                (
                    param_store.get(('shape', 'initial_cell_volume'))
                    * param_store.get(('shape', 'periplasm_fraction'))
                ).to(units.L).magnitude,
            ),
            0,
        ),
    },
    'tetracycline': {
        'Efflux': (
            _get_michaelis_menten_derivative(
                param_store.get(('tetracycline', 'efflux', 'kcat')).to(
                    1 / units.sec).magnitude,
                param_store.get(('tetracycline', 'efflux', 'km')).to(
                    units.mM).magnitude,
                param_store.get(('tetracycline', 'efflux', 'n')).to(
                    units.count).magnitude,
                param_store.get(('concs', 'initial_pump')).to(
                    units.mM).magnitude,
            ),
            # Assume diffusion is fast.
            param_store.get(('tetracycline', 'mic')).to(
                units.mM).magnitude,
        ),
        'Outer Diffusion': (
            _get_fickian_derivative(
                (
                    param_store.get(('shape', 'initial_outer_area'))
                    * param_store.get((
                        'tetracycline', 'permeability',
                        'outer_with_porins'))
                ).to(units.L / units.sec).magnitude,
                param_store.get(('tetracycline', 'mic')).to(
                    units.mM).magnitude,
                param_store.get(
                    ('shape', 'initial_periplasm_volume')
                ).to(units.L).magnitude,
                np.exp((
                    param_store.get(('tetracycline', 'charge'))
                    * param_store.get(('faraday_constant',))
                    * param_store.get(('donnan_potential',))
                    / param_store.get(('gas_constant',))
                    / param_store.get(('temperature',))
                ).to(units.count).magnitude),
            ),
            0,
        ),
        'Inner Diffusion': (
            _get_fickian_derivative(
                (
                    param_store.get(('shape', 'initial_outer_area'))
                    * param_store.get((
                        'tetracycline', 'permeability', 'inner'))
                ).to(units.L / units.sec).magnitude,
                # Assume outer diffusion is fast.
                (
                    param_store.get(('tetracycline', 'mic'))
                    * np.exp((
                        param_store.get(('tetracycline', 'charge'))
                        * param_store.get(('faraday_constant',))
                        * param_store.get(('donnan_potential',))
                        / param_store.get(('gas_constant',))
                        / param_store.get(('temperature',))
                    ).to(units.count).magnitude)
                ).to(units.mM).magnitude,
                param_store.get(
                    ('shape', 'initial_cytoplasm_volume')
                ).to(units.L).magnitude,
            ),
            0,
        ),
    },
}


def test_cephaloridine_diffusion_timescale():
    derivative = ODES['cephaloridine']['Diffusion'][0]
    y0 = param_store.get(('cephaloridine', 'mic'))
    _assert_timescale(derivative, y0.magnitude, MAX_TIMESCALE)


def test_tetracycline_outer_diffusion_timescale():
    derivative = ODES['tetracycline']['Outer Diffusion'][0]
    _assert_timescale(derivative, 0, MAX_TIMESCALE)


def main():
    for duration in (0.1, 10, 1000):
        fig, axes = plt.subplots(
            nrows=len(ODES) * 2, figsize=(6.4, 4.8 * len(ODES)))
        t = np.linspace(0, duration, 100)
        for i, (plot_name, ode_definitions) in enumerate(ODES.items()):
            ax1 = axes[i * 2]
            ax2 = axes[i * 2 + 1]
            for label, (derivative_func, y0) in ode_definitions.items():
                y = odeint(derivative_func, y0, t)[:,0]
                ax1.plot(t, y, label=label, alpha=0.5)
                ax2.plot(
                    t, (y - y[0]) / (y[-1] - y[0]), label=label, alpha=0.5)
            ax1.legend()
            ax2.legend()
            ax1.set_ylabel('Amount')
            ax2.set_ylabel('Amount (Normalized)')
            ax1.set_xlabel('Time (s)')
            ax2.set_xlabel('Time (s)')
            ax1.set_title(plot_name)
            ax2.set_title(plot_name)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, f'timescales_{duration}.png'))


if __name__ == '__main__':
    main()

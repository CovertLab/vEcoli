from matplotlib import pyplot as plt
import numpy as np
import os
from scipy.integrate import odeint
from scipy.optimize import root_scalar, root

from scripts.generate_ceph_tet_sbml import (
    CEPH_PUMP_KCAT,
    CEPH_PUMP_KM,
    CEPH_BETA_LACTAMASE_KCAT,
    CEPH_BETA_LACTAMASE_KM,
    CEPH_EXPORT_N,
    DEFAULT_CEPH_OUTER_PERM,

    TET_PUMP_KCAT,
    TET_PUMP_KM,
    DEFAULT_TET_OUTER_PERM,
    DEFAULT_TET_INNER_PERM,
    MEMBRANE_POTENTIAL,
    TETRACYCLINE_CHARGE,
    FARADAY_CONSTANT,
    GAS_CONSTANT,
    TEMPERATURE,

    INITIAL_ENVIRONMENT_CEPH,
    INITIAL_ENVIRONMENT_TET,
    INITIAL_PUMP,
    INITIAL_BETA_LACTAMASE,

    AREA_MASS_RATIO,
    CYTO_AREA_MASS_RATIO,
    CELL_MASS,
    PERIPLASM_VOLUME,
    CYTOPLASM_VOLUME,
)


CHANGE_RELATIVE_THRESHOLD = 0.0001  # 0.01% change acceptable
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
                CEPH_PUMP_KCAT, CEPH_PUMP_KM, CEPH_EXPORT_N, INITIAL_PUMP),
            INITIAL_ENVIRONMENT_CEPH,  # Assume diffusion is fast.
        ),
        'Hydrolysis': (
            _get_michaelis_menten_derivative(
                CEPH_BETA_LACTAMASE_KCAT, CEPH_BETA_LACTAMASE_KM, 1,
                INITIAL_BETA_LACTAMASE),
            INITIAL_ENVIRONMENT_CEPH,    # Assume diffusion is fast.
        ),
        'Diffusion': (
            _get_fickian_derivative(
                (
                    CELL_MASS * AREA_MASS_RATIO * DEFAULT_CEPH_OUTER_PERM
                    * 1e-3
                ),
                INITIAL_ENVIRONMENT_CEPH,
                PERIPLASM_VOLUME,
            ),
            0,
        ),
    },
    'tetracycline': {
        'Efflux': (
            _get_michaelis_menten_derivative(
                TET_PUMP_KCAT, TET_PUMP_KM, 1, INITIAL_PUMP),
            INITIAL_ENVIRONMENT_TET,  # Assume diffusion is fast.
        ),
        'Outer Diffusion': (
            _get_fickian_derivative(
                (
                    CELL_MASS * AREA_MASS_RATIO
                    * DEFAULT_TET_OUTER_PERM * 1e-3
                ),
                INITIAL_ENVIRONMENT_TET,
                PERIPLASM_VOLUME,
                np.exp(
                    TETRACYCLINE_CHARGE * FARADAY_CONSTANT
                    * MEMBRANE_POTENTIAL / GAS_CONSTANT / TEMPERATURE),
            ),
            0,
        ),
        'Inner Diffusion': (
            _get_fickian_derivative(
                (
                    CELL_MASS * CYTO_AREA_MASS_RATIO
                    * DEFAULT_TET_INNER_PERM * 1e-3
                ),
                INITIAL_ENVIRONMENT_TET,  # Assume outer diffusion fast.
                CYTOPLASM_VOLUME,
            ),
            0,
        ),
    },
}


def cephaloridine_periplasm_derivative(y):
    diffusion = ODES['cephaloridine']['Diffusion'][0]
    efflux = ODES['cephaloridine']['Efflux'][0]
    hydrolysis = ODES['cephaloridine']['Hydrolysis'][0]
    return y + diffusion(y, 0) - efflux(y, 0) - hydrolysis(y, 0)


def tetracycline_derivative(y):
    periplasmic, cytoplasmic = y
    inner_diffusion = ODES['tetracycline']['Inner Diffusion'][0]
    outer_diffusion = ODES['tetracycline']['Outer Diffusion'][0]
    efflux = ODES['tetracycline']['Efflux'][0]

    periplasmic_derivative = (
        outer_diffusion(periplasmic, 0) - efflux(periplasmic, 0)
        - inner_diffusion(cytoplasmic, 0)
    )
    cytoplasmic_derivative = inner_diffusion(cytoplasmic, 0)
    return periplasmic_derivative, cytoplasmic_derivative


def find_equilibrium():
    ceph = root_scalar(
        cephaloridine_periplasm_derivative,
        bracket=[0, INITIAL_ENVIRONMENT_CEPH])
    assert ceph.converged
    print('Periplasmic Cephaloridine Equilibrium (mM):', ceph.root)

    tet = root(
        tetracycline_derivative,
        [INITIAL_ENVIRONMENT_TET, INITIAL_ENVIRONMENT_TET])
    assert tet.success
    print('Equilibrium (periplasmic, cytoplasmic) in mM:', tet.x)


def test_cephaloridine_diffusion_timescale():
    derivative = ODES['cephaloridine']['Diffusion'][0]
    y0 = INITIAL_ENVIRONMENT_CEPH
    _assert_timescale(derivative, y0, MAX_TIMESCALE)


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
    #find_equilibrium()

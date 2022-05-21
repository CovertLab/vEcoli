from scipy import constants

from vivarium.library.topology import get_in, assoc_path
from vivarium.library.units import units

class Parameter:

    def __init__(self, value, source='', canonicalize=None, note=''):
        self.value = value
        self.source = source
        self.canonicalize = canonicalize or (lambda x: x)
        self.note = note


class ParameterStore:

    def __init__(self, parameters, derivation_rules=None):
        self._parameters = parameters
        self.derive_parameters(derivation_rules or {})

    def get(self, path):
        param_obj = get_in(self._parameters, path)
        if not param_obj:
            raise RuntimeError(
                f'No parameter found at path {path}')
        return param_obj.canonicalize(param_obj.value)

    def add(self, path, parameter):
        assert get_in(self._parameters, path) is None
        assoc_path(self._parameters, path, parameter)

    def derive_parameters(self, derivation_rules):
        for path, deriver in derivation_rules.items():
            new_param = deriver(self)
            self.add(path, new_param)


PARAMETER_DICT = {
    'ampicillin': {
        'permeability': {
            'outer': Parameter(
                0.28e-5 * units.cm / units.sec,
                'Kojima and Nikaido (2013)',
                note='This is total, not per-porin, permeability.',
            ),
        },
        'mic': Parameter(
            2 * units.micrograms / units.mL,
            'Mazzariol, Cornaglia, and Nikaido (2000)',
            lambda x: (
                # Divide by molecular weight from PubChem.
                x / (349.4 * units.g / units.mol)
            ).to(units.mM),
        ),
        'efflux': {
            'vmax': Parameter(
                0.085 * units.nmol / units.mg / units.sec,
                'Kojima and Nikaido (2013)',
            ),
            'km': Parameter(
                2.16e-3 * units.mM,
                'Kojima and Nikaido (2013)',
            ),
            'n': Parameter(
                1.9 * units.count,
                'Kojima and Nikaido (2013)',
            ),
        },
        'hydrolysis': {
            'kcat': Parameter(
                6.5 / units.sec,
                'Mazzariol, Cornaglia, and Nikaido (2000)',
            ),
            'km': Parameter(
                0.9e-3 * units.mM,
                'Mazzariol, Cornaglia, and Nikaido (2000)',
            ),
            'n': Parameter(
                1 * units.count,
                'Mazzariol, Cornaglia, and Nikaido (2000)',
            ),
        },
    },
    'cephaloridine': {
        'permeability': {
            'outer': Parameter(
                (52.6e-5 + 4.5e-5) * units.cm / units.sec,
                'Nikaido, Rosenberg, and Foulds (1983)',
                note='This is total, not per-porin, permeability',
            ),
        },
        # Cell-wide permeability with only one porin present.
        'porin_specific_permeability': {
            'outer': {
                'ompf': Parameter(
                    52.6e-5 * units.cm / units.sec,
                    'Nikaido, Rosenberg, and Foulds (1983)',
                ),
                'ompc': Parameter(
                    4.5e-5 * units.cm / units.sec,
                    'Nikaido, Rosenberg, and Foulds (1983)',
                ),
            },
        },
        'mic': Parameter(
            0.5 * units.micrograms / units.mL,
            'Rolinson (1980)',
            lambda x: (
                # Divide by molecular weight from PubChem.
                x / (415.5 * units.g / units.mol)
            ).to(units.mM),
        ),
        'efflux': {
            'vmax': Parameter(
                1.82 * units.nmol / units.mg / units.sec,
                'Nagano and Nikaido (2009)',
            ),
            'km': Parameter(
                0.288 * units.mM,
                'Nagano and Nikaido (2009)',
            ),
            'n': Parameter(
                1.75 * units.count,
                'Nagano and Nikaido (2009)',
            ),
        },
        'hydrolysis': {
            'kcat': Parameter(
                130 / units.sec,
                'Galleni et al. (1988)',
                note='Not confirmed',
            ),
            'km': Parameter(
                0.17 * units.mM,
                'Galleni et al. (1988)',
                note='Not confirmed',
            ),
            'n': Parameter(
                1 * units.count
            ),
        },
    },
    'tetracycline': {
        'permeability': {
            'outer_without_porins': Parameter(
                0.7e-7 * units.cm / units.sec,
                'Thananassi, Suh, and Nikaido (1995) p. 1004',
            ),
            'outer_with_porins': Parameter(
                1e-5 * units.cm / units.sec,
                'Thananassi, Suh, and Nikaido (1995) p. 1005',
            ),
            'inner': Parameter(
                3e-6 * units.cm / units.sec,
                'Thananassi, Suh, and Nikaido (1995) p. 1004',
            ),
        },
        'potential': {
            'membrane_potential': Parameter(
                0.12 * units.volt,
                'Berg, Howard C., E. coli in Mtion. 1934. Page 105',
            ),
            'tetracycline_charge': Parameter(
                1,
            ),
            'faraday_constant': Parameter(
                constants.value(
                    'Faraday constant') * units.C / units.mol
            ),
            'gas_constant': Parameter(
                constants.R * units.J / units.mol / units.K,
            ),
            'temperature': Parameter(
                298 * units.K,
            ),
        },
        'efflux': {
            'vmax': Parameter(
                0.2 * units.nmol / units.mg / units.min,
                'Thananassi, Suh, and Nikaido (1995) p. 1004',
            ),
            'km': Parameter(
                200 * units.uM,
                'Thananassi, Suh, and Nikaido (1995) p. 1004',
            ),
            'n': Parameter(
                1 * units.count,
            ),
        },
    },
    'shape': {
        'periplasm_fraction': Parameter(
            0.2,
            'Stock et al. (1977)',
        ),
        'initial_cell_mass': Parameter(
            1170 * units.fg,
            'Model',
        ),
        'initial_cell_volume': Parameter(
            1.2 * units.fL,
            'Model',
        ),
        'initial_area': Parameter(
            4.52 * units.um**2,
            'Model',
        ),
    },
    'concs': {
        'initial_pump': Parameter(
            6.7e-4 * units.mM,
            'Simulation c33d8283af0bed4a6a598774ac5d8aec19d169bf',
        ),
        'initial_hydrolase': Parameter(
            7.1e-4 * units.mM,
            'Simulation c33d8283af0bed4a6a598774ac5d8aec19d169bf',
        ),
    },
    'counts': {
        'initial_ompf': Parameter(
            18975 * units.count,
            'Simulation c33d8283af0bed4a6a598774ac5d8aec19d169bf',
        ),
        'initial_ompc': Parameter(
            5810 * units.count,
            'Simulation c33d8283af0bed4a6a598774ac5d8aec19d169bf',
        ),
    },
    'avogadro': constants.N_A / units.mol,
}

DERIVATION_RULES = {
    ('shape', 'initial_periplasm_volume'): lambda params: Parameter(
        (
            params.get(('shape', 'initial_cell_volume'))
            * params.get(('shape', 'periplasm_fraction'))
        ),
    ),
    ('ampicillin', 'efflux', 'kcat'): lambda params: Parameter(
        (
            params.get(('ampicillin', 'efflux', 'vmax'))
            / params.get(('concs', 'initial_pump'))
            * params.get(('shape', 'initial_cell_mass'))
            / params.get(('shape', 'initial_periplasm_volume'))
        )
    ),
    (
        'ampicillin', 'per_porin_permeability', 'outer', 'ompf'
    ): lambda params: Parameter(
        (
            params.get(('ampicillin', 'permeability', 'outer')) / (
                params.get(('counts', 'initial_ompf'))
                / params.get(('shape', 'initial_area'))
            )
        ),
    ),
    ('cephaloridine', 'efflux', 'kcat'): lambda params: Parameter(
        (
            params.get(('cephaloridine', 'efflux', 'vmax'))
            / params.get(('concs', 'initial_pump'))
            * params.get(('shape', 'initial_cell_mass'))
            / params.get(('shape', 'initial_periplasm_volume'))
        )
    ),
    (
        'cephaloridine', 'per_porin_permeability', 'outer', 'ompf'
    ): lambda params: Parameter(
        (
            params.get((
                'cephaloridine', 'porin_specific_permeability', 'outer',
                'ompf'
            )) / (
                params.get(('counts', 'initial_ompf'))
                / params.get(('shape', 'initial_area'))
            )
        ),
    ),
    (
        'cephaloridine', 'per_porin_permeability', 'outer', 'ompc'
    ): lambda params: Parameter(
        (
            params.get((
                'cephaloridine', 'porin_specific_permeability', 'outer',
                'ompc'
            )) / (
                params.get(('counts', 'initial_ompc'))
                / params.get(('shape', 'initial_area'))
            )
        ),
    ),
    ('tetracycline', 'efflux', 'kcat'): lambda params: Parameter(
        (
            params.get(('tetracycline', 'efflux', 'vmax'))
            / params.get(('concs', 'initial_pump'))
            * params.get(('shape', 'initial_cell_mass'))
            / params.get(('shape', 'initial_periplasm_volume'))
        )
    ),
    ('tetracycline', 'permeability', 'outer', 'ompf'): lambda params: Parameter(
        (
            params.get((
                'tetracycline', 'permeability', 'outer_with_porins'))
            - params.get((
                'tetracycline', 'permeability', 'outer_without_porins'))
        ),
    ),
    (
        'tetracycline', 'per_porin_permeability', 'outer', 'ompf'
    ): lambda params: Parameter(
        (
            params.get((
                'tetracycline', 'permeability', 'outer', 'ompf'
            )) / (
                params.get(('counts', 'initial_ompf'))
                / params.get(('shape', 'initial_area'))
            )
        ),
    ),
}

param_store = ParameterStore(PARAMETER_DICT, DERIVATION_RULES)

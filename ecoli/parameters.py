from vivarium.library.topology import get_in
from vivarium.library.units import units

class Parameter:

    def __init__(self, value, source, canonicalize=None, note=''):
        self.value = value
        self.source = source
        self.canonicalize = canonicalize or (lambda x: x)
        self.note = note


class ParameterStore:

    def __init__(self, parameters):
        self.parameters = parameters

    def get(self, path):
        param_obj = get_in(self.parameters, path)
        if not param_obj:
            raise RuntimeError(
                f'No parameter found at path {path}')
        return param_obj.canonicalize(param_obj.value)


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
                x / (349.4 * units.g / units.mol)
            ).to(units.mM),
        ),
        'efflux': {
            'kcat': Parameter(
                0.085 * units.nmol / units.mg / units.sec,  # Vmax
                'Kojima and Nikaido (2013)',
                lambda x: (
                    x / (6.7e-4 * units.mM)  # Pump concentration
                    * 1170 * units.fg  # Total cell mass
                    / (0.212799 * units.fL)  # Periplasm volume
                )
            ),
            'km': Parameter(
                2.16e-3 * units.mM,
                'Kojima and Nikaido (2013)',
            ),
            'n': Parameter(
                1.9,
                'Kojima and Nikaido (2013)',
            ),
        },
        'hydrolysis': {
            'kcat': Parameter(
                6.5e-3 / units.sec,
                'Mazzariol, Cornaglia, and Nikaido (2000)',
            ),
            'km': Parameter(
                0.9e-3 * units.mM,
                'Mazzariol, Cornaglia, and Nikaido (2000)',
            ),
            'n': Parameter(
                1,
                'Mazzariol, Cornaglia, and Nikaido (2000)',
            ),
        },
    },
    'shape': {
        'periplasm_fraction': Parameter(
            0.2,
            'Stock et al. (1977)'
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
            'Model',
        ),
        'initial_hydrolase': Parameter(
            7.1e-4 * units.mM,
            'Model',
        ),
    },
}

param_store = ParameterStore(PARAMETER_DICT)

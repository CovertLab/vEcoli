import copy

import numpy as np
from scipy import constants

from vivarium.library.units import units
from vivarium.core.process import Step

AVOGADRO = constants.N_A * 1 / units.mol


class NonSpatialEnvironment(Step):
    '''A non-spatial environment with volume'''

    name = 'nonspatial_environment'
    defaults = {
        'internal_volume': 1 * units.fL,
        'env_volume': 1 * units.fL,
        'concentrations': {},
    }

    def ports_schema(self):
        bin_x = 1 * units.um
        bin_y = 1 * units.um
        depth = self.parameters['env_volume'] / bin_x / bin_y
        n_bin_x = 1
        n_bin_y = 1
        schema = {
            'external': {
                '*': {
                    '_value': 0,
                },
            },
            'exchanges': {
                '*': {
                    '_value': 0,
                },
            },
            'fields': {
                '*': {
                    '_default': np.ones((1, 1)),
                },
            },
            'dimensions': {
                'depth': {
                    '_value': depth.to(units.um)
                },
                'n_bins': {
                    '_value': [n_bin_x, n_bin_y],
                },
                'bounds': {
                    '_value': [
                        n_bin_x * bin_x.to(units.um),
                        n_bin_y * bin_y.to(units.um),
                    ],
                },
            },
            'global': {
                'location': {
                    '_value': [0.5, 0.5],
                },
                'volume': {
                    '_value': self.parameters['internal_volume'],
                },
                'mmol_to_counts': {
                    '_value': (
                        AVOGADRO * self.parameters['internal_volume']
                    ).to('L/mmol')
                },
            },
        }
        # add field concentrations
        field_schema = {}
        for field_id, conc in self.parameters['concentrations'].items():
            if not isinstance(self.parameters['concentrations']['cephaloridine'], float):
                conc = conc.magnitude
            field_schema[field_id] = {
                '_value': np.array([[conc]])
            }
        return schema

    def next_update(self, timestep, states):
        fields = states['fields']
        new_fields = copy.deepcopy(fields)

        exchanges = states['exchanges']
        for molecule, exchange in exchanges.items():
            conc_delta = (
                exchange / AVOGADRO / self.parameters['env_volume'])
            new_fields[molecule][0, 0] += conc_delta.to(
                units.millimolar).magnitude

        update = {
            'external': {
                mol_id: {
                    '_updater': 'set',
                    '_value': field[0][0],
                }
                for mol_id, field in new_fields.items()
            },
            'fields': {
                molecule: {
                    '_updater': 'set',
                    '_value': new_field,
                }
                for molecule, new_field in new_fields.items()
            },
            'exchanges': {
                molecule: {
                    '_updater': 'accumulate',
                    '_value': -exchange,
                }
                for molecule, exchange in exchanges.items()
            },
        }

        return update

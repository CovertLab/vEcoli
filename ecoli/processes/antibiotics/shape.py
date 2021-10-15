import copy

from scipy.constants import N_A

from vivarium.core.process import Deriver
from vivarium.library.units import units


AVOGADRO = N_A / units.mol


class ShapeDeriver(Deriver):
    name = 'shape_deriver'
    defaults = {
        'periplasm_fraction': 0.3,
        # https://bionumbers.hms.harvard.edu/bionumber.aspx?&id=103904&ver=17
        'water_fraction': 0.7,
        'initial_state': {
            'cell_global': {
                'volume': 1.2 * units.fL,
                'mass': 1339 * units.fg,
            },
        },
    }

    def ports_schema(self):
        schema = {
            'cell_global': {
                'volume': {
                    '_default': 0 * units.fL,
                    '_emit': True,
                    '_divider': 'split',
                },
                'mass': {
                    '_default': 0 * units.fg,
                    '_emit': True,
                    '_divider': 'split',
                },
                'dry_mass': {
                    '_default': 0 * units.fg,
                    '_emit': True,
                    '_divider': 'split',
                },
                'mmol_to_counts': {
                    '_default': 0 * units.L / units.mmol,
                    '_divider': 'split',
                },
            },
            'periplasm_global': {
                'volume': {
                    '_default': 0 * units.fL,
                    '_emit': True,
                    '_divider': 'split',
                },
                'mmol_to_counts': {
                    '_default': 0 * units.L / units.mmol,
                    '_divider': 'split',
                },
            },
        }
        return schema

    def _calculate_periplasm_volume(self, state):
        cell_volume = state['cell_global']['volume']
        periplasm_volume = cell_volume * self.parameters[
            'periplasm_fraction']
        return periplasm_volume

    @staticmethod
    def _calculate_mmol_to_counts(volume):
        mmol_to_counts = volume * AVOGADRO
        return mmol_to_counts

    def _calculate_dry_mass(self, state):
        mass = state['cell_global']['mass']
        dry_mass = mass * (1 - self.parameters[
            'water_fraction'])
        return dry_mass

    def initial_state(self, config=None):
        initial_state = copy.deepcopy(self.parameters['initial_state'])
        periplasm_volume = self._calculate_periplasm_volume(
            initial_state)
        dry_mass = self._calculate_dry_mass(initial_state)
        initial_state['periplasm_global'] = {
            'volume': periplasm_volume,
            'mmol_to_counts': self._calculate_mmol_to_counts(
                periplasm_volume)
        }
        initial_state['cell_global']['dry_mass'] = dry_mass
        return initial_state

    def next_update(self, _, states):
        periplasm_volume = self._calculate_periplasm_volume(states)
        dry_mass = self._calculate_dry_mass(states)
        update = {
            'periplasm_global': {
                'volume': {
                    '_value': periplasm_volume,
                    '_updater': 'set',
                },
                'mmol_to_counts': {
                    '_value': self._calculate_mmol_to_counts(
                        periplasm_volume),
                    '_updater': 'set',
                },
            },
            'cell_global': {
                'dry_mass': {
                    '_value': dry_mass,
                    '_updater': 'set',
                }
            }
        }
        return update

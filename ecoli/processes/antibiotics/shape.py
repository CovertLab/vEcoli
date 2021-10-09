import copy

from vivarium.core.process import Deriver
from vivarium.library.units import units


class ShapeDeriver(Deriver):
    name = 'shape_deriver'
    defaults = {
        'periplasm_fraction': 0.3,
        'initial_state': {
            'cell_global': {
                'volume': 1.2 * units.fL,
            },
        },
    }

    def ports_schema(self):
        schema = {
            'cell_global': {
                'volume': {
                    '_default': 0 * units.fL,
                },
            },
            'periplasm_global': {
                'volume': {
                    '_default': 0 * units.fL,
                },
            },
        }
        return schema

    def _calculate_periplasm_volume(self, state):
        cell_volume = state['cell_global']['volume']
        periplasm_volume = cell_volume * self.parameters[
            'periplasm_fraction']
        return periplasm_volume


    def initial_state(self, config=None):
        initial_state = copy.deepcopy(self.parameters['initial_state'])
        periplasm_volume = self._calculate_periplasm_volume(
            initial_state)
        initial_state['periplasm_global'] = {
            'volume': periplasm_volume,
        }
        return initial_state

    def next_update(self, _, states):
        periplasm_volume = self._calculate_periplasm_volume(states)
        update = {
            'periplasm_global': {
                'volume': {
                    '_value': periplasm_volume,
                    '_updater': 'set',
                },
            },
        }
        return update

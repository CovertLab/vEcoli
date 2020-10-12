"""
====
Mass
====
"""

from scipy import constants

from vivarium.core.process import Deriver
from ecoli.library.schema import mw_schema
# from wholecell.utils import units


AVOGADRO = constants.N_A  #* 1 / units.mol


def calculate_mass(value, path, node):
    '''
    Reducer for summing masses in hierarchy
    '''
    if 'mw' in node.properties:
        count = node.value
        mw = node.properties['mw']
        added_mass = mass_from_count(count, mw)
        return value + added_mass
    else:
        return value

def mass_from_count(count, mw):
    mol = count / AVOGADRO
    return mw * mol


class Mass(Deriver):
    name = 'mass-ecoli'
    defaults = {
        'molecular_weights': {},
        'cellDensity': 1100.0,
        'bulk_path': ('..', '..', '..', 'bulk'),
        'water_key': 'WATER[c]'
    }

    def __init__(self, initial_parameters=None):
        super(Mass, self).__init__(initial_parameters)
        self.molecular_weights = self.parameters['molecular_weights']
        self.bulk_path = self.parameters['bulk_path']
        self.water_key = self.parameters.get('water_key')
        self.water_mw = self.parameters['molecular_weights'].get(self.water_key, 18.015)
        # TODO -- molecular_weights to units.g / units.mol

    def ports_schema(self):
        return {
            'bulk': mw_schema(self.molecular_weights),
            'listeners': {
                'mass': {
                    'cell_mass': {
                        '_default': 0.0,
                        '_emit': True},
                    'dry_mass': {
                        '_default': 0.0,
                        '_updater': 'set',
                        '_emit': True}
                }
            }
        }

    def next_update(self, timestep, states):
        # TODO -- dry_mass will be 1 ts delayed from cell_mass
        cell_mass = states['listeners']['mass']['cell_mass']
        bulk = states['bulk']
        water_count = bulk.get(self.water_key, 0.0)
        water_mass = mass_from_count(water_count, self.water_mw) * 1e+15  # convert g to fg
        dry_mass = cell_mass - water_mass

        return {
            'listeners': {
                'mass': {
                    'cell_mass': {
                        '_reduce': {
                            'reducer': calculate_mass,
                            'from': self.bulk_path,
                            'initial': 0.0
                        }
                    },
                    'dry_mass': dry_mass
                },
            }
        }

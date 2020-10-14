"""
====
Mass
====
"""

import os

from scipy import constants

from vivarium.core.process import Deriver
from ecoli.library.schema import mw_schema
# from wholecell.utils import units

from vivarium.core.experiment import pp
from vivarium.core.composition import process_in_experiment
from vivarium.library.units import units


AVOGADRO = constants.N_A * 1 / units.mol


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
    name = 'mass'
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


# test functions
def test_mass():

    water_mw = 1.0 * units.g / units.mol
    biomass_mw = 1.0 * units.g / units.mol

    # declare schema override to get mw properties
    parameters = {
        'initial_mass': 0 * units.g,  # in grams
        '_schema': {
            'bulk': {
                'water': {
                    '_emit': True,
                    '_properties': {'mw': water_mw}},
                'biomass': {
                    '_emit': True,
                    '_properties': {'mw': biomass_mw}},
            },
        }
    }
    mass_process = Mass(parameters)

    # declare initial state
    state = {
        'bulk': {
            'water': 0.7 * AVOGADRO.magnitude,
            'biomass': 0.3 * AVOGADRO.magnitude,
        },
        'global': {
            'initial_mass': 0.0,
            'mass': 0.0,
        }
    }

    # make the experiment with initial state
    settings = {'initial_state': state}
    experiment = process_in_experiment(mass_process, settings)

    # run experiment and get output
    experiment.update(1)
    output = experiment.emitter.get_data()
    experiment.end()

    # assert output[0.0]['global']['mass']  == 4
    return output


def run_mass():
    output = test_mass()
    pp(output)


if __name__ == '__main__':
    run_mass()

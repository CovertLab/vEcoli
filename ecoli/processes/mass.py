"""
====
Mass
====
"""

import os

import numpy as np
from scipy import constants

from vivarium.core.process import Deriver
from ecoli.library.schema import mw_schema
# from wholecell.utils import units

from vivarium.core.engine import pp
from vivarium.core.composition import process_in_experiment
from vivarium.library.units import units

AVOGADRO = constants.N_A #* 1 / units.mol

def mass_from_counts_array(counts, mw):
    return np.array([mass_from_count(count, mw) for count in counts])

def mass_from_count(count, mw):
    mol = count / AVOGADRO
    return mw * mol


class Mass(Deriver):
    name = 'ecoli-mass'
    defaults = {
        'molecular_weights': {},
        'unique_masses': {},
        'cellDensity': 1100.0,
        'water_id': 'water'
    }

    def __init__(self, parameters=None):
        super().__init__(parameters)
        self.molecular_weights = self.parameters['molecular_weights']
        self.unique_masses = self.parameters['unique_masses']

    def ports_schema(self):
        return {
            'bulk': {
                mol_id: {'_default': 0}
                for mol_id in self.molecular_weights.keys()
            },
            'unique': {
                mol_id: {'*': {}}
                for mol_id in self.unique_masses.keys()
            },
            'listeners': {
                'mass': {
                    'cell_mass': {
                        '_default': 0.0,
                        '_updater': 'set',
                        '_emit': True},
                    'water_mass': {
                        '_default': 0.0,
                        '_updater': 'set',
                        '_emit': True},
                    'dry_mass': {
                        '_default': 0.0,
                        '_updater': 'set',
                        '_emit': True},
                }
            }
        }

    def get_bulk_mass(self, molecule_id, state):
        count = state['bulk'][molecule_id]
        return mass_from_count(count, self.molecular_weights.get(molecule_id))

    def next_update(self, timestep, states):

        # TODO -- run these with reducers, to avoid deepcopy of states

        # calculate bulk molecule mass
        bulk_mass = 0.0
        for molecule_id, count in states['bulk'].items():
            if count > 0:
                added_mass = mass_from_count(count, self.molecular_weights.get(molecule_id))
                bulk_mass += added_mass

        # calculate unique molecule mass
        unique_masses = np.array([])
        for molecule_id, molecules in states['unique'].items():
            n_molecules = len(molecules)
            if unique_masses.any():
                unique_masses += self.unique_masses[molecule_id] * n_molecules
            else:
                unique_masses = self.unique_masses[molecule_id] * n_molecules
        unique_mass = np.sum(unique_masses)

        # calculate masses
        cell_mass = bulk_mass + unique_mass
        water_mass = self.get_bulk_mass(self.parameters['water_id'], states)
        dry_mass = cell_mass - water_mass

        return {
            'listeners': {
                'mass': {
                    'cell_mass': cell_mass,
                    'water_mass': water_mass,
                    'dry_mass': dry_mass,
                },
            }
        }


# test functions
def test_mass():

    water_mw = 1.0  #* units.g / units.mol
    biomass_mw = 1.0  #* units.g / units.mol

    # declare schema override to get mw properties
    parameters = {
        'initial_mass': 0 * units.g,  # in grams
        'molecular_weights': {
            'water': water_mw,
            'biomass': biomass_mw,
        }
    }
    mass_process = Mass(parameters)

    # declare initial state
    state = {
        'bulk': {
            'water': 0.7 * AVOGADRO,  # .magnitude,
            'biomass': 0.3 * AVOGADRO,  # .magnitude,
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

    assert output[0.0]['listeners']['mass']['cell_mass'] == 1.0
    return output


def run_mass():
    output = test_mass()
    pp(output)


if __name__ == '__main__':
    run_mass()

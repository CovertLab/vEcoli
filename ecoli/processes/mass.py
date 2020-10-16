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

from vivarium.core.experiment import pp
from vivarium.core.composition import process_in_experiment
from vivarium.library.units import units


AVOGADRO = constants.N_A #* 1 / units.mol


def calculate_mass(value, path, node):
    '''
    Reducer for summing masses in hierarchy
    '''
    if 'mw' in node.properties:
        count = node.value
        mw = node.properties['mw']
        node_mass = mass_from_count(count, mw)
        return value + node_mass
    else:
        return value

def mass_from_count(count, mw):
    mol = count / AVOGADRO
    return mw * mol


class Mass(Deriver):
    name = 'mass'
    defaults = {
        'molecular_weights': {},
        'unique_masses': {},
        'cellDensity': 1100.0,
        'bulk_path': ('..', '..', '..', 'bulk'),
        'water_path': ('..', '..', '..', 'bulk', 'water')
    }

    def __init__(self, initial_parameters=None):
        super(Mass, self).__init__(initial_parameters)
        self.molecular_weights = self.parameters['molecular_weights']
        self.bulk_path = self.parameters['bulk_path']
        self.water_path = self.parameters['water_path']
        self.unique_masses = self.parameters['unique_masses']

    def ports_schema(self):
        return {
            'bulk': mw_schema(self.molecular_weights),
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
                    'unique_mass': {
                        '_default': 0.0,
                        '_updater': 'set',
                        '_emit': True}
                }
            }
        }

    def next_update(self, timestep, states):
        cell_mass = states['listeners']['mass']['cell_mass']
        water_mass = states['listeners']['mass']['water_mass']
        dry_mass = cell_mass - water_mass  # dry mass will be 1 ts delayed from cell and water mass


        # TODO -- get bulk mass in a loop
        # import ipdb; ipdb.set_trace()

        # calculate unique molecule mass
        unique_masses = np.array([])
        for molecule_id, molecules in states['unique'].items():
            n_molecules = len(molecules)
            if unique_masses.any():
                unique_masses += self.unique_masses[molecule_id] * n_molecules
            else:
                unique_masses = self.unique_masses[molecule_id] * n_molecules
        unique_mass = np.sum(unique_masses)



        # self.waterMass = all_submasses[self.waterIndex]
        # self.dryMass = self.cellMass - self.waterMass
        # self.rnaMass = all_submasses[self.rnaIndexes].sum()
        # self.rRnaMass = all_submasses[self.rRnaIndex]
        # self.tRnaMass = all_submasses[self.tRnaIndex]
        # self.mRnaMass = all_submasses[self.mRnaIndex]
        # self.dnaMass = all_submasses[self.dnaIndex]
        # self.proteinMass = all_submasses[self.proteinIndex]
        # self.smallMoleculeMass = all_submasses[self.smallMoleculeIndex]

        # self.instantaniousGrowthRate = self.growth / self.timeStepSec() / self.dryMass
        # self.proteinMassFraction = self.proteinMass / self.dryMass
        # self.rnaMassFraction = self.rnaMass / self.dryMass
        #
        # if not self.setInitial:
        #     self.setInitial = True
        #
        #     self.timeInitial = self.time()
        #
        #     self.dryMassInitial = self.dryMass
        #     self.proteinMassInitial = self.proteinMass
        #     self.rnaMassInitial = self.rnaMass
        #     self.smallMoleculeMassInitial = self.smallMoleculeMass
        #
        # self.dryMassFoldChange = self.dryMass / self.dryMassInitial
        # self.proteinMassFoldChange = self.proteinMass / self.proteinMassInitial
        # self.rnaMassFoldChange = self.rnaMass / self.rnaMassInitial
        # self.smallMoleculeFoldChange = self.smallMoleculeMass / self.smallMoleculeMassInitial
        #
        # self.expectedMassFoldChange = np.exp(np.log(2) * (self.time() - self.timeInitial) / self.cellCycleLen)

        return {
            'listeners': {
                'mass': {
                    'cell_mass': {
                        '_reduce': {
                            'reducer': calculate_mass,
                            'from': self.bulk_path,
                            'initial': 0.0,
                        }
                    },
                    'water_mass': {
                        '_reduce': {
                            'reducer': calculate_mass,
                            'from': self.water_path,
                            'initial': 0.0,
                        }
                    },
                    'dry_mass': dry_mass,
                    'unique_mass': unique_mass,
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

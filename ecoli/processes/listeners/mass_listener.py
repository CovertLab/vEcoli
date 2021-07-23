"""
Mass

Mass listener. Represents the total cellular mass.
"""

import numpy as np
from vivarium.core.process import Deriver
from vivarium.library.units import units
from ecoli.library.schema import bulk_schema, array_from

from vivarium.core.engine import pp


class MassListener(Deriver):
    """ MassListener """

    name = 'ecoli-mass-listener'

    defaults = {
        'cellDensity': 1100.0,
        'bulk_ids': [],
        'bulk_masses': np.zeros([1, 9]),
        'unique_ids': [],
        'unique_masses': np.zeros([1, 9]),
        'submass_indices': {
            'rna': [],
            'rRna': [],
            'tRna': [],
            'mRna': [],
            'dna': [],
            'protein': [],
            'smallMolecule': [],
            'water' : -1,
        },
        'n_avogadro': 6.0221409e23,  # 1/mol
        'time_step' : 2.0
    }

    def __init__(self, parameters=None):
        super().__init__(parameters)

        # molecule indexes and masses
        self.bulk_ids = self.parameters['bulk_ids']
        self.bulk_masses = self.parameters['bulk_masses']
        self.unique_ids = self.parameters['unique_ids']
        self.unique_masses = self.parameters['unique_masses']

        self.submass_indices = self.parameters['submass_indices']
        self.waterIndex = self.parameters['submass_indices']['water']

        # compartment indexes
        # self.sim_data.internal_state.compartments ?
        # self.projection_index = sim_data.compartment_id_to_index["CCO-CELL-PROJECTION"]
        # self.cytosol_index = sim_data.compartment_id_to_index["CCO-CYTOSOL"]
        # self.extracellular_index = sim_data.compartment_id_to_index["CCO-EXTRACELLULAR"]
        # self.flagellum_index = sim_data.compartment_id_to_index["CCO-FLAGELLUM"]
        # self.membrane_index = sim_data.compartment_id_to_index["CCO-MEMBRANE"]
        # self.outer_membrane_index = sim_data.compartment_id_to_index["CCO-OUTER-MEM"]
        # self.periplasm_index = sim_data.compartment_id_to_index["CCO-PERI-BAC"]
        # self.pilus_index = sim_data.compartment_id_to_index["CCO-PILUS"]
        # self.inner_membrane_index = sim_data.compartment_id_to_index["CCO-PM-BAC-NEG"]

        # units and constants
        self.cellDensity = self.parameters['cellDensity']
        self.n_avogadro = self.parameters['n_avogadro']

        self.time_step = self.parameters['time_step']
        self.first_time_step = True

    def ports_schema(self):
        return {
            'bulk': bulk_schema(self.bulk_ids),
            'unique': {
                mol_id: {'*': {}}
                for mol_id in self.unique_ids
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
        

    def next_update(self, timestep, states):
        # Initialize update with 0's for each submass
        mass_update = {key + "Mass": 0 for key in self.submass_indices}

        # Get previous dry mass, for calculating growth later
        old_dry_mass = states['listeners']['mass']['dry_mass']

        # get submasses from bulk and unique
        bulk_counts = np.array([states['bulk'][id] for id in self.bulk_ids])
        bulk_submasses = np.dot(bulk_counts, self.bulk_masses)

        unique_counts = np.array([len(states['unique'][id])
                                 for id in self.unique_ids])
        unique_submasses = np.dot(unique_counts, self.unique_masses)

        all_submasses = bulk_submasses + unique_submasses

        # Store cell mass, water mass, dry mass
        mass_update['cell_mass'] = all_submasses.sum()
        mass_update['water_mass'] = all_submasses[self.waterIndex]
        mass_update['dry_mass'] = (
            mass_update['cell_mass'] - mass_update['water_mass'])

        # Store submasses
        for submass, indices in self.submass_indices.items():
            if submass != 'water':
                mass_update[submass + "Mass"] = all_submasses[indices].sum()

        mass_update['volume'] = mass_update['cell_mass'] / self.cellDensity

        mass_update['proteinMassFraction'] = (
            mass_update['proteinMass'] / mass_update['dry_mass'])
        mass_update['rnaMassFraction'] = (
            mass_update['rnaMass'] / mass_update['dry_mass'])

        if self.first_time_step:
            mass_update['growth'] = np.nan

            # TODO: How to get time?
            # self.timeInitial = self.time()

            self.dryMassInitial = mass_update['dry_mass']
            self.proteinMassInitial = mass_update['proteinMass']
            self.rnaMassInitial = mass_update['rnaMass']
            self.smallMoleculeMassInitial = mass_update['smallMoleculeMass']
        else:
            mass_update['growth'] = mass_update['dry_mass'] - old_dry_mass

            mass_update['instantaniousGrowthRate'] = (
                mass_update['growth'] / self.time_step / mass_update['dry_mass'])

        # Unique molecules assumed to be in cytosol, after wcEcoli

        # mass_update['projection_mass'] = compartment_submasses[self.projection_index, :].sum()
        # mass_update['cytosol_mass'] = compartment_submasses[self.cytosol_index, :].sum()
        # mass_update['extracellular_mass'] = compartment_submasses[self.extracellular_index, :].sum()
        # mass_update['flagellum_mass'] = compartment_submasses[self.flagellum_index, :].sum()
        # mass_update['membrane_mass'] = compartment_submasses[self.membrane_index, :].sum()
        # mass_update['outer_membrane_mass'] = compartment_submasses[self.outer_membrane_index, :].sum()
        # mass_update['periplasm_mass'] = compartment_submasses[self.periplasm_index, :].sum()
        # mass_update['pilus_mass'] = compartment_submasses[self.pilus_index, :].sum()
        # mass_update['inner_membrane_mass'] = compartment_submasses[self.inner_membrane_index, :].sum()

        # From wcEcoli, don't need (?) (mass difference due to partitioning (?))
        # mass_update['processMassDifferences'] = sum(
        #     state.process_mass_diffs() for state in self.internal_states.values()
        # ).sum(axis=1)

        # These are "logged quantities" in wcEcoli - keep separate?
        mass_update['dryMassFoldChange'] = mass_update['dry_mass'] / \
            self.dryMassInitial
        mass_update['proteinMassFoldChange'] = mass_update['proteinMass'] / \
            self.proteinMassInitial
        mass_update['rnaMassFoldChange'] = mass_update['rnaMass'] / \
            self.rnaMassInitial
        mass_update['smallMoleculeFoldChange'] = mass_update['smallMoleculeMass'] / \
            self.smallMoleculeMassInitial

        # TODO: Implement later? Need Clock process
        # update['expectedMassFoldChange'] = np.exp(np.log(2) * (self.time() - self.timeInitial) / self.cellCycleLen)

        self.first_time_step = False

        pp(mass_update)

        return {
            'listeners': {
                'mass': mass_update
            }
        }

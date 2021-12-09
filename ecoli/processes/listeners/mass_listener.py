"""
=============
Mass Listener
=============

Represents the total cellular mass.
"""

import numpy as np
from vivarium.core.process import Deriver
from vivarium.library.units import units
from ecoli.library.schema import bulk_schema, arrays_from, array_from, dict_value_schema, submass_schema

from vivarium.core.engine import pp

from ecoli.processes.registries import topology_registry
from ecoli.library.convert_update import convert_numpy_to_builtins

# Register default topology for this process, associating it with process name
NAME = 'ecoli-mass-listener'
topology_registry.register(
    NAME,
    {
        "bulk": ("bulk",),
        "unique": ("unique",),
        "listeners": ("listeners",)
    })


class MassListener(Deriver):
    """ MassListener """
    name = NAME

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
            'water': -1,
        },
        'compartment_indices': {
            'projection': [],
            'cytosol': [],
            'extracellular': [],
            'flagellum': [],
            'membrane': [],
            'outer_membrane': [],
            'periplasm': [],
            'pilus': [],
            'inner_membrane': [],
        },
        'compartment_id_to_index': {},
        'n_avogadro': 6.0221409e23,  # 1/mol
        'time_step': 2.0
    }

    def __init__(self, parameters=None):
        super().__init__(parameters)

        # molecule indexes and masses
        self.bulk_ids = self.parameters['bulk_ids']
        self.bulk_masses = self.parameters['bulk_masses']
        self.unique_ids = self.parameters['unique_ids']
        self.unique_masses = self.parameters['unique_masses']

        self.submass_indices = self.parameters['submass_indices']
        self.water_index = self.parameters['submass_indices']['water']
        self.submass_indices.pop('water')

        # compartment indexes
        self.compartment_id_to_index = self.parameters['compartment_id_to_index']
        self.projection_index = self.parameters['compartment_indices']['projection']
        self.cytosol_index = self.parameters['compartment_indices']['cytosol']
        self.extracellular_index = self.parameters['compartment_indices']['extracellular']
        self.flagellum_index = self.parameters['compartment_indices']['flagellum']
        self.membrane_index = self.parameters['compartment_indices']['membrane']
        self.outer_membrane_index = self.parameters['compartment_indices']['outer_membrane']
        self.periplasm_index = self.parameters['compartment_indices']['periplasm']
        self.pilus_index = self.parameters['compartment_indices']['pilus']
        self.inner_membrane_index = self.parameters['compartment_indices']['inner_membrane']

        # Set up matrix for compartment mass calculation
        self.compartment_abbrev_to_index = self.parameters['compartment_abbrev_to_index']
        self._bulk_molecule_by_compartment = np.stack(
            [np.core.defchararray.chararray.endswith(self.bulk_ids, abbrev + ']'
                                                     ) for abbrev in self.compartment_abbrev_to_index])

        # units and constants
        self.cellDensity = self.parameters['cellDensity']
        self.n_avogadro = self.parameters['n_avogadro']

        self.time_step = self.parameters['time_step']
        self.first_time_step = True

        self.mass_diffs = ['rRNA', 'tRNA', 'mRNA', 'miscRNA', 'nonspecific_RNA',
                           'protein', 'metabolite', 'water', 'DNA']

    def ports_schema(self):
        default_schema = {
            '_default': 0.0,
            '_updater': 'set',
            '_emit': True}

        ports = {
            'bulk': bulk_schema(self.bulk_ids),
            'unique': {
                mol_id: dict_value_schema(mol_id + 's')
                for mol_id in self.unique_ids
                if mol_id not in ['DnaA_box', 'active_ribosome']
            },
            'listeners': {
                'mass': {
                    'cell_mass': default_schema,
                    'water_mass': default_schema,
                    'dry_mass': default_schema,
                    **{submass + 'Mass': default_schema
                       for submass in self.submass_indices.keys()},
                    'volume': default_schema,
                    'proteinMassFraction': default_schema,
                    'rnaMassFraction': default_schema,
                    'growth': default_schema,
                    'instantaniousGrowthRate': default_schema,
                    'dryMassFoldChange': default_schema,
                    'proteinMassFoldChange': default_schema,
                    'rnaMassFoldChange': default_schema,
                    'smallMoleculeFoldChange': default_schema,
                    # compartment mass
                    'projection_mass': default_schema,
                    'cytosol_mass': default_schema,
                    'extracellular_mass': default_schema,
                    'flagellum_mass': default_schema,
                    'membrane_mass': default_schema,
                    'outer_membrane_mass': default_schema,
                    'periplasm_mass': default_schema,
                    'pilus_mass': default_schema,
                    'inner_membrane_mass': default_schema,
                }
            }
        }
        ports['unique'].update({
            'active_ribosome': dict_value_schema('active_ribosome'),
            'DnaA_box': dict_value_schema('DnaA_boxes')
        })
        return ports

    def get_compartment_submasses(self, states):

        # Compute bulk summed masses for each compartment
        bulk_state = array_from(states['bulk'])
        bulk_compartment_submasses = np.dot(
            bulk_state * self._bulk_molecule_by_compartment, self.bulk_masses)

        # Compute unique summed masses for each compartment
        unique_compartment_submasses = np.zeros_like(bulk_compartment_submasses)
        for molecule_id, molecule_mass in zip(
                self.unique_ids, self.unique_masses):
            molecules = states['unique'][molecule_id]
            n_molecules = len(molecules)
            if n_molecules == 0:
                continue

            mass = molecule_mass * n_molecules
            # TODO: include other compartments for unique molecules
            unique_compartment_submasses[self.compartment_abbrev_to_index['c'], :] += mass

        compartment_submasses = np.add(
            bulk_compartment_submasses,
            unique_compartment_submasses)

        return compartment_submasses

    def next_update(self, timestep, states):
        # Initialize update with 0's for each submass
        mass_update = {key + "Mass": 0 for key in self.submass_indices}

        # Get previous dry mass, for calculating growth later
        old_dry_mass = states['listeners']['mass']['dry_mass']

        # get submasses from bulk and unique
        bulk_counts = np.array([states['bulk'][id] for id in self.bulk_ids])
        bulk_submasses = np.dot(bulk_counts, self.bulk_masses)

        unique_counts = np.array([len(states['unique'][unique_id])
                                  for unique_id in self.unique_ids])
        unique_submasses = np.dot(unique_counts, self.unique_masses)
        unique_mass_diffs = np.zeros(len(self.mass_diffs))
        for unique_id in self.unique_ids:
            for molecule in states['unique'][unique_id].values():
                unique_mass_diffs += molecule['submass']
        unique_submasses += unique_mass_diffs

        # all of the submasses
        all_submasses = bulk_submasses + unique_submasses

        # save cell mass, water mass, dry mass
        mass_update['cell_mass'] = all_submasses.sum()
        mass_update['water_mass'] = all_submasses[self.water_index]
        mass_update['dry_mass'] = (
                mass_update['cell_mass'] - mass_update['water_mass'])

        # Store submasses
        for submass, indices in self.submass_indices.items():
            mass_update[submass + "Mass"] = all_submasses[indices].sum()

        mass_update['volume'] = mass_update['cell_mass'] / self.cellDensity

        mass_update['proteinMassFraction'] = (
                mass_update['proteinMass'] / mass_update['dry_mass'])
        mass_update['rnaMassFraction'] = (
                mass_update['rnaMass'] / mass_update['dry_mass'])

        if self.first_time_step:
            mass_update['growth'] = np.nan
            self.dryMassInitial = mass_update['dry_mass']
            self.proteinMassInitial = mass_update['proteinMass']
            self.rnaMassInitial = mass_update['rnaMass']
            self.smallMoleculeMassInitial = mass_update['smallMoleculeMass']
        else:
            mass_update['growth'] = mass_update['dry_mass'] - old_dry_mass

        mass_update['instantaniousGrowthRate'] = (
                mass_update['growth'] / self.time_step / mass_update['dry_mass'])

        # Compartment submasses
        compartment_submasses = self.get_compartment_submasses(states)
        mass_update['projection_mass'] = compartment_submasses[self.projection_index, :].sum()
        mass_update['cytosol_mass'] = compartment_submasses[self.cytosol_index, :].sum()
        mass_update['extracellular_mass'] = compartment_submasses[self.extracellular_index, :].sum()
        mass_update['flagellum_mass'] = compartment_submasses[self.flagellum_index, :].sum()
        mass_update['membrane_mass'] = compartment_submasses[self.membrane_index, :].sum()
        mass_update['outer_membrane_mass'] = compartment_submasses[self.outer_membrane_index, :].sum()
        mass_update['periplasm_mass'] = compartment_submasses[self.periplasm_index, :].sum()
        mass_update['pilus_mass'] = compartment_submasses[self.pilus_index, :].sum()
        mass_update['inner_membrane_mass'] = compartment_submasses[self.inner_membrane_index, :].sum()

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

        self.first_time_step = False

        update = {
            'listeners': {
                'mass': mass_update
            }
        }
        return convert_numpy_to_builtins(update)

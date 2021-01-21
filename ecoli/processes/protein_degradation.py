"""
ProteinDegradation

Protein degradation sub-model. Encodes molecular simulation of protein degradation as a Poisson process

TODO:
- protein complexes
- add protease functionality
"""

from __future__ import absolute_import, division, print_function

import numpy as np

from vivarium.core.process import Process
from vivarium.core.composition import simulate_process_in_experiment

from wholecell.utils.constants import REQUEST_PRIORITY_DEGRADATION
from wholecell.utils import units

from functools import reduce


class ProteinDegradation(Process):
    name = 'ecoli-protein-degradation'

    defaults = {
        'raw_degradation_rate': [],
        'shuffle_indexes': None,
        'water_id': 'h20',
        'amino_acid_ids': [],
        'amino_acid_counts': [],
        'protein_ids': [],
        'protein_lengths': [],
        'seed': 0}

    # Constructor
    def __init__(self, initial_parameters):
        super(ProteinDegradation, self).__init__(initial_parameters)

        self.raw_degradation_rate = self.parameters['raw_degradation_rate']
        self.shuffle_indexes = self.parameters['shuffle_indexes']
        if self.shuffle_indexes:
            self.raw_degradation_rate = self.raw_degradation_rate[self.shuffle_indexes]

        self.water_id = self.parameters['water_id']
        self.amino_acid_ids = self.parameters['amino_acid_ids']
        self.amino_acid_counts = self.parameters['amino_acid_counts']

        self.metabolite_ids = self.amino_acid_ids + [self.water_id]
        self.amino_acid_indexes = np.arange(0, len(self.amino_acid_ids))
        self.water_index = self.metabolite_ids.index(self.water_id)

        # Build protein IDs for S matrix
        self.protein_ids = self.parameters['protein_ids']
        self.protein_lengths = self.parameters['protein_lengths']

        self.seed = self.parameters['seed']
        self.random_state = np.random.RandomState(seed=self.seed)

        # Build S matrix
        self.degradation_matrix = np.zeros((
            len(self.metabolite_ids),
            len(self.protein_ids)), np.int64)
        self.degradation_matrix[self.amino_acid_indexes, :] = np.transpose(
            self.amino_acid_counts)
        self.degradation_matrix[self.water_index, :] = -(np.sum(
            self.degradation_matrix[self.amino_acid_indexes, :], axis=0) - 1)

    def ports_schema(self):
        return {
            'metabolites': {
                metabolite: {
                    '_default': 0,
                    '_emit': True}
                for metabolite in self.metabolite_ids},
            'proteins': {
                protein: {
                    '_default': 0,
                    '_emit': True}
                for protein in self.protein_ids}}

    def next_update(self, timestep, states):
        proteins = states['proteins']
        protein_counts = np.array(list(proteins.values()))
        rates = self.raw_degradation_rate * timestep

        degrade = np.fmin(
            self.random_state.poisson(rates * protein_counts),
            protein_counts)

        # Determine the number of hydrolysis reactions
        reactions = np.dot(self.protein_lengths, degrade)

        # Determine the amount of water required to degrade the selected proteins
        # Assuming one N-1 H2O is required per peptide chain length N
        # TODO(Ryan): It seems this water request is never used?
        # self.h2o.requestIs(nReactions - np.sum(nProteinsToDegrade))
        # self.proteins.requestIs(nProteinsToDegrade)

        # Degrade selected proteins, release amino acids from those proteins back into the cell, 
        # and consume H_2O that is required for the degradation process
        metabolites_delta = np.dot(
            self.degradation_matrix,
            degrade).astype(int)

        update = {
            'metabolites': {
                metabolite: metabolites_delta[index]
                for index, metabolite in enumerate(self.metabolite_ids)},
            'proteins': {
                protein: -degrade[index]
                for index, protein in enumerate(proteins.keys())}}

        return update


def test_protein_degradation():
    test_config = {
        'raw_degradation_rate': np.array([0.05, 0.08, 0.13, 0.21]),
        'water_id': 'H2O',
        'amino_acid_ids': ['A', 'B', 'C'],
        'amino_acid_counts': np.array([
            [5, 7, 13],
            [1, 3, 5],
            [4, 4, 4],
            [13, 11, 5]]),
        'protein_ids': ['w', 'x', 'y', 'z'],
        'protein_lengths': np.array([
            25, 9, 12, 29])}

    protein_degradation = ProteinDegradation(test_config)

    state = {
        'metabolites': {
            'A': 10,
            'B': 20,
            'C': 30,
            'H2O': 1000},
        'proteins': {
            'w': 5,
            'x': 6,
            'y': 7,
            'z': 8}}

    settings = {
        'total_time': 10,
        'initial_state': state}

    data = simulate_process_in_experiment(protein_degradation, settings)

    # Assertions =======================================================
    # Proteins are monotonically decreasing, never <0:
    for protein in test_config['protein_ids']:
        protein_data = data["proteins"][protein]
        assert all(x >= y >= 0
                   for x, y in zip(protein_data, protein_data[1:])), \
            f"Protein {protein} is not monotonically decreasing or falls below zero."

    # Amino acids are monotonically increasing
    for aa in test_config['amino_acid_ids']:
        aa_data = data["metabolites"][aa]
        assert all(x <= y
                   for x, y in zip(aa_data, aa_data[1:])), \
            f"Amino acid {aa} is not monotonically increasing."

    # H20 is monotonically decreasing, never < 0
    h20_data = data["metabolites"][test_config["water_id"]]
    assert all(x >= y >= 0
               for x, y in zip(h20_data, h20_data[1:])), \
        f"H20 is not monotonically decreasing or falls below zero."

    # Amino acids are released, H20 consumed whenever a protein is degraded

    # Protein degradation events follow a Poisson distribution with specified rate

    print(data)


if __name__ == "__main__":
    test_protein_degradation()

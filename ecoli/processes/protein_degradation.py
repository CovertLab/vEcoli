"""
===================
Protein Degradation
===================

This process accounts for the degradation of protein monomers.
Specific proteins to be degraded are selected as a Poisson process.

TODO:
 - protein complexes
 - add protease functionality
"""
import numpy
import numpy as np

from vivarium.core.composition import simulate_process

from ecoli.library.data_predicates import (
    monotonically_increasing, monotonically_decreasing, all_nonnegative)
from ecoli.library.schema import array_to, array_from, numpy_schema

from ecoli.processes.registries import topology_registry
from ecoli.processes.partition import PartitionedProcess


# Register default topology for this process, associating it with process name
NAME = 'ecoli-protein-degradation'
TOPOLOGY = {
    "bulk": ('bulk',)
}
topology_registry.register(NAME, TOPOLOGY)


class ProteinDegradation(PartitionedProcess):
    """ Protein Degradation PartitionedProcess """

    name = NAME
    topology = TOPOLOGY
    defaults = {
        'raw_degradation_rate': [],
        'shuffle_indexes': None,
        'water_id': 'h2o',
        'amino_acid_ids': [],
        'amino_acid_counts': [],
        'protein_ids': [],
        'protein_lengths': [],
        'seed': 0}

    # Constructor
    def __init__(self, parameters=None):
        super().__init__(parameters)

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
        # Assuming N-1 H2O is required per peptide chain length N
        self.degradation_matrix[self.water_index, :] = -(np.sum(
            self.degradation_matrix[self.amino_acid_indexes, :], axis=0) - 1)

    def ports_schema(self):
        return {'bulk': numpy_schema('bulk')} #right now a dictionary, now need to connect to entire bulk store
    #replace them both with a single port called bulk, and have entire bulk array and want to pull out specific protein counts

    def calculate_request(self, timestep, states):
        #have to do water id if it's not a string then do.. self.water_id = bulk_array idx
        # Determine how many proteins to degrade based on the degradation rates and counts of each protein
        if isinstance(self.water_id, str):
            self.water_id = np.where(states[
                'bulk']['id'] == self.water_id)[0][0]
        if isinstance(self.protein_ids, numpy.ndarray):
            protein_list = self.protein_ids.tolist()
            self.protein_ids = []
            for protein in protein_list:
                self.protein_ids.append(np.where(states['bulk']['id'] == protein)[0][0])

        #protein_data = array_from(states['proteins']) #get the rows for protein ids and want to make sure that the proteins are in order!
        protein_data = states['bulk'][self.protein_ids]['count'] #protein_ids is currently a list of row numbers, which one to index?
        nProteinsToDegrade = np.fmin(
            self.random_state.poisson(self._proteinDegRates(timestep) * protein_data),
            protein_data
            )

        # Determine the number of hydrolysis reactions
        # TODO(vivarium): Missing asNumber() and other unit-related things
        nReactions = np.dot(self.protein_lengths, nProteinsToDegrade) #protein_lengths already numpy array

        # Determine the amount of water required to degrade the selected proteins
        # Assuming one N-1 H2O is required per peptide chain length N
        # requests = {'metabolites': {self.water_id: nReactions - np.sum(nProteinsToDegrade)},
        #             'proteins': (array_to(states['proteins'], nProteinsToDegrade))}
        #switch to just giving indices array and update array to pass in indexes
        index_update = self.protein_ids.append(self.water_id)
        parameter_update = nProteinsToDegrade.tolist().append(nReactions - np.sum(nProteinsToDegrade))
        requests = {'bulk': tuple((index_update, parameter_update))}
        return requests

    def evolve_state(self, timestep, states):
        # Degrade selected proteins, release amino acids from those proteins back into the cell, 
        # and consume H_2O that is required for the degradation process
        if isinstance(self.protein_ids, numpy.ndarray):
            protein_list = self.protein_ids.tolist()
            self.protein_ids = []
            for protein in protein_list:
                self.protein_ids.append(np.where(states['bulk']['id'] == protein)[0][0])
        if isinstance(self.metabolite_ids, numpy.ndarray):
            metabolite_list = self.metabolite_ids.tolist()
            self.metabolite_ids = []
            for metabolite in metabolite_list:
                self.metabolite_ids.append(np.where(states['bulk']['id'] == metabolite)[0][0])
        #allocated_proteins = array_from(states['proteins'])
        allocated_proteins = states['bulk'][self.protein_ids]['count']
        metabolites_delta = np.dot(
            self.degradation_matrix,
            allocated_proteins).astype(int)

        index_update = self.metabolite_ids + self.protein_ids
        value_update = metabolites_delta.tolist() + (-allocated_proteins).tolist()


        update = {'bulk': tuple((index_update, value_update))}

        # update = {
        #     'metabolites': {
        #         metabolite: metabolites_delta[index]
        #         for index, metabolite in enumerate(self.metabolite_ids)},
        #     'proteins': {
        #         protein: -allocated_proteins[index]
        #         for index, protein in enumerate(states['proteins'])}}

        return update


    def _proteinDegRates(self, timestep):
        return self.raw_degradation_rate * timestep


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
            'H2O': 10000},
        'proteins': {
            'w': 50,
            'x': 60,
            'y': 70,
            'z': 80}}

    settings = {
        'total_time': 100,
        'initial_state': state}

    data = simulate_process(protein_degradation, settings)

    # Assertions =======================================================
    protein_data = np.concatenate([[data["proteins"][protein]] for protein in test_config['protein_ids']], axis=0)
    protein_delta = protein_data[:, 1:] - protein_data[:, :-1]

    aa_data = np.concatenate([[data["metabolites"][aa]] for aa in test_config['amino_acid_ids']], axis=0)
    aa_delta = aa_data[:, 1:] - aa_data[:, :-1]

    h2o_data = np.array(data["metabolites"][test_config["water_id"]])
    h2o_delta = h2o_data[1:] - h2o_data[:-1]

    # Proteins are monotonically decreasing, never <0:
    for i in range(protein_data.shape[0]):
        assert monotonically_decreasing(protein_data[i, :]), (
            f"Protein {test_config['protein_ids'][i]} is not monotonically decreasing.")
        assert all_nonnegative(protein_data), f"Protein {test_config['protein_ids'][i]} falls below 0."

    # Amino acids are monotonically increasing
    for i in range(aa_data.shape[0]):
        assert monotonically_increasing(aa_data[i, :]), (
            f"Amino acid {test_config['amino_acid_ids'][i]} is not monotonically increasing.")

    # H2O is monotonically decreasing, never < 0
    assert monotonically_decreasing(h2o_data), f"H2O is not monotonically decreasing."
    assert all_nonnegative(h2o_data), f"H2O falls below 0."

    # Amino acids are released in specified numbers whenever a protein is degraded
    aa_delta_expected = map(lambda i: [test_config['amino_acid_counts'].T @ -protein_delta[:, i]],
                            range(protein_delta.shape[1]))
    aa_delta_expected = np.concatenate(list(aa_delta_expected)).T
    np.testing.assert_array_equal(aa_delta, aa_delta_expected,
                                  "Mismatch between expected release of amino acids, and counts actually released.")

    # N-1 molecules H2O is consumed whenever a protein of length N is degraded
    h2o_delta_expected = (protein_delta.T * (test_config['protein_lengths'] - 1)).T
    h2o_delta_expected = np.sum(h2o_delta_expected, axis=0)
    np.testing.assert_array_equal(h2o_delta, h2o_delta_expected,
                                  ("Mismatch between number of water molecules consumed\n"
                                   "and expected to be consumed in degradation."))

    print("Passed all tests.")

    return data


if __name__ == "__main__":
    test_protein_degradation()

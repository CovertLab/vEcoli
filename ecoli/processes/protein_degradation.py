from __future__ import absolute_import, division, print_function

import numpy as np

from vivarium.core.process import Process
from vivarium.core.composition import simulate_process_in_experiment

import wholecell.processes.process
from wholecell.utils.constants import REQUEST_PRIORITY_DEGRADATION
from wholecell.utils import units


class ProteinDegradation(Process):
    name = "protein-degradation"

    defaults = {
        'raw_degradation_rate': [],
        'shuffle_indexes': None,
        'water_id': 'h20',
        'amino_acid_ids': [],
        'amino_acid_counts': [],
        'protein_ids': [],
        'protein_lengths': []}

    # Constructor
    def __init__(self):
        if not initial_parameters:
            initial_parameters = {}

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

        # Build S matrix
        self.degradation_matrix = np.zeros((
            len(self.metabolite_ids),
            len(self.protein_ids)), np.int64)
        self.degradation_matrix[self.amino_acid_indexes, :] = np.transpose(
            self.amino_acid_counts)
        self.degradation_matrix[self.water_index, :]  = -(np.sum(
            self.degradation_matrix[self.amino_acid_indexes, :], axis = 0) - 1)

    def ports_schema(self):
        return {
            'metabolites': {
                metabolite: {
                    '_default': 0,
                    '_emit': True}
                for metabolite in self.metabolite_ids},
            'water': {
                self.water_id: {
                    '_default': 0,
                    '_emit': True}},
            'proteins': {
                protein: {
                    '_default': 0,
                    '_emit': True}
                for protein in self.protein_ids}}

    def next_update(self, timestep, states):
        proteins = states['proteins']
        protein_counts = np.array(proteins.values())
        rates = self.raw_degradation_rate * timestep

        degrade = np.fmin(
            self.randomState.poisson(rates * protein_counts),
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
            degrade)

        return {
            'metabolites': {
                metabolite: metabolites_delta[index]
                for index, metabolite in enumerate(self.metabolites.keys())},
            'proteins': {
                protein: -degrade[index]
                for index, protein in enumerate(proteins.keys())}}



    # # Construct object graph
    # def initialize(self, sim, sim_data):
    #     super(ProteinDegradation, self).initialize(sim, sim_data)

    #     # Load protein degradation rates (based on N-end rule)
    #     self.rawDegRate = sim_data.process.translation.monomerData['degRate'].asNumber(1 / units.s)

    #     shuffleIdxs = None
    #     if hasattr(sim_data.process.translation, "monomerDegRateShuffleIdxs") and sim_data.process.translation.monomerDegRateShuffleIdxs is not None:
    #         shuffleIdxs = sim_data.process.translation.monomerDegRateShuffleIdxs
    #         self.rawDegRate = self.rawDegRate[shuffleIdxs]

    #     # Build metabolite IDs for S matrix
    #     h2oId = [sim_data.moleculeIds.water]
    #     metaboliteIds = sim_data.moleculeGroups.amino_acids + h2oId
    #     aaIdxs = np.arange(0, len(sim_data.moleculeGroups.amino_acids))
    #     h2oIdx = metaboliteIds.index(sim_data.moleculeIds.water)

    #     # Build protein IDs for S matrix
    #     proteinIds = sim_data.process.translation.monomerData['id']

    #     # Load protein length
    #     self.proteinLengths = sim_data.process.translation.monomerData['length']

    #     # Build S matrix
    #     self.proteinDegSMatrix = np.zeros((len(metaboliteIds), len(proteinIds)), np.int64)
    #     self.proteinDegSMatrix[aaIdxs, :] = np.transpose(sim_data.process.translation.monomerData["aaCounts"].asNumber())
    #     self.proteinDegSMatrix[h2oIdx, :]  = -(np.sum(self.proteinDegSMatrix[aaIdxs, :], axis = 0) - 1)

    #     # Build Views
    #     self.metabolites = self.bulkMoleculesView(metaboliteIds)
    #     self.h2o = self.bulkMoleculeView(sim_data.moleculeIds.water)
    #     self.proteins = self.bulkMoleculesView(proteinIds)

    #     self.bulkMoleculesRequestPriorityIs(REQUEST_PRIORITY_DEGRADATION)

    # def calculateRequest(self):

    #     # Determine how many proteins to degrade based on the degradation rates and counts of each protein
    #     nProteinsToDegrade = np.fmin(
    #         self.randomState.poisson(self._proteinDegRates() * self.proteins.total_counts()),
    #         self.proteins.total_counts()
    #         )

    #     # Determine the number of hydrolysis reactions
    #     nReactions = np.dot(self.proteinLengths.asNumber(), nProteinsToDegrade)

    #     # Determine the amount of water required to degrade the selected proteins
    #     # Assuming one N-1 H2O is required per peptide chain length N
    #     self.h2o.requestIs(nReactions - np.sum(nProteinsToDegrade))
    #     self.proteins.requestIs(nProteinsToDegrade)


    # def evolveState(self):

    #     # Degrade selected proteins, release amino acids from those proteins back into the cell, 
    #     # and consume H_2O that is required for the degradation process
    #     self.metabolites.countsInc(np.dot(
    #         self.proteinDegSMatrix,
    #         self.proteins.counts()
    #         ))
    #     self.proteins.countsIs(0)

    # def _proteinDegRates(self):
    #     return self.rawDegRate * self.timeStepSec()

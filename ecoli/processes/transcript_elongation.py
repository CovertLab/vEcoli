"""
TranscriptElongation

Transcription elongation sub-model.

TODO:
- use transcription units instead of single genes
- account for energy
"""

from __future__ import absolute_import, division, print_function

import numpy as np

from vivarium.core.process import Process
from vivarium.core.composition import simulate_process

from ecoli.library.schema import array_from, array_to, arrays_from, arrays_to, listener_schema, bulk_schema

from wholecell.utils.polymerize import buildSequences, polymerize, computeMassIncrease
from wholecell.utils import units

class TranscriptElongation(Process):
    name = 'ecoli-transcript-elongation'

    defaults = {
        'max_time_step': 0.0,
        'rnaPolymeraseElongationRateDict': {},
        'rnaIds': [],
        'rnaLengths': np.array([]),
        'rnaSequences': np.array([[]]),
        'ntWeights': np.array([]),
        'endWeight': np.array([]),
        'replichore_lengths': np.array([]),
        'idx_16S_rRNA': np.array([]),
        'idx_23S_rRNA': np.array([]),
        'idx_5S_rRNA': np.array([]),
        'is_mRNA': np.array([]),
        'ppi': '',
        'inactive_RNAP': '',
        'ntp_ids': [],
        'variable_elongation': False,
        'make_elongation_rates': lambda random, rates, timestep, variable: rates,
        'seed': 0}

    def __init__(self, initial_parameters):
        super(TranscriptElongation, self).__init__(initial_parameters)

        self.max_time_step = self.parameters['max_time_step']

        # Load parameters
        self.rnaPolymeraseElongationRateDict = self.parameters['rnaPolymeraseElongationRateDict']
        self.rnaIds = self.parameters['rnaIds']
        self.rnaLengths = self.parameters['rnaLengths']
        self.rnaSequences = self.parameters['rnaSequences']
        self.ppi = self.parameters['ppi']
        self.inactive_RNAP = self.parameters['inactive_RNAP']
        self.ntp_ids = self.parameters['ntp_ids']
        self.ntWeights = self.parameters['ntWeights']
        self.endWeight = self.parameters['endWeight']
        self.replichore_lengths = self.parameters['replichore_lengths']
        self.chromosome_length = self.replichore_lengths.sum()

        # ID Groups of rRNAs
        self.idx_16S_rRNA = self.parameters['idx_16S_rRNA']
        self.idx_23S_rRNA = self.parameters['idx_23S_rRNA']
        self.idx_5S_rRNA = self.parameters['idx_5S_rRNA']

        # Mask for mRNAs
        self.is_mRNA = self.parameters['is_mRNA']
        self.molecule_ids = [self.ppi, self.inactive_RNAP]

        self.variable_elongation = self.parameters['variable_elongation']
        self.make_elongation_rates = self.parameters['make_elongation_rates']

        self.seed = self.parameters['seed']
        self.random_state = np.random.RandomState(seed = self.seed)

    def ports_schema(self):
        return {
            'environment': {
                'media_id': {'_default': ''}},

            'RNAs': {
                '*': {
                    'unique_index': {'_default': 0, '_updater': 'set'},
                    'TU_index': {'_default': 0, '_updater': 'set'},
                    'transcript_length': {'_default': 0, '_updater': 'set', '_emit': True},
                    'is_mRNA': {'_default': False, '_updater': 'set'},
                    'is_full_transcript': {'_default': False, '_updater': 'set'},
                    'can_translate': {'_default': False, '_updater': 'set'},
                    'RNAP_index': {'_default': 0, '_updater': 'set'}}},

            'active_RNAPs': {
                '*': {
                    'unique_index': {'_default': 0, '_updater': 'set'},
                    'domain_index': {'_default': 0, '_updater': 'set'},
                    'coordinates': {'_default': 0, '_updater': 'set', '_emit': True},
                    'direction': {'_default': 0, '_updater': 'set'}}},

            'bulk_RNAs': bulk_schema(self.rnaIds),
            'ntps': bulk_schema(self.ntp_ids),
            'molecules': bulk_schema(self.molecule_ids),

            'listeners': {}}

    def next_update(self, timestep, states):
        # Calculate elongation rate based on the current media
        current_media_id = states['environment']['media_id']

        self.rnapElongationRate = self.rnaPolymeraseElongationRateDict[current_media_id].asNumber(units.nt / units.s)

        self.elongation_rates = self.make_elongation_rates(
            self.random_state,
            self.rnapElongationRate,
            timestep,
            self.variable_elongation)

        # If there are no active RNA polymerases, return immediately
        if len(states['active_RNAPs']) == 0:
            return {}

        # Determine total possible sequences of nucleotides that can be
        # transcribed in this time step for each partial transcript
        # Get attributes from existing RNAs
        TU_index_all_RNAs, length_all_RNAs, is_full_transcript, is_mRNA_all_RNAs, RNAP_index_all_RNAs = arrays_from(
            states['RNAs'].values(),
            ['TU_index', 'transcript_length', 'is_full_transcript', 'is_mRNA', 'RNAP_index'])

        TU_indexes = TU_index_all_RNAs
        transcript_lengths = length_all_RNAs

        is_partial_transcript = np.logical_not(is_full_transcript)
        TU_indexes_partial = TU_indexes[is_partial_transcript]
        transcript_lengths_partial = transcript_lengths[is_partial_transcript]

        sequences = buildSequences(
            self.rnaSequences,
            TU_indexes_partial,
            transcript_lengths_partial,
            self.elongation_rates)

        sequenceComposition = np.bincount(
            sequences[sequences != polymerize.PAD_VALUE], minlength = 4)

        # Calculate if any nucleotides are limited and request up to the number
        # in the sequences or number available
        # ntpsTotal = self.ntps.total_counts()
        ntpsTotal = array_from(states['ntps'])
        maxFractionalReactionLimit = np.fmin(1, ntpsTotal / sequenceComposition)
        ntpsCounts = maxFractionalReactionLimit * sequenceComposition

        update = {
            'listeners': {
                'growth_limits': {}}}

        update['listeners']['growth_limits']['ntp_pool_size'] = ntpsTotal
        update['listeners']['growth_limits']['ntp_request_size'] = ntpsCounts
        update['listeners']['growth_limits']['ntp_allocated'] = ntpsCounts

        ntpCounts = array_from(states['ntps'])

        # Determine sequences of RNAs that should be elongated
        is_partial_transcript = np.logical_not(is_full_transcript)
        partial_transcript_indexes = np.where(is_partial_transcript)[0]
        TU_index_partial_RNAs = TU_index_all_RNAs[is_partial_transcript]
        length_partial_RNAs = length_all_RNAs[is_partial_transcript]
        is_mRNA_partial_RNAs = is_mRNA_all_RNAs[is_partial_transcript]
        RNAP_index_partial_RNAs = RNAP_index_all_RNAs[is_partial_transcript]

        sequences = buildSequences(
            self.rnaSequences,
            TU_index_partial_RNAs,
            length_partial_RNAs,
            self.elongation_rates)

        # Polymerize transcripts based on sequences and available nucleotides
        reactionLimit = ntpCounts.sum()
        result = polymerize(
            sequences,
            ntpCounts,
            reactionLimit,
            self.random_state,
            self.elongation_rates[TU_index_partial_RNAs])

        sequence_elongations = result.sequenceElongation
        ntps_used = result.monomerUsages

        # Calculate changes in mass associated with polymerization
        added_mass = computeMassIncrease(sequences, sequence_elongations,
            self.ntWeights)
        did_initialize = (length_partial_RNAs == 0) & (sequence_elongations > 0)
        added_mass[did_initialize] += self.endWeight

        # Calculate updated transcript lengths
        updated_transcript_lengths = length_partial_RNAs + sequence_elongations

        # Get attributes of active RNAPs
        coordinates, domain_index, direction, RNAP_unique_index = arrays_from(
            states['active_RNAPs'].values(),
            ['coordinates', 'domain_index', 'direction', 'unique_index'])

        # Active RNAP count should equal partial transcript count
        assert len(RNAP_unique_index) == len(RNAP_index_partial_RNAs)

        # All partial RNAs must be linked to an RNAP
        assert (np.count_nonzero(RNAP_index_partial_RNAs == -1) == 0)

        # Get mapping indexes between partial RNAs to RNAPs
        partial_RNA_to_RNAP_mapping, RNAP_to_partial_RNA_mapping = get_mapping_arrays(
            RNAP_index_partial_RNAs, RNAP_unique_index)

        # Rescale boolean array of directions to an array of 1's and -1's.
        # True is converted to 1, False is converted to -1.
        direction_rescaled = (2*(direction - 0.5)).astype(np.int64)

        # Compute the updated coordinates of RNAPs. Coordinates of RNAPs
        # moving in the positive direction are increased, whereas coordinates
        # of RNAPs moving in the negative direction are decreased.
        updated_coordinates = coordinates + np.multiply(
            direction_rescaled, sequence_elongations[partial_RNA_to_RNAP_mapping])

        # Reset coordinates of RNAPs that cross the boundaries between right
        # and left replichores
        updated_coordinates[
            updated_coordinates > self.replichore_lengths[0]
            ] -= self.chromosome_length
        updated_coordinates[
            updated_coordinates < -self.replichore_lengths[1]
            ] += self.chromosome_length

        # Update transcript lengths of RNAs and coordinates of RNAPs
        length_all_RNAs[is_partial_transcript] = updated_transcript_lengths

        # Update added submasses of RNAs. Masses of partial mRNAs are counted
        # as mRNA mass as they are already functional, but the masses of other
        # types of partial RNAs are counted as nonspecific RNA mass.
        added_nsRNA_mass_all_RNAs = np.zeros_like(
            TU_index_all_RNAs, dtype=np.float64)
        added_mRNA_mass_all_RNAs = np.zeros_like(
            TU_index_all_RNAs, dtype=np.float64)

        added_nsRNA_mass_all_RNAs[is_partial_transcript] = np.multiply(
            added_mass, np.logical_not(is_mRNA_partial_RNAs))
        added_mRNA_mass_all_RNAs[is_partial_transcript] = np.multiply(
            added_mass, is_mRNA_partial_RNAs)

        # Determine if transcript has reached the end of the sequence
        terminal_lengths = self.rnaLengths[TU_index_partial_RNAs]
        did_terminate_mask = (updated_transcript_lengths == terminal_lengths)
        terminated_RNAs = np.bincount(
            TU_index_partial_RNAs[did_terminate_mask],
            minlength = self.rnaSequences.shape[0])

        # Assume transcription from all rRNA genes produce rRNAs from the first
        # operon. This is done to simplify the complexation reactions that
        # produce ribosomal subunits.
        n_total_16Srrna = terminated_RNAs[self.idx_16S_rRNA].sum()
        n_total_23Srrna = terminated_RNAs[self.idx_23S_rRNA].sum()
        n_total_5Srrna = terminated_RNAs[self.idx_5S_rRNA].sum()

        terminated_RNAs[self.idx_16S_rRNA] = 0
        terminated_RNAs[self.idx_23S_rRNA] = 0
        terminated_RNAs[self.idx_5S_rRNA] = 0

        terminated_RNAs[self.idx_16S_rRNA[0]] = n_total_16Srrna
        terminated_RNAs[self.idx_23S_rRNA[0]] = n_total_23Srrna
        terminated_RNAs[self.idx_5S_rRNA[0]] = n_total_5Srrna

        # Update is_full_transcript attribute of RNAs
        is_full_transcript_updated = is_full_transcript.copy()
        is_full_transcript_updated[
            partial_transcript_indexes[did_terminate_mask]] = True

        n_terminated = did_terminate_mask.sum()
        n_initialized = did_initialize.sum()
        n_elongations = ntps_used.sum()

        # Get counts of new bulk RNAs
        n_new_bulk_RNAs = terminated_RNAs.copy()
        n_new_bulk_RNAs[self.is_mRNA] = 0

        rna_indexes = list(states['RNAs'].keys())
        rnas_update = arrays_to(
            len(states['RNAs']), {
                'transcript_length': length_all_RNAs,
                'is_full_transcript': is_full_transcript_updated,
                'submass': arrays_to(
                    len(states['RNAs']), {
                        'nonspecific_RNA': added_nsRNA_mass_all_RNAs,
                        'mRNA': added_mRNA_mass_all_RNAs})})

        delete_rnas = partial_transcript_indexes[np.logical_and(
            did_terminate_mask, np.logical_not(is_mRNA_partial_RNAs))]        

        update = {
            'listeners': {},
            'RNAs': {
                rna_indexes[index]: rna
                for index, rna in enumerate(rnas_update)}}
        update['RNAs']['_delete'] = [rna_indexes[index] for index in delete_rnas]

        rnap_indexes = list(states['active_RNAPs'].keys())
        rnaps_update = arrays_to(
            len(states['active_RNAPs']), {
                'coordinates': updated_coordinates})

        delete_rnaps = np.where(did_terminate_mask[partial_RNA_to_RNAP_mapping])[0]

        update['active_RNAPs'] = {
            rnap_indexes[index]: rnap
            for index, rnap in enumerate(rnaps_update)}

        update['active_RNAPs']['_delete'] = [rnap_indexes[index] for index in delete_rnaps]

        update['ntps'] = array_to(self.ntp_ids, -ntps_used)
        update['bulk_RNAs'] = array_to(self.rnaIds, n_new_bulk_RNAs)
        update['molecules'] = array_to(self.molecule_ids, [
            n_elongations - n_initialized,
            n_terminated])

        # self.RNAs.attrIs(transcript_length=length_all_RNAs)
        # self.RNAs.attrIs(is_full_transcript=is_full_transcript_updated)
        # self.RNAs.add_submass_by_name("nonspecific_RNA", added_nsRNA_mass_all_RNAs)
        # self.RNAs.add_submass_by_name("mRNA", added_mRNA_mass_all_RNAs)

        # # Remove partial transcripts that have finished transcription and are
        # # not mRNAs from unique molecules (these are moved to bulk molecules)
        # self.RNAs.delByIndexes(
        #     partial_transcript_indexes[np.logical_and(
        #         did_terminate_mask, np.logical_not(is_mRNA_partial_RNAs))])

        # self.active_RNAPs.attrIs(coordinates=updated_coordinates)

        # # Remove RNAPs that have finished transcription
        # self.active_RNAPs.delByIndexes(
        #     np.where(did_terminate_mask[partial_RNA_to_RNAP_mapping]))

        # # Update bulk molecule counts
        # self.ntps.countsDec(ntps_used)
        # self.bulk_RNAs.countsInc(n_new_bulk_RNAs)
        # self.inactive_RNAPs.countInc(n_terminated)
        # self.ppi.countInc(n_elongations - n_initialized)

        # Write outputs to listeners
        update['listeners']['transcript_elongation_listener'] = {
            "countRnaSynthesized": terminated_RNAs,
            "countNTPsUSed": n_elongations}
        update['listeners']['growth_limits'] = {
            "ntpUsed": ntps_used}
        update['listeners']['rnap_data'] = {
            "actualElongations": sequence_elongations.sum(),
            "didTerminate": did_terminate_mask.sum(),
            "terminationLoss": (terminal_lengths - length_partial_RNAs)[did_terminate_mask].sum()}

        return update

    def isTimeStepShortEnough(self, inputTimeStep, timeStepSafetyFraction):
        return inputTimeStep <= self.max_time_step


def get_mapping_arrays(x, y):
    """
    Returns the array of indexes of each element of array x in array y, and
    vice versa. Assumes that the elements of x and y are unique, and
    set(x) == set(y).
    """
    def argsort_unique(idx):
        """
        Quicker argsort for arrays that are permutations of np.arange(n).
        """
        n = idx.size
        argsort_idx = np.empty(n, dtype=np.int64)
        argsort_idx[idx] = np.arange(n)
        return argsort_idx

    x_argsort = np.argsort(x)
    y_argsort = np.argsort(y)

    x_to_y = x_argsort[argsort_unique(y_argsort)]
    y_to_x = y_argsort[argsort_unique(x_argsort)]

    return x_to_y, y_to_x

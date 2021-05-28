"""
TranscriptElongation

Transcription elongation sub-model.

TODO:
- use transcription units instead of single genes
- account for energy
"""

import numpy as np

from vivarium.core.process import Process
from vivarium.core.composition import simulate_process

from ecoli.library.schema import array_from, array_to, arrays_from, arrays_to, listener_schema, bulk_schema

from wholecell.utils.random import stochasticRound
from wholecell.utils.polymerize import buildSequences, polymerize, computeMassIncrease
from wholecell.utils import units

from ecoli.library.data_predicates import monotonically_increasing

class TranscriptElongation(Process):
    # TODO: comment out terminationLoss - doesn't seem to be used/informative?
    """TranscriptElongation

    defaults:
        - max_time_step (float) : ???
        - rnaPolymeraseElongationRateDict (dict): Array with elongation rate set points
                                                  for different media environments.
        - rnaIds (array[str]) : array of names for each TU
        - rnaLengths (array[int]) : array of lengths for each TU (in nucleotides?)
        - rnaSequences (2D array[int]) : Array with the nucleotide sequences of each TU.
                                         This is in the form of a 2D array where
                                         each row is a TU, and each column is a position in
                                         the TU's sequence. Nucleotides are stored as an index
                                         {0, 1, 2, 3}, and the row is padded with -1's on the right
                                         to indicate where the sequence ends.
        - ntWeights (array[float]): Array of nucleotide weights
        - endWeight (array[float]): ???,
        - replichore_lengths (array[int]): lengths of replichores (in nucleotides?),
        - idx_16S_rRNA (array[int]): indexes of TUs for 16SrRNA
        - idx_23S_rRNA (array[int]): indexes of TUs for 23SrRNA
        - idx_5S_rRNA (array[int]): indexes of TUs for 5SrRNA
        - is_mRNA (array[bool]): Mask for mRNAs
        - ppi (str): ID of PPI
        - inactive_RNAP (str): ID of inactive RNAP
        - ntp_ids list[str]: IDs of ntp's (A, C, G, U)
        - variable_elongation (bool): Whether to use variable elongation.
                                      False by default.
        - make_elongation_rates: Function to make elongation rates, of the form:
                                 lambda random, rates, timestep, variable: rates
    """


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
        super().__init__(initial_parameters)

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


        # TODO: Remove request code once migration is complete
        self.request_on = False

        self.stop=False

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

            'listeners': {
                'transcript_elongation_listener': {
                    'countNTPsUsed': {
                        '_default': 0,
                        '_updater': 'set',
                        '_emit': True
                    },
                    'countRnaSynthesized': {
                        '_default': np.zeros(len(self.rnaIds)),
                        '_updater': 'set',
                        '_emit': True
                    }
                    },
                'growth_limits': {
                    'ntpUsed': {
                        '_default': np.zeros(len(self.ntp_ids)),
                        '_updater': 'set',
                        '_emit': True
                    }
                },
                'rnap_data': {
                    'actualElongations': {
                        '_default': 0,
                        '_updater': 'set',
                        '_emit': True},
                    'didTerminate': {
                        '_default': 0,
                        '_updater': 'set',
                        '_emit': True},
                    'terminationLoss': {
                        '_default': 0,
                        '_updater': 'set',
                        '_emit': True}}
            }}

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
            # TODO (Eran): replace with custom updater that zeros if not given update
            return {
                       'listeners': {
                           'transcript_elongation_listener': {
                               'countNTPsUsed' : 0,
                               'countRnaSynthesized': np.zeros(len(self.rnaIds))
                           },
                           'growth_limits': {
                               'ntpUsed' : np.zeros(len(self.ntp_ids))
                           },
                           'rnap_data': {
                              'actualElongations' : 0,
                              'didTerminate' : 0,
                              'terminationLoss' : 0
                           }
                       }
            }

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
            sequences[sequences != polymerize.PAD_VALUE], minlength=4)

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

        ntpCounts = array_from(states['ntps']) #TODO: redundant? same as ntpsTotal?

        if self.request_on:  # Equivalent to calculateRequest in wcEcoli
            pass
        else:
            # retrieve states
            pass

        # Start of evolveState equivalent ==============================================================================

        # Determine sequences of RNAs that should be elongated
        is_partial_transcript = np.logical_not(is_full_transcript) # redundant
        partial_transcript_indexes = np.where(is_partial_transcript)[0]
        TU_index_partial_RNAs = TU_index_all_RNAs[is_partial_transcript]
        length_partial_RNAs = length_all_RNAs[is_partial_transcript]
        is_mRNA_partial_RNAs = is_mRNA_all_RNAs[is_partial_transcript]
        RNAP_index_partial_RNAs = RNAP_index_all_RNAs[is_partial_transcript]

        sequences = buildSequences(
            self.rnaSequences,
            TU_index_partial_RNAs,
            length_partial_RNAs,
            self.elongation_rates) # redundant?

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
            n_elongations - n_initialized,  # ppi
            n_terminated])                  # inactve RNAPs

        # TODO: make sure everything here is represented above, delete
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
            "countNTPsUsed": n_elongations}
        update['listeners']['growth_limits'] = {
            "ntpUsed": ntps_used }#if ntps_used else np.zeros(4)}
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

def test_transcript_elongation():
    def make_elongation_rates(random, base, time_step, variable_elongation=False):
        size = 9  # number of TUs
        lengths = time_step * np.full(size, base, dtype=np.int64)
        lengths = stochasticRound(random, lengths) if random else np.round(lengths)

        return lengths.astype(np.int64)

    test_config = TranscriptElongation.defaults
    test_config['max_time_step'] = 2.0

    with open('data/elongation_sequences.npy', 'rb') as f:
        sequences = np.load(f)

    test_config = {
        'max_time_step': 2.0,
        'rnaPolymeraseElongationRateDict': {'minimal' : 49.24 * units.nt / units.s},
        'rnaIds': ['16S rRNA', '23S rRNA', '5S rRNA', 'mRNA'],
        'rnaLengths': np.array([1542, 2905, 120, 1080]),
        'rnaSequences': sequences,
        'ntWeights': np.array([5.44990582e-07, 5.05094471e-07, 5.71557547e-07, 5.06728441e-07]),
        'endWeight': np.array([2.90509649e-07]),
        'replichore_lengths': np.array([2322985, 2316690]),
        'idx_16S_rRNA': np.array([0]),
        'idx_23S_rRNA': np.array([1]),
        'idx_5S_rRNA': np.array([2]),
        'is_mRNA': np.array([False, False, False, True]),
        'ppi': 'PPI[c]',
        'inactive_RNAP': 'APORNAP-CPLX[c]',
        'ntp_ids': ['ATP[c]', 'CTP[c]', 'GTP[c]', 'UTP[c]'],
        'variable_elongation': False,
        'make_elongation_rates': make_elongation_rates,
        'seed': 0}


    transcript_elongation = TranscriptElongation(test_config)

    initial_state = {
        'environment': {'media_id': 'minimal'},

        'RNAs': {str(i) : {'unique_index': i,
                           'TU_index': i,
                           'transcript_length': 0,
                           'is_mRNA': test_config['is_mRNA'][i],
                           'is_full_transcript': False,
                           'can_translate': True,
                           'RNAP_index': i}
                 for i in range(len(test_config['rnaIds']))},

        'active_RNAPs': {str(i) : {'unique_index': i,
                                   'domain_index': 2,
                                   'coordinates': i * 1000,
                                   'direction': True}
                         for i in range(4)},

        'bulk_RNAs': {'16S rRNA' : 0,
                      '23S rRNA' : 0,
                      '5S rRNA' : 0,
                      'mRNA' : 0
                      },
        'ntps': {'ATP[c]': 6178058, 'CTP[c]': 1152211, 'GTP[c]': 1369694, 'UTP[c]': 3024874},
        'molecules': {'PPI[c]': 320771, 'APORNAP-CPLX[c]': 2768}
    }

    settings = {
        'total_time': 100,
        'initial_state': initial_state}

    data = simulate_process(transcript_elongation, settings)

    plots(data)
    assertions(test_config, data)

    # Test running out of ntps

    initial_state['ntps'] = {'ATP[c]': 100, 'CTP[c]': 100, 'GTP[c]': 100, 'UTP[c]': 100}

    data = simulate_process(transcript_elongation, settings)

    plots(data, "transcript_elongation_toymodel_100_ntps.png")
    assertions(test_config, data)

    # Test no ntps

    initial_state['ntps'] = {'ATP[c]': 0, 'CTP[c]': 0, 'GTP[c]': 0, 'UTP[c]': 0}

    data = simulate_process(transcript_elongation, settings)

    plots(data, "transcript_elongation_toymodel_no_ntps.png")
    assertions(test_config, data)


def plots(actual_update, filename="transcript_elongation_toymodel.png"):
    import matplotlib.pyplot as plt

    # unpack update
    rnas_synthesized = actual_update['listeners']['transcript_elongation_listener']['countRnaSynthesized']
    ntps_used = actual_update['listeners']['growth_limits']['ntpUsed']
    total_ntps_used = actual_update['listeners']['transcript_elongation_listener']['countNTPsUsed']

    ntps = actual_update['ntps']

    plt.figure()

    plt.subplot(2, 1, 1)
    plt.plot(range(len(rnas_synthesized)), rnas_synthesized)
    plt.xlabel("TU")
    plt.ylabel("Count")
    plt.title("Counts synthesized")

    plt.subplot(2, 1, 2)
    t = np.array(actual_update['time'])
    width = 0.25
    for i, ntp in enumerate(np.array(ntps_used).transpose()):
        plt.bar(t + (i-2) * width, ntp, width, label=str(i))
    plt.ylabel('Count')
    plt.title('NTP Counts Used')
    plt.legend()

    plt.subplots_adjust(hspace=0.5)
    plt.gcf().set_size_inches(10, 6)
    plt.savefig(f"out/migration/{filename}")


def assertions(config, actual_update):
    # unpack update
    trans_lengths = [r['transcript_length'] for r in actual_update["RNAs"].values()]
    rnas_synthesized = actual_update['listeners']['transcript_elongation_listener']['countRnaSynthesized']
    bulk_16SrRNA = actual_update['bulk_RNAs']['16S rRNA']
    bulk_5SrRNA = actual_update['bulk_RNAs']['5S rRNA']
    bulk_23SrRNA = actual_update['bulk_RNAs']['23S rRNA']
    bulk_mRNA = actual_update['bulk_RNAs']['mRNA']

    ntps_used = actual_update['listeners']['growth_limits']['ntpUsed']
    total_ntps_used = actual_update['listeners']['transcript_elongation_listener']['countNTPsUsed']
    ntps = actual_update['ntps']

    RNAP_coordinates = [v['coordinates'] for v in actual_update['active_RNAPs'].values()]
    RNAP_elongations = actual_update['listeners']['rnap_data']['actualElongations']
    terminations = actual_update['listeners']['rnap_data']['didTerminate']

    ppi = actual_update['molecules']['PPI[c]']
    inactive_RNAP = actual_update['molecules']['APORNAP-CPLX[c]']


    # transcript lengths are monotonically increasing
    assert np.all(list(map(monotonically_increasing, trans_lengths)))

    # RNAP positions are monotonically increasing
    assert np.all(list(map(monotonically_increasing, RNAP_coordinates)))

    # RNAP positions match transcript lengths?
    RNAP_coordinates_realigned = [np.array(v) - v[0] for v in RNAP_coordinates]
    for t_length, expect, mRNA in zip(trans_lengths, RNAP_coordinates_realigned, config['is_mRNA']):
        # truncate lengths arrays of terminated mRNAs, since these stay
        # in the unique molecules pool after termination
        if mRNA:
            t_length = t_length[:len(expect)]
        np.testing.assert_array_equal(t_length, expect)

    # bulk RNAs monotonically increasing?
    assert monotonically_increasing(bulk_16SrRNA)
    assert monotonically_increasing(bulk_5SrRNA)
    assert monotonically_increasing(bulk_23SrRNA)
    assert monotonically_increasing(bulk_mRNA)

    # Change in PPI matches #elongations - #initiations
    d_ppi = np.array(ppi)[1:] - ppi[:-1]
    expected_d_ppi = RNAP_elongations[1:]
    # Hacky way to find number of initiations
    # Assuming initiations only happen in the first time step (valid for toy model,
    # where there is no initiation process)
    expected_d_ppi[0] -= np.sum((np.array([t[0] for t in trans_lengths]) == 0) & (np.array([t[1] for t in trans_lengths]) > 0))

    np.testing.assert_array_equal(d_ppi, expected_d_ppi)

    # Change in APORNAP-CPLX matches terminations
    d_inactive_RNAP = np.array(inactive_RNAP[1:]) - inactive_RNAP[:-1]
    np.testing.assert_array_equal(d_inactive_RNAP, terminations[1:])

    # RNAP elongations matches total ntps used at each timestep
    np.testing.assert_array_equal(RNAP_elongations, [sum(v) for v in ntps_used])

    # terminations match rnas_synthesized
    assert all(np.array([sum(v) for v in rnas_synthesized]) == terminations)
    d_16SrRNA = np.array(bulk_16SrRNA)[1:] - bulk_16SrRNA[:-1]
    d_5SrRNA = np.array(bulk_5SrRNA)[1:] - bulk_5SrRNA[:-1]
    d_23SrRNA = np.array(bulk_23SrRNA)[1:] - bulk_23SrRNA[:-1]
    np.testing.assert_array_equal(d_16SrRNA, np.array(rnas_synthesized)[:, config['idx_16S_rRNA']].transpose()[0, 1:])
    np.testing.assert_array_equal(d_5SrRNA, np.array(rnas_synthesized)[:, config['idx_5S_rRNA']].transpose()[0, 1:])
    np.testing.assert_array_equal(d_23SrRNA, np.array(rnas_synthesized)[:, config['idx_23S_rRNA']].transpose()[0, 1:])

    # bulk NTP counts decrease by numbers used
    ntps_arr = np.array([v for v in ntps.values()])
    d_ntps = ntps_arr[:, 1:] - ntps_arr[:, :-1]

    np.testing.assert_array_equal(d_ntps, -np.array(ntps_used[1:]).transpose())

    # total NTPS used matches sum of ntps_used,
    # total of each type of NTP used matches rna sequences
    assert np.all(np.sum(ntps_used, axis=1) == np.array(total_ntps_used))
    actual = np.sum(ntps_used, axis=0)
    n_term = np.sum(rnas_synthesized, axis=0)
    sequence_ntps = np.array([[sum(seq==0), sum(seq==1), sum(seq==2), sum(seq==3)]
                              for seq in config['rnaSequences']])
    expect = np.array([np.array(seq) * n
                       for seq, n in zip(sequence_ntps, n_term)
                       ]).sum(axis=0)
    partial = np.array([0 < t_length[-1] < final_length   # length > 0 and not completed
                        for t_length, final_length
                        in zip(trans_lengths, config['rnaLengths'])])

    for a, e, is_partial in zip(actual, expect, partial):
        if is_partial:
            assert a >= e
        else:
            assert a == e

    #import ipdb; ipdb.set_trace()
    #np.testing.assert_array_equal(actual, expect)


if __name__ == "__main__":
    test_transcript_elongation()
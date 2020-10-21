'''
TranscriptInitiation

Transcription initiation sub-model.

TODO:
- use transcription units instead of single genes
- match sigma factors to promoters
'''

from __future__ import absolute_import, division, print_function

import numpy as np
import scipy.sparse
from typing import cast

from six.moves import zip

from vivarium.core.process import Process
from vivarium.core.composition import simulate_process_in_experiment

from ecoli.library.schema import arrays_from, arrays_to, add_elements, listener_schema, bulk_schema

from wholecell.utils import units


class TranscriptInitiation(Process):
    name = 'ecoli-transcript-initiation'

    defaults = {
        'fracActiveRnapDict': {},
        'rnaLengths': np.array([]),
        'rnaPolymeraseElongationRateDict': {},
        'variable_elongation': False,
        'make_elongation_rates': lambda random, rate, timestep, variable: np.array([]),
        'basal_prob': np.array([]),
        'delta_prob': {'deltaI': [], 'deltaJ': [], 'deltaV': []},
        'perturbations': {},
        'rna_data': {},
        'shuffleIdxs': None,

        'idx_16SrRNA': np.array([]),
        'idx_23SrRNA': np.array([]),
        'idx_5SrRNA': np.array([]),
        'idx_rRNA': np.array([]),
        'idx_mRNA': np.array([]),
        'idx_tRNA': np.array([]),
        'idx_rprotein': np.array([]),
        'idx_rnap': np.array([]),
        'rnaSynthProbFractions': {},
        'rnaSynthProbRProtein': {},
        'rnaSynthProbRnaPolymerase': {},
        'replication_coordinate': np.array([]),
        'transcription_direction': np.array([]),
        'n_avogadro': 6.02214076e+23 / units.mol,
        'cell_density': 1100 * units.g / units.L,
        'ppgpp': 'ppGpp',
        'inactive_RNAP': 'APORNAP-CPLX[c]',
        'synth_prob': lambda concentration, copy: 0.0,
        'copy_number': lambda x: x,
        'ppgpp_regulation': False,
        'seed': 0}

    # Constructor
    def __init__(self, initial_parameters):
        super(TranscriptInitiation, self).__init__(initial_parameters)

        # Load parameters
        self.fracActiveRnapDict = self.parameters['fracActiveRnapDict']
        self.rnaLengths = self.parameters['rnaLengths']
        self.rnaPolymeraseElongationRateDict = self.parameters['rnaPolymeraseElongationRateDict']
        self.variable_elongation = self.parameters['variable_elongation']
        self.make_elongation_rates = self.parameters['make_elongation_rates']

        # Initialize matrices used to calculate synthesis probabilities
        self.basal_prob = self.parameters['basal_prob']
        self.n_TUs = len(self.basal_prob)
        self.delta_prob = self.parameters['delta_prob']
        self.delta_prob_matrix = scipy.sparse.csr_matrix(
            (self.delta_prob['deltaV'],
            (self.delta_prob['deltaI'], self.delta_prob['deltaJ'])),
            shape=self.delta_prob['shape']
            ).toarray()

        # Determine changes from genetic perturbations
        self.genetic_perturbations = {}
        self.perturbations = self.parameters['perturbations']
        self.rna_data = self.parameters['rna_data']

        if len(self.perturbations) > 0:
            probability_indexes = [
                (index, self.perturbations[rna_data['id']])
                    for index, rna_data in enumerate(self.rna_data)
                    if rna_data['id'] in self.perturbations]

            self.genetic_perturbations = {
                'fixedRnaIdxs': [pair[0] for pair in probability_indexes],
                'fixedSynthProbs': [pair[1] for pair in probability_indexes]}

        # If initiationShuffleIdxs does not exist, set value to None
        self.shuffleIdxs = self.parameters['shuffleIdxs']

        # ID Groups
        self.idx_16SrRNA = self.parameters['idx_16SrRNA']
        self.idx_23SrRNA = self.parameters['idx_23SrRNA']
        self.idx_5SrRNA = self.parameters['idx_5SrRNA']
        self.idx_rRNA = self.parameters['idx_rRNA']
        self.idx_mRNA = self.parameters['idx_mRNA']
        self.idx_tRNA = self.parameters['idx_tRNA']
        self.idx_rprotein = self.parameters['idx_rprotein']
        self.idx_rnap = self.parameters['idx_rnap']

        # Synthesis probabilities for different categories of genes
        self.rnaSynthProbFractions = self.parameters['rnaSynthProbFractions']
        self.rnaSynthProbRProtein = self.parameters['rnaSynthProbRProtein']
        self.rnaSynthProbRnaPolymerase = self.parameters['rnaSynthProbRnaPolymerase']

        # Coordinates and transcription directions of transcription units
        self.replication_coordinate = self.parameters['replication_coordinate']
        self.transcription_direction = self.parameters['transcription_direction']

        self.inactive_RNAP = self.parameters['inactive_RNAP']

        # ppGpp control related
        self.n_avogadro = self.parameters['n_avogadro']
        self.cell_density = self.parameters['cell_density']
        self.ppgpp = self.parameters['ppgpp']
        self.synth_prob = self.parameters['synth_prob']
        self.copy_number = self.parameters['copy_number']
        self.ppgpp_regulation = self.parameters['ppgpp_regulation']

        self.seed = self.parameters['seed']
        self.random_state = np.random.RandomState(seed = self.seed)

        self.rnap_index = 60000

    def ports_schema(self):
        return {
            'environment': {
                'media_id': {
                    '_default': '',
                    '_updater': 'set'}},

            'molecules': bulk_schema([self.inactive_RNAP, self.ppgpp]),

            'full_chromosomes': {
                '*': {
                    'unique_index': {'_default': 0}}},

            'promoters': {
                '*': {
                    'TU_index': {'_default': 0},
                    'coordinates': {'_default': 0},
                    'domain_index': {'_default': 0},
                    'bound_TF': {'_default': 0}}},

            'RNAs': {
                '*': {
                    'unique_index': {'_default': 0, '_updater': 'set'},
                    'TU_index': {'_default': 0, '_updater': 'set'},
                    'transcript_length': {'_default': 0, '_updater': 'set', '_emit': True},
                    'is_mRNA': {'_default': 0, '_updater': 'set'},
                    'is_full_transcript': {'_default': 0, '_updater': 'set'},
                    'can_translate': {'_default': 0, '_updater': 'set'},
                    'RNAP_index': {'_default': 0, '_updater': 'set'}}},

            'active_RNAPs': {
                '*': {
                    'unique_index': {'_default': 0, '_updater': 'set'},
                    'domain_index': {'_default': 0, '_updater': 'set'},
                    'coordinates': {'_default': 0, '_updater': 'set', '_emit': True},
                    'direction': {'_default': 0, '_updater': 'set'}}},

            'listeners': {
                'mass': {
                    'cell_mass': {'_default': 0.0},
                    'dry_mass': {'_default': 0.0}},
                'rna_synth_prob': {
                    'rna_synth_prob': {
                        '_default': 0.0, '_updater': 'set', '_emit': True}},
                'ribosome_data': listener_schema({
                    'rrn16S_produced': 0,
                    'rrn23S_produced': 0,
                    'rrn5S_produced': 0,
                    'rrn16S_init_prob': 0,
                    'rrn23S_init_prob': 0,
                    'rrn5S_init_prob': 0,
                    'total_rna_init': 0}),
                'rnap_data': listener_schema({
                    'didInitialize': 0,
                    'rnaInitEvent': 0})}}

    def next_update(self, timestep, states):
        current_media_id = states['environment']['media_id']
        n_promoters = len(states['promoters'])

        # Get attributes of promoters
        TU_index, coordinates_promoters, domain_index_promoters, bound_TF = arrays_from(
            states['promoters'].values(),
            ['TU_index', 'coordinates', 'domain_index', 'bound_TF'])

        if len(states['full_chromosomes']) > 0:
            if self.ppgpp_regulation:
                cell_mass = states['listeners']['mass']['cell_mass'] * units.fg
                cell_volume = cell_mass / self.cell_density
                counts_to_molar = 1 / (self.n_avogadro * cell_volume)
                ppgpp_conc = states['molecules'][self.ppgpp] * counts_to_molar
                basal_prob = self.synth_prob(ppgpp_conc, self.copy_number)
            else:
                basal_prob = self.basal_prob

            # Calculate probabilities of the RNAP binding to each promoter
            self.promoter_init_probs = (basal_prob[TU_index] +
                np.multiply(self.delta_prob_matrix[TU_index, :], bound_TF).sum(axis=1))

            if len(self.genetic_perturbations) > 0:
                self._rescale_initiation_probs(
                    self.genetic_perturbations['fixedRnaIdxs'],
                    self.genetic_perturbations['fixedSynthProbs'],
                    TU_index)

            # Adjust probabilities to not be negative
            self.promoter_init_probs[self.promoter_init_probs < 0] = 0.0
            self.promoter_init_probs /= self.promoter_init_probs.sum()

            if not self.ppgpp_regulation:
                # Adjust synthesis probabilities depending on environment
                synthProbFractions = self.rnaSynthProbFractions[current_media_id]

                # Create masks for different types of RNAs
                is_mrna = np.isin(TU_index, self.idx_mRNA)
                is_trna = np.isin(TU_index, self.idx_tRNA)
                is_rrna = np.isin(TU_index, self.idx_rRNA)
                is_rprotein = np.isin(TU_index, self.idx_rprotein)
                is_rnap = np.isin(TU_index, self.idx_rnap)
                is_fixed = is_trna | is_rrna | is_rprotein | is_rnap

                # Rescale initiation probabilities based on type of RNA
                self.promoter_init_probs[is_mrna] *= synthProbFractions['mRna'] / self.promoter_init_probs[is_mrna].sum()
                self.promoter_init_probs[is_trna] *= synthProbFractions['tRna'] / self.promoter_init_probs[is_trna].sum()
                self.promoter_init_probs[is_rrna] *= synthProbFractions['rRna'] / self.promoter_init_probs[is_rrna].sum()

                # Set fixed synthesis probabilities for RProteins and RNAPs
                self._rescale_initiation_probs(
                    np.concatenate((self.idx_rprotein, self.idx_rnap)),
                    np.concatenate((
                        self.rnaSynthProbRProtein[current_media_id],
                        self.rnaSynthProbRnaPolymerase[current_media_id]
                        )),
                    TU_index)

                assert self.promoter_init_probs[is_fixed].sum() < 1.0

                # Scale remaining synthesis probabilities accordingly
                scaleTheRestBy = (1. - self.promoter_init_probs[is_fixed].sum()) / self.promoter_init_probs[~is_fixed].sum()
                self.promoter_init_probs[~is_fixed] *= scaleTheRestBy

        # If there are no chromosomes in the cell, set all probs to zero
        else:
            self.promoter_init_probs = np.zeros(n_promoters)

        self.fracActiveRnap = self.fracActiveRnapDict[current_media_id]
        self.rnaPolymeraseElongationRate = self.rnaPolymeraseElongationRateDict[current_media_id]
        self.elongation_rates = self.make_elongation_rates(
            self.random_state,
            self.rnaPolymeraseElongationRate.asNumber(units.nt / units.s),
            1,  # want elongation rate, not lengths adjusted for time step
            self.variable_elongation)

        update = {
            'listeners': {
                'rna_synth_prob': {
                    'rna_synth_prob': np.zeros(self.n_TUs)}}}

        # no synthesis if no chromosome
        if len(states['full_chromosomes']) == 0:
            return update

        # Construct matrix that maps promoters to transcription units
        TU_to_promoter = scipy.sparse.csr_matrix(
            (np.ones(n_promoters), (TU_index, np.arange(n_promoters))),
            shape = (self.n_TUs, n_promoters))

        # Compute synthesis probabilities of each transcription unit
        TU_synth_probs = TU_to_promoter.dot(self.promoter_init_probs)
        update['listeners']['rna_synth_prob']['rna_synth_prob'] = TU_synth_probs

        # Shuffle synthesis probabilities if we're running the variant that
        # calls this (In general, this should lead to a cell which does not
        # grow and divide)
        if self.shuffleIdxs is not None:
            self._rescale_initiation_probs(
                np.arange(self.n_TUs),
                TU_synth_probs[self.shuffleIdxs],
                TU_index)

        # Calculate RNA polymerases to activate based on probabilities
        self.activationProb = self._calculateActivationProb(
            timestep,
            self.fracActiveRnap,
            self.rnaLengths,
            (units.nt / units.s) * self.elongation_rates,
            TU_synth_probs)

        n_RNAPs_to_activate = np.int64(
            self.activationProb * states['molecules'][self.inactive_RNAP])

        if n_RNAPs_to_activate == 0:
            return

        #### Growth control code ####

        # Sample a multinomial distribution of initiation probabilities to
        # determine what promoters are initialized
        n_initiations = self.random_state.multinomial(
            n_RNAPs_to_activate, self.promoter_init_probs)

        # Build array of transcription unit indexes for partially transcribed
        # RNAs and domain indexes for RNAPs
        TU_index_partial_RNAs = np.repeat(TU_index, n_initiations)
        domain_index_rnap = np.repeat(domain_index_promoters, n_initiations)

        # Build arrays of starting coordinates and transcription directions
        coordinates = self.replication_coordinate[TU_index_partial_RNAs]
        direction = self.transcription_direction[TU_index_partial_RNAs]

        new_RNAPs = arrays_to(
            n_RNAPs_to_activate, {
                'unique_index': np.arange(self.rnap_index, self.rnap_index + n_RNAPs_to_activate).astype(str),
                'domain_index': domain_index_rnap,
                'coordinates': coordinates,
                'direction': direction})

        RNAP_indexes = [
            RNAP['unique_index']
            for RNAP in new_RNAPs]

        update['active_RNAPs'] = add_elements(new_RNAPs, 'unique_index')

        # Decrement counts of inactive RNAPs
        update['molecules'] = {
            'inactive_RNAPs': -n_initiations.sum()}

        # Add partially transcribed RNAs
        is_mRNA = np.isin(TU_index_partial_RNAs, self.idx_mRNA)
        new_RNAs = arrays_to(
            n_RNAPs_to_activate, {
                'unique_index': np.arange(self.rnap_index, self.rnap_index + n_RNAPs_to_activate).astype(str),
                'TU_index': TU_index_partial_RNAs,
                'transcript_length': np.zeros(cast(int, n_RNAPs_to_activate)),
                'is_mRNA': is_mRNA,
                'is_full_transcript': np.zeros(cast(int, n_RNAPs_to_activate), dtype=np.bool),
                'can_translate': is_mRNA,
                'RNAP_index': RNAP_indexes})
        update['RNAs'] = add_elements(new_RNAs, 'unique_index')

        self.rnap_index += n_RNAPs_to_activate

        # Create masks for ribosomal RNAs
        is_5Srrna = np.isin(TU_index, self.idx_5SrRNA)
        is_16Srrna = np.isin(TU_index, self.idx_16SrRNA)
        is_23Srrna = np.isin(TU_index, self.idx_23SrRNA)

        # Write outputs to listeners
        update['listeners']['ribosome_data'] = {
            'rrn16S_produced': n_initiations[is_16Srrna].sum(),
            'rrn23S_produced': n_initiations[is_23Srrna].sum(),
            'rrn5S_produced': n_initiations[is_5Srrna].sum(),

            'rrn16S_init_prob': n_initiations[is_16Srrna].sum() / float(n_RNAPs_to_activate),
            'rrn23S_init_prob': n_initiations[is_23Srrna].sum() / float(n_RNAPs_to_activate),
            'rrn5S_init_prob': n_initiations[is_5Srrna].sum() / float(n_RNAPs_to_activate),
            'total_rna_init': n_RNAPs_to_activate}

        update['listeners']['rnap_data'] = {
            'didInitialize': n_RNAPs_to_activate,
            'rnaInitEvent': TU_to_promoter.dot(n_initiations)}

        return update

    def _calculateActivationProb(self, timestep, fracActiveRnap, rnaLengths, rnaPolymeraseElongationRates, synthProb):
        """
        Calculate expected RNAP termination rate based on RNAP elongation rate
        - allTranscriptionTimes: Vector of times required to transcribe each
        transcript
        - allTranscriptionTimestepCounts: Vector of numbers of timesteps
        required to transcribe each transcript
        - averageTranscriptionTimeStepCounts: Average number of timesteps
        required to transcribe a transcript, weighted by synthesis
        probabilities of each transcript
        - expectedTerminationRate: Average number of terminations in one
        timestep for one transcript
        """
        allTranscriptionTimes = 1. / rnaPolymeraseElongationRates * rnaLengths
        timesteps = (1. / (timestep * units.s) * allTranscriptionTimes).asNumber()
        allTranscriptionTimestepCounts = np.ceil(timesteps)
        averageTranscriptionTimestepCounts = np.dot(
            synthProb, allTranscriptionTimestepCounts)
        expectedTerminationRate = 1. / averageTranscriptionTimestepCounts

        """
        Modify given fraction of active RNAPs to take into account early
        terminations in between timesteps
        - allFractionTimeInactive: Vector of probabilities an "active" RNAP
        will in effect be "inactive" because it has terminated during a
        timestep
        - averageFractionTimeInactive: Average probability of an "active" RNAP
        being in effect "inactive", weighted by synthesis probabilities
        - effectiveFracActiveRnap: New higher "goal" for fraction of active
        RNAP, considering that the "effective" fraction is lower than what the
        listener sees
        """
        allFractionTimeInactive = 1 - (
            1. / (timestep * units.s) * allTranscriptionTimes).asNumber() / allTranscriptionTimestepCounts
        averageFractionTimeInactive = np.dot(allFractionTimeInactive, synthProb)
        effectiveFracActiveRnap = fracActiveRnap * 1/(1 - averageFractionTimeInactive)

        # Return activation probability that will balance out the expected termination rate
        return effectiveFracActiveRnap * expectedTerminationRate /(1 - effectiveFracActiveRnap)


    def _rescale_initiation_probs(
            self, fixed_indexes, fixed_synth_probs, TU_index):
        """
        Rescales the initiation probabilities of each promoter such that the
        total synthesis probabilities of certain types of RNAs are fixed to
        a predetermined value. For instance, if there are two copies of
        promoters for RNA A, whose synthesis probability should be fixed to
        0.1, each promoter is given an initiation probability of 0.05.
        """
        for idx, synth_prob in zip(fixed_indexes, fixed_synth_probs):
            fixed_mask = (TU_index == idx)
            self.promoter_init_probs[fixed_mask] = synth_prob / fixed_mask.sum()

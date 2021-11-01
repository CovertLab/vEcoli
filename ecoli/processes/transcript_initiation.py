"""
=====================
Transcript Initiation
=====================

This process models the binding of RNA polymerase to each gene.
The number of RNA polymerases to activate in each time step is determined
such that the average fraction of RNA polymerases that are active throughout
the simulation matches measured fractions, which are dependent on the
cellular growth rate. This is done by assuming a steady state concentration
of active RNA polymerases.

TODO:
  - use transcription units instead of single genes
  - match sigma factors to promoters
"""

import numpy as np
import scipy.sparse
import matplotlib.pyplot as plt
from typing import cast

from six.moves import zip

from vivarium.core.composition import simulate_process

from ecoli.library.schema import (
    create_unique_indexes, arrays_from, arrays_to,
    add_elements, dict_value_schema, listener_schema,
    bulk_schema,
)

from wholecell.utils import units
from wholecell.utils.random import stochasticRound
from wholecell.utils.unit_struct_array import UnitStructArray

from ecoli.library.data_predicates import monotonically_decreasing, all_nonnegative
from scipy.stats import chisquare

from ecoli.processes.registries import topology_registry
from ecoli.processes.partition import PartitionedProcess


# Register default topology for this process, associating it with process name
NAME = 'ecoli-transcript-initiation'
TOPOLOGY = {
        "environment": ("environment",),
        "full_chromosomes": ("unique", "full_chromosome"),
        "RNAs": ("unique", "RNA"),
        "active_RNAPs": ("unique", "active_RNAP"),
        "promoters": ("unique", "promoter"),
        "molecules": ("bulk",),
        "listeners": ("listeners",)
}
topology_registry.register(NAME, TOPOLOGY)


class TranscriptInitiation(PartitionedProcess):
    """ Transcript Initiation PartitionedProcess

    defaults:
        - fracActiveRnapDict (dict): Dictionary with keys corresponding to media, values being
                                     the fraction of active RNA Polymerase (RNAP) for that media.
        - rnaLengths (1d array[int]): lengths of RNAs for each transcription unit (TU), in nucleotides
        - rnaPolymeraseElongationRateDict (dict): Dictionary with keys corresponding to media, values being
                                                  RNAP's elongation rate in that media, in nucleotides/s
        - variable_elongation (bool): Whether to add amplified elongation rates for rRNAs.
                                      False by default.
        - make_elongation_rates (func): Function for making elongation rates
                                        (see Transcription.make_elongation_rates).
                                        - parameters: random, rate, timestep, variable
                                        - returns: a numpy array
        - basal_prob (1d array[float]): Baseline probability of synthesis for every TU.
        - delta_prob (dict): Dictionary with four keys, used to create a matrix encoding the effect
                             of transcription factors (TFs) on transcription probabilities.
                             - 'deltaV' (array[float]): deltas associated with the effects of TFs on TUs,
                             - 'deltaI' (array[int]): index of the affected TU for each delta,
                             - 'deltaJ' (array[int]): index of the acting TF for each delta, and
                             - 'shape' (tuple): (m, n) = (# of TUs, # of TFs)
        - perturbations (dict): Dictionary of genetic perturbations (optional, can be empty)
        - rna_data (1d array): Structured array with an entry for each TU, where entries look like:

                    (id, deg_rate, length (nucleotides), counts_AGCU, mw (molecular weight),
                    is_mRNA, is_miscRNA, is_rRNA, is_tRNA, is_23S_rRNA, is_16S_rRNA, is_5S_rRNA,
                    is_ribosomal_protein, is_RNAP, gene_id, Km_endoRNase, replication_coordinate, direction)

                    NOTE: This array has some redundancy with other parameters - we could just
                    get rid of some parameters and pull them from this table instead

        - shuffleIdxs (1D array or None): A permutation of the TU indices, used to shuffle
                                          probabilities around if given. Can be None,
                                          in which case no shuffling is performed.

        - idx_16SrRNA (1D array): indexes of TUs for 16SrRNA, an RNA component of the ribosome
        - idx_23SrRNA (1D array): indexes of TUs for 23SrRNA, an RNA component of the ribosome
        - idx_5SrRNA (1D array):  indexes of TUs for 5SrRNA, an RNA component of the ribosome
        - idx_rRNA (1D array): indexes of TUs corresponding to rRNAs
        - idx_mRNA (1D array): indexes of TUs corresponding to mRNAs
        - idx_tRNA (1D array): indexes of TUs corresponding to tRNAs
        - idx_rprotein (1D array): indexes of TUs corresponding ribosomal proteins
        - idx_rnap (1D array): indexes of TU corresponding to RNAP
        - rnaSynthProbFractions (dict): Dictionary where keys are media types, values are sub-dictionaries with
                                        keys 'mRna', 'tRna', 'rRna', and values being probabilities of synthesis
                                        for each RNA type. These should sum to 1 (?)
        - rnaSynthProbRProtein (dict): Dictionary where keys are media types, values are
                                       arrays storing the (fixed) probability of synthesis for each
                                       rProtein TU, under that media condition.
        - rnaSynthProbRnaPolymerase (dict): Dictionary where keys are media types, values are
                                            arrays storing the (fixed) probability of synthesis for each
                                            RNAP TU, under that media condition.
        - replication_coordinate (1d array): Array with chromosome coordinates for each TU
        - transcription_direction (1d array[bool]): Array of transcription directions for each TU (T/F corresponding to which direction?)
        - n_avogadro: Avogadro's number (constant)
        - cell_density: Density of cell (constant)
        - ppgpp (str): id of ppGpp
        - inactive_RNAP (str): id of inactive RNAP
        - synth_prob (func): Function used in model of ppGpp regulation (see Transcription.synth_prob_from_ppgpp).
                             - parameters: ppGpp_concentration (mol/volume), copy_number (Callable[float, int])
                             - returns (ndarray[float]): normalized synthesis probability for each gene
        - copy_number (func): see Replication.get_average_copy_number.
                              - parameters: tau (float): expected doubling time in minutes
                                            coords (int or ndarray[int]): chromosome coordinates of genes
                              - returns: average copy number of each gene expected at a doubling time, tau
        - ppgpp_regulation (bool): Whether to include model of ppGpp regulation
        - seed (int): random seed
    """

    name = NAME
    topology = TOPOLOGY
    defaults = {
        'fracActiveRnapDict': {},
        'rnaLengths': np.array([]),
        'rnaPolymeraseElongationRateDict': {},
        'variable_elongation': False,
        'make_elongation_rates': lambda random, rate, timestep, variable: np.array([]),
        'basal_prob': np.array([]),
        'delta_prob': {'deltaI': [], 'deltaJ': [], 'deltaV': [], 'shape': tuple()},
        'get_delta_prob_matrix': None,
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

        # attenuation
        'trna_attenuation': False,
        'attenuated_rna_indices': np.array([]),
        'attenuation_adjustments': np.array([]),

        # random seed
        'seed': 0}

    # Constructor
    def __init__(self, parameters=None):
        super().__init__(parameters)

        # Load parameters
        self.fracActiveRnapDict = self.parameters['fracActiveRnapDict']
        self.rnaLengths = self.parameters['rnaLengths']
        self.rnaPolymeraseElongationRateDict = self.parameters['rnaPolymeraseElongationRateDict']
        self.variable_elongation = self.parameters['variable_elongation']
        self.make_elongation_rates = self.parameters['make_elongation_rates']

        # Initialize matrices used to calculate synthesis probabilities
        self.basal_prob = self.parameters['basal_prob'].copy()
        self.trna_attenuation = self.parameters['trna_attenuation']
        if self.trna_attenuation:
            self.attenuated_rna_indices = self.parameters['attenuated_rna_indices']
            self.attenuation_adjustments = self.parameters['attenuation_basal_prob_adjustments']
            self.basal_prob[self.attenuated_rna_indices] += self.attenuation_adjustments

        self.n_TUs = len(self.basal_prob)
        self.delta_prob = self.parameters['delta_prob']
        if self.parameters['get_delta_prob_matrix'] is not None:
            self.delta_prob_matrix = self.parameters['get_delta_prob_matrix'](dense=True)
        else:
            # make delta_prob_matrix without adjustments
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
        self.random_state = np.random.RandomState(seed=self.seed)

        self.rnap_index = 6000000
        self.rna_index = 7000000

    def ports_schema(self):
        return {
            'environment': {
                'media_id': {
                    '_default': '',
                    '_updater': 'set'}},

            'molecules': bulk_schema([self.inactive_RNAP, self.ppgpp]),

            'full_chromosomes': dict_value_schema('full_chromosomes'),
            'promoters': dict_value_schema('promoters'),
            'RNAs': dict_value_schema('RNAs'),
            'active_RNAPs': dict_value_schema('active_RNAPs'),

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

    def calculate_request(self, timestep, states):
        # Get all inactive RNA polymerases
        requests = {'molecules': {self.inactive_RNAP: states['molecules'][self.inactive_RNAP]}}

        # Read current environment
        current_media_id = states['environment']['media_id']

        if len(states['full_chromosomes']) > 0:
            # Get attributes of promoters
            TU_index, bound_TF = arrays_from(
                states['promoters'].values(), ['TU_index', 'bound_TF'])

            if self.ppgpp_regulation:
                cell_mass = states['listeners']['mass']['cell_mass'] * units.fg
                cell_volume = cell_mass / self.cell_density
                counts_to_molar = 1 / (self.n_avogadro * cell_volume)
                ppgpp_conc = states['molecules'][self.ppgpp] * counts_to_molar
                basal_prob, _ = self.synth_prob(ppgpp_conc, self.copy_number)
                if self.trna_attenuation:
                    basal_prob[self.attenuated_rna_indices] += self.attenuation_adjustments
            else:
                basal_prob = self.basal_prob

            # Calculate probabilities of the RNAP binding to each promoter
            self.promoter_init_probs = (basal_prob[TU_index] +
                np.multiply(self.delta_prob_matrix[TU_index, :], bound_TF).sum(axis=1))

            if len(self.genetic_perturbations) > 0:
                self._rescale_initiation_probs(
                    self.genetic_perturbations["fixedRnaIdxs"],
                    self.genetic_perturbations["fixedSynthProbs"],
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
                self.promoter_init_probs[is_mrna] *= synthProbFractions["mRna"] / self.promoter_init_probs[
                    is_mrna].sum()
                self.promoter_init_probs[is_trna] *= synthProbFractions["tRna"] / self.promoter_init_probs[
                    is_trna].sum()
                self.promoter_init_probs[is_rrna] *= synthProbFractions["rRna"] / self.promoter_init_probs[
                    is_rrna].sum()

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
                scaleTheRestBy = (1. - self.promoter_init_probs[is_fixed].sum()) / self.promoter_init_probs[
                    ~is_fixed].sum()
                self.promoter_init_probs[~is_fixed] *= scaleTheRestBy

        # If there are no chromosomes in the cell, set all probs to zero
        else:
            self.promoter_init_probs = np.zeros(len(states['promoters']))

        self.fracActiveRnap = self.fracActiveRnapDict[current_media_id]
        self.rnaPolymeraseElongationRate = self.rnaPolymeraseElongationRateDict[current_media_id]
        self.elongation_rates = self.make_elongation_rates(
            self.random_state,
            self.rnaPolymeraseElongationRate.asNumber(units.nt / units.s),
            1,  # want elongation rate, not lengths adjusted for time step
            self.variable_elongation)
        return requests

    def evolve_state(self, timestep, states):
        update = {
            'listeners': {
                'rna_synth_prob': {
                    'rna_synth_prob': np.zeros(self.n_TUs)}}}

        # no synthesis if no chromosome
        if len(states['full_chromosomes']) == 0:
            return update

        # Get attributes of promoters
        TU_index, coordinates_promoters, domain_index_promoters, bound_TF = arrays_from(
            states['promoters'].values(),
            ['TU_index', 'coordinates', 'domain_index', 'bound_TF'])

        n_promoters = len(states['promoters'])
        # Construct matrix that maps promoters to transcription units
        TU_to_promoter = scipy.sparse.csr_matrix(
            (np.ones(n_promoters), (TU_index, np.arange(n_promoters))),
            shape=(self.n_TUs, n_promoters))

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
            return update

        # --- Growth control code ---

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

        # new RNAPs
        RNAP_indexes = create_unique_indexes(n_RNAPs_to_activate)
        RNAP_indexes = np.array(RNAP_indexes)
        new_RNAPs = arrays_to(
            n_RNAPs_to_activate, {
                'unique_index': RNAP_indexes,
                'domain_index': domain_index_rnap,
                'coordinates': coordinates,
                'direction': direction})

        update['active_RNAPs'] = add_elements(new_RNAPs, 'unique_index')

        # Decrement counts of inactive RNAPs
        update['molecules'] = {
            self.inactive_RNAP: -n_initiations.sum()}

        # Add partially transcribed RNAs
        is_mRNA = np.isin(TU_index_partial_RNAs, self.idx_mRNA)
        rna_indices = create_unique_indexes(n_RNAPs_to_activate)
        rna_indices = np.array(rna_indices)
        new_RNAs = arrays_to(
            n_RNAPs_to_activate, {
                'unique_index': rna_indices,
                'TU_index': TU_index_partial_RNAs,
                'transcript_length': np.zeros(cast(int, n_RNAPs_to_activate)),
                'is_mRNA': is_mRNA,
                'is_full_transcript': np.zeros(cast(int, n_RNAPs_to_activate), dtype=bool).tolist(),
                'can_translate': is_mRNA,
                'RNAP_index': RNAP_indexes})

        update['RNAs'] = add_elements(new_RNAs, 'unique_index')

        # Create masks for ribosomal RNAs
        is_5Srrna = np.isin(TU_index, self.idx_5SrRNA)
        is_16Srrna = np.isin(TU_index, self.idx_16SrRNA)
        is_23Srrna = np.isin(TU_index, self.idx_23SrRNA)

        # Write outputs to listeners
        update['listeners']['ribosome_data'] = {
            'rrn16S_produced': n_initiations[is_16Srrna].sum(),  # should go in transcript_elongation?
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
        effectiveFracActiveRnap = fracActiveRnap * 1 / (1 - averageFractionTimeInactive)

        # Return activation probability that will balance out the expected termination rate
        return effectiveFracActiveRnap * expectedTerminationRate / (1 - effectiveFracActiveRnap)

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


def test_transcript_initiation():
    def make_elongation_rates(random, base, time_step, variable_elongation=False):
        size = 9  # number of TUs
        lengths = time_step * np.full(size, base, dtype=np.int64)
        lengths = stochasticRound(random, lengths) if random else np.round(lengths)

        return lengths.astype(np.int64)

    rna_data = UnitStructArray(
        # id, deg_rate, len, counts, _ACGU mw, mRNA?, miscRNA?, rRNA?, tRNA?, 23S?, 16S?, 5S?, rProt?, RNAP?, geneid,
        # Km, coord, direction
        np.array([('16SrRNA', .002, 45, [10, 11, 12, 12], 13500, False, False, True, False, False, True, False, False,
                   False, '16SrRNA', 2e-4, 0, True),
                  ('23SrRNA', .002, 450, [100, 110, 120, 120], 135000, False, False, True, False, True, False, False,
                   False, False, '23SrRNA', 2e-4, 1000, True),
                  ('5SrRNA', .002, 600, [150, 150, 150, 150], 180000, False, False, True, False, False, False, True,
                   False, False, '5SrRNA', 2e-4, 2000, True),
                  ('rProtein', .002, 700, [175, 175, 175, 175], 210000, True, False, False, False, False, False, False,
                   True, False, 'rProtein', 2e-4, 3000, False),
                  ('RNAP', .002, 800, [200, 200, 200, 200], 240000, True, False, False, False, False, False, False,
                   False, True, 'RNAP', 2e-4, 4000, False),
                  ('miscProt', .002, 900, [225, 225, 225, 225], 270000, True, False, False, False, False, False, False,
                   False, False, 'miscProt', 2e-4, 5000, True),
                  ('tRNA1', .002, 1200, [300, 300, 300, 300], 360000, False, False, False, True, False, False, False,
                   False, False, 'tRNA1', 2e-4, 6000, False),
                  ('tRNA2', .002, 4000, [1000, 1000, 1000, 1000], 1200000, False, False, False, True, False, False,
                   False, False, False, 'tRNA2', 2e-4, 7000, False),
                  ('tRNA3', .002, 7000, [1750, 1750, 1750, 1750], 2100000, False, False, False, True, False, False,
                   False, False, False, 'tRNA3', 2e-4, 8000, True)
                  ],
                 dtype=[('id', '<U15'), ('deg_rate', '<f8'), ('length', '<i8'),
                        ('counts_ACGU', '<i8', (4,)), ('mw', '<f8'), ('is_mRNA', '?'),
                        ('is_miscRNA', '?'), ('is_rRNA', '?'), ('is_tRNA', '?'),
                        ('is_23S_rRNA', '?'), ('is_16S_rRNA', '?'), ('is_5S_rRNA', '?'),
                        ('is_ribosomal_protein', '?'), ('is_RNAP', '?'), ('gene_id', '<U8'),
                        ('Km_endoRNase', '<f8'), ('replication_coordinate', '<i8'), ('direction', '?')]),
        {'id': None, 'deg_rate': 1.0 / units.s, 'length': units.nt, 'counts_ACGU': units.nt,
         'mw': units.g / units.mol, 'is_mRNA': None, 'is_miscRNA': None, 'is_rRNA': None, 'is_tRNA': None,
         'is_23S_rRNA': None, 'is_16S_rRNA': None, 'is_5S_rRNA': None, 'is_ribosomal_protein': None,
         'is_RNAP': None, 'gene_id': None, 'Km_endoRNase': units.mol / units.L, 'replication_coordinate': None,
         'direction': None}
    )

    test_config = {
        'fracActiveRnapDict': {'minimal': 0.2},
        'rnaLengths': np.array([x[2] for x in rna_data.fullArray()]),
        'rnaPolymeraseElongationRateDict': {'minimal': 50 * units.nt / units.s},
        'make_elongation_rates': make_elongation_rates,
        'basal_prob': np.array([1e-7, 1e-7, 1e-7, 1e-7, 1e-6, 1e-6, 1e-6, 1e-5, 1e-5]),
        'delta_prob': {
            'deltaV': [-1e-3, -1e-5, -1e-6, 1e-7, 1e-6, 1e-6, 1e-5],
            'deltaI': [0, 1, 2, 3, 4, 5, 6],
            'deltaJ': [0, 1, 2, 3, 0, 1, 2],
            'shape': (9, 4)
        },
        'rna_data': rna_data,

        'idx_16SrRNA': np.array([0]),
        'idx_23SrRNA': np.array([1]),
        'idx_5SrRNA': np.array([2]),
        'idx_rRNA': np.array([0, 1, 2]),
        'idx_mRNA': np.array([3, 4, 5]),
        'idx_tRNA': np.array([6, 7, 8]),
        'idx_rprotein': np.array([3]),
        'idx_rnap': np.array([4]),
        'rnaSynthProbFractions': {'minimal': {'mRna': 0.25, 'tRna': 0.6, 'rRna': 0.15}},
        'rnaSynthProbRProtein': {'minimal': np.array([.06])},
        'rnaSynthProbRnaPolymerase': {'minimal': np.array([.04])},
        'replication_coordinate': np.array([x[-2] for x in rna_data.fullArray()]),
        'transcription_direction': np.array([x[-1] for x in rna_data.fullArray()]),
        'inactive_RNAP': 'APORNAP-CPLX[c]',

        'seed': 0,
        'time_step': 2,
        #'_schema' : {'molecules' : {'APORNAP-CPLX[c]' : {'_updater' : 'null'}}}
    }

    transcript_initiation = TranscriptInitiation(test_config)

    initial_state = {
        'environment': {'media_id': 'minimal'},
        'molecules': {'APORNAP-CPLX[c]': 1000, 'GUANOSINE-5DP-3DP[c]': 0},  # RNAP breaks at 132
        'full_chromosomes': {'0': {'unique_index': 0}},
        'promoters': {},
        'RNAs': {},
        'active_RNAPs': {},
        'listeners': {
            'mass': {
                'cell_mass': 1000,
                'dry_mass': 350}
        }
    }

    # add promoter data to initial_state
    for i in range(len(rna_data)):
        rna = rna_data[i]
        p = {
            'TU_index': i,
            'coordinates': rna['replication_coordinate'],
            'domain_index': 0,  # 0, 1, 2??
            'bound_TF': [False, False, False, False]
        }
        initial_state['promoters'][str(i)] = p

    settings = {
        'total_time': 100,
        'initial_state': initial_state}

    data_noTF = simulate_process(transcript_initiation, settings)

    # Also gather data where TFs are bound:

    # Assertions =========================================================
    # TODO(Michael):
    #  1) When no initiations occurred in a timestep, the inits_by_TU is a scalar 0
    #  2) Weird things happen when RNAP is limiting, including affecting RNA synth probs
    #     - for toy model, initial RNAPs <= 132 results in no initiation
    #     - 1000 initial RNAPs, run for t=1000s results in initiations without depletion of inactive RNAP
    #  3) rnaps in data['active_RNAPs'] seem to be missing direction, domain index data (likewise with rnas)
    #  4) Effect of TF binding is not tested in the toy model
    #     - simple test is to compare gene under up/down-regulation with data from same gene without regulation
    #  5) Test of fixed synthesis probabilties does not pass

    # Unpack data
    inactive_RNAP = np.array(data_noTF['molecules'][test_config['inactive_RNAP']])
    d_inactive_RNAP = inactive_RNAP[1:] - inactive_RNAP[:-1]
    d_active_RNAP = np.array(data_noTF['listeners']['rnap_data']['didInitialize'][1:])

    inits_by_TU = np.stack(data_noTF['listeners']['rnap_data']['rnaInitEvent'][1:])

    rnap_inits = inits_by_TU[:, test_config['idx_rnap']]
    rprotein_inits = inits_by_TU[:, test_config['idx_rprotein']]

    # Sanity checks
    assert monotonically_decreasing(inactive_RNAP), "Inactive RNAPs are not monotonically decreasing"
    assert all_nonnegative(inactive_RNAP), "Inactive RNAPs fall below zero"
    assert all_nonnegative(inits_by_TU), "Negative initiations (!?)"
    assert monotonically_decreasing(d_active_RNAP), "Change in active RNAPs is not monotonically decreasing"
    assert all_nonnegative(d_active_RNAP), "One or more timesteps has decrease in active RNAPs"
    assert np.sum(d_active_RNAP) == np.sum(inits_by_TU), "# of active RNAPs does not match number of initiations"

    # Inactive RNAPs deplete as they are activated
    np.testing.assert_array_equal(-d_inactive_RNAP,
                                  d_active_RNAP,
                                  "Depletion of inactive RNAPs does not match counts of RNAPs activated.")

    # Fixed synthesis probability TUs (RNAP, rProtein) and non-fixed TUs synthesized in correct proportion
    expected = np.array([test_config['rnaSynthProbRProtein']['minimal'][0],
                         test_config['rnaSynthProbRnaPolymerase']['minimal'][0],
                         1])
    expected[2] -= expected[0] + expected[1]
    expected *= np.sum(inits_by_TU)

    actual = np.array([np.sum(rprotein_inits),
                       np.sum(rnap_inits),
                       np.sum(inits_by_TU) - np.sum(rprotein_inits) - np.sum(rnap_inits)])

    fixed_prob_test = chisquare(actual, f_exp=expected)

    assert fixed_prob_test.pvalue > 0.05, ("Distribution of RNA types synthesized does "
                                          "not (approximately) match set points for fixed synthesis"
                                          f"(p = {fixed_prob_test.pvalue} <= 0.05)")

    # mRNA, tRNA, rRNA synthesized in correct proportion
    RNA_dist = np.array([np.sum(inits_by_TU[:, test_config['idx_mRNA']]),
                         np.sum(inits_by_TU[:, test_config['idx_tRNA']]),
                         np.sum(inits_by_TU[:, test_config['idx_rRNA']])])
    RNA_dist /= sum(RNA_dist)

    RNA_synth_prob_test = chisquare(RNA_dist,
                                    [v for v in test_config['rnaSynthProbFractions']['minimal'].values()])

    assert (RNA_synth_prob_test.pvalue > 0.05), ("Distribution of RNA types synthesized does"
                                                 "not (approximately) match values set by media")

    return test_config, data_noTF


def run_plot(config, data):
    N = len(data['time'])
    timestep = config['time_step']
    inits_by_TU = np.stack(data['listeners']['rnap_data']['rnaInitEvent'][1:])
    synth_probs = np.array(data['listeners']['rna_synth_prob']['rna_synth_prob'][1:])

    # plot synthesis probabilities over time
    plt.subplot(2, 2, 1)
    prev = np.zeros(N - 1)
    for TU in range(synth_probs.shape[1]):
        plt.bar(data['time'][1:], synth_probs[:, TU], bottom=prev, width=timestep)
        prev += synth_probs[:, TU]
    plt.xlabel('Time (s)')
    plt.ylabel('Probability of Synthesis')
    plt.title('Theoretical Synthesis Probabilities over Time')

    # plot actual probability of synthesis for each RNA
    plt.subplot(2, 2, 2)
    probs = np.sum(inits_by_TU, axis=0) / np.sum(inits_by_TU)
    prev = 0
    for i in range(len(probs)):
        prob = probs[i]
        plt.bar([0], [prob], bottom=prev, width=1)
        plt.text(i / len(probs) - 0.5, prev + prob/2, config['rna_data'][i][0])
        prev += prob
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.ylabel("Probability")
    plt.title("Actual Probability of Synthesis by TU")

    # plot which RNAs are transcribed
    plt.subplot(2, 2, 3)

    prev = np.zeros(N - 1)
    for TU in range(inits_by_TU.shape[1]):
        plt.bar(data['time'][1:], inits_by_TU[:, TU], bottom=prev, width=timestep)
        prev += inits_by_TU[:, TU]
    plt.xlabel("Time (s)")
    plt.ylabel("Transcripts")
    plt.title("Transcripts over time, for all TUs")

    # plot which RNAs are transcribed, grouped by mRNA/tRNA/rRNA
    plt.subplot(2, 2, 4)

    grouped_inits = np.concatenate([[np.sum(inits_by_TU[:, config['idx_tRNA']], axis=1)],
                                    [np.sum(inits_by_TU[:, config['idx_mRNA']], axis=1)],
                                    [np.sum(inits_by_TU[:, config['idx_rRNA']], axis=1)]]).T

    prev = np.zeros(N - 1)
    for TU in range(grouped_inits.shape[1]):
        plt.bar(data['time'][1:], grouped_inits[:, TU], bottom=prev, width=timestep)
        prev += grouped_inits[:, TU]
    plt.xlabel("Time (s)")
    plt.ylabel("Transcripts")
    plt.title("Transcripts over time by RNA type")
    plt.legend(["tRNA", "mRNA", "rRNA"], bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    plt.subplots_adjust(hspace=0.5, wspace=0.25)
    plt.gcf().set_size_inches(10, 6)
    plt.savefig("out/migration/transcript_initiation_toy_model.png")


def main():
    config, data = test_transcript_initiation()
    run_plot(config, data)


if __name__ == "__main__":
    main()

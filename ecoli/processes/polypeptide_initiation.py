"""
PolypeptideInitiation

Polypeptide initiation sub-model.
"""

from typing import cast

import numpy as np

from vivarium.core.process import Process
from vivarium.core.composition import simulate_process

from ecoli.library.schema import arrays_from, arrays_to, add_elements

from wholecell.utils import units
from wholecell.utils.fitting import normalize
from six.moves import zip


class PolypeptideInitiation(Process):
    name = 'ecoli-polypeptide-initiation'

    defaults = {
        'protein_lengths': [],
        'translation_efficiencies': [],
        'active_ribosome_fraction': {},
        'elongation_rates': {},
        'variable_elongation': False,
        'make_elongation_rates': lambda x: [],
        'protein_index_to_TU_index': [],
        'all_TU_ids': [],
        'all_mRNA_ids': [],
        'ribosome30S': 'ribosome30S',
        'ribosome50S': 'ribosome50S',
        'seed': 0,
        'shuffle_indexes': None}

    def __init__(self, parameters=None):
        super().__init__(parameters)

        # Load parameters
        self.protein_lengths = self.parameters['protein_lengths']
        self.translation_efficiencies = self.parameters['translation_efficiencies']
        self.active_ribosome_fraction = self.parameters['active_ribosome_fraction']
        self.elongation_rates = self.parameters['elongation_rates']
        self.variable_elongation = self.parameters['variable_elongation']
        self.make_elongation_rates = self.parameters['make_elongation_rates']
        
        # Get indexes from proteins to transcription units
        self.protein_index_to_TU_index = self.parameters['protein_index_to_TU_index']
        
        # Build matrix to convert transcription unit counts to mRNA counts
        self.all_TU_ids = self.parameters['all_TU_ids']
        self.all_mRNA_ids = self.parameters['all_mRNA_ids']
        self.n_TUs = len(self.all_TU_ids)
        self.n_mRNAs = len(self.all_mRNA_ids)

        self.TU_counts_to_mRNA_counts = np.zeros(
            (self.n_mRNAs, self.n_TUs), dtype=np.float64)

        self.TU_id_to_index = {TU_id: i for i, TU_id in enumerate(self.all_TU_ids)}
        for i, mRNA_id in enumerate(self.all_mRNA_ids):
            self.TU_counts_to_mRNA_counts[i, self.TU_id_to_index[mRNA_id]] = 1

        # Determine changes from parameter shuffling variant
        self.shuffle_indexes = self.parameters['shuffle_indexes']
        if self.shuffle_indexes:
            self.translation_efficiencies = self.translation_efficiencies[self.shuffle_indexes]

        self.ribosome30S = self.parameters['ribosome30S']
        self.ribosome50S = self.parameters['ribosome50S']

        self.seed = self.parameters['seed']
        self.random_state = np.random.RandomState(seed = self.seed)
        self.ribosome_index = 30000

        self.empty_update = {
            'listeners': {
                'ribosome_data': {
                    'ribosomes_initialized': 0,
                    'prob_translation_per_transcript': 0.0}}}

    def ports_schema(self):
        return {
            'environment': {
                'media_id': {
                    '_default': '',
                    '_updater': 'set'}},
            'listeners': {
                'ribosome_data': {
                    'ribosomes_initialized': {
                        '_default': 0,
                        '_updater': 'set',
                        '_emit': True},
                    'prob_translation_per_transcript': {
                        '_default': [],
                        '_updater': 'set',
                        '_emit': True},
                    'effective_elongation_rate': {
                        '_default': 0.0,
                        '_updater': 'set',
                        '_emit': True}}},
            'active_ribosome': {
                '*': {
                    'unique_index': {'_default': 0},
                    'protein_index': {'_default': 0},
                    'peptide_length': {'_default': 0, '_emit': True},
                    'mRNA_index': {'_default': 0},
                    'pos_on_mRNA': {'_default': 0, '_emit': True}}},
            'RNA': {
                '*': {
                    'TU_index': {'_default': 0},
                    'can_translate': {'_default': False},
                    'unique_index': {'_default': 0}}},
            'subunits': {
                self.ribosome30S: {
                    '_default': 0,
                    '_emit': True},
                self.ribosome50S: {
                    '_default': 0,
                    '_emit': True}}}
    
    def calculate_request(self, timestep, states):
        current_media_id = self['environment']['media_id']
        
        requests = {}
        requests['requested'] = {subunit: count for (subunit, count) 
                                 in states['subunits']}

        self.fracActiveRibosome = self.active_ribosome_fraction[current_media_id]

        # Read ribosome elongation rate from last timestep
        self.ribosomeElongationRate = states['listeners']['ribosome_data'][
            'effective_elongation_rate']
        # If the ribosome elongation rate is zero (which is always the case for
        # the first timestep), set ribosome elongation rate to the one in
        # dictionary
        if self.ribosomeElongationRate == 0:
            self.ribosomeElongationRate = self.elongation_rates[
                current_media_id].asNumber(units.aa / units.s)
        self.elongation_rates = self.make_elongation_rates(
            self.random_state,
            self.ribosomeElongationRate,
            1,  # want elongation rate, not lengths adjusted for time step
            self.variable_elongation)

        # Ensure rates are never zero
        self.elongation_rates = np.fmax(self.elongation_rates, 1)
        
        
    def evolve_state(self, timestep, states):
        # Calculate number of ribosomes that could potentially be initialized
        # based on counts of free 30S and 50S subunits
        inactive_ribosome_count = np.min([
            states['allocated'][self.ribosome30S],
            states['allocated'][self.ribosome50S]])

        # Get attributes of active (translatable) mRNAs
        TU_index_RNAs, can_translate, unique_index_RNAs = arrays_from(
            states['RNA'].values(),
            ['TU_index', 'can_translate', 'unique_index'])
        TU_index_active_mRNAs = TU_index_RNAs[can_translate]
        unique_index_active_mRNAs = unique_index_RNAs[can_translate]

        # Get counts of each type of active mRNA
        TU_counts = np.bincount(TU_index_active_mRNAs, minlength=self.n_TUs)
        mRNA_counts = self.TU_counts_to_mRNA_counts.dot(TU_counts)

        # Calculate initiation probabilities for ribosomes based on mRNA counts
        # and associated mRNA translational efficiencies
        protein_init_prob = normalize(
            mRNA_counts * self.translation_efficiencies)

        # Calculate actual number of ribosomes that should be activated based
        # on probabilities
        activation_prob = self.calculate_activation_prob(
            self.active_ribosome_fraction,
            self.protein_lengths,
            self.elongation_rates,
            protein_init_prob,
            timestep)

        n_ribosomes_to_activate = np.int64(activation_prob * inactive_ribosome_count)

        if n_ribosomes_to_activate == 0:
            return self.empty_update

        # Sample multinomial distribution to determine which mRNAs have full
        # 70S ribosomes initialized on them
        n_new_proteins = self.random_state.multinomial(
            n_ribosomes_to_activate,
            protein_init_prob)

        # Build attributes for active ribosomes.
        # Each ribosome is assigned a protein index for the protein that
        # corresponds to the polypeptide it will polymerize. This is done in
        # blocks of protein ids for efficiency.
        protein_indexes = np.empty(n_ribosomes_to_activate, np.int64)
        mRNA_indexes = np.empty(n_ribosomes_to_activate, np.int64)
        nonzero_count = (n_new_proteins > 0)
        start_index = 0

        for protein_index, counts in zip(
                np.arange(n_new_proteins.size)[nonzero_count],
                n_new_proteins[nonzero_count]):

            # Set protein index
            protein_indexes[start_index:start_index+counts] = protein_index

            # Get mask for active mRNA molecules that produce this protein
            mask = (TU_index_active_mRNAs == self.protein_index_to_TU_index[protein_index])
            n_mRNAs = np.count_nonzero(mask)

            # Distribute ribosomes among these mRNAs
            n_ribosomes_per_RNA = self.random_state.multinomial(
                counts, np.full(n_mRNAs, 1./n_mRNAs))

            # Get unique indexes of each mRNA
            mRNA_indexes[start_index:start_index + counts] = np.repeat(
                unique_index_active_mRNAs[mask], n_ribosomes_per_RNA)

            start_index += counts

        # Create active 70S ribosomes and assign their attributes
        new_ribosomes = arrays_to(
            n_ribosomes_to_activate, {
                'unique_index': np.arange(self.ribosome_index, self.ribosome_index + n_ribosomes_to_activate),
                'protein_index': protein_indexes,
                'peptide_length': np.zeros(cast(int, n_ribosomes_to_activate), dtype=np.int64),
                'mRNA_index': mRNA_indexes,
                'pos_on_mRNA': np.zeros(cast(int, n_ribosomes_to_activate), dtype=np.int64)})

        self.ribosome_index += n_ribosomes_to_activate
        # TODO -- this tracking of index seems messy -- can it automatically create new index?

        update = {
            'subunits': {
                self.ribosome30S: -n_new_proteins.sum(),
                self.ribosome50S: -n_new_proteins.sum()},
            'active_ribosome': add_elements(new_ribosomes, 'unique_index'), # {
                # '_add': [{
                #     'path': (ribosome['unique_index'],),
                #     'state': ribosome}
                #     for ribosome in new_ribosomes]},
            'listeners': {
                'ribosome_data': {
                    'ribosomes_initialized': n_new_proteins.sum(),
                    'prob_translation_per_transcript': protein_init_prob}}}

        return update

    def next_update(self, timestep, states):
        if not states['RNA']:
            return self.empty_update

        media_id = states['environment']['media_id']
        active_ribosome_fraction = self.active_ribosome_fraction[media_id]
        ribosome_elongation_rate = states['listeners']['ribosome_data']['effective_elongation_rate']

        # If the ribosome elongation rate is zero (which is always the case for
        # the first timestep), set ribosome elongation rate to the one in
        # dictionary
        if ribosome_elongation_rate == 0:
            ribosome_elongation_rate = self.elongation_rates[media_id].asNumber(units.aa / units.s)

        elongation_rates = self.make_elongation_rates(
            self.random_state,
            ribosome_elongation_rate,
            1,  # want elongation rate, not lengths adjusted for time step
            self.variable_elongation)

        elongation_rates = np.fmax(elongation_rates, 1)

        # Calculate number of ribosomes that could potentially be initialized
        # based on counts of free 30S and 50S subunits
        inactive_ribosome_count = np.min([
            states['subunits'][self.ribosome30S],
            states['subunits'][self.ribosome50S]])

        # Get attributes of active (translatable) mRNAs
        TU_index_RNAs, can_translate, unique_index_RNAs = arrays_from(
            states['RNA'].values(),
            ['TU_index', 'can_translate', 'unique_index'])
        TU_index_active_mRNAs = TU_index_RNAs[can_translate]
        unique_index_active_mRNAs = unique_index_RNAs[can_translate]

        # Get counts of each type of active mRNA
        TU_counts = np.bincount(TU_index_active_mRNAs, minlength=self.n_TUs)
        mRNA_counts = self.TU_counts_to_mRNA_counts.dot(TU_counts)

        # Calculate initiation probabilities for ribosomes based on mRNA counts
        # and associated mRNA translational efficiencies
        protein_init_prob = normalize(
            mRNA_counts * self.translation_efficiencies)

        # Calculate actual number of ribosomes that should be activated based
        # on probabilities
        activation_prob = self.calculate_activation_prob(
            active_ribosome_fraction,
            self.protein_lengths,
            elongation_rates,
            protein_init_prob,
            timestep)

        n_ribosomes_to_activate = np.int64(activation_prob * inactive_ribosome_count)

        if n_ribosomes_to_activate == 0:
            return self.empty_update

        # Sample multinomial distribution to determine which mRNAs have full
        # 70S ribosomes initialized on them
        n_new_proteins = self.random_state.multinomial(
            n_ribosomes_to_activate,
            protein_init_prob)

        # Build attributes for active ribosomes.
        # Each ribosome is assigned a protein index for the protein that
        # corresponds to the polypeptide it will polymerize. This is done in
        # blocks of protein ids for efficiency.
        protein_indexes = np.empty(n_ribosomes_to_activate, np.int64)
        mRNA_indexes = np.empty(n_ribosomes_to_activate, np.int64)
        nonzero_count = (n_new_proteins > 0)
        start_index = 0

        for protein_index, counts in zip(
                np.arange(n_new_proteins.size)[nonzero_count],
                n_new_proteins[nonzero_count]):

            # Set protein index
            protein_indexes[start_index:start_index+counts] = protein_index

            # Get mask for active mRNA molecules that produce this protein
            mask = (TU_index_active_mRNAs == self.protein_index_to_TU_index[protein_index])
            n_mRNAs = np.count_nonzero(mask)

            # Distribute ribosomes among these mRNAs
            n_ribosomes_per_RNA = self.random_state.multinomial(
                counts, np.full(n_mRNAs, 1./n_mRNAs))

            # Get unique indexes of each mRNA
            mRNA_indexes[start_index:start_index + counts] = np.repeat(
                unique_index_active_mRNAs[mask], n_ribosomes_per_RNA)

            start_index += counts

        # Create active 70S ribosomes and assign their attributes
        new_ribosomes = arrays_to(
            n_ribosomes_to_activate, {
                'unique_index': np.arange(self.ribosome_index, self.ribosome_index + n_ribosomes_to_activate),
                'protein_index': protein_indexes,
                'peptide_length': np.zeros(cast(int, n_ribosomes_to_activate), dtype=np.int64),
                'mRNA_index': mRNA_indexes,
                'pos_on_mRNA': np.zeros(cast(int, n_ribosomes_to_activate), dtype=np.int64)})

        self.ribosome_index += n_ribosomes_to_activate
        # TODO -- this tracking of index seems messy -- can it automatically create new index?

        update = {
            'subunits': {
                self.ribosome30S: -n_new_proteins.sum(),
                self.ribosome50S: -n_new_proteins.sum()},
            'active_ribosome': add_elements(new_ribosomes, 'unique_index'), # {
                # '_add': [{
                #     'path': (ribosome['unique_index'],),
                #     'state': ribosome}
                #     for ribosome in new_ribosomes]},
            'listeners': {
                'ribosome_data': {
                    'ribosomes_initialized': n_new_proteins.sum(),
                    'prob_translation_per_transcript': protein_init_prob}}}

        return update

    def calculate_activation_prob(
            self,
            fracActiveRibosome,
            proteinLengths,
            ribosomeElongationRates,
            proteinInitProb,
            timeStepSec):
        """
        Calculates the expected ribosome termination rate based on the ribosome
        elongation rate
        Params:
            - allTranslationTimes: Vector of times required to translate each
            protein
            - allTranslationTimestepCounts: Vector of numbers of timesteps
            required to translate each protein
            - averageTranslationTimeStepCounts: Average number of timesteps
            required to translate a protein, weighted by initiation
            probabilities
            - expectedTerminationRate: Average number of terminations in one
            timestep for one protein
        """
        allTranslationTimes = 1. / ribosomeElongationRates * proteinLengths
        allTranslationTimestepCounts = np.ceil(allTranslationTimes / timeStepSec)
        averageTranslationTimestepCounts = np.dot(allTranslationTimestepCounts, proteinInitProb)
        expectedTerminationRate = 1.0 / averageTranslationTimestepCounts

        # Modify given fraction of active ribosomes to take into account early
        # terminations in between timesteps
        # allFractionTimeInactive: Vector of probabilities an "active" ribosome
        #   will in effect be "inactive" because it has terminated during a
        #   timestep
        # averageFractionTimeInactive: Average probability of an "active"
        #   ribosome being in effect "inactive", weighted by initiation
        #   probabilities
        # effectiveFracActiveRnap: New higher "goal" for fraction of active
        #   ribosomes, considering that the "effective" fraction is lower than
        #   what the listener sees
        allFractionTimeInactive = 1 - allTranslationTimes / timeStepSec / allTranslationTimestepCounts
        averageFractionTimeInactive = np.dot(allFractionTimeInactive, proteinInitProb)
        effectiveFracActiveRibosome = fracActiveRibosome * 1 / (1 - averageFractionTimeInactive)

        # Return activation probability that will balance out the expected
        # termination rate
        activationProb = effectiveFracActiveRibosome * expectedTerminationRate / (1 - effectiveFracActiveRibosome)

        # The upper bound for the activation probability is temporarily set to
        # 1.0 to prevent negative molecule counts. This will lower the fraction
        # of active ribosomes for timesteps longer than roughly 1.8s.
        if activationProb >= 1.0:
            activationProb = 1

        return activationProb


def test_polypeptide_initiation():
    def make_elongation_rates(
            self,
            random,
            base,
            time_step,
            variable_elongation=False):
        return base

    test_config = {
        'protein_lengths': np.array([25, 9, 12, 29]),
        'translation_efficiencies': normalize(np.array([0.1, 0.2, 0.3, 0.4])),
        'active_ribosome_fraction': {'minimal': 0.1},
        'elongation_rates': {'open': 10},
        'variable_elongation': False,
        'make_elongation_rates': make_elongation_rates,
        'protein_index_to_TU_index': np.arange(4),
        'all_TU_ids': ['wRNA', 'xRNA', 'yRNA', 'zRNA'],
        'all_mRNA_ids': ['wRNA', 'xRNA', 'yRNA', 'zRNA'],
        'ribosome30S': '30S',
        'ribosome50S': '50S',
        'seed': 0}

    polypeptide_initiation = PolypeptideInitiation(test_config)

    state = {
        'environment': {
            'media_id': 'minimal'},
        'listeners': {
            'ribosome_data': {
                'effective_elongation_rate': 11}},
        'subunits': {
            '30S': 2000,
            '50S': 3000},
        'RNA': {
            0: {
                'TU_index': 0,
                'can_translate': True,
                'unique_index': 0},
            1: {
                'TU_index': 0,
                'can_translate': False,
                'unique_index': 1},
            2: {
                'TU_index': 1,
                'can_translate': True,
                'unique_index': 2},
            3: {
                'TU_index': 2,
                'can_translate': True,
                'unique_index': 3},
            4: {
                'TU_index': 2,
                'can_translate': True,
                'unique_index': 4}}}

    settings = {
        'total_time': 10,
        'initial_state': state}

    data = simulate_process(polypeptide_initiation, settings)

    print(data)


if __name__ == "__main__":
    test_polypeptide_initiation()

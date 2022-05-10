"""
======================
Polypeptide Elongation
======================

This process models the polymerization of amino acids into polypeptides
by ribosomes using an mRNA transcript as a template. Elongation terminates
once a ribosome has reached the end of an mRNA transcript. Polymerization
occurs across all ribosomes simultaneously and resources are allocated to
maximize the progress of all ribosomes within the limits of the maximum ribosome
elongation rate, available amino acids and GTP, and the length of the transcript.
"""

import numpy as np
import logging as log

# wcEcoli imports
from wholecell.utils.polymerize import buildSequences, polymerize, computeMassIncrease
from wholecell.utils.random import stochasticRound
from wholecell.utils import units

# vivarium imports
from vivarium.core.process import Process
from vivarium.core.composition import simulate_process
from vivarium.library.dict_utils import deep_merge
from vivarium.plots.simulation_output import plot_variables

# vivarium-ecoli imports
from ecoli.library.schema import bulk_schema, dict_value_schema, listener_schema, arrays_from, array_from
from ecoli.models.polypeptide_elongation_models import BaseElongationModel, MICROMOLAR_UNITS
from ecoli.states.wcecoli_state import MASSDIFFS
from ecoli.processes.registries import topology_registry
from ecoli.processes.partition import PartitionedProcess


# Register default topology for this process, associating it with process name
NAME = 'ecoli-polypeptide-elongation'
TOPOLOGY = {
    "environment": ("environment",),
    "listeners": ("listeners",),
    "active_ribosome": ("unique", "active_ribosome"),
    "molecules": ("bulk",),
    "monomers": ("bulk",),
    "amino_acids": ("bulk",),
    "ppgpp_reaction_metabolites": ("bulk",),
    "uncharged_trna": ("bulk",),
    "charged_trna": ("bulk",),
    "charging_molecules": ("bulk",),
    "synthetases": ("bulk",),
    "subunits": ("bulk",),
    "polypeptide_elongation": ("process_state", "polypeptide_elongation"),
    # Non-partitioned counts
    'molecules_total': ('bulk',),
    'amino_acids_total': ('bulk',),
    'charged_trna_total': ('bulk',),
    'uncharged_trna_total': ('bulk',)
}
topology_registry.register(NAME, TOPOLOGY)

DEFAULT_AA_NAMES = [
            'L-ALPHA-ALANINE[c]', 'ARG[c]', 'ASN[c]', 'L-ASPARTATE[c]', 'CYS[c]', 'GLT[c]', 'GLN[c]', 'GLY[c]',
            'HIS[c]', 'ILE[c]', 'LEU[c]', 'LYS[c]', 'MET[c]', 'PHE[c]', 'PRO[c]', 'SER[c]', 'THR[c]', 'TRP[c]',
            'TYR[c]', 'L-SELENOCYSTEINE[c]', 'VAL[c]']


class PolypeptideElongation(PartitionedProcess):
    """ Polypeptide Elongation PartitionedProcess

    defaults:
        proteinIds: array length n of protein names
    """

    name = NAME
    topology = TOPOLOGY
    defaults = {
        'max_time_step': 2.0,
        'n_avogadro': 6.02214076e+23 / units.mol,
        'proteinIds': np.array([]),
        'proteinLengths': np.array([]),
        'proteinSequences': np.array([[]]),
        'aaWeightsIncorporated': np.array([]),
        'endWeight': np.array([2.99146113e-08]),
        'variable_elongation': False,
        'make_elongation_rates': lambda random, rate, timestep, variable: np.array([]),
        'next_aa_pad': 1,
        'ribosomeElongationRate': 17.388824902723737,
        'translation_aa_supply': {'minimal': np.array([])},
        'import_threshold': 1e-05,
        'aa_from_trna': np.zeros(21),
        'gtpPerElongation': 4.2,
        'ppgpp_regulation': False,
        'trna_charging': False,
        'translation_supply': False,
        'mechanistic_supply': False,
        'ribosome30S': 'ribosome30S',
        'ribosome50S': 'ribosome50S',
        'amino_acids': DEFAULT_AA_NAMES,
        'basal_elongation_rate': 22.0,
        'ribosomeElongationRateDict': {
            'minimal': 17.388824902723737 * units.aa / units.s},
        'uncharged_trna_names': np.array([]),
        'aaNames': DEFAULT_AA_NAMES,
        'aa_enzymes': [],
        'proton': 'PROTON',
        'water': 'H2O',
        'cellDensity': 1100 * units.g / units.L,
        'elongation_max': 22 * units.aa / units.s,
        'aa_from_synthetase': np.array([[]]),
        'charging_stoich_matrix': np.array([[]]),
        'charged_trna_names': [],
        'charging_molecule_names': [],
        'synthetase_names': [],
        'ppgpp_reaction_names': [],
        'ppgpp_reaction_metabolites': [],
        'ppgpp_reaction_stoich': np.array([[]]),
        'ppgpp_synthesis_reaction': 'GDPPYPHOSKIN-RXN',
        'ppgpp_degradation_reaction': 'PPGPPSYN-RXN',
        'rela': 'RELA',
        'spot': 'SPOT',
        'ppgpp': 'ppGpp',
        'kS': 100.0,
        'KMtf': 1.0,
        'KMaa': 100.0,
        'krta': 1.0,
        'krtf': 500.0,
        'KD_RelA': 0.26,
        'k_RelA': 75.0,
        'k_SpoT_syn': 2.6,
        'k_SpoT_deg': 0.23,
        'KI_SpoT': 20.0,
        'aa_supply_scaling': lambda aa_conc, aa_in_media: 0,
        'seed': 0,
        'submass_indexes': MASSDIFFS,
    }

    def __init__(self, parameters=None):
        super().__init__(parameters)

        self.max_time_step = self.parameters['max_time_step']

        # Load parameters
        self.n_avogadro = self.parameters['n_avogadro']
        self.proteinIds = self.parameters['proteinIds']
        self.protein_lengths = self.parameters['proteinLengths']
        self.proteinSequences = self.parameters['proteinSequences']
        self.aaWeightsIncorporated = self.parameters['aaWeightsIncorporated']
        self.endWeight = self.parameters['endWeight']
        self.variable_elongation = self.parameters['variable_elongation']
        self.make_elongation_rates = self.parameters['make_elongation_rates']
        self.next_aa_pad = self.parameters['next_aa_pad']

        self.ribosome30S = self.parameters['ribosome30S']
        self.ribosome50S = self.parameters['ribosome50S']
        self.amino_acids = self.parameters['amino_acids']
        self.aaNames = self.parameters['aaNames']
        self.aa_enzymes = self.parameters['aa_enzymes']

        self.ribosomeElongationRate = self.parameters['ribosomeElongationRate']

        # Amino acid supply calculations
        self.translation_aa_supply = self.parameters['translation_aa_supply']
        self.import_threshold = self.parameters['import_threshold']

        # Used for figure in publication
        self.trpAIndex = np.where(self.proteinIds == "TRYPSYN-APROTEIN[c]")[0][0]

        self.elngRateFactor = 1.

        # Data structures for charging
        self.aa_from_trna = self.parameters['aa_from_trna']

        # Set modeling method
        # TODO (Eran): bring these different elongations models back
        # if self.parameters['trna_charging']:
        #     self.elongation_model = SteadyStateElongationModel(self.parameters, self)
        # elif self.parameters['translation_supply']:
        #     self.elongation_model = TranslationSupplyElongationModel(self.parameters, self)
        # else:
        #     self.elongation_model = BaseElongationModel(self.parameters, self)
        self.elongation_model = BaseElongationModel(self.parameters, self)
        self.ppgpp_regulation = self.parameters['ppgpp_regulation']
        self.mechanistic_supply = self.parameters['mechanistic_supply']

        # Growth associated maintenance energy requirements for elongations
        self.gtpPerElongation = self.parameters['gtpPerElongation']
        # Need to account for ATP hydrolysis for charging that has been
        # removed from measured GAM (ATP -> AMP is 2 hydrolysis reactions)
        # if charging reactions are not explicitly modeled

        if not self.parameters['trna_charging']:
            self.gtpPerElongation += 2

        # Variable for metabolism to read to consume required energy
        self.gtp_to_hydrolyze = 0

        # basic molecule names
        self.proton = self.parameters['proton']
        self.water = self.parameters['water']
        self.rela = self.parameters['rela']
        self.spot = self.parameters['spot']
        self.ppgpp = self.parameters['ppgpp']

        # Names of molecules associated with tRNA charging
        self.ppgpp_reaction_metabolites = self.parameters['ppgpp_reaction_metabolites']
        self.uncharged_trna_names = self.parameters['uncharged_trna_names']
        self.charged_trna_names = self.parameters['charged_trna_names']
        self.charging_molecule_names = self.parameters['charging_molecule_names']
        self.synthetase_names = self.parameters['synthetase_names']

        # Index of protein submass in submass vector
        self.protein_submass_idx = self.parameters['submass_indexes']['massDiff_protein']

        self.seed = self.parameters['seed']
        self.random_state = np.random.RandomState(seed = self.seed)

    def ports_schema(self):
        return {
            'environment': {
                'media_id': {
                    '_default': '',
                    '_updater': 'set'},
            },
            'listeners': {
                'mass': {
                    'cell_mass': {'_default': 0.0},
                    'dry_mass': {'_default': 0.0}},

                'growth_limits': listener_schema({
                    'fraction_trna_charged': 0,
                    'aa_pool_size': 0,
                    'aa_request_size': 0,
                    'active_ribosomes_allocated': 0,
                    'net_charged': [],
                    'aasUsed': 0,
                    'aa_supply': 0,
                    'aa_supply_enzymes': 0,
                    'aa_supply_aa_conc': 0,
                    'aa_supply_fraction': 0}),

                'ribosome_data': listener_schema({
                    'translation_supply': 0,
                    'effective_elongation_rate': 0,
                    'aaCountInSequence': 0,
                    'aaCounts': 0,
                    'actualElongations': 0,
                    'actualElongationHist': 0,
                    'elongationsNonTerminatingHist': 0,
                    'didTerminate': 0,
                    'terminationLoss': 0,
                    'numTrpATerminated': 0,
                    'processElongationRate': 0})},

            'molecules': bulk_schema([
                self.proton,
                self.water,
                self.rela,
                self.spot,
                self.ppgpp]),

            'molecules_total': bulk_schema([
                self.proton,
                self.water,
                self.rela,
                self.spot,
                self.ppgpp],
                partition=False),

            'monomers': bulk_schema(self.proteinIds),
            'amino_acids': bulk_schema(self.amino_acids),
            'aa_enzymes': bulk_schema(self.aa_enzymes),
            'ppgpp_reaction_metabolites': bulk_schema(self.ppgpp_reaction_metabolites),
            'uncharged_trna': bulk_schema(self.uncharged_trna_names),
            'charged_trna': bulk_schema(self.charged_trna_names),
            'charging_molecules': bulk_schema(self.charging_molecule_names),
            'synthetases': bulk_schema(self.synthetase_names),

            'amino_acids_total': bulk_schema(self.amino_acids, partition=False),
            'uncharged_trna_total': bulk_schema(self.uncharged_trna_names, partition=False),
            'charged_trna_total': bulk_schema(self.charged_trna_names, partition=False),

            'active_ribosome': dict_value_schema('active_ribosome'),

            'subunits': bulk_schema([self.ribosome30S, self.ribosome50S]),

            'polypeptide_elongation': {
                'aa_count_diff': {
                    '_default': {},
                    '_updater': 'set',
                    '_emit': True},
                'gtp_to_hydrolyze': {
                    '_default': 0,
                    '_updater': 'set',
                    '_emit': True}}}

    def calculate_request(self, timestep, states):
        """
        Set ribosome elongation rate based on simulation medium environment and elongation rate factor
        which is used to create single-cell variability in growth rate
        The maximum number of amino acids that can be elongated in a single timestep is set to 22
        intentionally as the minimum number of padding values on the protein sequence matrix is set to 22.
        If timesteps longer than 1.0s are used, this feature will lead to errors in the effective ribosome
        elongation rate.
        """

        current_media_id = states['environment']['media_id']

        # MODEL SPECIFIC: get ribosome elongation rate
        self.ribosomeElongationRate = self.elongation_model.elongation_rate(current_media_id)

        # If there are no active ribosomes, return immediately
        if len(states['active_ribosome']) == 0:
            return {'listeners': {}}

        # Build sequences to request appropriate amount of amino acids to
        # polymerize for next timestep
        # The int64 dtype is important (may break otherwise)
        proteinIndexes = np.array([states['active_ribosome'][i]['protein_index']
                          for i in states['active_ribosome']], dtype = np.int64)
        peptideLengths = np.array([states['active_ribosome'][i]['peptide_length']
                          for i in states['active_ribosome']], dtype = np.int64)

        self.elongation_rates = self.make_elongation_rates(
            self.random_state,
            self.ribosomeElongationRate,
            timestep,
            self.variable_elongation)

        sequences = buildSequences(
            self.proteinSequences,
            proteinIndexes,
            peptideLengths,
            self.elongation_rates)

        sequenceHasAA = (sequences != polymerize.PAD_VALUE)
        aasInSequences = np.bincount(sequences[sequenceHasAA], minlength=21)

        # Calculate AA supply for expected doubling of protein
        dryMass = (states['listeners']['mass']['dry_mass'] * units.fg)
        translation_supply_rate = self.translation_aa_supply[current_media_id] \
            * self.elngRateFactor
        mol_aas_supplied = translation_supply_rate * dryMass * timestep * units.s
        self.aa_supply = units.strip_empty_units(mol_aas_supplied * self.n_avogadro)


        # MODEL SPECIFIC: Calculate AA request
        fraction_charged, aa_counts_for_translation, requests = \
            self.elongation_model.request(timestep, states, aasInSequences)

        # Write to listeners
        requests['listeners'] = {
                'ribosome_data': {},
                'growth_limits': {}}
        requests['listeners']['ribosome_data'] = {'translation_supply':
                        translation_supply_rate.asNumber()}
        requests['listeners']['growth_limits']['fraction_trna_charged'] = np.dot(fraction_charged, self.aa_from_trna)
        requests['listeners']['growth_limits']['aa_pool_size'] = array_from(states['amino_acids'])
        requests['listeners']['growth_limits']['aa_request_size'] = aa_counts_for_translation
        return requests


    def evolve_state(self, timestep, states):
        """
        Set ribosome elongation rate based on simulation medium environment and elongation rate factor
        which is used to create single-cell variability in growth rate
        The maximum number of amino acids that can be elongated in a single timestep is set to 22
        intentionally as the minimum number of padding values on the protein sequence matrix is set to 22.
        If timesteps longer than 1.0s are used, this feature will lead to errors in the effective ribosome
        elongation rate.
        """

        update = {
            'molecules': {
                self.water: 0,
            },
            'listeners': {
                'ribosome_data': {},
                'growth_limits': {}},
            'polypeptide_elongation': {}}

        # Begin wcEcoli evolveState()
        # Set value to 0 for metabolism in case of early return
        update['polypeptide_elongation']['gtp_to_hydrolyze'] = 0

        # Get number of active ribosomes
        n_active_ribosomes = len(states['active_ribosome'])
        update['listeners']['growth_limits']['active_ribosomes_allocated'] = n_active_ribosomes

        # If there are no active ribosomes, return immediately
        if n_active_ribosomes == 0:
            return update

        # Build amino acids sequences for each ribosome to polymerize
        protein_indexes, peptide_lengths, positions_on_mRNA = arrays_from(
            states['active_ribosome'].values(),
            ['protein_index', 'peptide_length', 'pos_on_mRNA'])

        all_sequences = buildSequences(
            self.proteinSequences,
            protein_indexes,
            peptide_lengths,
            self.elongation_rates + self.next_aa_pad)
        sequences = all_sequences[:, :-self.next_aa_pad].copy()

        if sequences.size == 0:
            return update

        # Calculate elongation resource capacity
        aaCountInSequence = np.bincount(sequences[(sequences != polymerize.PAD_VALUE)])
        total_aa_counts = array_from(states['amino_acids'])

        # MODEL SPECIFIC: Get amino acid counts
        aa_counts_for_translation = self.elongation_model.final_amino_acids(total_aa_counts)
        aa_counts_for_translation = aa_counts_for_translation.astype(int)

        # Using polymerization algorithm elongate each ribosome up to the limits
        # of amino acids, sequence, and GTP
        result = polymerize(
            sequences,
            aa_counts_for_translation,
            10000000,  # Set to a large number, the limit is now taken care of in metabolism
            self.random_state,
            self.elongation_rates[protein_indexes])

        sequence_elongations = result.sequenceElongation
        aas_used = result.monomerUsages
        nElongations = result.nReactions

        next_amino_acid = all_sequences[np.arange(len(sequence_elongations)), sequence_elongations]
        next_amino_acid_count = np.bincount(next_amino_acid[next_amino_acid != polymerize.PAD_VALUE], minlength=21)

        # Update masses of ribosomes attached to polymerizing polypeptides
        added_protein_mass = computeMassIncrease(
            sequences,
            sequence_elongations,
            self.aaWeightsIncorporated)

        updated_lengths = peptide_lengths + sequence_elongations
        updated_positions_on_mRNA = positions_on_mRNA + 3*sequence_elongations

        didInitialize = (
            (sequence_elongations > 0) &
            (peptide_lengths == 0))

        added_protein_mass[didInitialize] += self.endWeight

        # Write current average elongation to listener
        currElongRate = (sequence_elongations.sum() / n_active_ribosomes) / timestep
        update['listeners']['ribosome_data']['effective_elongation_rate'] = currElongRate

        # Ribosomes that reach the end of their sequences are terminated and
        # dissociated into 30S and 50S subunits. The polypeptide that they are polymerizing
        # is converted into a protein in BulkMolecules
        terminalLengths = self.protein_lengths[protein_indexes]

        didTerminate = (updated_lengths == terminalLengths)

        terminatedProteins = np.bincount(
            protein_indexes[didTerminate],
            minlength = self.proteinSequences.shape[0])

        update['active_ribosome'] = {'_delete': []}
        for index, ribosome in enumerate(states['active_ribosome'].values()):
            added_submass = np.zeros(9)
            unique_index = str(ribosome['unique_index'])
            if didTerminate[index]:
                update['active_ribosome']['_delete'].append(unique_index)
            else:
                added_submass[self.protein_submass_idx] = added_protein_mass[index]
                update['active_ribosome'][unique_index] = {
                    'peptide_length': updated_lengths[index],
                    'pos_on_mRNA': updated_positions_on_mRNA[index],
                    'submass': ribosome['submass'] + added_submass}

        update['monomers'] = {}
        for index, count in enumerate(terminatedProteins):
            update['monomers'][self.proteinIds[index]] = count

        nTerminated = didTerminate.sum()
        nInitialized = didInitialize.sum()

        update['subunits'] = {}
        update['subunits'][self.ribosome30S] = nTerminated
        update['subunits'][self.ribosome50S] = nTerminated

        # MODEL SPECIFIC: evolve
        # TODO(wcEcoli): use something other than a class attribute to pass aa diff to metabolism
        # TODO(vivarium): Why is aa_counts_for_translation is used here instead of total_aa_counts?
        net_charged, aa_count_diff, evolve_update = self.elongation_model.evolve(
            timestep,
            states,
            aa_counts_for_translation,
            aas_used,
            next_amino_acid_count,
            nElongations,
            nInitialized)

        update = deep_merge(update, evolve_update)

        # GTP hydrolysis is carried out in Metabolism process for growth
        # associated maintenance. This is set here for metabolism to use.
        self.gtp_to_hydrolyze = self.gtpPerElongation * nElongations

        update['polypeptide_elongation']['aa_count_diff'] = aa_count_diff
        update['polypeptide_elongation']['gtp_to_hydrolyze'] = self.gtp_to_hydrolyze

        # Write data to listeners
        update['listeners']['growth_limits']['net_charged'] = net_charged
        update['listeners']['growth_limits']['aasUsed'] = aas_used

        update['listeners']['ribosome_data']['effective_elongation_rate'] = currElongRate
        update['listeners']['ribosome_data']['aaCountInSequence'] = aaCountInSequence
        update['listeners']['ribosome_data']['aaCounts'] = aa_counts_for_translation
        update['listeners']['ribosome_data']['actualElongations'] = sequence_elongations.sum()
        update['listeners']['ribosome_data']['actualElongationHist'] = np.histogram(
            sequence_elongations, bins = np.arange(0,23))[0]
        update['listeners']['ribosome_data']['elongationsNonTerminatingHist'] = np.histogram(
            sequence_elongations[~didTerminate], bins=np.arange(0,23))[0]
        update['listeners']['ribosome_data']['didTerminate'] = didTerminate.sum()
        update['listeners']['ribosome_data']['terminationLoss'] = (terminalLengths - peptide_lengths)[
            didTerminate].sum()
        update['listeners']['ribosome_data']['numTrpATerminated'] = terminatedProteins[self.trpAIndex]
        update['listeners']['ribosome_data']['processElongationRate'] = self.ribosomeElongationRate / timestep

        log.info('polypeptide elongation terminated: {}'.format(nTerminated))
        return update

    def isTimeStepShortEnough(self, inputTimeStep, timeStepSafetyFraction):
        model_specific = self.elongation_model.isTimeStepShortEnough(inputTimeStep, timeStepSafetyFraction)
        max_time_step = inputTimeStep <= self.max_time_step
        return model_specific and max_time_step


def test_polypeptide_elongation():
    def make_elongation_rates(random, base, time_step, variable_elongation=False):
        size = 1
        lengths = time_step * np.full(size, base, dtype=np.int64)
        lengths = stochasticRound(random, lengths) if random else np.round(lengths)
        return lengths.astype(np.int64)

    test_config = {
        'time_step': 2,
        'proteinIds': np.array(['TRYPSYN-APROTEIN[c]']),
        'ribosome30S': 'CPLX0-3953[c]',
        'ribosome50S': 'CPLX0-3962[c]',
        'make_elongation_rates': make_elongation_rates,
        'proteinLengths': np.array([245]),  # this is the length of proteins in proteinSequences
        'translation_aa_supply': {
            'minimal': (units.mol / units.fg / units.min) * np.array([
                6.73304301e-21, 3.63835219e-21, 2.89772671e-21, 3.88086822e-21,
                5.04645651e-22, 4.45295877e-21, 2.64600664e-21, 5.35711230e-21,
                1.26817689e-21, 3.81168405e-21, 5.66834531e-21, 4.30576056e-21,
                1.70428208e-21, 2.24878356e-21, 2.49335033e-21, 3.47019761e-21,
                3.83858460e-21, 6.34564026e-22, 1.86880523e-21, 1.40959498e-27,
                5.20884460e-21])},
        'proteinSequences': np.array([[
            12, 10, 18, 9, 13, 1, 10, 9, 9, 16, 20, 9, 18, 15, 9, 10, 20, 4, 20, 13, 7, 15, 9, 18, 4, 10, 13, 15, 14, 1,
            2, 14, 11, 8, 20, 0, 16, 13, 7, 8, 12, 13, 7, 1, 10, 0, 14, 10, 13, 7, 10, 11, 20, 5, 4, 1, 11, 14, 16, 3,
            0, 5, 15, 18, 7, 2, 0, 9, 18, 9, 0, 2, 8, 6, 2, 2, 18, 3, 12, 20, 16, 0, 15, 2, 9, 20, 6, 14, 14, 16, 20,
            16, 20, 7, 11, 11, 15, 10, 10, 17, 9, 14, 13, 13, 7, 6, 10, 18, 17, 10, 16, 7, 2, 10, 10, 9, 3, 1, 2, 2, 1,
            16, 11, 0, 8, 7, 16, 9, 0, 5, 20, 20, 2, 8, 13, 11, 11, 1, 1, 9, 15, 9, 17, 12, 13, 14, 5, 7, 16, 1, 15, 1,
            7, 1, 7, 10, 10, 14, 13, 11, 16, 7, 0, 13, 8, 0, 0, 9, 0, 0, 7, 20, 14, 9, 9, 14, 20, 4, 20, 15, 16, 16, 15,
            2, 11, 9, 2, 10, 2, 1, 10, 8, 2, 7, 10, 20, 9, 20, 5, 12, 10, 14, 14, 9, 3, 20, 15, 6, 18, 7, 11, 3, 6, 20,
            1, 5, 10, 0, 0, 8, 4, 1, 15, 9, 12, 5, 6, 11, 9, 0, 5, 10, 3, 11, 5, 20, 0, 5, 1, 5, 0, 0, 7, 11, 20, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]]).astype(np.int8)
    }

    polypep_elong = PolypeptideElongation(test_config)

    initial_state = {
        'environment': {'media_id': 'minimal'},
        'subunits': {
            'CPLX0-3953[c]': 100,
            'CPLX0-3962[c]': 100
        },
        'monomers': {
            'TRYPSYN-APROTEIN[c]': 0,
        },
        'amino_acids': {
            aa: 100 for aa in DEFAULT_AA_NAMES
        },
        'active_ribosome': {
            '1': {'unique_index': 1, 'protein_index': 0, 'peptide_length': 1, 'pos_on_mRNA': 1,
                  'submass': np.zeros(len(MASSDIFFS))}
        },
        'listeners': {
            'mass': {
                'dry_mass': 350.0
            }
        }
    }

    settings = {
        'total_time': 200,
        'initial_state': initial_state}
    data = simulate_process(polypep_elong, settings)

    return data, test_config


def run_plot(data, config):

    # plot a list of variables
    proteins = [('monomers', prot_id) for prot_id in config['proteinIds']]
    aa = [('amino_acids', aa) for aa in DEFAULT_AA_NAMES]
    variables = proteins + aa

    plot_variables(
        data,
        variables=variables,
        out_dir='out/processes/polypeptide_elongation',
        filename='variables'
    )


def main():
    data, config = test_polypeptide_elongation()
    run_plot(data, config)


if __name__ == '__main__':
    main()

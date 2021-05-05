"""
PolypeptideElongation

Translation elongation sub-model.

TODO:
- see the initiation process for more TODOs
"""

import numpy as np
from scipy.integrate import odeint
from six.moves import range, zip
import logging as log

from vivarium.core.process import Process
from vivarium.library.dict_utils import deep_merge
from vivarium.core.composition import simulate_process

from ecoli.library.schema import bulk_schema, listener_schema, arrays_from, array_from, array_to

from wholecell.utils.polymerize import buildSequences, polymerize, computeMassIncrease
from wholecell.utils.random import stochasticRound
from wholecell.utils import units

MICROMOLAR_UNITS = units.umol / units.L


class PolypeptideElongation(Process):
    name = 'ecoli-polypeptide-elongation'

    defaults = {
        'max_time_step': 2.0,
        'n_avogadro': 6.02214076e+23 / units.mol,
        'proteinIds': [],
        'proteinLengths': np.array([]),
        'proteinSequences': np.array([[]]),
        'aaWeightsIncorporated': np.array([]),
        'endWeight': np.array([2.99146113e-08]),
        'variable_elongation': False,
        'make_elongation_rates': lambda random, rate, timestep, variable: np.array([]),
        'ribosomeElongationRate': 17.388824902723737,
        'translation_aa_supply': {'minimal': np.array([])},
        'import_threshold': 1e-05,
        'aa_from_trna': np.array([[]]),
        'gtpPerElongation': 4.2,
        'ppgpp_regulation': False,
        'trna_charging': False,
        'translation_supply': False,
        'ribosome30S': 'ribosome30S',
        'ribosome50S': 'ribosome50S',
        'amino_acids': [],

        'basal_elongation_rate': 22.0,
        'ribosomeElongationRateDict': {},
        'uncharged_trna_names': np.array([]),
        'aaNames': [],
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
        'seed': 0}

    def __init__(self, initial_parameters):
        super().__init__(initial_parameters)

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
        self.ribosome30S = self.parameters['ribosome30S']
        self.ribosome50S = self.parameters['ribosome50S']
        self.amino_acids = self.parameters['amino_acids']
        self.aaNames = self.parameters['aaNames']

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
        if self.parameters['trna_charging']:
            self.elongation_model = SteadyStateElongationModel(self.parameters, self)
        elif self.parameters['translation_supply']:
            self.elongation_model = TranslationSupplyElongationModel(self.parameters, self)
        else:
            self.elongation_model = BaseElongationModel(self.parameters, self)
        self.ppgpp_regulation = self.parameters['ppgpp_regulation']

        # Growth associated maintenance energy requirements for elongations
        self.gtpPerElongation = self.parameters['gtpPerElongation']
        ## Need to account for ATP hydrolysis for charging that has been
        ## removed from measured GAM (ATP -> AMP is 2 hydrolysis reactions)
        ## if charging reactions are not explicitly modeled
        if not self.parameters['trna_charging']:
            self.gtpPerElongation += 2
        ## Variable for metabolism to read to consume required energy
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

        self.seed = self.parameters['seed']
        self.random_state = np.random.RandomState(seed = self.seed)

    def ports_schema(self):
        return {
            'environment': {
                'media_id': {
                    '_default': '',
                    '_updater': 'set'},
                'amino_acids': bulk_schema([aa[:-3] for aa in self.aaNames])},

            'listeners': {
                'mass': {
                    'cell_mass': {'_default': 0.0},
                    'dry_mass': {'_default': 0.0}},

                'growth_limits': listener_schema({
                    'fraction_trna_charged': 0,
                    'aa_pool_size': 0,
                    'aa_request_size': 0,
                    'aa_allocated': 0,
                    'active_ribosomes_allocated': 0,
                    'net_charged': 0,
                    'aasUsed': 0}),

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

            'monomers': bulk_schema(self.proteinIds),
            'amino_acids': bulk_schema(self.amino_acids),
            'ppgpp_reaction_metabolites': bulk_schema(self.ppgpp_reaction_metabolites),
            'uncharged_trna': bulk_schema(self.uncharged_trna_names),
            'charged_trna': bulk_schema(self.charged_trna_names),
            'charging_molecules': bulk_schema(self.charging_molecule_names),
            'synthetases': bulk_schema(self.synthetase_names),

            'active_ribosome': {
                '*': {
                    'unique_index': {'_default': 0, '_updater': 'set'},
                    'protein_index': {'_default': 0, '_updater': 'set'},
                    'peptide_length': {'_default': 0, '_updater': 'set', '_emit': True},
                    'pos_on_mRNA': {'_default': 0, '_updater': 'set', '_emit': True},
                    'submass': {
                        'protein': {'_default': 0, '_emit': True}}}},

            'subunits': {
                self.ribosome30S: {
                    '_default': 0,
                    '_emit': True},
                self.ribosome50S: {
                    '_default': 0,
                    '_emit': True}},

            'polypeptide_elongation': {
                'aa_count_diff': {
                    '_default': {},
                    '_updater': 'set',
                    '_emit': True},
                'gtp_to_hydrolyze': {
                    '_default': 0,
                    '_updater': 'set',
                    '_emit': True}}}

    def next_update(self, timestep, states):
        # Set ribosome elongation rate based on simulation medium environment and elongation rate factor
        # which is used to create single-cell variability in growth rate
        # The maximum number of amino acids that can be elongated in a single timestep is set to 22 intentionally as the minimum number of padding values
        # on the protein sequence matrix is set to 22. If timesteps longer than 1.0s are used, this feature will lead to errors in the effective ribosome
        # elongation rate.

        update = {
            'molecules': {
                self.water: 0,
            },
            'listeners': {
                'ribosome_data': {},
                'growth_limits': {}}}

        current_media_id = states['environment']['media_id']

        # MODEL SPECIFIC: get ribosome elongation rate
        self.ribosomeElongationRate = self.elongation_model.elongation_rate(current_media_id)

        # If there are no active ribosomes, return immediately
        if len(states['active_ribosome']) == 0:
            return {}

        # Build sequences to request appropriate amount of amino acids to
        # polymerize for next timestep
        protein_indexes, peptide_lengths, positions_on_mRNA = arrays_from(
            states['active_ribosome'].values(),
            ['protein_index', 'peptide_length', 'pos_on_mRNA'])

        self.elongation_rates = self.make_elongation_rates(
            self.random_state,
            self.ribosomeElongationRate,
            timestep,
            self.variable_elongation)

        sequences = buildSequences(
            self.proteinSequences,
            protein_indexes,
            peptide_lengths,
            self.elongation_rates)

        sequenceHasAA = (sequences != polymerize.PAD_VALUE)
        aasInSequences = np.bincount(sequences[sequenceHasAA], minlength=21)

        # Calculate AA supply for expected doubling of protein
        dryMass = states['listeners']['mass']['dry_mass'] * units.fg
        self.cell_mass = states['listeners']['mass']['cell_mass'] * units.fg
        translation_supply_rate = self.translation_aa_supply[current_media_id] * self.elngRateFactor
        mol_aas_supplied = translation_supply_rate * dryMass * timestep * units.s
        self.aa_supply = units.strip_empty_units(mol_aas_supplied * self.n_avogadro)
        update['listeners']['ribosome_data']['translation_supply'] = translation_supply_rate.asNumber()

        assert not any(i<0 for i in states['uncharged_trna'].values())

        # # MODEL SPECIFIC: Calculate AA request
        # fraction_charged, aa_counts_requested, requests = self.elongation_model.request(
        #     timestep, states, aasInSequences)

        # save request
        aa_counts_for_translation = array_from(states['amino_acids'])

        # Write to listeners
        # update['listeners']['growth_limits']['fraction_trna_charged'] = np.dot(fraction_charged, self.aa_from_trna)
        # update['listeners']['growth_limits']['aa_pool_size'] = array_from(states['amino_acids'])
        # update['listeners']['growth_limits']['aa_request_size'] = aa_counts_for_translation


        # import ipdb; ipdb.set_trace()


        ## Begin wcEcoli evolveState()

        # Set value to 0 for metabolism in case of early return
        self.gtp_to_hydrolyze = 0

        # Write allocation data to listener
        # update['listeners']['growth_limits']['aa_allocated'] = aa_counts_for_translation

        # Get number of active ribosomes
        n_active_ribosomes = len(states['active_ribosome'])
        update['listeners']['growth_limits']['active_ribosomes_allocated'] = n_active_ribosomes

        if n_active_ribosomes == 0:
            return update

        if sequences.size == 0:
            return update

        # Calculate elongation resource capacity
        aaCountInSequence = np.bincount(sequences[(sequences != polymerize.PAD_VALUE)])
        # total_aa_counts = array_from(states['amino_acids'])
        # total_aa_counts = self.aas.counts()

        # MODEL SPECIFIC: Get amino acid counts
        aa_counts_for_translation = self.elongation_model.final_amino_acids(aa_counts_for_translation)
        aa_counts_for_translation = aa_counts_for_translation.astype(int)

        # Using polymerization algorithm elongate each ribosome up to the limits
        # of amino acids, sequence, and GTP
        result = polymerize(
            sequences,
            aa_counts_for_translation,
            10000000, # Set to a large number, the limit is now taken care of in metabolism
            self.random_state,
            self.elongation_rates[protein_indexes])

        sequence_elongations = result.sequenceElongation
        aas_used = result.monomerUsages
        nElongations = result.nReactions

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

        # Update active ribosomes, terminating if necessary
        # self.active_ribosomes.attrIs(
        #     peptide_length=updated_lengths,
        #     pos_on_mRNA=updated_positions_on_mRNA)
        # self.active_ribosomes.add_submass_by_name("protein", added_protein_mass)

        # Ribosomes that reach the end of their sequences are terminated and
        # dissociated into 30S and 50S subunits. The polypeptide that they are polymerizing
        # is converted into a protein in BulkMolecules
        terminalLengths = self.protein_lengths[protein_indexes]

        didTerminate = (updated_lengths == terminalLengths)

        terminatedProteins = np.bincount(
            protein_indexes[didTerminate],
            minlength = self.proteinSequences.shape[0])

        # self.active_ribosomes.delByIndexes(termination)

        update['active_ribosome'] = {'_delete': []}
        for index, ribosome in enumerate(states['active_ribosome'].values()):
            if didTerminate[index]:
                update['active_ribosome']['_delete'].append((ribosome['unique_index'],))
            else:
                update['active_ribosome'][ribosome['unique_index']] = {
                    'peptide_length': updated_lengths[index],
                    'pos_on_mRNA': updated_positions_on_mRNA[index],
                    'submass': {
                        'protein': added_protein_mass[index]}}

        update['monomers'] = {}
        for index, count in enumerate(terminatedProteins):
            update['monomers'][self.proteinIds[index]] = count

        # self.bulkMonomers.countsInc(terminatedProteins)

        nTerminated = didTerminate.sum()
        nInitialized = didInitialize.sum()

        update['subunits'] = {}
        update['subunits'][self.ribosome30S] = nTerminated
        update['subunits'][self.ribosome50S] = nTerminated
        # self.ribosome30S.countInc(nTerminated)
        # self.ribosome50S.countInc(nTerminated)

        # MODEL SPECIFIC: evolve
        # TODO: use something other than a class attribute to pass aa diff to metabolism
        net_charged, aa_count_diff, evolve_update = self.elongation_model.evolve(
            timestep,
            states,
            requests,
            aa_counts_for_translation,
            aas_used,
            nElongations,
            nInitialized)

        update = deep_merge(update, evolve_update)

        # GTP hydrolysis is carried out in Metabolism process for growth
        # associated maintenance. This is set here for metabolism to use.
        self.gtp_to_hydrolyze = self.gtpPerElongation * nElongations

        update['polypeptide_elongation'] = {}
        update['polypeptide_elongation']['aa_count_diff'] = aa_count_diff
        update['polypeptide_elongation']['gtp_to_hydrolyze'] = self.gtp_to_hydrolyze

        # Write data to listeners
        update['listeners']['growth_limits']['net_charged'] = net_charged

        update['listeners']["ribosome_data"]["effective_elongation_rate"] = currElongRate
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
        return inputTimeStep <= self.max_time_step


class BaseElongationModel(object):
    """
    Base Model: Request amino acids according to upcoming sequence, assuming
    max ribosome elongation.
    """
    def __init__(self, parameters, process):
        self.parameters = parameters
        self.process = process
        self.basal_elongation_rate = self.parameters['basal_elongation_rate']
        self.ribosomeElongationRateDict = self.parameters['ribosomeElongationRateDict']
        self.aaNames = self.parameters['aaNames']

    def elongation_rate(self, current_media_id):
        rate = self.process.elngRateFactor * self.ribosomeElongationRateDict[
            current_media_id].asNumber(units.aa / units.s)
        return np.min([self.basal_elongation_rate, rate])

    def amino_acid_counts(self, aasInSequences):
        return aasInSequences

    def request(self, timestep, states, aasInSequences):
        aa_counts_for_translation = self.amino_acid_counts(aasInSequences)

        # self.process.aas.requestIs(aa_counts_for_translation)

        # Not modeling charging so set fraction charged to 0 for all tRNA
        fraction_charged = np.zeros(len(self.aaNames))

        return fraction_charged, aa_counts_for_translation, {}

    def final_amino_acids(self, total_aa_counts):
        return total_aa_counts

    def evolve(self, timestep, states, requests, total_aa_counts, aas_used, nElongations, nInitialized):
        # Update counts of amino acids and water to reflect polymerization reactions
        net_charged = np.zeros(len(self.parameters['uncharged_trna_names']))

        return net_charged, {}, {
            'amino_acids': array_to(states['amino_acids'].keys(), -aas_used),
            'molecules': {
                self.process.water: nElongations - nInitialized}}

class TranslationSupplyElongationModel(BaseElongationModel):
    """
    Translation Supply Model: Requests minimum of 1) upcoming amino acid
    sequence assuming max ribosome elongation (ie. Base Model) and 2) estimation
    based on doubling the proteome in one cell cycle (does not use ribosome
    elongation, computed in Parca).
    """
    def __init__(self, parameters, process):
        super(TranslationSupplyElongationModel, self).__init__(parameters, process)

    def elongation_rate(self, current_media_id):
        return self.basal_elongation_rate

    def amino_acid_counts(self, aasInSequences):
        return np.fmin(self.process.aa_supply, aasInSequences)  # Check if this is required. It is a better request but there may be fewer elongations.

class SteadyStateElongationModel(TranslationSupplyElongationModel):
    """
    Steady State Charging Model: Requests amino acids based on the
    Michaelis-Menten competitive inhibition model.
    """
    def __init__(self, parameters, process):
        super(SteadyStateElongationModel, self).__init__(parameters, process)

        # Cell parameters
        self.cellDensity = self.parameters['cellDensity']
        self.maxRibosomeElongationRate = float(self.parameters['elongation_max'].asNumber(units.aa / units.s))

        # Data structures for charging
        self.aa_from_synthetase = self.parameters['aa_from_synthetase']
        self.charging_stoich_matrix = self.parameters['charging_stoich_matrix']

        # ppGpp synthesis
        self.ppgpp = self.parameters['ppgpp']
        self.rela = self.parameters['rela']
        self.spot = self.parameters['spot']
        self.ppgpp_reaction_names = self.parameters['ppgpp_reaction_names']
        self.ppgpp_reaction_metabolites = self.parameters['ppgpp_reaction_metabolites']
        self.ppgpp_reaction_stoich = self.parameters['ppgpp_reaction_stoich']
        self.ppgpp_synthesis_reaction = self.parameters['ppgpp_synthesis_reaction']
        self.ppgpp_degradation_reaction = self.parameters['ppgpp_degradation_reaction']
        self.synthesis_index = self.ppgpp_reaction_names.index(self.ppgpp_synthesis_reaction)
        self.degradation_index = self.ppgpp_reaction_names.index(self.ppgpp_degradation_reaction)

        # Parameters for tRNA charging and ribosome elongation
        self.kS = self.parameters['kS']
        self.KMtf = self.parameters['KMtf']
        self.KMaa = self.parameters['KMaa']
        self.krta = self.parameters['krta']
        self.krtf = self.parameters['krtf']
        aa_removed_from_charging = {'L-SELENOCYSTEINE[c]'}
        self.aa_charging_mask = np.array([aa not in aa_removed_from_charging for aa in self.aaNames])

        # ppGpp parameters
        self.KD_RelA = self.parameters['KD_RelA']
        self.k_RelA = self.parameters['k_RelA']
        self.k_SpoT_syn = self.parameters['k_SpoT_syn']
        self.k_SpoT_deg = self.parameters['k_SpoT_deg']
        self.KI_SpoT = self.parameters['KI_SpoT']

        # Amino acid supply calculations
        self.aa_supply_scaling = self.parameters['aa_supply_scaling']

    def request(self, timestep, states, aasInSequences):
        # Conversion from counts to molarity
        cell_mass = self.process.cell_mass
        cell_volume = cell_mass / self.cellDensity
        self.counts_to_molar = 1 / (self.process.n_avogadro * cell_volume)

        # Get counts and convert synthetase and tRNA to a per AA basis
        synthetase_counts = np.dot(
            self.aa_from_synthetase,
            array_from(states['synthetases']))
        aa_counts = array_from(states['amino_acids'])
        uncharged_trna_array = array_from(states['uncharged_trna'])
        charged_trna_array = array_from(states['charged_trna'])
        uncharged_trna_counts = np.dot(self.process.aa_from_trna, uncharged_trna_array)
        charged_trna_counts = np.dot(self.process.aa_from_trna, charged_trna_array)
        ribosome_counts = len(states['active_ribosome'])

        # Get concentration
        f = aasInSequences / aasInSequences.sum()
        synthetase_conc = self.counts_to_molar * synthetase_counts
        aa_conc = self.counts_to_molar * aa_counts
        uncharged_trna_conc = self.counts_to_molar * uncharged_trna_counts
        charged_trna_conc = self.counts_to_molar * charged_trna_counts
        ribosome_conc = self.counts_to_molar * ribosome_counts

        # Calculate steady state tRNA levels and resulting elongation rate
        fraction_charged, v_rib = self.calculate_trna_charging(
            synthetase_conc,
            uncharged_trna_conc,
            charged_trna_conc,
            aa_conc,
            ribosome_conc,
            f,
            timestep)

        aa_counts_for_translation = v_rib * f * timestep / self.counts_to_molar.asNumber(MICROMOLAR_UNITS)

        total_trna = charged_trna_array + uncharged_trna_array
        final_charged_trna = np.dot(fraction_charged, self.process.aa_from_trna * total_trna)

        charged_trna_request = charged_trna_array - final_charged_trna
        charged_trna_request[charged_trna_request < 0] = 0
        uncharged_trna_request = final_charged_trna - charged_trna_array
        uncharged_trna_request[uncharged_trna_request < 0] = 0

        self.aa_counts_for_translation = np.array(aa_counts_for_translation)

        fraction_trna_per_aa = total_trna / np.dot(np.dot(self.process.aa_from_trna, total_trna), self.process.aa_from_trna)
        total_charging_reactions = (
                np.dot(aa_counts_for_translation, self.process.aa_from_trna)
                * fraction_trna_per_aa + uncharged_trna_request)

        # Adjust aa_supply higher if amino acid concentrations are low
        # Improves stability of charging and mimics amino acid synthesis
        # inhibition and export
        aa_in_media = array_from(states['environment']['amino_acids']) # self.aa_environment.import_present()
        # TODO (Travis): add to listener?
        self.process.aa_supply *= self.aa_supply_scaling(aa_conc, aa_in_media)

        # Only request molecules that will be consumed in the charging reactions
        requested_molecules = -np.dot(self.charging_stoich_matrix, total_charging_reactions)
        requested_molecules[requested_molecules < 0] = 0
        # self.charging_molecules.requestIs(requested_molecules)

        # Request charged tRNA that will become uncharged
        # self.charged_trna.requestIs(charged_trna_request)

        # Request water for transfer of AA from tRNA for initial polypeptide.
        # This is severe overestimate assuming the worst case that every
        # elongation is initializing a polypeptide. This excess of water
        # shouldn't matter though.
        # self.water.requestIs(aa_counts_for_translation.sum())

        # ppGpp reactions based on charged tRNA
        request_ppgpp_metabolites = np.zeros(len(self.process.ppgpp_reaction_metabolites))
        if self.process.ppgpp_regulation:
            total_trna_conc = self.counts_to_molar * (uncharged_trna_counts + charged_trna_counts)
            updated_charged_trna_conc = total_trna_conc * fraction_charged
            updated_uncharged_trna_conc = total_trna_conc - updated_charged_trna_conc
            ppgpp_conc = self.counts_to_molar * states['molecules'][self.ppgpp]
            rela_conc = self.counts_to_molar * states['molecules'][self.rela]
            spot_conc = self.counts_to_molar * states['molecules'][self.spot]
            delta_metabolites, _, _, _, _, _ = self.ppgpp_metabolite_changes(
                updated_uncharged_trna_conc, updated_charged_trna_conc, ribosome_conc,
                f, rela_conc, spot_conc, ppgpp_conc, self.counts_to_molar, v_rib, timestep, request=True
            )

            request_ppgpp_metabolites = -delta_metabolites
            # self.ppgpp_reaction_metabolites.requestIs(request_ppgpp_metabolites)
            # self.ppgpp.requestAll()

        return fraction_charged, aa_counts_for_translation, {
            'charging_molecules': requested_molecules,
            'charged_trna': charged_trna_request,
            self.process.water: aa_counts_for_translation.sum(),
            'ppgpp_reaction_metabolites': request_ppgpp_metabolites}

    def final_amino_acids(self, total_aa_counts):
        return np.fmin(total_aa_counts, self.aa_counts_for_translation)

    def evolve(self, timestep, states, requests, total_aa_counts, aas_used, nElongations, nInitialized):
        update = {
            'molecules': {},
            'listeners': {},
        }

        # Get tRNA counts
        uncharged_trna = array_from(states['uncharged_trna'])
        charged_trna = requests['charged_trna'].astype(int)
        total_trna = uncharged_trna + charged_trna

        # Adjust molecules for number of charging reactions that occurred
        ## Net charged is tRNA that can be charged minus allocated charged tRNA for uncharging
        aa_for_charging = total_aa_counts - aas_used
        n_aa_charged = np.fmin(aa_for_charging, np.dot(self.process.aa_from_trna, uncharged_trna))
        n_trna_charged = self.distribution_from_aa(n_aa_charged, uncharged_trna, True)
        net_charged = n_trna_charged - charged_trna

        ## Reactions that are charged and elongated in same time step
        charged_and_elongated = self.distribution_from_aa(aas_used, total_trna)
        total_charging_reactions = charged_and_elongated + net_charged
        update['charging_molecules'] = array_to(
            states['charging_molecules'].keys(),
            np.dot(self.charging_stoich_matrix, total_charging_reactions).astype(int))

        ## Account for uncharging of tRNA during elongation
        update['charged_trna'] = array_to(states['charged_trna'].keys(), -charged_and_elongated)
        update['uncharged_trna'] = array_to(states['uncharged_trna'].keys(), charged_and_elongated)

        # Create ppGpp
        ## Concentrations of interest
        if self.process.ppgpp_regulation:
            v_rib = (nElongations * self.counts_to_molar).asNumber(MICROMOLAR_UNITS) / timestep
            ribosome_conc = self.counts_to_molar * len(states['active_ribosome'])
            updated_uncharged_trna_counts = array_from(states['uncharged_trna']) - net_charged
            updated_charged_trna_counts = array_from(states['charged_trna']) + net_charged
            uncharged_trna_conc = self.counts_to_molar * np.dot(
                self.process.aa_from_trna, updated_uncharged_trna_counts)
            charged_trna_conc = self.counts_to_molar * np.dot(
                self.process.aa_from_trna, updated_charged_trna_counts)
            ppgpp_conc = self.counts_to_molar * states['molecules'][self.ppgpp]
            rela_conc = self.counts_to_molar * states['molecules'][self.rela]
            spot_conc = self.counts_to_molar * states['molecules'][self.spot]

            f = aas_used / aas_used.sum()
            limits = requests['ppgpp_reaction_metabolites']
            delta_metabolites, ppgpp_syn, ppgpp_deg, rela_syn, spot_syn, spot_deg = self.ppgpp_metabolite_changes(
                uncharged_trna_conc, charged_trna_conc, ribosome_conc, f, rela_conc,
                spot_conc, ppgpp_conc, self.counts_to_molar, v_rib, timestep, limits=limits)

            if 'growth_limits' not in update['listeners']:
                update['listeners']['growth_limits'] = {}
            update['listeners']['growth_limits']['rela_syn'] = rela_syn
            update['listeners']['growth_limits']['spot_syn'] = spot_syn
            update['listeners']['growth_limits']['spot_deg'] = spot_deg

            update['ppgpp_reaction_metabolites'] = array_to(self.ppgpp_reaction_metabolites, delta_metabolites)

        # Update proton counts to reflect polymerization reactions and transfer of AA from tRNA
        # Peptide bond formation releases a water but transferring AA from tRNA consumes a OH-
        # Net production of H+ for each elongation, consume extra water for each initialization
        # since a peptide bond doesn't form
        update['molecules'][self.process.proton] = nElongations
        update['molecules'][self.process.water] = nInitialized

        # Use the difference between expected AA supply based on expected doubling time
        # and current DCW and AA used to charge tRNA to update the concentration target
        # in metabolism during the next time step
        aa_diff = self.process.aa_supply - np.dot(self.process.aa_from_trna, total_charging_reactions)

        return net_charged, {aa: diff for aa, diff in zip(self.aaNames, aa_diff)}, update

    def calculate_trna_charging(self, synthetase_conc, uncharged_trna_conc, charged_trna_conc, aa_conc, ribosome_conc, f, time_limit=1000, use_disabled_aas=False):
        '''
        Calculates the steady state value of tRNA based on charging and incorporation through polypeptide elongation.
        The fraction of charged/uncharged is also used to determine how quickly the ribosome is elongating.

        Inputs:
            synthetase_conc (array of floats with concentration units) - concentration of synthetases associated
                with each amino acid
            uncharged_trna_conc (array of floats with concentration units) - concentration of uncharged tRNA associated
                with each amino acid
            charged_trna_conc (array of floats with concentration units) - concentration of charged tRNA associated
                with each amino acid
            aa_conc (array of floats with concentration units) - concentration of each amino acid
            ribosome_conc (float with concentration units) - concentration of active ribosomes
            f (array of floats) - fraction of each amino acid to be incorporated to total amino acids incorporated
            time_limit (float) - time limit to reach steady state
            use_disabled_aas (bool) - if True, all amino acids will be used for charging calculations,
                if False, some will be excluded as determined in initialize

        Returns:
            fraction_charged (array of floats) - fraction of total tRNA that is charged for each tRNA species
            v_rib (float) - ribosomal elongation rate in units of uM/s
        '''

        def negative_check(trna1, trna2):
            '''
            Check for floating point precision issues that can lead to small
            negative numbers instead of 0. Adjusts both species of tRNA to
            bring concentration of trna1 to 0 and keep the same total concentration.

            Args:
                trna1 (ndarray[float]): concentration of one tRNA species (charged or uncharged)
                trna2 (ndarray[float]): concentration of another tRNA species (charged or uncharged)
            '''

            mask = trna1 < 0
            trna2[mask] = trna1[mask] + trna2[mask]
            trna1[mask] = 0

        def dcdt(c, t):
            '''
            Function for odeint to integrate

            Args:
                c (ndarray[float]): 1D array of concentrations of uncharged and charged tRNAs
                    dims: 2 * number of amino acids (uncharged tRNA come first, then charged)
                t (float): time of integration step

            Returns:
                ndarray[float]: dc/dt for tRNA concentrations
                    dims: 2 * number of amino acids (uncharged tRNA come first, then charged)
            '''

            uncharged_trna_conc = c[:n_aas]
            charged_trna_conc = c[n_aas:]

            v_charging = (self.kS * synthetase_conc * uncharged_trna_conc * aa_conc / (self.KMaa * self.KMtf)
                / (1 + uncharged_trna_conc/self.KMtf + aa_conc/self.KMaa + uncharged_trna_conc*aa_conc/self.KMtf/self.KMaa))
            numerator_ribosome = 1 + np.sum(f * (self.krta / charged_trna_conc + uncharged_trna_conc / charged_trna_conc * self.krta / self.krtf))
            v_rib = self.maxRibosomeElongationRate * ribosome_conc / numerator_ribosome

            # Handle case when f is 0 and charged_trna_conc is 0
            if not np.isfinite(v_rib):
                v_rib = 0

            dc = v_charging - v_rib*f

            return np.hstack((-dc, dc))

        # Convert inputs for integration
        synthetase_conc = synthetase_conc.asNumber(MICROMOLAR_UNITS)
        uncharged_trna_conc = uncharged_trna_conc.asNumber(MICROMOLAR_UNITS)
        charged_trna_conc = charged_trna_conc.asNumber(MICROMOLAR_UNITS)
        aa_conc = aa_conc.asNumber(MICROMOLAR_UNITS)
        ribosome_conc = ribosome_conc.asNumber(MICROMOLAR_UNITS)

        # Remove disabled amino acids from calculations
        n_total_aas = len(aa_conc)
        if use_disabled_aas:
            mask = np.ones(n_total_aas, bool)
        else:
            mask = self.aa_charging_mask
        synthetase_conc = synthetase_conc[mask]
        uncharged_trna_conc = uncharged_trna_conc[mask]
        charged_trna_conc = charged_trna_conc[mask]
        aa_conc = aa_conc[mask]
        f = f[mask]

        n_aas = len(aa_conc)

        # Integrate rates of charging and elongation
        dt = 0.001
        t = np.arange(0, time_limit, dt)
        c_init = np.hstack((uncharged_trna_conc, charged_trna_conc))
        sol = odeint(dcdt, c_init, t)

        # Determine new values from integration results
        uncharged_trna_conc = sol[-1, :n_aas]
        charged_trna_conc = sol[-1, n_aas:]
        negative_check(uncharged_trna_conc, charged_trna_conc)
        negative_check(charged_trna_conc, uncharged_trna_conc)

        fraction_charged = charged_trna_conc / (uncharged_trna_conc + charged_trna_conc)
        numerator_ribosome = 1 + np.sum(f * (self.krta / charged_trna_conc + uncharged_trna_conc / charged_trna_conc * self.krta / self.krtf))
        v_rib = self.maxRibosomeElongationRate * ribosome_conc / numerator_ribosome

        # Replace SEL fraction charged with average
        new_fraction_charged = np.zeros(n_total_aas)
        new_fraction_charged[mask] = fraction_charged
        new_fraction_charged[~mask] = fraction_charged.mean()

        return new_fraction_charged, v_rib

    def distribution_from_aa(self, n_aa, n_trna, limited=False):
        '''
        Distributes counts of amino acids to tRNAs that are associated with each amino acid.
        Uses self.process.aa_from_trna mapping to distribute from amino acids to tRNA based on the
        fraction that each tRNA species makes up for all tRNA species that code for the
        same amino acid.

        Inputs:
            n_aa (array of ints) - counts of each amino acid to distribute to each tRNA
            n_trna (array of ints) - counts of each tRNA to determine the distribution
            limited (bool) - optional, if True, limits the amino acids distributed to
                each tRNA to the number of tRNA that are available (n_trna)

        Returns:
            array of ints - distributed counts for each tRNA
        '''

        # Determine the fraction each tRNA species makes up out of all tRNA of the
        # associated amino acid
        f_trna = n_trna / np.dot(np.dot(self.process.aa_from_trna, n_trna), self.process.aa_from_trna)
        f_trna[~np.isfinite(f_trna)] = 0

        trna_counts = np.zeros(f_trna.shape, np.int64)
        for count, row in zip(n_aa, self.process.aa_from_trna):
            idx = (row == 1)
            frac = f_trna[idx]

            counts = np.floor(frac * count)
            diff = int(count - counts.sum())

            # Add additional counts to get up to counts to distribute
            # Prevent adding over the number of tRNA available if limited
            if diff > 0:
                if limited:
                    for _ in range(diff):
                        frac[(n_trna[idx] - counts) == 0] = 0
                        frac /= frac.sum()  # normalize for multinomial distribution
                        adjustment = self.process.random_state.multinomial(1, frac)
                        counts += adjustment
                else:
                    adjustment = self.process.random_state.multinomial(diff, frac)
                    counts += adjustment

            trna_counts[idx] = counts

        return trna_counts

    def ppgpp_metabolite_changes(self, uncharged_trna_conc, charged_trna_conc,
            ribosome_conc, f, rela_conc, spot_conc, ppgpp_conc, counts_to_molar,
            v_rib, timestep, request=False, limits=None):
        '''
        Calculates the changes in metabolite counts based on ppGpp synthesis and
        degradation reactions.

        Args:
            uncharged_trna_conc (np.array[float] with concentration units):
                concentration of uncharged tRNA associated with each amino acid
            charged_trna_conc (np.array[float] with concentration units):
                concentration of charged tRNA associated with each amino acid
            ribosome_conc (float with concentration units): concentration of active ribosomes
            f (np.array[float]): fraction of each amino acid to be incorporated
                to total amino acids incorporated
            rela_conc (float with concentration units): concentration of RelA
            spot_conc (float with concentration units): concentration of SpoT
            ppgpp_conc (float with concentration units): concentration of ppGpp
            counts_to_molar (float with concentration units): conversion factor
                from counts to molarity
            v_rib (float): rate of amino acid incorporation at the ribosome,
                in units of uM/s
            request (bool): if True, only considers reactant stoichiometry,
                otherwise considers reactants and products. For use in
                calculateRequest. GDP appears as both a reactant and product
                and the request can be off the actual use if not handled in this
                manner.
            limits (np.array[float]): counts of molecules that are available to prevent
                negative total counts as a result of delta_metabolites.
                If None, no limits are placed on molecule changes.

        Returns:
            delta_metabolites (np.array[int]): the change in counts of each metabolite
                involved in ppGpp reactions
            n_syn_reactions (int): the number of ppGpp synthesis reactions
            n_deg_reactions (int): the number of ppGpp degradation reactions
            v_rela_syn (float): rate of synthesis from RelA
            v_spot_syn (float): rate of synthesis from SpoT
            v_deg (float): rate of degradation from SpoT
        '''

        uncharged_trna_conc = uncharged_trna_conc.asNumber(MICROMOLAR_UNITS)
        charged_trna_conc = charged_trna_conc.asNumber(MICROMOLAR_UNITS)
        ribosome_conc = ribosome_conc.asNumber(MICROMOLAR_UNITS)
        rela_conc = rela_conc.asNumber(MICROMOLAR_UNITS)
        spot_conc = spot_conc.asNumber(MICROMOLAR_UNITS)
        ppgpp_conc = ppgpp_conc.asNumber(MICROMOLAR_UNITS)
        counts_to_micromolar = counts_to_molar.asNumber(MICROMOLAR_UNITS)

        numerator = 1 + charged_trna_conc / self.krta + uncharged_trna_conc / self.krtf
        saturated_charged = charged_trna_conc / self.krta / numerator
        saturated_uncharged = uncharged_trna_conc / self.krtf / numerator
        fraction_a_site = f * v_rib / (saturated_charged * self.maxRibosomeElongationRate)
        ribosomes_bound_to_uncharged = fraction_a_site * saturated_uncharged

        # Handle rare cases when tRNA concentrations are 0
        # Can result in inf and nan so assume a fraction of ribosomes
        # bind to the uncharged tRNA if any tRNA are present or 0 if not
        mask = ~np.isfinite(ribosomes_bound_to_uncharged)
        ribosomes_bound_to_uncharged[mask] = ribosome_conc * f[mask] * np.array(
            uncharged_trna_conc[mask] + charged_trna_conc[mask] > 0)

        # Calculate rates for synthesis and degradation
        frac_rela = 1 / (1 + self.KD_RelA / ribosomes_bound_to_uncharged.sum())
        v_rela_syn = self.k_RelA * rela_conc * frac_rela
        v_spot_syn = self.k_SpoT_syn * spot_conc
        v_syn = v_rela_syn + v_spot_syn
        v_deg = self.k_SpoT_deg * spot_conc * ppgpp_conc / (1 + uncharged_trna_conc.sum() / self.KI_SpoT)

        # Convert to discrete reactions
        n_syn_reactions = stochasticRound(self.process.random_state, v_syn * timestep / counts_to_micromolar)[0]
        n_deg_reactions = stochasticRound(self.process.random_state, v_deg * timestep / counts_to_micromolar)[0]

        # Only look at reactant stoichiometry if requesting molecules to use
        if request:
            ppgpp_reaction_stoich = np.zeros_like(self.ppgpp_reaction_stoich)
            reactants = self.ppgpp_reaction_stoich < 0
            ppgpp_reaction_stoich[reactants] = self.ppgpp_reaction_stoich[reactants]
        else:
            ppgpp_reaction_stoich = self.ppgpp_reaction_stoich

        # Calculate the change in metabolites and adjust to limits if provided
        # Possible reactions are adjusted down to limits if the change in any
        # metabolites would result in negative counts
        max_iterations = int(n_deg_reactions + n_syn_reactions + 1)
        old_counts = None
        for it in range(max_iterations):
            delta_metabolites = (ppgpp_reaction_stoich[:, self.synthesis_index] * n_syn_reactions
                + ppgpp_reaction_stoich[:, self.degradation_index] * n_deg_reactions)

            if limits is None:
                break
            else:
                final_counts = delta_metabolites + limits

                if np.all(final_counts >= 0) or (old_counts is not None and np.all(final_counts == old_counts)):
                    break

                limited_index = np.argmin(final_counts)
                if ppgpp_reaction_stoich[limited_index, self.synthesis_index] < 0:
                    limited = np.ceil(final_counts[limited_index] / ppgpp_reaction_stoich[limited_index, self.synthesis_index])
                    n_syn_reactions -= min(limited, n_syn_reactions)
                if ppgpp_reaction_stoich[limited_index, self.degradation_index] < 0:
                    limited = np.ceil(final_counts[limited_index] / ppgpp_reaction_stoich[limited_index, self.degradation_index])
                    n_deg_reactions -= min(limited, n_deg_reactions)

                old_counts = final_counts
        else:
            raise ValueError('Failed to meet molecule limits with ppGpp reactions.')

        return delta_metabolites, n_syn_reactions, n_deg_reactions, v_rela_syn, v_spot_syn, v_deg


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

# wcEcoli imports
from wholecell.utils.polymerize import (buildSequences, polymerize,
    computeMassIncrease)
from wholecell.utils.random import stochasticRound
from wholecell.utils import units

from wholecell.utils._trna_charging import get_initiations, get_codons_read

# vivarium imports
from vivarium.core.composition import simulate_process
from vivarium.library.dict_utils import deep_merge
from vivarium.plots.simulation_output import plot_variables

# vivarium-ecoli imports
from ecoli.library.schema import (listener_schema, numpy_schema, counts, attrs,
    bulk_name_to_idx)
from ecoli.models.polypeptide_elongation_models import (BaseElongationModel,
    TranslationSupplyElongationModel, SteadyStateElongationModel,
    KineticTrnaChargingModel, CoarseKineticTrnaChargingModel, MICROMOLAR_UNITS)
from ecoli.processes.registries import topology_registry
from ecoli.processes.partition import PartitionedProcess


# Register default topology for this process, associating it with process name
NAME = 'ecoli-polypeptide-elongation'
TOPOLOGY = {
    "environment": ("environment",),
    "boundary": ("boundary",),
    "listeners": ("listeners",),
    "active_ribosome": ("unique", "active_ribosome"),
    "bulk": ("bulk",),
    "polypeptide_elongation": ("process_state", "polypeptide_elongation"),
    # Non-partitioned counts
    "bulk_total": ("bulk",),
    "timestep": ("timestep",)
}
topology_registry.register(NAME, TOPOLOGY)

DEFAULT_AA_NAMES = [
    'L-ALPHA-ALANINE[c]', 'ARG[c]', 'ASN[c]', 'L-ASPARTATE[c]', 'CYS[c]',
    'GLT[c]', 'GLN[c]', 'GLY[c]', 'HIS[c]', 'ILE[c]', 'LEU[c]', 'LYS[c]',
    'MET[c]', 'PHE[c]', 'PRO[c]', 'SER[c]', 'THR[c]', 'TRP[c]', 'TYR[c]',
    'L-SELENOCYSTEINE[c]', 'VAL[c]']


class PolypeptideElongation(PartitionedProcess):
    """ Polypeptide Elongation PartitionedProcess

    defaults:
        proteinIds: array length n of protein names
    """

    name = NAME
    topology = TOPOLOGY
    defaults = {
        'time_step': 1,
        'max_time_step': 2.0,
        'n_avogadro': 6.02214076e+23 / units.mol,
        'protein_ids': np.array([]),
        'protein_lengths': np.array([]),
        'protein_sequences': np.array([[]]),
        'monomer_weights_incorporated': np.array([]),
        'endWeight': np.array([2.99146113e-08]),
        'make_elongation_rates': (lambda random, rate, timestep, variable:
            np.array([])),
        'next_aa_pad': 1,
        'ribosomeElongationRate': 17.388824902723737,
        'translation_aa_supply': {'minimal': np.array([])},
        'aa_from_trna': np.zeros(21),
        'gtpPerElongation': 4.2,
        'aa_supply_in_charging': False,
        'adjust_timestep_for_charging': False,
        'mechanistic_translation_supply': False,
        'mechanistic_aa_transport': False,
        'ppgpp_regulation': False,
        'disable_ppgpp_elongation_inhibition': False,
        'variable_elongation': False,
        'steady_state_trna_charging': False,
        'kinetic_trna_charging': False,
        'coarse_kinetic_elongation': False,
        'codon_sequences': [],
        'n_codons': 0,
        'translation_supply': False,
        'ribosome30S': 'ribosome30S',
        'ribosome50S': 'ribosome50S',
        'amino_acids': DEFAULT_AA_NAMES,
        'aa_exchange_names': DEFAULT_AA_NAMES,
        'basal_elongation_rate': 22.0,
        'ribosomeElongationRateDict': {
            'minimal': 17.388824902723737 * units.aa / units.s},
        'uncharged_trna_names': np.array([]),
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
        'aa_importers': [],
        'amino_acid_export': None,
        'synthesis_index': 0,
        'aa_exporters': [],
        'get_pathway_enzyme_counts_per_aa': None,
        'import_constraint_threshold': 0,
        'unit_conversion': 0,
        'elong_rate_by_ppgpp': 0,
        'amino_acid_import': None,
        'degradation_index': 1,
        'amino_acid_synthesis': None, 
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
        'emit_unique': False,
        'ribosome_profiling_molecules': {},
        'ribosome_profiling_molecule_indexes': {},
        'ribosome_profiling_listener_sizes': {},
        'n_proteins': 0,
        'n_monomers': 0,
    }

    def __init__(self, parameters=None):
        super().__init__(parameters)

        self.max_time_step = self.parameters['max_time_step']

        # Simulation options
        self.aa_supply_in_charging = self.parameters['aa_supply_in_charging']
        self.adjust_timestep_for_charging = self.parameters[
            'adjust_timestep_for_charging']
        self.mechanistic_translation_supply = self.parameters[
            'mechanistic_translation_supply']
        self.mechanistic_aa_transport = self.parameters[
            'mechanistic_aa_transport']
        self.ppgpp_regulation = self.parameters['ppgpp_regulation']
        self.disable_ppgpp_elongation_inhibition = self.parameters[
            'disable_ppgpp_elongation_inhibition']
        self.variable_elongation = self.parameters['variable_elongation']
        self.variable_polymerize = self.ppgpp_regulation or self.variable_elongation
        translation_supply = self.parameters['translation_supply']
        steady_state_trna_charging = self.parameters['steady_state_trna_charging']
        kinetic_trna_charging = self.parameters['kinetic_trna_charging']
        coarse_kinetic_elongation = self.parameters['coarse_kinetic_elongation']

        # Load parameters
        self.n_avogadro = self.parameters['n_avogadro']
        self.protein_ids = self.parameters['protein_ids']
        self.protein_lengths = self.parameters['protein_lengths']
        self.endWeight = self.parameters['endWeight']
        self.make_elongation_rates = self.parameters['make_elongation_rates']
        self.next_aa_pad = self.parameters['next_aa_pad']

        self.ribosome30S = self.parameters['ribosome30S']
        self.ribosome50S = self.parameters['ribosome50S']
        self.amino_acids = self.parameters['amino_acids']
        self.aa_exchange_names = self.parameters['aa_exchange_names']
        self.aa_environment_names = [aa[:-3] for aa in self.aa_exchange_names]
        self.aa_enzymes = self.parameters['aa_enzymes']

        self.ribosomeElongationRate = self.parameters['ribosomeElongationRate']

        # Amino acid supply calculations
        self.translation_aa_supply = self.parameters['translation_aa_supply']

        # Used for figure in publication
        self.trpAIndex = np.where(self.protein_ids ==
            "TRYPSYN-APROTEIN[c]")[0][0]

        self.elngRateFactor = 1.

        # Data structures for charging
        self.aa_from_trna = self.parameters['aa_from_trna']

        # Set modeling method
        # TODO: Test that these models all work properly
        if kinetic_trna_charging:
            self.elongation_model = KineticTrnaChargingModel(self.parameters, self)
        elif coarse_kinetic_elongation:
            self.elongation_model = CoarseKineticTrnaChargingModel(
                self.parameters, self)
        elif steady_state_trna_charging:
            self.elongation_model = SteadyStateElongationModel(
                self.parameters, self)
        elif translation_supply:
            self.elongation_model = TranslationSupplyElongationModel(
                self.parameters, self)
        else:
            self.elongation_model = BaseElongationModel(self.parameters, self)

        # Growth associated maintenance energy requirements for elongations
        self.gtpPerElongation = self.parameters['gtpPerElongation']
        # Need to account for ATP hydrolysis for charging that has been
        # removed from measured GAM (ATP -> AMP is 2 hydrolysis reactions)
        # if charging reactions are not explicitly modeled
        if not (steady_state_trna_charging or kinetic_trna_charging):
            self.gtpPerElongation += 2

        # basic molecule names
        self.proton = self.parameters['proton']
        self.water = self.parameters['water']
        self.rela = self.parameters['rela']
        self.spot = self.parameters['spot']
        self.ppgpp = self.parameters['ppgpp']
        self.aa_importers = self.parameters['aa_importers']
        self.aa_exporters = self.parameters['aa_exporters']
        # Numpy index for bulk molecule
        self.proton_idx = None

        # Names of molecules associated with tRNA charging
        self.ppgpp_reaction_metabolites = self.parameters[
            'ppgpp_reaction_metabolites']
        self.uncharged_trna_names = self.parameters['uncharged_trna_names']
        self.charged_trna_names = self.parameters['charged_trna_names']
        self.charging_molecule_names = self.parameters[
            'charging_molecule_names']
        self.synthetase_names = self.parameters['synthetase_names']

        self.seed = self.parameters['seed']
        self.random_state = np.random.RandomState(seed = self.seed)

        self.zero_aa_exchange_rates = MICROMOLAR_UNITS / units.s * np.zeros(
            len(self.amino_acids))
        
        self.n_proteins = self.parameters['n_proteins']
        self.n_codons = self.parameters['n_codons']
        self.codon_sequences = self.parameters['codon_sequences']

        self.ribosome_profiling_molecules = self.parameters[
            'ribosome_profiling_molecules']
        self.ribosome_profiling_molecule_indexes = self.parameters[
            'ribosome_profiling_molecule_indexes']
        self.ribosome_profiling_listener_sizes = self.parameters[
            'ribosome_profiling_listener_sizes']

    def ports_schema(self):
        schema = {
            'environment': {
                'media_id': {
                    '_default': '',
                    '_updater': 'set'},
                'exchange': {'*': {'_default': 0}}
            },
            'boundary': {
                'external': {
                    aa: {'_default': 0}
                    for aa in sorted(self.aa_environment_names)
                }
            },
            'listeners': {
                'mass': listener_schema({
                    'cell_mass': 0.0,
                    'dry_mass': 0.0}),

                'growth_limits': listener_schema({
                    'fraction_trna_charged': ([0] * len(self.uncharged_trna_names),
                        self.uncharged_trna_names),
                    'aa_allocated': ([0] * len(self.amino_acids),
                        self.amino_acids),
                    'aa_pool_size': ([0] * len(self.amino_acids),
                        self.amino_acids),
                    'aa_request_size': ([0] * len(self.amino_acids),
                        self.amino_acids),
                    'active_ribosome_allocated': 0,
                    'net_charged': ([0] * len(self.uncharged_trna_names),
                        self.uncharged_trna_names),
                    'aas_used': ([0] * len(self.amino_acids),
                        self.amino_acids),
                    'aa_count_diff': ([0] * len(self.amino_acids),
                        self.amino_acids),
                    # Below only if trna_charging enbaled
                    'original_aa_supply': ([0] * len(self.amino_acids),
                        self.amino_acids),
                    'aa_in_media': ([False] * len(self.amino_acids),
                        self.amino_acids),
                    'synthetase_conc': ([0] * len(self.amino_acids),
                        self.amino_acids),
                    'uncharged_trna_conc': (
                        [0] * len(self.amino_acids), self.amino_acids),
                    'charged_trna_conc': ([0] * len(self.amino_acids),
                        self.amino_acids),
                    'aa_conc': ([0] * len(self.amino_acids),
                        self.amino_acids),
                    'ribosome_conc': 0,
                    'fraction_aa_to_elongate': ([0] * len(self.amino_acids),
                        self.amino_acids),
                    'aa_supply': ([0] * len(self.amino_acids),
                        self.amino_acids),
                    'aa_synthesis': ([0] * len(self.amino_acids),
                        self.amino_acids),
                    'aa_import': ([0] * len(self.amino_acids),
                        self.amino_acids),
                    'aa_export': ([0] * len(self.amino_acids),
                        self.amino_acids),
                    'aa_importers': ([0] * len(self.aa_importers),
                        self.aa_importers),
                    'aa_exporters': ([0] * len(self.aa_exporters),
                        self.aa_exporters),
                    'aa_supply_enzymes_fwd': ([0] * len(self.amino_acids),
                        self.amino_acids),
                    'aa_supply_enzymes_rev': ([0] * len(self.amino_acids),
                        self.amino_acids),
                    'aa_supply_aa_conc': ([0] * len(self.amino_acids),
                        self.amino_acids),
                    'aa_supply_fraction_fwd': ([0] * len(self.amino_acids),
                        self.amino_acids),
                    'aa_supply_fraction_rev': ([0] * len(self.amino_acids),
                        self.amino_acids),
                    'ppgpp_conc': 0,
                    'rela_conc': 0,
                    'spot_conc': 0,
                    'rela_syn': ([0] * len(self.amino_acids),
                        self.amino_acids),
                    'spot_syn': 0,
                    'spot_deg': 0,
                    'spot_deg_inhibited': ([0] * len(self.amino_acids),
                        self.amino_acids),
                    'trna_charged': ([0] * len(self.amino_acids),
                        self.amino_acids)}),

                'ribosome_data': listener_schema({
                    'translation_supply': ([0] * len(self.amino_acids),
                        self.amino_acids),
                    'effective_elongation_rate': 0,
                    'aa_count_in_sequence': ([0] * len(self.amino_acids),
                        self.amino_acids),
                    'aa_counts': ([0] * len(self.amino_acids),
                        self.amino_acids),
                    'actual_elongations': 0,
                    'actual_elongation_hist': [0] * 22,
                    'elongations_non_terminating_hist': [0] * 22,
                    'did_terminate': 0,
                    'termination_loss': 0,
                    'num_trpA_terminated': 0,
                    'process_elongation_rate': 0,
                    'n_initialized': 0,
                    'n_terminated': 0}),
                    
                'trna_charging': listener_schema({
                    'charging_events': 0,
                    'cleaved': 0,
                    'codons_read': 0,
                    'codons_to_trnas_counter': 0,
                    'initial_disagreements': 0,
                    'initiated': 0,
                    'not_cleaved': 0,
                    'reading_events': 0,
                    'saturation_trna': 0,
                    'turnover': 0})},

            'bulk': numpy_schema('bulk'),
            'bulk_total': numpy_schema('bulk'),

            'active_ribosome': numpy_schema('active_ribosome',
                emit=self.parameters['emit_unique']),

            'polypeptide_elongation': {
                'aa_count_diff': {
                    '_default': {},
                    '_emit': True,
                    '_updater': 'set',
                    '_divider': 'empty_dict'},
                'gtp_to_hydrolyze': {
                    '_default': 0,
                    '_emit': True,
                    '_updater': 'set',
                    '_divider': 'zero',
                },
                'aa_exchange_rates': {
                    '_default': 0,
                    '_emit': True,
                    '_updater': 'set',
                    '_divider': 'zero'
                },
            },
            'timestep': {'_default': self.parameters['time_step']}
        }
        schema['listeners']['trna_charging'] = deep_merge(
            schema['listeners']['trna_charging'],
            listener_schema({
                f'ribosome_positions_{gene}': 0
                for gene in self.ribosome_profiling_molecules.values()})
        )

        return schema

    def calculate_request(self, timestep, states):
        """
        Set ribosome elongation rate based on simulation medium environment and elongation rate factor
        which is used to create single-cell variability in growth rate
        The maximum number of amino acids that can be elongated in a single timestep is set to 22
        intentionally as the minimum number of padding values on the protein sequence matrix is set to 22.
        If timesteps longer than 1.0s are used, this feature will lead to errors in the effective ribosome
        elongation rate.
        """

        if self.proton_idx is None:
            bulk_ids = states['bulk']['id']
            self.proton_idx = bulk_name_to_idx(self.proton, bulk_ids)
            self.water_idx = bulk_name_to_idx(self.water, bulk_ids)
            self.rela_idx = bulk_name_to_idx(self.rela, bulk_ids)
            self.spot_idx = bulk_name_to_idx(self.spot, bulk_ids)
            self.ppgpp_idx = bulk_name_to_idx(self.ppgpp, bulk_ids)
            self.monomer_idx = bulk_name_to_idx(self.protein_ids, bulk_ids)
            self.amino_acid_idx = bulk_name_to_idx(self.amino_acids, bulk_ids)
            self.aa_enzyme_idx = bulk_name_to_idx(self.aa_enzymes, bulk_ids)
            self.ppgpp_rxn_metabolites_idx = bulk_name_to_idx(
                self.ppgpp_reaction_metabolites, bulk_ids)
            self.uncharged_trna_idx = bulk_name_to_idx(
                self.uncharged_trna_names, bulk_ids)
            self.charged_trna_idx = bulk_name_to_idx(
                self.charged_trna_names, bulk_ids)
            self.charging_molecule_idx = bulk_name_to_idx(
                self.charging_molecule_names, bulk_ids)
            self.synthetase_idx = bulk_name_to_idx(
                self.synthetase_names, bulk_ids)
            self.ribosome30S_idx = bulk_name_to_idx(
                self.ribosome30S, bulk_ids)
            self.ribosome50S_idx = bulk_name_to_idx(
                self.ribosome50S, bulk_ids)
            self.aa_importer_idx = bulk_name_to_idx(
                self.aa_importers, bulk_ids)
            self.aa_exporter_idx = bulk_name_to_idx(
                self.aa_exporters, bulk_ids)

        # If there are no active ribosomes, return immediately
        if states['active_ribosome']['_entryState'].sum() == 0:
            return {'listeners': {'ribosome_data': {}, 'growth_limits': {}}}

        # Get active ribosome attributes
        protein_indexes, peptide_lengths, = attrs(states['active_ribosome'],
            ['protein_index', 'peptide_length'])

        # Set ribosome elongation rate based on simulation medium
        # environment and elongation rate factor, which is used to
        # create single-cell variability in growth rate.
        # Elongation Model Dependent: Get ribosome elongation rate
        self.ribosomeElongationRate = self.elongation_model.elongation_rate(
            states, protein_indexes, peptide_lengths)

        self.elongation_rates = self.make_elongation_rates(
            self.random_state,
            self.ribosomeElongationRate,
            states['timestep'],
            self.variable_elongation)

        # Build sequences to request appropriate amount of amino acids to
        # polymerize for next timestep
        sequences = buildSequences(
            self.elongation_model.protein_sequences,
            protein_indexes,
            peptide_lengths,
            self.elongation_rates)

        # Calculate AA supply for expected doubling of protein
        dryMass = (states['listeners']['mass']['dry_mass'] * units.fg)
        current_media_id = states['environment']['media_id']
        translation_supply_rate = self.translation_aa_supply[current_media_id] \
            * self.elngRateFactor
        mol_aas_supplied = translation_supply_rate * dryMass * states['timestep'] * units.s
        self.aa_supply = units.strip_empty_units(mol_aas_supplied * self.n_avogadro)


        # Elongation Model Dependent: Calculate monomer request
        monomers_in_sequences = np.bincount(
            sequences[sequences != polymerize.PAD_VALUE],
            minlength=self.elongation_model.n_monomers)
        fraction_charged, aa_request, requests = \
            self.elongation_model.request(states, monomers_in_sequences, 
                                          protein_indexes, peptide_lengths)

        # Write to listeners
        listeners = requests.setdefault('listeners', {})
        ribosome_data_listener = listeners.setdefault(
            'ribosome_data', {})
        ribosome_data_listener['translation_supply'
            ] = translation_supply_rate.asNumber()
        growth_limits_listener = requests['listeners'].setdefault(
            'growth_limits', {})
        growth_limits_listener['fraction_trna_charged'
            ] = fraction_charged
        growth_limits_listener['aa_pool_size'] = counts(states['bulk_total'],
            self.amino_acid_idx)
        growth_limits_listener['aa_request_size'] = aa_request

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
            'listeners': {
                'ribosome_data': {},
                'growth_limits': {}},
            'polypeptide_elongation': {},
            'active_ribosome': {},
            'bulk': []
        }

        # Begin wcEcoli evolveState()
        # Set values for metabolism in case of early return
        update['polypeptide_elongation']['gtp_to_hydrolyze'] = 0
        update['polypeptide_elongation']['aa_count_diff'] = {}

        # Get number of active ribosomes
        n_active_ribosomes = states['active_ribosome']['_entryState'].sum()
        update['listeners']['growth_limits'][
            'active_ribosome_allocated'] = n_active_ribosomes
        update['listeners']['growth_limits']['aa_allocated'] = counts(
            states['bulk'], self.amino_acid_idx)

        # If there are no active ribosomes, return immediately
        if n_active_ribosomes == 0:
            return update
        
        # Polypeptide elongation requires counts to be updated in real-time
        # so make a writeable copy of bulk counts to do so
        states['bulk'] = counts(states['bulk'], range(len(states['bulk'])))

        # Build amino acids sequences for each ribosome to polymerize
        protein_indexes, peptide_lengths, positions_on_mRNA = attrs(
            states['active_ribosome'],
            ['protein_index', 'peptide_length', 'pos_on_mRNA'])

        all_sequences = buildSequences(
            self.elongation_model.protein_sequences,
            protein_indexes,
            peptide_lengths,
            self.elongation_rates + self.next_aa_pad)
        sequences = all_sequences[:, :-self.next_aa_pad].copy()

        if sequences.size == 0:
            return update

        # Build codon sequences
        codon_sequences_width = self.elongation_model.codon_sequences_width(
            self.elongation_rates)
        codon_sequences = buildSequences(
            self.codon_sequences,
            protein_indexes,
            peptide_lengths,
            codon_sequences_width)
        
        # Calculate elongation resource capacity
        monomer_count_in_sequence = np.bincount(
            sequences[(sequences != polymerize.PAD_VALUE)],
            minlength=self.elongation_model.n_monomers)
        monomer_count_in_sequence_in_aas = self.elongation_model.monomer_to_aa(
            monomer_count_in_sequence)

        # Elongation Model Dependent: Get monomer counts
        monomer_limit, monomer_limit_in_aas = self.elongation_model.monomer_limit(
            states, monomer_count_in_sequence)

        # Use the polymerize algorithm to elongate each ribosome up to
        # the limits of monomers, sequences, and GTP.
        result = polymerize(
            sequences,
            monomer_limit,
            10000000,  # Set to a large number, the limit is now taken care of in metabolism
            self.random_state,
            self.elongation_rates[protein_indexes],
            variable_elongation=self.variable_polymerize)
        
        # Elongation Model Dependent: Verify consistency
        result, aas_used, net_charged, listeners = \
            self.elongation_model.reconcile(states, result)
        update = deep_merge(update, {'listeners': listeners})

        sequence_elongations = result.sequenceElongation
        n_elongations = result.nReactions

        # Elongation Model Dependent
        next_amino_acid_count = self.elongation_model.next_amino_acids(
            all_sequences, sequence_elongations)

        # Update masses of ribosomes attached to polymerizing polypeptides
        sequences = self.elongation_model.sequences(sequences)
        added_protein_mass = computeMassIncrease(
            sequences,
            sequence_elongations,
            self.elongation_model.monomer_weights_incorporated)

        updated_lengths = peptide_lengths + sequence_elongations
        updated_positions_on_mRNA = positions_on_mRNA + 3*sequence_elongations

        did_initialize = (
            (sequence_elongations > 0) &
            (peptide_lengths == 0))

        added_protein_mass[did_initialize] += self.endWeight

        # Write current average elongation to listener
        currElongRate = (sequence_elongations.sum() /
            n_active_ribosomes) / states['timestep']

        # Ribosomes that reach the end of their sequences are terminated and
        # dissociated into 30S and 50S subunits. The polypeptide that they are
        # polymerizing is converted into a protein in BulkMolecules
        terminal_lengths = self.protein_lengths[protein_indexes]

        did_terminate = (updated_lengths == terminal_lengths)

        terminated_proteins = np.bincount(
            protein_indexes[did_terminate],
            minlength = self.n_proteins)
        
        # Mature completed polypeptides
        (did_terminate, terminated_proteins, initial_methionines_cleaved,
            listeners) = self.elongation_model.protein_maturation(states, 
            did_terminate, terminated_proteins, protein_indexes)
        update = deep_merge(update, {'listeners': listeners})

        protein_mass, = attrs(states['active_ribosome'], ['massDiff_protein'])
        update['active_ribosome'].update({
            'delete': np.where(did_terminate)[0],
            'set': {
                'massDiff_protein': protein_mass + added_protein_mass,
                'peptide_length': updated_lengths,
                'pos_on_mRNA': updated_positions_on_mRNA,
            }
        })

        update['bulk'].append((self.monomer_idx, terminated_proteins))
        # In wcEcoli, in/decrementing a bulk molecule count is immediately 
        # reflected in the counts seen by the process
        states['bulk'][self.monomer_idx] += terminated_proteins

        n_terminated = did_terminate.sum()
        n_initialized = did_initialize.sum()

        update['bulk'].append((self.ribosome30S_idx, n_terminated))
        update['bulk'].append((self.ribosome50S_idx, n_terminated))
        # In wcEcoli, in/decrementing a bulk molecule count is immediately 
        # reflected in the counts seen by the process
        states['bulk'][self.ribosome30S_idx] += n_terminated
        states['bulk'][self.ribosome50S_idx] += n_terminated

        # Elongation Model Dependent: evolve
        net_charged, aa_count_diff, evolve_update = self.elongation_model.evolve(
            states,
            aas_used,
            next_amino_acid_count,
            n_elongations,
            n_initialized,
            net_charged,
            result.monomerUsages,
            initial_methionines_cleaved)

        evolve_bulk_update = evolve_update.pop('bulk')
        update = deep_merge(update, evolve_update)
        update['bulk'].extend(evolve_bulk_update)

        update['polypeptide_elongation']['aa_count_diff'] = aa_count_diff
        # GTP hydrolysis is carried out in Metabolism process for growth
        # associated maintenance. This is passed to metabolism.
        update['polypeptide_elongation'][
            'gtp_to_hydrolyze'] = self.gtpPerElongation * n_elongations

        # Write data to listeners
        update['listeners']['growth_limits']['net_charged'] = net_charged
        update['listeners']['growth_limits']['aas_used'] = aas_used
        update['listeners']['growth_limits']['aa_count_diff'] = [
            aa_count_diff.get(id_, 0) for id_ in self.amino_acids]
        
        ribosome_data_listener = update['listeners'].setdefault(
            'ribosome_data', {})
        ribosome_data_listener['effective_elongation_rate'] = currElongRate
        ribosome_data_listener['aa_count_in_sequence'] = \
            monomer_count_in_sequence_in_aas
        ribosome_data_listener['aa_counts'] = monomer_limit_in_aas
        ribosome_data_listener['actual_elongations'
            ] = sequence_elongations.sum()
        ribosome_data_listener['actual_elongation_hist'] = np.histogram(
            sequence_elongations, bins = np.arange(0,23))[0]
        ribosome_data_listener['elongations_non_terminating_hist'
            ] = np.histogram(sequence_elongations[~did_terminate],
                             bins=np.arange(0,23))[0]
        ribosome_data_listener['did_terminate'] = did_terminate.sum()
        ribosome_data_listener['termination_loss'] = (
            terminal_lengths - peptide_lengths)[did_terminate].sum()
        ribosome_data_listener['num_trpA_terminated'] = terminated_proteins[
            self.trpAIndex]
        ribosome_data_listener['n_terminated'] = np.bincount(
            protein_indexes[did_terminate], minlength=self.n_proteins)
        ribosome_data_listener['n_initialized'] = n_initialized
        ribosome_data_listener['process_elongation_rate'] = (
            self.ribosomeElongationRate / states['timestep'])
        
        trna_charging_listener = update['listeners'].setdefault(
            'trna_charging', {})
        for molecule, gene in self.ribosome_profiling_molecules.items():
            molecule_index = self.ribosome_profiling_molecule_indexes[molecule]
            listener_size = self.ribosome_profiling_listener_sizes[molecule]
            ribosome_positions = np.zeros(listener_size, np.int64)
            ribosomes_on_MOI = protein_indexes == molecule_index
            for ribosome in np.where(ribosomes_on_MOI)[0]:
                ribosome_positions[updated_lengths[ribosome]] += 1
            
            trna_charging_listener[f'ribosome_positions_{gene}'] = ribosome_positions

        # Calculate codons read and initiations performed
        codons_read = get_codons_read(
            codon_sequences,
            sequence_elongations,
            self.n_codons)

        n_initiations = get_initiations(
            sequence_elongations,
            peptide_lengths,
            protein_indexes,
            )
        trna_charging_listener['codons_read'] = codons_read
        trna_charging_listener['initiated'] = n_initiations

        return update

    def isTimeStepShortEnough(self, inputTimeStep, timeStepSafetyFraction):
        model_specific = self.elongation_model.isTimeStepShortEnough(
            inputTimeStep, timeStepSafetyFraction)
        max_time_step = inputTimeStep <= self.max_time_step
        return model_specific and max_time_step


def test_polypeptide_elongation(return_data=False):
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
        'protein_lengths': np.array([245]),  # this is the length of proteins in protein_sequences
        'translation_aa_supply': {
            'minimal': (units.mol / units.fg / units.min) * np.array([
                6.73304301e-21, 3.63835219e-21, 2.89772671e-21, 3.88086822e-21,
                5.04645651e-22, 4.45295877e-21, 2.64600664e-21, 5.35711230e-21,
                1.26817689e-21, 3.81168405e-21, 5.66834531e-21, 4.30576056e-21,
                1.70428208e-21, 2.24878356e-21, 2.49335033e-21, 3.47019761e-21,
                3.83858460e-21, 6.34564026e-22, 1.86880523e-21, 1.40959498e-27,
                5.20884460e-21])},
        'protein_sequences': np.array([[
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
        'bulk': np.array([
            ('CPLX0-3953[c]', 100),
            ('CPLX0-3962[c]', 100),
            ('TRYPSYN-APROTEIN[c]', 0),
            ('RELA', 0),
            ('SPOT', 0),
            ('H2O', 0),
            ('PROTON', 0),
            ('ppGpp', 0),
            ] + [(aa, 100) for aa in DEFAULT_AA_NAMES],
            dtype=[('id', 'U40'), ('count', int)]),
        'unique': {
            'active_ribosome': np.array([
                (1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
            ], dtype=[
                ('_entryState', np.bool_), ('unique_index', int),
                ('protein_index', int), ('peptide_length', int),
                ('pos_on_mRNA', int), ('massDiff_DNA', '<f8'),
                ('massDiff_mRNA', '<f8'), ('massDiff_metabolite', '<f8'),
                ('massDiff_miscRNA', '<f8'), ('massDiff_nonspecific_RNA',
                '<f8'), ('massDiff_protein', '<f8'), ('massDiff_rRNA',
                '<f8'), ('massDiff_tRNA', '<f8'), ('massDiff_water', '<f8')])
        },
        'listeners': {
            'mass': {
                'dry_mass': 350.0
            }
        }
    }
    
    settings = {
        'total_time': 200,
        'initial_state': initial_state,
        'topology': TOPOLOGY}
    data = simulate_process(polypep_elong, settings)

    if return_data:
        return data, test_config


def run_plot(data, config):

    # plot a list of variables
    bulk_ids = ['CPLX0-3953[c]', 'CPLX0-3962[c]', 'TRYPSYN-APROTEIN[c]',
        'RELA', 'SPOT', 'H2O', 'PROTON', 'ppGpp'] + [
            aa for aa in DEFAULT_AA_NAMES]
    variables = [(bulk_id,) for bulk_id in bulk_ids]

    # format data
    bulk_timeseries = np.array(data['bulk'])
    for i, bulk_id in enumerate(bulk_ids):
        data[bulk_id] = bulk_timeseries[:, i]

    plot_variables(
        data,
        variables=variables,
        out_dir='out/processes/polypeptide_elongation',
        filename='variables'
    )


def main():
    data, config = test_polypeptide_elongation(return_data=True)
    run_plot(data, config)


if __name__ == '__main__':
    main()

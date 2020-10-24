"""
========================
E. coli master composite
========================
"""

import json
import numpy as np
import uuid

from six.moves import cPickle

from vivarium.core.process import Generator
from vivarium.core.composition import simulate_compartment_in_experiment
from vivarium.core.experiment import pp

# vivarium processes
from vivarium.processes.divide_condition import DivideCondition
from vivarium.processes.meta_division import MetaDivision

# vivarium-ecoli processes
from ecoli.processes.tf_binding import TfBinding
from ecoli.processes.transcript_initiation import TranscriptInitiation
from ecoli.processes.transcript_elongation import TranscriptElongation
from ecoli.processes.rna_degradation import RnaDegradation
from ecoli.processes.polypeptide_initiation import PolypeptideInitiation
from ecoli.processes.polypeptide_elongation import PolypeptideElongation, MICROMOLAR_UNITS
from ecoli.processes.complexation import Complexation
from ecoli.processes.two_component_system import TwoComponentSystem
from ecoli.processes.equilibrium import Equilibrium
from ecoli.processes.protein_degradation import ProteinDegradation
from ecoli.processes.metabolism import Metabolism
from ecoli.processes.mass import Mass

from wholecell.utils import units
from wholecell.utils.fitting import normalize

RAND_MAX = 2**31
SIM_DATA_PATH = '../wcEcoli/out/underscore/kb/simData.cPickle'


class Ecoli(Generator):

    defaults = {
        'time_step': 2.0,
        'parallel': False,
        'seed': 0,
        'sim_data_path': SIM_DATA_PATH,
        'daughter_path': tuple(),
    }

    def __init__(self, config):
        super(Ecoli, self).__init__(config)

        self.seed = np.uint32(self.config['seed'] % np.iinfo(np.uint32).max)
        self.random_state = np.random.RandomState(seed = self.seed)

    def initialize_tf_binding(self, sim_data, time_step=1, parallel=False):
        tf_binding_config = {
            'time_step': time_step,
            '_parallel': parallel,

            'tf_ids': sim_data.process.transcription_regulation.tf_ids,
            'delta_prob': sim_data.process.transcription_regulation.delta_prob,
            'n_avogadro': sim_data.constants.n_avogadro,
            'cell_density': sim_data.constants.cell_density,
            'pPromoter_bound_tf': sim_data.process.transcription_regulation.p_promoter_bound_tf,
            'tf_to_tf_type': sim_data.process.transcription_regulation.tf_to_tf_type,
            'active_to_bound': sim_data.process.transcription_regulation.active_to_bound,
            'get_unbound': sim_data.process.equilibrium.get_unbound,
            'active_to_inactive_tf': sim_data.process.two_component_system.active_to_inactive_tf,
            'bulk_molecule_ids': sim_data.internal_state.bulk_molecules.bulk_data["id"],
            'bulk_mass_data': sim_data.internal_state.bulk_molecules.bulk_data["mass"],
            'seed': self.random_state.randint(RAND_MAX)}

        tf_binding = TfBinding(tf_binding_config)
        return tf_binding

    def initialize_transcript_initiation(self, sim_data, time_step=1, parallel=False):
        transcript_initiation_config = {
            'time_step': time_step,
            '_parallel': parallel,

            'fracActiveRnapDict': sim_data.process.transcription.rnapFractionActiveDict,
            'rnaLengths': sim_data.process.transcription.rna_data["length"],
            'rnaPolymeraseElongationRateDict': sim_data.process.transcription.rnaPolymeraseElongationRateDict,
            'variable_elongation': False,
            'make_elongation_rates': sim_data.process.transcription.make_elongation_rates,
            'basal_prob': sim_data.process.transcription_regulation.basal_prob,
            'delta_prob': sim_data.process.transcription_regulation.delta_prob,
            'perturbations': getattr(sim_data, "genetic_perturbations", {}),
            'rna_data': sim_data.process.transcription.rna_data,
            'shuffleIdxs': getattr(sim_data.process.transcription, "initiationShuffleIdxs", None),

            'idx_16SrRNA': np.where(sim_data.process.transcription.rna_data['is_16S_rRNA'])[0],
            'idx_23SrRNA': np.where(sim_data.process.transcription.rna_data['is_23S_rRNA'])[0],
            'idx_5SrRNA': np.where(sim_data.process.transcription.rna_data['is_5S_rRNA'])[0],
            'idx_rRNA': np.where(sim_data.process.transcription.rna_data['is_rRNA'])[0],
            'idx_mRNA': np.where(sim_data.process.transcription.rna_data['is_mRNA'])[0],
            'idx_tRNA': np.where(sim_data.process.transcription.rna_data['is_tRNA'])[0],
            'idx_rprotein': np.where(sim_data.process.transcription.rna_data['is_ribosomal_protein'])[0],
            'idx_rnap': np.where(sim_data.process.transcription.rna_data['is_RNAP'])[0],
            'rnaSynthProbFractions': sim_data.process.transcription.rnaSynthProbFraction,
            'rnaSynthProbRProtein': sim_data.process.transcription.rnaSynthProbRProtein,
            'rnaSynthProbRnaPolymerase': sim_data.process.transcription.rnaSynthProbRnaPolymerase,
            'replication_coordinate': sim_data.process.transcription.rna_data["replication_coordinate"],
            'transcription_direction': sim_data.process.transcription.rna_data["direction"],
            'n_avogadro': sim_data.constants.n_avogadro,
            'cell_density': sim_data.constants.cell_density,
            'inactive_RNAP': 'APORNAP-CPLX[c]',
            'ppgpp': sim_data.molecule_ids.ppGpp,
            'synth_prob': sim_data.process.transcription.synth_prob_from_ppgpp,
            'copy_number': sim_data.process.replication.get_average_copy_number,
            'ppgpp_regulation': False,
            'seed': self.random_state.randint(RAND_MAX)}

        transcript_initiation = TranscriptInitiation(transcript_initiation_config)
        return transcript_initiation

    def initialize_transcript_elongation(self, sim_data, time_step=1, parallel=False):
        transcript_elongation_config = {
            'time_step': time_step,
            '_parallel': parallel,

            'max_time_step': sim_data.process.transcription.max_time_step,
            'rnaPolymeraseElongationRateDict': sim_data.process.transcription.rnaPolymeraseElongationRateDict,
            'rnaIds': sim_data.process.transcription.rna_data['id'],
            'rnaLengths': sim_data.process.transcription.rna_data["length"].asNumber(),
            'rnaSequences': sim_data.process.transcription.transcription_sequences,
            'ntWeights': sim_data.process.transcription.transcription_monomer_weights,
            'endWeight': sim_data.process.transcription.transcription_end_weight,
            'replichore_lengths': sim_data.process.replication.replichore_lengths,
            'idx_16S_rRNA': np.where(sim_data.process.transcription.rna_data['is_16S_rRNA'])[0],
            'idx_23S_rRNA': np.where(sim_data.process.transcription.rna_data['is_23S_rRNA'])[0],
            'idx_5S_rRNA': np.where(sim_data.process.transcription.rna_data['is_5S_rRNA'])[0],
            'is_mRNA': sim_data.process.transcription.rna_data['is_mRNA'],
            'ppi': sim_data.molecule_ids.ppi,
            'inactive_RNAP': "APORNAP-CPLX[c]",
            'ntp_ids': ["ATP[c]", "CTP[c]", "GTP[c]", "UTP[c]"],
            'variable_elongation': False,
            'make_elongation_rates': sim_data.process.transcription.make_elongation_rates,
            'seed': self.random_state.randint(RAND_MAX)}

        transcript_elongation = TranscriptElongation(transcript_elongation_config)
        return transcript_elongation

    def initialize_rna_degradation(self, sim_data, time_step=1, parallel=False):
        rna_degradation_config = {
            'time_step': time_step,
            '_parallel': parallel,

            'rnaIds': sim_data.process.transcription.rna_data['id'],
            'n_avogadro': sim_data.constants.n_avogadro,
            'cell_density': sim_data.constants.cell_density,
            'endoRnaseIds': sim_data.process.rna_decay.endoRNase_ids,
            'exoRnaseIds': sim_data.molecule_groups.exoRNases,
            'KcatExoRNase': sim_data.constants.kcat_exoRNase,
            'KcatEndoRNases': sim_data.process.rna_decay.kcats,
            'charged_trna_names': sim_data.process.transcription.charged_trna_names,
            'rnaDegRates': sim_data.process.transcription.rna_data['deg_rate'],
            'shuffle_indexes': sim_data.process.transcription.rnaDegRateShuffleIdxs if hasattr(sim_data.process.transcription, "rnaDegRateShuffleIdxs") and sim_data.process.transcription.rnaDegRateShuffleIdxs is not None else None,
            'is_mRNA': sim_data.process.transcription.rna_data['is_mRNA'].astype(np.int64),
            'is_rRNA': sim_data.process.transcription.rna_data['is_rRNA'].astype(np.int64),
            'is_tRNA': sim_data.process.transcription.rna_data['is_tRNA'].astype(np.int64),
            'rna_lengths': sim_data.process.transcription.rna_data['length'].asNumber(),
            'polymerized_ntp_ids': sim_data.molecule_groups.polymerized_ntps,
            'water_id': sim_data.molecule_ids.water,
            'ppi_id': sim_data.molecule_ids.ppi,
            'proton_id': sim_data.molecule_ids.proton,
            'counts_ACGU': units.transpose(sim_data.process.transcription.rna_data['counts_ACGU']).asNumber(),
            'nmp_ids': ["AMP[c]", "CMP[c]", "GMP[c]", "UMP[c]"],
            'rrfaIdx': sim_data.process.transcription.rna_data["id"].tolist().index("RRFA-RRNA[c]"),
            'rrlaIdx': sim_data.process.transcription.rna_data["id"].tolist().index("RRLA-RRNA[c]"),
            'rrsaIdx': sim_data.process.transcription.rna_data["id"].tolist().index("RRSA-RRNA[c]"),
            'Km': sim_data.process.transcription.rna_data['Km_endoRNase'],
            'EndoRNaseCoop': sim_data.constants.endoRNase_cooperation,
            'EndoRNaseFunc': sim_data.constants.endoRNase_function,
            'ribosome30S': sim_data.molecule_ids.s30_full_complex,
            'ribosome50S': sim_data.molecule_ids.s50_full_complex,
            'seed': self.random_state.randint(RAND_MAX)}

        rna_degradation = RnaDegradation(rna_degradation_config)
        return rna_degradation

    def initialize_polypeptide_initiation(self, sim_data, time_step=1, parallel=False):
        polypeptide_initiation_config = {
            'time_step': time_step,
            '_parallel': parallel,

            'protein_lengths': sim_data.process.translation.monomer_data["length"].asNumber(),
            'translation_efficiencies': normalize(sim_data.process.translation.translation_efficiencies_by_monomer),
            'active_ribosome_fraction': sim_data.process.translation.ribosomeFractionActiveDict,
            'elongation_rates': sim_data.process.translation.ribosomeElongationRateDict,
            'variable_elongation': False,
            'make_elongation_rates': sim_data.process.translation.make_elongation_rates,
            'protein_index_to_TU_index': sim_data.relation.rna_index_to_monomer_mapping,
            'all_TU_ids': sim_data.process.transcription.rna_data['id'],
            'all_mRNA_ids': sim_data.process.translation.monomer_data['rna_id'],
            'ribosome30S': sim_data.molecule_ids.s30_full_complex,
            'ribosome50S': sim_data.molecule_ids.s50_full_complex,
            'seed': self.random_state.randint(RAND_MAX),
            'shuffle_indexes': sim_data.process.translation.monomer_deg_rate_shuffle_idxs if hasattr(
                sim_data.process.translation, "monomer_deg_rate_shuffle_idxs") else None,
            'seed': self.random_state.randint(RAND_MAX)}

        polypeptide_initiation = PolypeptideInitiation(polypeptide_initiation_config)
        return polypeptide_initiation

    def initialize_polypeptide_elongation(self, sim_data, time_step=1, parallel=False):
        constants = sim_data.constants
        molecule_ids = sim_data.molecule_ids
        translation = sim_data.process.translation
        transcription = sim_data.process.transcription
        metabolism = sim_data.process.metabolism

        variable_elongation = False

        polypeptide_elongation_config = {
            'time_step': time_step,
            '_parallel': parallel,

            # base parameters
            'max_time_step': translation.max_time_step,
            'n_avogadro': constants.n_avogadro,
            'proteinIds': translation.monomer_data['id'],
            'proteinLengths': translation.monomer_data["length"].asNumber(),
            'proteinSequences': translation.translation_sequences,
            'aaWeightsIncorporated': translation.translation_monomer_weights,
            'endWeight': translation.translation_end_weight,
            'variable_elongation': variable_elongation,
            'make_elongation_rates': translation.make_elongation_rates,
            'ribosomeElongationRate': float(sim_data.growth_rate_parameters.ribosomeElongationRate.asNumber(units.aa / units.s)),
            'translation_aa_supply': sim_data.translation_supply_rate,
            'import_threshold': sim_data.external_state.import_constraint_threshold,
            'aa_from_trna': transcription.aa_from_trna,
            'gtpPerElongation': constants.gtp_per_translation,
            'ppgpp_regulation': False,
            'trna_charging': False,
            'translation_supply': False,
            'ribosome30S': sim_data.molecule_ids.s30_full_complex,
            'ribosome50S': sim_data.molecule_ids.s50_full_complex,
            'amino_acids': sim_data.molecule_groups.amino_acids,

            # parameters for specific elongation models
            'basal_elongation_rate': sim_data.constants.ribosome_elongation_rate_basal.asNumber(units.aa / units.s),
            'ribosomeElongationRateDict': sim_data.process.translation.ribosomeElongationRateDict,
            'uncharged_trna_names': sim_data.process.transcription.rna_data['id'][sim_data.process.transcription.rna_data['is_tRNA']],
            'aaNames': sim_data.molecule_groups.amino_acids,
            'proton': sim_data.molecule_ids.proton,
            'water': sim_data.molecule_ids.water,
            'cellDensity': constants.cell_density,
            'elongation_max': constants.ribosome_elongation_rate_max if variable_elongation else constants.ribosome_elongation_rate_basal,
            'aa_from_synthetase': transcription.aa_from_synthetase,
            'charging_stoich_matrix': transcription.charging_stoich_matrix(),
            'charged_trna_names': transcription.charged_trna_names,
            'charging_molecule_names': transcription.charging_molecules,
            'synthetase_names': transcription.synthetase_names,
            'ppgpp_reaction_names': metabolism.ppgpp_reaction_names,
            'ppgpp_reaction_metabolites': metabolism.ppgpp_reaction_metabolites,
            'ppgpp_reaction_stoich': metabolism.ppgpp_reaction_stoich,
            'ppgpp_synthesis_reaction': metabolism.ppgpp_synthesis_reaction,
            'ppgpp_degradation_reaction': metabolism.ppgpp_degradation_reaction,
            'rela': molecule_ids.RelA,
            'spot': molecule_ids.SpoT,
            'ppgpp': molecule_ids.ppGpp,
            'kS': constants.synthetase_charging_rate.asNumber(1 / units.s),
            'KMtf': constants.Km_synthetase_uncharged_trna.asNumber(MICROMOLAR_UNITS),
            'KMaa': constants.Km_synthetase_amino_acid.asNumber(MICROMOLAR_UNITS),
            'krta': constants.Kdissociation_charged_trna_ribosome.asNumber(MICROMOLAR_UNITS),
            'krtf': constants.Kdissociation_uncharged_trna_ribosome.asNumber(MICROMOLAR_UNITS),
            'KD_RelA': constants.KD_RelA_ribosome.asNumber(MICROMOLAR_UNITS),
            'k_RelA': constants.k_RelA_ppGpp_synthesis.asNumber(1 / units.s),
            'k_SpoT_syn': constants.k_SpoT_ppGpp_synthesis.asNumber(1 / units.s),
            'k_SpoT_deg': constants.k_SpoT_ppGpp_degradation.asNumber(1 / (MICROMOLAR_UNITS * units.s)),
            'KI_SpoT': constants.KI_SpoT_ppGpp_degradation.asNumber(MICROMOLAR_UNITS),
            'aa_supply_scaling': metabolism.aa_supply_scaling,
            'seed': self.random_state.randint(RAND_MAX)}

        polypeptide_elongation = PolypeptideElongation(polypeptide_elongation_config)
        return polypeptide_elongation

    def initialize_complexation(self, sim_data, time_step=1, parallel=False):
        complexation_config = {
            'time_step': time_step,
            '_parallel': parallel,

            'stoichiometry': sim_data.process.complexation.stoich_matrix().astype(np.int64).T,
            'rates': sim_data.process.complexation.rates,
            'molecule_names': sim_data.process.complexation.molecule_names,
            'seed': self.random_state.randint(RAND_MAX)}

        complexation = Complexation(complexation_config)
        return complexation

    def initialize_two_component_system(self, sim_data, time_step=1, parallel=False):
        two_component_system_config = {
            'time_step': time_step,
            '_parallel': parallel,

            'jit': False,
            'n_avogadro': sim_data.constants.n_avogadro.asNumber(1 / units.mmol),
            'cell_density': sim_data.constants.cell_density.asNumber(units.g / units.L),
            'moleculesToNextTimeStep': sim_data.process.two_component_system.molecules_to_next_time_step,
            'moleculeNames': sim_data.process.two_component_system.molecule_names,
            'seed': self.random_state.randint(RAND_MAX)}

        two_component_system = TwoComponentSystem(two_component_system_config)
        return two_component_system

    def initialize_equilibrium(self, sim_data, time_step=1, parallel=False):
        equilibrium_config = {
            'time_step': time_step,
            '_parallel': parallel,

            'jit': False,
            'n_avogadro': sim_data.constants.n_avogadro.asNumber(1 / units.mmol),
            'cell_density': sim_data.constants.cell_density.asNumber(units.g / units.L),
            'stoichMatrix': sim_data.process.equilibrium.stoich_matrix().astype(np.int64),
            'fluxesAndMoleculesToSS': sim_data.process.equilibrium.fluxes_and_molecules_to_SS,
            'moleculeNames': sim_data.process.equilibrium.molecule_names,
            'seed': self.random_state.randint(RAND_MAX)}

        equilibrium = Equilibrium(equilibrium_config)
        return equilibrium

    def initialize_protein_degradation(self, sim_data, time_step=1, parallel=False):
        protein_degradation_config = {
            'time_step': time_step,
            '_parallel': parallel,

            'raw_degradation_rate': sim_data.process.translation.monomer_data['deg_rate'].asNumber(1 / units.s),
            'shuffle_indexes': sim_data.process.translation.monomer_deg_rate_shuffle_idxs if hasattr(
                sim_data.process.translation, "monomer_deg_rate_shuffle_idxs") else None,
            'water_id': sim_data.molecule_ids.water,
            'amino_acid_ids': sim_data.molecule_groups.amino_acids,
            'amino_acid_counts': sim_data.process.translation.monomer_data["aa_counts"].asNumber(),
            'protein_ids': sim_data.process.translation.monomer_data['id'],
            'protein_lengths': sim_data.process.translation.monomer_data['length'].asNumber(),
            'seed': self.random_state.randint(RAND_MAX)}

        protein_degradation = ProteinDegradation(protein_degradation_config)
        return protein_degradation

    def initialize_metabolism(self, sim_data, time_step=1, parallel=False):
        metabolism_config = {
            'time_step': time_step,
            '_parallel': parallel,

            'get_import_constraints': sim_data.external_state.get_import_constraints,
            'nutrientToDoublingTime': sim_data.nutrient_to_doubling_time,
            'aa_names': sim_data.molecule_groups.amino_acids,

            # these are options given to the wholecell.sim.simulation
            'use_trna_charging': False,
            'include_ppgpp': False,

            # these values came from the initialized environment state
            'current_timeline': None,
            'media_id': 'minimal',

            'condition': sim_data.condition,
            'nutrients': sim_data.conditions[sim_data.condition]['nutrients'],
            'metabolism': sim_data.process.metabolism,
            'non_growth_associated_maintenance': sim_data.constants.non_growth_associated_maintenance,
            'avogadro': sim_data.constants.n_avogadro,
            'cell_density': sim_data.constants.cell_density,
            'dark_atp': sim_data.constants.darkATP,
            'cell_dry_mass_fraction': sim_data.mass.cell_dry_mass_fraction,
            'get_biomass_as_concentrations': sim_data.mass.getBiomassAsConcentrations,
            'ppgpp_id': sim_data.molecule_ids.ppGpp,
            'get_ppGpp_conc': sim_data.growth_rate_parameters.get_ppGpp_conc,
            'exchange_data_from_media': sim_data.external_state.exchange_data_from_media,
            'get_mass': sim_data.getter.get_mass,
            'doubling_time': sim_data.condition_to_doubling_time[sim_data.condition],
            'amino_acid_ids': sorted(sim_data.amino_acid_code_to_id_ordered.values()),
            'seed': self.random_state.randint(RAND_MAX)}

        metabolism = Metabolism(metabolism_config)
        return metabolism

    def initialize_mass(self, sim_data, time_step=1, parallel=False):
        bulk_ids = sim_data.internal_state.bulk_molecules.bulk_data['id']

        # molecule weight is converted to femtograms/mol
        molecular_weights = {
                molecule_id: sim_data.getter.get_mass([molecule_id]).asNumber(units.fg / units.mol)[0]
                for molecule_id in bulk_ids}

        # unique molecule masses
        unique_masses = {}
        uniqueMoleculeMasses = sim_data.internal_state.unique_molecule.unique_molecule_masses
        for (id_, mass) in zip(uniqueMoleculeMasses["id"], uniqueMoleculeMasses["mass"]):
            unique_masses[id_] = (mass / sim_data.constants.n_avogadro).asNumber(units.fg)

        mass_config = {
            'molecular_weights': molecular_weights,
            'unique_masses': unique_masses,
            'cellDensity': sim_data.constants.cell_density.asNumber(units.g / units.L),
            'water_id': 'WATER[c]',
        }
        mass = Mass(mass_config)
        return mass

    def initialize_division(self, sim_data, time_step=1, parallel=False):
        # TODO -- get mass for division from sim_data
        # TODO -- set divider to binomial division
        divide_config = {
            'threshold': 2220  # fg
        }
        divide_condition = DivideCondition(divide_config)
        return divide_condition

    def initialize_meta_division(self, config, time_step=1, parallel=False):
        daughter_path = config['daughter_path']
        agent_id = config.get('agent_id', str(uuid.uuid1()))
        division_config = dict(
            config.get('division', {}),
            daughter_path=daughter_path,
            agent_id=agent_id,
            compartment=self)
        meta_division = MetaDivision(division_config)
        return meta_division

    def initial_state(self, config=None):
        return get_state_from_file()

    def generate_processes(self, config):
        time_step = config['time_step']
        parallel = config['parallel']  # TODO (Eran) -- which processes can be parallelized?
        sim_data_path = config['sim_data_path']

        # load sim_data
        with open(sim_data_path, 'rb') as sim_data_file:
            sim_data = cPickle.load(sim_data_file)

        # initialize processes
        tf_binding = self.initialize_tf_binding(sim_data, time_step, )
        transcript_initiation = self.initialize_transcript_initiation(sim_data, time_step)
        transcript_elongation = self.initialize_transcript_elongation(sim_data, time_step)
        rna_degradation = self.initialize_rna_degradation(sim_data, time_step)
        polypeptide_initiation = self.initialize_polypeptide_initiation(sim_data, time_step)
        polypeptide_elongation = self.initialize_polypeptide_elongation(sim_data, time_step)
        complexation = self.initialize_complexation(sim_data, time_step)
        two_component_system = self.initialize_two_component_system(sim_data, time_step)
        equilibrium = self.initialize_equilibrium(sim_data, time_step)
        protein_degradation = self.initialize_protein_degradation(sim_data, time_step)
        metabolism = self.initialize_metabolism(sim_data, time_step)
        mass = self.initialize_mass(sim_data, time_step)
        divide_condition = self.initialize_division(sim_data, time_step)
        # meta_division = self.initialize_meta_division(config)

        return {
            'tf_binding': tf_binding,
            'transcript_initiation': transcript_initiation,
            'transcript_elongation': transcript_elongation,
            'rna_degradation': rna_degradation,
            'polypeptide_initiation': polypeptide_initiation,
            'polypeptide_elongation': polypeptide_elongation,
            'complexation': complexation,
            'two_component_system': two_component_system,
            'equilibrium': equilibrium,
            'protein_degradation': protein_degradation,
            'metabolism': metabolism,
            'mass': mass,
            'divide_condition': divide_condition,
            # 'division': meta_division,
        }

    def generate_topology(self, config):
        return {
            'tf_binding': {
                'promoters': ('unique', 'promoter'),
                'active_tfs': ('bulk',),
                'inactive_tfs': ('bulk',),
                'listeners': ('listeners',)},

            'transcript_initiation': {
                'environment': ('environment',),
                'full_chromosomes': ('unique', 'full_chromosome'),
                'RNAs': ('unique', 'RNA'),
                'active_RNAPs': ('unique', 'active_RNAP'),
                'promoters': ('unique', 'promoter'),
                'molecules': ('bulk',),
                'listeners': ('listeners',)},

            'transcript_elongation': {
                'environment': ('environment',),
                'RNAs': ('unique', 'RNA'),
                'active_RNAPs': ('unique', 'active_RNAP'),
                'molecules': ('bulk',),
                'bulk_RNAs': ('bulk',),
                'ntps': ('bulk',),
                'listeners': ('listeners',)},

            'rna_degradation': {
                'charged_trna': ('bulk',),
                'bulk_RNAs': ('bulk',),
                'nmps': ('bulk',),
                'fragmentMetabolites': ('bulk',),
                'fragmentBases': ('bulk',),
                'endoRnases': ('bulk',),
                'exoRnases': ('bulk',),
                'subunits': ('bulk',),
                'molecules': ('bulk',),
                'RNAs': ('unique', 'RNA'),
                'active_ribosome': ('unique', 'active_ribosome'),
                'listeners': ('listeners',)},

            'polypeptide_initiation': {
                'environment': ('environment',),
                'listeners': ('listeners',),
                'active_ribosome': ('unique', 'active_ribosome'),
                'RNA': ('unique', 'RNA'),
                'subunits': ('bulk',)},

            'polypeptide_elongation': {
                'environment': ('environment',),
                'listeners': ('listeners',),
                'active_ribosome': ('unique', 'active_ribosome'),
                'molecules': ('bulk',),
                'monomers': ('bulk',),
                'amino_acids': ('bulk',),
                'ppgpp_reaction_metabolites': ('bulk',),
                'uncharged_trna': ('bulk',),
                'charged_trna': ('bulk',),
                'charging_molecules': ('bulk',),
                'synthetases': ('bulk',),
                'subunits': ('bulk',),
                'polypeptide_elongation': ('process_state', 'polypeptide_elongation')},

            'complexation': {
                'molecules': ('bulk',)},

            'two_component_system': {
                'listeners': ('listeners',),
                'molecules': ('bulk',)},

            'equilibrium': {
                'listeners': ('listeners',),
                'molecules': ('bulk',)},

            'protein_degradation': {
                'metabolites': ('bulk',),
                'proteins': ('bulk',)},

            'metabolism': {
                'metabolites': ('bulk',),
                'catalysts': ('bulk',),
                'kinetics_enzymes': ('bulk',),
                'kinetics_substrates': ('bulk',),
                'amino_acids': ('bulk',),
                'listeners': ('listeners',),
                'environment': ('environment',),
                'polypeptide_elongation': ('process_state', 'polypeptide_elongation')},

            'mass': {
                'bulk': ('bulk',),
                'unique': ('unique',),
                'listeners': ('listeners',)},

            'divide_condition': {
                'variable': ('listeners', 'mass', 'cell_mass'),
                'divide': ('globals', 'divide',),
            },

            # 'division': {
            #     'global': boundary_path,
            #     'agents': agents_path
            # },
        }


def infinitize(value):
    if value == '__INFINITY__':
        return float('inf')
    else:
        return value

def load_states(path):
    with open(path, 'r') as states_file:
        states = json.load(states_file)

    states['environment'] = {
        key: infinitize(value)
        for key, value in states['environment'].items()}

    return states

def get_state_from_file(path='data/wcecoli_t10.json'):

    states = load_states(path)

    initial_state = {
        'environment': {
            'media_id': 'minimal',
            # TODO(Ryan): pull in environmental amino acid levels
            'amino_acids': {},
            'exchange_data': {
                'unconstrained': {
                    'CL-[p]',
                    'FE+2[p]',
                    'CO+2[p]',
                    'MG+2[p]',
                    'NA+[p]',
                    'CARBON-DIOXIDE[p]',
                    'OXYGEN-MOLECULE[p]',
                    'MN+2[p]',
                    'L-SELENOCYSTEINE[c]',
                    'K+[p]',
                    'SULFATE[p]',
                    'ZN+2[p]',
                    'CA+2[p]',
                    'PI[p]',
                    'NI+2[p]',
                    'WATER[p]',
                    'AMMONIUM[c]'},
                'constrained': {
                    'GLC[p]': 20.0 * units.mmol / (units.g * units.h)}},
            'external_concentrations': states['environment']},
        # TODO(Eran): deal with mass
        # add mw property to bulk and unique molecules
        # and include any "submass" attributes from unique molecules
        'listeners': states['listeners'],
        'bulk': states['bulk'],
        'unique': states['unique'],
        'process_state': {
            'polypeptide_elongation': {}}}

    return initial_state


def test_ecoli():
    ecoli = Ecoli({'agent_id': '1'})
    initial_state = get_state_from_file()
    settings = {
        'timestep': 1,
        'total_time': 10,
        'initial_state': initial_state}

    data = simulate_compartment_in_experiment(ecoli, settings)

    return data



def run_ecoli():
    output = test_ecoli()

    # separate data by port
    bulk = output['bulk']
    unique = output['unique']
    listeners = output['listeners']
    process_state = output['process_state']
    environment = output['environment']

    # print(bulk)
    # print(unique.keys())
    pp(listeners['mass'])


if __name__ == '__main__':
    run_ecoli()

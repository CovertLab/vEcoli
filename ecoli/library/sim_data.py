import numpy as np
from six.moves import cPickle
from wholecell.utils import units
from wholecell.utils.fitting import normalize

from ecoli.processes.polypeptide_elongation import MICROMOLAR_UNITS

RAND_MAX = 2**31
SIM_DATA_PATH = 'reconstruction/sim_data/kb/simData.cPickle'


class LoadSimData:

    def __init__(
        self, 
        sim_data_path=SIM_DATA_PATH, 
        seed=0,
        trna_charging=False,
        ppgpp_regulation=False,
        ):

        self.seed = np.uint32(seed % np.iinfo(np.uint32).max)
        self.random_state = np.random.RandomState(seed = self.seed)
        
        self.trna_charging = trna_charging
        self.ppgpp_regulation = ppgpp_regulation

        # load sim_data
        with open(sim_data_path, 'rb') as sim_data_file:
            self.sim_data = cPickle.load(sim_data_file)


    def get_chromosome_replication_config(self, time_step=2, parallel=False):
        get_dna_critical_mass = self.sim_data.mass.get_dna_critical_mass
        doubling_time = self.sim_data.condition_to_doubling_time[self.sim_data.condition]
        chromosome_replication_config = {
            'time_step': time_step,
            '_parallel': parallel,

            'max_time_step': self.sim_data.process.replication.max_time_step,
            'get_dna_critical_mass': get_dna_critical_mass,
            'criticalInitiationMass': get_dna_critical_mass(doubling_time),
            'nutrientToDoublingTime': self.sim_data.nutrient_to_doubling_time,
            'replichore_lengths': self.sim_data.process.replication.replichore_lengths,
            'sequences': self.sim_data.process.replication.replication_sequences,
            'polymerized_dntp_weights': self.sim_data.process.replication.replication_monomer_weights,
            'replication_coordinate': self.sim_data.process.transcription.rna_data['replication_coordinate'],
            'D_period': self.sim_data.growth_rate_parameters.d_period.asNumber(units.s),
            'no_child_place_holder': self.sim_data.process.replication.no_child_place_holder,
            'basal_elongation_rate': int(round(
                self.sim_data.growth_rate_parameters.replisome_elongation_rate.asNumber(units.nt / units.s))),
            'make_elongation_rates': self.sim_data.process.replication.make_elongation_rates,

            # sim options
            'mechanistic_replisome': True,

            # molecules
            'replisome_trimers_subunits': self.sim_data.molecule_groups.replisome_trimer_subunits,
            'replisome_monomers_subunits': self.sim_data.molecule_groups.replisome_monomer_subunits,
            'dntps': self.sim_data.molecule_groups.dntps,
            'ppi': [self.sim_data.molecule_ids.ppi],

            # random state
            'seed': self.random_state.randint(RAND_MAX),
        }


        return chromosome_replication_config

    def get_tf_config(self, time_step=2, parallel=False):
        tf_binding_config = {
            'time_step': time_step,
            '_parallel': parallel,

            'tf_ids': self.sim_data.process.transcription_regulation.tf_ids,
            'delta_prob': self.sim_data.process.transcription_regulation.delta_prob,
            'n_avogadro': self.sim_data.constants.n_avogadro,
            'cell_density': self.sim_data.constants.cell_density,
            'p_promoter_bound_tf': self.sim_data.process.transcription_regulation.p_promoter_bound_tf,
            'tf_to_tf_type': self.sim_data.process.transcription_regulation.tf_to_tf_type,
            'active_to_bound': self.sim_data.process.transcription_regulation.active_to_bound,
            'get_unbound': self.sim_data.process.equilibrium.get_unbound,
            'active_to_inactive_tf': self.sim_data.process.two_component_system.active_to_inactive_tf,
            'bulk_molecule_ids': self.sim_data.internal_state.bulk_molecules.bulk_data["id"],
            'bulk_mass_data': self.sim_data.internal_state.bulk_molecules.bulk_data["mass"],
            'seed': self.random_state.randint(RAND_MAX)}

        return tf_binding_config

    def get_transcript_initiation_config(self, time_step=2, parallel=False):
        transcript_initiation_config = {
            'time_step': time_step,
            '_parallel': parallel,

            'fracActiveRnapDict': self.sim_data.process.transcription.rnapFractionActiveDict,
            'rnaLengths': self.sim_data.process.transcription.rna_data["length"],
            'rnaPolymeraseElongationRateDict': self.sim_data.process.transcription.rnaPolymeraseElongationRateDict,
            'variable_elongation': False,
            'make_elongation_rates': self.sim_data.process.transcription.make_elongation_rates,
            'basal_prob': self.sim_data.process.transcription_regulation.basal_prob,
            'delta_prob': self.sim_data.process.transcription_regulation.delta_prob,
            'get_delta_prob_matrix': self.sim_data.process.transcription_regulation.get_delta_prob_matrix,
            'perturbations': getattr(self.sim_data, "genetic_perturbations", {}),
            'rna_data': self.sim_data.process.transcription.rna_data,
            'shuffleIdxs': getattr(self.sim_data.process.transcription, "initiationShuffleIdxs", None),

            'idx_16SrRNA': np.where(self.sim_data.process.transcription.rna_data['is_16S_rRNA'])[0],
            'idx_23SrRNA': np.where(self.sim_data.process.transcription.rna_data['is_23S_rRNA'])[0],
            'idx_5SrRNA': np.where(self.sim_data.process.transcription.rna_data['is_5S_rRNA'])[0],
            'idx_rRNA': np.where(self.sim_data.process.transcription.rna_data['is_rRNA'])[0],
            'idx_mRNA': np.where(self.sim_data.process.transcription.rna_data['is_mRNA'])[0],
            'idx_tRNA': np.where(self.sim_data.process.transcription.rna_data['is_tRNA'])[0],
            'idx_rprotein': np.where(self.sim_data.process.transcription.rna_data['is_ribosomal_protein'])[0],
            'idx_rnap': np.where(self.sim_data.process.transcription.rna_data['is_RNAP'])[0],
            'rnaSynthProbFractions': self.sim_data.process.transcription.rnaSynthProbFraction,
            'rnaSynthProbRProtein': self.sim_data.process.transcription.rnaSynthProbRProtein,
            'rnaSynthProbRnaPolymerase': self.sim_data.process.transcription.rnaSynthProbRnaPolymerase,
            'replication_coordinate': self.sim_data.process.transcription.rna_data["replication_coordinate"],
            'transcription_direction': self.sim_data.process.transcription.rna_data["direction"],
            'n_avogadro': self.sim_data.constants.n_avogadro,
            'cell_density': self.sim_data.constants.cell_density,
            'inactive_RNAP': 'APORNAP-CPLX[c]',
            'ppgpp': self.sim_data.molecule_ids.ppGpp,
            'synth_prob': self.sim_data.process.transcription.synth_prob_from_ppgpp,
            'copy_number': self.sim_data.process.replication.get_average_copy_number,
            'ppgpp_regulation': self.ppgpp_regulation,

            # attenuation
            'trna_attenuation': False,
            'attenuated_rna_indices': self.sim_data.process.transcription.attenuated_rna_indices,
            'attenuation_adjustments': self.sim_data.process.transcription.attenuation_basal_prob_adjustments,

            # random seed
            'seed': self.random_state.randint(RAND_MAX)
        }

        return transcript_initiation_config

    def get_transcript_elongation_config(self, time_step=2, parallel=False):
        transcript_elongation_config = {
            'time_step': time_step,
            '_parallel': parallel,

            'max_time_step': self.sim_data.process.transcription.max_time_step,
            'rnaPolymeraseElongationRateDict': self.sim_data.process.transcription.rnaPolymeraseElongationRateDict,
            'rnaIds': self.sim_data.process.transcription.rna_data['id'],
            'rnaLengths': self.sim_data.process.transcription.rna_data["length"].asNumber(),
            'rnaSequences': self.sim_data.process.transcription.transcription_sequences,
            'ntWeights': self.sim_data.process.transcription.transcription_monomer_weights,
            'endWeight': self.sim_data.process.transcription.transcription_end_weight,
            'replichore_lengths': self.sim_data.process.replication.replichore_lengths,
            'idx_16S_rRNA': np.where(self.sim_data.process.transcription.rna_data['is_16S_rRNA'])[0],
            'idx_23S_rRNA': np.where(self.sim_data.process.transcription.rna_data['is_23S_rRNA'])[0],
            'idx_5S_rRNA': np.where(self.sim_data.process.transcription.rna_data['is_5S_rRNA'])[0],
            'is_mRNA': self.sim_data.process.transcription.rna_data['is_mRNA'],
            'ppi': self.sim_data.molecule_ids.ppi,
            'inactive_RNAP': "APORNAP-CPLX[c]",
            'ntp_ids': ["ATP[c]", "CTP[c]", "GTP[c]", "UTP[c]"],
            'variable_elongation': False,
            'make_elongation_rates': self.sim_data.process.transcription.make_elongation_rates,

            # attenuation
            'trna_attenuation': False,
            'charged_trna_names': self.sim_data.process.transcription.charged_trna_names,
            'polymerized_ntps': self.sim_data.molecule_groups.polymerized_ntps,
            'cell_density': self.sim_data.constants.cell_density,
            'n_avogadro': self.sim_data.constants.n_avogadro,
            'stop_probabilities': self.sim_data.process.transcription.get_attenuation_stop_probabilities,
            'attenuated_rna_indices': self.sim_data.process.transcription.attenuated_rna_indices,
            'location_lookup': self.sim_data.process.transcription.attenuation_location,

            # random seed
            'seed': self.random_state.randint(RAND_MAX)
        }

        return transcript_elongation_config

    def get_rna_degradation_config(self, time_step=2, parallel=False):
        rna_degradation_config = {
            'time_step': time_step,
            '_parallel': parallel,

            'rnaIds': self.sim_data.process.transcription.rna_data['id'],
            'n_avogadro': self.sim_data.constants.n_avogadro,
            'cell_density': self.sim_data.constants.cell_density,
            'endoRnaseIds': self.sim_data.process.rna_decay.endoRNase_ids,
            'exoRnaseIds': self.sim_data.molecule_groups.exoRNases,
            'KcatExoRNase': self.sim_data.constants.kcat_exoRNase,
            'KcatEndoRNases': self.sim_data.process.rna_decay.kcats,
            'charged_trna_names': self.sim_data.process.transcription.charged_trna_names,
            'rnaDegRates': self.sim_data.process.transcription.rna_data['deg_rate'],
            'shuffle_indexes': self.sim_data.process.transcription.rnaDegRateShuffleIdxs if hasattr(self.sim_data.process.transcription, "rnaDegRateShuffleIdxs") and self.sim_data.process.transcription.rnaDegRateShuffleIdxs is not None else None,
            'is_mRNA': self.sim_data.process.transcription.rna_data['is_mRNA'].astype(np.int64),
            'is_rRNA': self.sim_data.process.transcription.rna_data['is_rRNA'].astype(np.int64),
            'is_tRNA': self.sim_data.process.transcription.rna_data['is_tRNA'].astype(np.int64),
            'rna_lengths': self.sim_data.process.transcription.rna_data['length'].asNumber(),
            'polymerized_ntp_ids': self.sim_data.molecule_groups.polymerized_ntps,
            'water_id': self.sim_data.molecule_ids.water,
            'ppi_id': self.sim_data.molecule_ids.ppi,
            'proton_id': self.sim_data.molecule_ids.proton,
            'counts_ACGU': units.transpose(self.sim_data.process.transcription.rna_data['counts_ACGU']).asNumber(),
            'nmp_ids': ["AMP[c]", "CMP[c]", "GMP[c]", "UMP[c]"],
            'rrfaIdx': self.sim_data.process.transcription.rna_data["id"].tolist().index("RRFA-RRNA[c]"),
            'rrlaIdx': self.sim_data.process.transcription.rna_data["id"].tolist().index("RRLA-RRNA[c]"),
            'rrsaIdx': self.sim_data.process.transcription.rna_data["id"].tolist().index("RRSA-RRNA[c]"),
            'Km': self.sim_data.process.transcription.rna_data['Km_endoRNase'],
            'EndoRNaseCoop': self.sim_data.constants.endoRNase_cooperation,
            'EndoRNaseFunc': self.sim_data.constants.endoRNase_function,
            'ribosome30S': self.sim_data.molecule_ids.s30_full_complex,
            'ribosome50S': self.sim_data.molecule_ids.s50_full_complex,
            'seed': self.random_state.randint(RAND_MAX)}

        return rna_degradation_config

    def get_polypeptide_initiation_config(self, time_step=2, parallel=False):
        polypeptide_initiation_config = {
            'time_step': time_step,
            '_parallel': parallel,

            'protein_lengths': self.sim_data.process.translation.monomer_data["length"].asNumber(),
            'translation_efficiencies': normalize(self.sim_data.process.translation.translation_efficiencies_by_monomer),
            'active_ribosome_fraction': self.sim_data.process.translation.ribosomeFractionActiveDict,
            'elongation_rates': self.sim_data.process.translation.ribosomeElongationRateDict,
            'variable_elongation': False,
            'make_elongation_rates': self.sim_data.process.translation.make_elongation_rates,
            'protein_index_to_TU_index': self.sim_data.relation.RNA_to_monomer_mapping,
            'all_TU_ids': self.sim_data.process.transcription.rna_data['id'],
            'all_mRNA_ids': self.sim_data.process.translation.monomer_data['rna_id'],
            'ribosome30S': self.sim_data.molecule_ids.s30_full_complex,
            'ribosome50S': self.sim_data.molecule_ids.s50_full_complex,
            'seed': self.random_state.randint(RAND_MAX),
            'shuffle_indexes': self.sim_data.process.translation.monomer_deg_rate_shuffle_idxs if hasattr(
                self.sim_data.process.translation, "monomer_deg_rate_shuffle_idxs") else None,
            'seed': self.random_state.randint(RAND_MAX)}

        return polypeptide_initiation_config

    def get_polypeptide_elongation_config(self, time_step=2, parallel=False):
        constants = self.sim_data.constants
        molecule_ids = self.sim_data.molecule_ids
        translation = self.sim_data.process.translation
        transcription = self.sim_data.process.transcription
        metabolism = self.sim_data.process.metabolism

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
            'next_aa_pad': translation.next_aa_pad,
            'ribosomeElongationRate': float(self.sim_data.growth_rate_parameters.ribosomeElongationRate.asNumber(units.aa / units.s)),
            'translation_aa_supply': self.sim_data.translation_supply_rate,
            'import_threshold': self.sim_data.external_state.import_constraint_threshold,
            'aa_from_trna': transcription.aa_from_trna,
            'gtpPerElongation': constants.gtp_per_translation,
            'ppgpp_regulation': self.ppgpp_regulation,
            'mechanistic_supply': False,
            'trna_charging': self.trna_charging,
            'translation_supply': False,
            'ribosome30S': self.sim_data.molecule_ids.s30_full_complex,
            'ribosome50S': self.sim_data.molecule_ids.s50_full_complex,
            'amino_acids': self.sim_data.molecule_groups.amino_acids,

            # parameters for specific elongation models
            'basal_elongation_rate': self.sim_data.constants.ribosome_elongation_rate_basal.asNumber(units.aa / units.s),
            'ribosomeElongationRateDict': self.sim_data.process.translation.ribosomeElongationRateDict,
            'uncharged_trna_names': self.sim_data.process.transcription.rna_data['id'][self.sim_data.process.transcription.rna_data['is_tRNA']],
            'aaNames': self.sim_data.molecule_groups.amino_acids,
            'proton': self.sim_data.molecule_ids.proton,
            'water': self.sim_data.molecule_ids.water,
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
            'aa_enzymes': metabolism.aa_enzymes,
            'amino_acid_synthesis': metabolism.amino_acid_synthesis,
            'amino_acid_import': metabolism.amino_acid_import,
            'seed': self.random_state.randint(RAND_MAX)}

        return polypeptide_elongation_config

    def get_complexation_config(self,  time_step=2, parallel=False):
        complexation_config = {
            'time_step': time_step,
            '_parallel': parallel,

            'stoichiometry': self.sim_data.process.complexation.stoich_matrix().astype(np.int64).T,
            'rates': self.sim_data.process.complexation.rates,
            'molecule_names': self.sim_data.process.complexation.molecule_names,
            'seed': self.random_state.randint(RAND_MAX)}

        return complexation_config

    def get_two_component_system_config(self, time_step=2, parallel=False, random_seed=None):
        two_component_system_config = {
            'time_step': time_step,
            '_parallel': parallel,

            'jit': False,
            'n_avogadro': self.sim_data.constants.n_avogadro.asNumber(1 / units.mmol),  # TODO -- wcEcoli has this in 1/mmol, why?
            'cell_density': self.sim_data.constants.cell_density.asNumber(units.g / units.L),
            'moleculesToNextTimeStep': self.sim_data.process.two_component_system.molecules_to_next_time_step,
            'moleculeNames': self.sim_data.process.two_component_system.molecule_names,
            'seed': random_seed or self.random_state.randint(RAND_MAX)}

        #return two_component_system_config, stoichI, stoichJ, stoichV
        return two_component_system_config

    def get_equilibrium_config(self, time_step=2, parallel=False):
        equilibrium_config = {
            'time_step': time_step,
            '_parallel': parallel,

            'jit': False,
            'n_avogadro': self.sim_data.constants.n_avogadro.asNumber(1 / units.mol),
            'cell_density': self.sim_data.constants.cell_density.asNumber(units.g / units.L),
            'stoichMatrix': self.sim_data.process.equilibrium.stoich_matrix().astype(np.int64),
            'fluxesAndMoleculesToSS': self.sim_data.process.equilibrium.fluxes_and_molecules_to_SS,
            'moleculeNames': self.sim_data.process.equilibrium.molecule_names,
            'seed': self.random_state.randint(RAND_MAX)}

        return equilibrium_config

    def get_protein_degradation_config(self, time_step=2, parallel=False):
        protein_degradation_config = {
            'time_step': time_step,
            '_parallel': parallel,

            'raw_degradation_rate': self.sim_data.process.translation.monomer_data['deg_rate'].asNumber(1 / units.s),
            'shuffle_indexes': self.sim_data.process.translation.monomer_deg_rate_shuffle_idxs if hasattr(
                self.sim_data.process.translation, "monomer_deg_rate_shuffle_idxs") else None,
            'water_id': self.sim_data.molecule_ids.water,
            'amino_acid_ids': self.sim_data.molecule_groups.amino_acids,
            'amino_acid_counts': self.sim_data.process.translation.monomer_data["aa_counts"].asNumber(),
            'protein_ids': self.sim_data.process.translation.monomer_data['id'],
            'protein_lengths': self.sim_data.process.translation.monomer_data['length'].asNumber(),
            'seed': self.random_state.randint(RAND_MAX)}

        return protein_degradation_config

    def get_metabolism_config(self, time_step=2, parallel=False, deriver_mode=False):
        metabolism_config = {
            'time_step': time_step,
            '_parallel': parallel,

            'get_import_constraints': self.sim_data.external_state.get_import_constraints,
            'nutrientToDoublingTime': self.sim_data.nutrient_to_doubling_time,
            'aa_names': self.sim_data.molecule_groups.amino_acids,

            # these are options given to the wholecell.sim.simulation
            'use_trna_charging': self.trna_charging,
            'include_ppgpp': not self.ppgpp_regulation or not self.trna_charging,

            # these values came from the initialized environment state
            'current_timeline': None,
            'media_id': 'minimal',

            'condition': self.sim_data.condition,
            'nutrients': self.sim_data.conditions[self.sim_data.condition]['nutrients'],
            'metabolism': self.sim_data.process.metabolism,
            'non_growth_associated_maintenance': self.sim_data.constants.non_growth_associated_maintenance,
            'avogadro': self.sim_data.constants.n_avogadro,
            'cell_density': self.sim_data.constants.cell_density,
            'dark_atp': self.sim_data.constants.darkATP,
            'cell_dry_mass_fraction': self.sim_data.mass.cell_dry_mass_fraction,
            'get_biomass_as_concentrations': self.sim_data.mass.getBiomassAsConcentrations,
            'ppgpp_id': self.sim_data.molecule_ids.ppGpp,
            'get_ppGpp_conc': self.sim_data.growth_rate_parameters.get_ppGpp_conc,
            'exchange_data_from_media': self.sim_data.external_state.exchange_data_from_media,
            'get_masses': self.sim_data.getter.get_masses,
            'doubling_time': self.sim_data.condition_to_doubling_time[self.sim_data.condition],
            'amino_acid_ids': sorted(self.sim_data.amino_acid_code_to_id_ordered.values()),
            'seed': self.random_state.randint(RAND_MAX),
            'linked_metabolites': self.sim_data.process.metabolism.linked_metabolites,
            # Whether to use metabolism as a deriver (with t=0 skipped)
            'deriver_mode': deriver_mode
        }

        return metabolism_config

    def get_mass_config(self, time_step=2, parallel=False):

        bulk_ids = self.sim_data.internal_state.bulk_molecules.bulk_data['id']
        molecular_weights = {}
        for molecule_id in bulk_ids:
            molecular_weights[molecule_id] = self.sim_data.getter.get_mass(
                molecule_id).asNumber(units.fg / units.mol)

        # unique molecule masses
        unique_masses = {}
        uniqueMoleculeMasses = self.sim_data.internal_state.unique_molecule.unique_molecule_masses
        for (id_, mass) in zip(uniqueMoleculeMasses["id"], uniqueMoleculeMasses["mass"]):
            unique_masses[id_] = (mass / self.sim_data.constants.n_avogadro).asNumber(units.fg)

        mass_config = {
            'molecular_weights': molecular_weights,
            'unique_masses': unique_masses,
            'cellDensity': self.sim_data.constants.cell_density.asNumber(units.g / units.L),
            'water_id': 'WATER[c]',
        }
        return mass_config
    
    def get_allocator_config(self, time_step=2, parallel=False, process_names=[]):
        allocator_config = {
            'time_step': time_step, 
            'molecule_names': self.sim_data.internal_state.bulk_molecules.bulk_data['id'],
            'seed': self.random_state.randint(2**31),
            'process_names': process_names,
        }
        return allocator_config

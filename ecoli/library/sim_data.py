import re
import binascii
import numpy as np
import networkx as nx
from six.moves import cPickle
from wholecell.utils import units
from wholecell.utils.unit_struct_array import UnitStructArray
from wholecell.utils.fitting import normalize

from ecoli.processes.polypeptide_elongation import MICROMOLAR_UNITS
from ecoli.library.parameters import param_store

RAND_MAX = 2**31
SIM_DATA_PATH = 'reconstruction/sim_data/kb/simData.cPickle'

class LoadSimData:

    def __init__(
        self,
        sim_data_path=SIM_DATA_PATH,
        seed=0,
        trna_charging=False,
        ppgpp_regulation=False,
        mar_regulon=False,
        rnai_data=None,
    ):

        self.seed = seed
        self.random_state = np.random.RandomState(seed = seed)

        self.trna_charging = trna_charging
        self.ppgpp_regulation = ppgpp_regulation
        
        # NEW to vivarium-ecoli: Whether to lump miscRNA with mRNAs
        # when calculating degradation
        self.degrade_misc = False

        # load sim_data
        with open(sim_data_path, 'rb') as sim_data_file:
            self.sim_data = cPickle.load(sim_data_file)

        self.submass_indexes = {
            f'massDiff_{submass}': idx
            for submass, idx in self.sim_data.submass_name_to_index.items()
        }
        
        # NEW to vivarium-ecoli
        # Changes gene expression upon tetracycline exposure
        if mar_regulon:
            # Define aliases to reduce code verbosity
            treg_alias = self.sim_data.process.transcription_regulation
            bulk_mol_alias =  self.sim_data.internal_state.bulk_molecules
            eq_alias = self.sim_data.process.equilibrium

            # Assume marA (PD00365) controls the entire tetracycline
            # gene expression program and marR (CPLX0-7710) is inactivated
            # by complexation with tetracycline
            treg_alias.tf_ids += ["PD00365", "CPLX0-7710"]
            treg_alias.delta_prob["shape"] = (
                    treg_alias.delta_prob["shape"][0], 
                    treg_alias.delta_prob["shape"][1]+2)
            treg_alias.tf_to_tf_type["PD00365"] = "0CS"
            treg_alias.tf_to_tf_type["CPLX0-7710"] = "1CS"
            treg_alias.active_to_bound["CPLX0-7710"] = "marR-tet"
            
            # TU index of genes for outer membrane proteins, regulators,
            # and inner membrane transporters
            TU_idxs = [2493, 2011, 1641, 1394, 2112, 1642, 1543, 662, 995,
                3289, 262, 1807, 2010, 659, 1395, 260, 259, 11, 944, 1631,
                1330, 660, 1399, 661]
            new_deltaI = np.array(TU_idxs)
            new_deltaJ = np.array([24]*24)
            # Values were chosen to recapitulate mRNA fold change when exposed 
            # to 1.5 mg/L tetracycline (Viveiros et al. 2007)
            new_deltaV = np.array([1.76e-3, 1.96e-4, 2.95e-5, 2.21e-5, 2.1e-6,
                2.02e-5, -5.3e-7, 7.8e-6, 8.08e-6, 1.58e-7, 7.42e-6, 1.51e-6,
                2.04e-6, 2.34e-4, 4.11e-6, 7.17e-8, 1.4e-8, 8.89e-8, 2.44e-5,
                1.68e-8, 8.09e-6, 5.77e-5, 8.52e-7, 5.4e-4])
            
            treg_alias.delta_prob["deltaI"] = np.concatenate(
                [treg_alias.delta_prob["deltaI"], new_deltaI])
            treg_alias.delta_prob["deltaJ"] = np.concatenate(
                [treg_alias.delta_prob["deltaJ"], new_deltaJ])
            treg_alias.delta_prob["deltaV"] = np.concatenate(
                [treg_alias.delta_prob["deltaV"], new_deltaV])
            
            # Add mass data for tetracycline and marR-tetracycline complex
            bulk_data = bulk_mol_alias.bulk_data.fullArray()
            bulk_data = np.resize(bulk_data, bulk_data.shape[0]+3)
            tet_mass = param_store.get(('tetracycline', 'mass')).magnitude
            bulk_data[-1] = ('tetracycline[c]', 
                [0, 0, 0, 0, 0, 0, tet_mass, 0, 0])
            bulk_data[-2] = ('tetracycline[p]', 
                [0, 0, 0, 0, 0, 0, tet_mass, 0, 0])
            # Protein mass is 6th element in 2nd column of each row in
            # the Numpy structured array bulk_data
            marR_mass = bulk_data[bulk_data['id'] == 'CPLX0-7710[c]'][0][1][5]
            bulk_data[-3] = ('marR-tet[c]', 
                [0, 0, 0, 0, 0, marR_mass, tet_mass, 0, 0])
            bulk_units = bulk_mol_alias.bulk_data.fullUnits()
            bulk_mol_alias.bulk_data = UnitStructArray(bulk_data, bulk_units)
            
            # Add equilibrium reaction for marR-tetracycline and 
            # reinitialize self.sim_data.process.equilibrium variables
            eq_alias._stoichMatrixI = np.concatenate(
                [eq_alias._stoichMatrixI, np.array([98, 99, 100])])
            eq_alias._stoichMatrixJ = np.concatenate(
                [eq_alias._stoichMatrixJ, np.array([34, 34, 34])])
            eq_alias._stoichMatrixV = np.concatenate(
                [eq_alias._stoichMatrixV, np.array([-1, -1, 1])])
            eq_alias.molecule_names += [
                'CPLX0-7710[c]', 'tetracycline[c]', 'marR-tet[c]']
            eq_alias.ids_complexes = [
                eq_alias.molecule_names[i] 
                for i in np.where(np.any(
                    eq_alias.stoich_matrix() > 0, axis=1))[0]]
            eq_alias.rxn_ids += ['marR-tet']
            # All existing equilibrium rxns use a forward rate of 1
            eq_alias.rates_fwd = np.concatenate(
                [eq_alias.rates_fwd, np.array([1])])
            # Existing equilibrium rxns use a default reverse rate of 1e-6
            # This happens to nearly perfectly yield full MarR inactivation
            # at 1.5 mg/L external tetracycline
            eq_alias.rates_rev = np.concatenate(
                [eq_alias.rates_rev, np.array([1e-6])])

            # Mass balance matrix
            eq_alias._stoichMatrixMass = np.concatenate(
                [eq_alias._stoichMatrixMass, np.array(
                    [marR_mass, tet_mass, marR_mass+tet_mass])])
            eq_alias.balance_matrix = (
                eq_alias.stoich_matrix() * 
                eq_alias.mass_matrix())

            # Find the mass balance of each equation in the balanceMatrix
            massBalanceArray = eq_alias.mass_balance()

            # The stoichometric matrix should balance out to numerical zero.
            assert np.max(np.absolute(massBalanceArray)) < 1e-9

            # Build matrices
            eq_alias._populateDerivativeAndJacobian()
            eq_alias._stoichMatrix = eq_alias.stoich_matrix()
        
        # NEW to vivarium-ecoli
        # Append new RNA IDs and degradation rates for sRNA-mRNA duplexes
        if rnai_data:
            # Define aliases to reduce code verbosity
            ts_alias = self.sim_data.process.transcription
            bulk_mol_alias = self.sim_data.internal_state.bulk_molecules
            treg_alias = self.sim_data.process.transcription_regulation

            self.duplex_ids = np.array(rnai_data['duplex_ids'])
            n_duplex_rnas = len(self.duplex_ids)
            duplex_deg_rates = np.array(rnai_data['duplex_deg_rates'])
            duplex_km = np.array(rnai_data['duplex_km'])
            duplex_na = np.zeros(n_duplex_rnas)
            # Mark duplexes as miscRNAs so they are degraded appropriately
            duplex_is_miscRNA = np.ones(n_duplex_rnas, dtype=np.bool_)
            
            self.srna_ids = np.array(rnai_data['srna_ids'])
            target_ids = np.array(rnai_data['target_ids'])
            self.target_tu_ids = np.zeros(len(target_ids), dtype=int)
            
            self.binding_probs = np.array(rnai_data['binding_probs'])
            
            # Get duplex length, ACGU content, molecular weight, and sequence
            duplex_lengths = np.zeros(n_duplex_rnas)
            duplex_ACGU = np.zeros((n_duplex_rnas, 4))
            duplex_mw = np.zeros(n_duplex_rnas)
            rna_data = ts_alias.rna_data.fullArray()
            rna_units = ts_alias.rna_data.fullUnits()
            rna_sequences = ts_alias.transcription_sequences
            duplex_sequences = np.full(
                (n_duplex_rnas, rna_sequences.shape[1]), -1)
            for i, (srna_id, target_id) in enumerate(
                zip(self.srna_ids, target_ids)):
                # Use first match for each sRNA and target mRNA
                srna_tu_id = np.where(rna_data['id']==srna_id)[0][0]
                self.target_tu_ids[i] = np.where(
                    rna_data['id']==target_id)[0][0]
                duplex_ACGU[i] = (rna_data['counts_ACGU'][srna_tu_id] + 
                    rna_data['counts_ACGU'][self.target_tu_ids[i]])
                duplex_mw[i] = (rna_data['mw'][srna_tu_id] + 
                    rna_data['mw'][self.target_tu_ids[i]])
                srna_length = rna_data['length'][srna_tu_id]
                target_length = rna_data['length'][self.target_tu_ids[i]]
                duplex_lengths[i] = srna_length + target_length
                if duplex_lengths[i] > duplex_sequences.shape[1]:
                    # Extend columns in sequence arrays to accomodate duplexes
                    # where the sum of the RNA lengths > # of columns
                    extend_length = (
                        duplex_lengths[i] - duplex_sequences.shape[1])
                    extend_duplex_sequences = np.full(
                        (duplex_sequences.shape[0], extend_length), 
                        -1, dtype=duplex_sequences.dtype)
                    duplex_sequences = np.append(
                        duplex_sequences, extend_duplex_sequences, axis=1)
                    extend_rna_sequences = np.full(
                        (rna_sequences.shape[0], extend_length), 
                        -1, dtype=rna_sequences.dtype)
                    rna_sequences = np.append(
                        rna_sequences, extend_rna_sequences, axis=1)
                duplex_sequences[i, :srna_length] = rna_sequences[
                    srna_tu_id][:srna_length] 
                duplex_sequences[i, srna_length:srna_length+target_length
                    ] = rna_sequences[self.target_tu_ids[i]][:target_length]
            
            # Make duplex metadata visible to all RNA-related processes
            old_n_rnas = rna_data.shape[0]
            rna_data = np.resize(rna_data, old_n_rnas+n_duplex_rnas)
            rna_sequences = np.resize(rna_sequences, (
                old_n_rnas+n_duplex_rnas, rna_sequences.shape[1]))
            for i, new_rna in enumerate(zip(self.duplex_ids, duplex_deg_rates,
                duplex_lengths, duplex_ACGU, duplex_mw, duplex_na,
                duplex_is_miscRNA, duplex_na, duplex_na, duplex_na, duplex_na,
                duplex_na, duplex_na, duplex_na, duplex_na, duplex_km,
                duplex_na, duplex_na)
            ):
                rna_data[old_n_rnas+i] = new_rna
                rna_sequences[old_n_rnas+i] = duplex_sequences[i]
            ts_alias.transcription_sequences = rna_sequences
            ts_alias.rna_data = UnitStructArray(rna_data, rna_units)
            
            # Add bulk mass data for duplexes to avoid errors (though mRNAs
            # should never go to bulk)
            bulk_data = bulk_mol_alias.bulk_data.fullArray()
            bulk_units = bulk_mol_alias.bulk_data.fullUnits()
            old_n_bulk = bulk_data.shape[0]
            bulk_data = np.resize(bulk_data, bulk_data.shape[0]+n_duplex_rnas)
            for i, duplex in enumerate(self.duplex_ids):
                duplex_submasses = np.zeros(9)
                duplex_submasses[2] = duplex_mw[i]
                bulk_data[old_n_bulk+i] = (duplex, duplex_submasses)
            bulk_mol_alias.bulk_data = UnitStructArray(bulk_data, bulk_units)
            
            # Add filler transcription data for duplex RNAs to prevent errors
            treg_alias.basal_prob = np.append(treg_alias.basal_prob, 0)
            treg_alias.delta_prob['shape'] = (
                treg_alias.delta_prob['shape'][0] + 1,
                treg_alias.delta_prob['shape'][1])

            # Set flag so miscRNA duplexes are degraded together with mRNAs
            self.degrade_misc = True

    def get_monomer_counts_indices(self, names):
        """Given a list of monomer names without location tags, this returns
        the indices of those monomers in the monomer_counts listener array.
        The "id" column of reconstruction/ecoli/flat/proteins.tsv contains
        nearly all supported monomer names."""
        monomer_ids = self.sim_data.process.translation.monomer_data["id"]
        # Strip location string (e.g. [c])
        monomer_ids = np.array([re.split(r'\[.\]', monomer)[0]
                                for monomer in monomer_ids])
        return [int(np.where(monomer_ids==name)[0][0]) for name in names]

    def get_mrna_counts_indices(self, names):
        """Given a list of mRNA names without location tags, this returns
        the indices of those mRNAs in the mRNA_counts listener array.
        The "id" column of reconstruction/ecoli/flat/rnas.tsv contains
        nearly all supported mRNA names."""
        is_mrna = self.sim_data.process.transcription.rna_data['is_mRNA']
        mrna_ids = self.sim_data.process.transcription.rna_data[
            'id'][is_mrna]
        # Strip location string (e.g. [c])
        mrna_ids = np.array([re.split(r'\[.\]', mrna)[0]
                            for mrna in mrna_ids])
        return [int(np.where(mrna_ids==name)[0][0]) for name in names]
    
    def get_rna_indices(self, names):
        """Given a list of RNA names without location tags, this returns
        the TU indices of those RNAs (for rnaInitEvents and rna_synth_prob).
        The "id" column of reconstruction/ecoli/flat/rnas.tsv contains
        nearly all supported RNA names."""
        rna_ids = self.sim_data.process.transcription.rna_data['id']
        # Strip location string (e.g. [c])
        rna_ids = np.array([re.split(r'\[.\]', mrna)[0]
                            for mrna in rna_ids])
        return [int(np.where(rna_ids==name)[0][0]) for name in names]
            
    def _seedFromName(self, name):
        return binascii.crc32(name.encode('utf-8'), self.seed) & 0xffffffff

    def get_config_by_name(self, name, time_step=2, parallel=False):
        name_config_mapping = {
            'ecoli-tf-binding': self.get_tf_config,
            'ecoli-transcript-initiation': self.get_transcript_initiation_config,
            'ecoli-transcript-elongation': self.get_transcript_elongation_config,
            'ecoli-rna-degradation': self.get_rna_degradation_config,
            'ecoli-polypeptide-initiation': self.get_polypeptide_initiation_config,
            'ecoli-polypeptide-elongation': self.get_polypeptide_elongation_config,
            'ecoli-complexation': self.get_complexation_config,
            'ecoli-two-component-system': self.get_two_component_system_config,
            'ecoli-equilibrium': self.get_equilibrium_config,
            'ecoli-protein-degradation': self.get_protein_degradation_config,
            'ecoli-metabolism': self.get_metabolism_config,
            'ecoli-metabolism-gradient-descent': self.get_metabolism_gd_config,
            'ecoli-metabolism-redux': self.get_metabolism_redux_config,
            'ecoli-chromosome-replication': self.get_chromosome_replication_config,
            'ecoli-mass': self.get_mass_config,
            'ecoli-mass-listener': self.get_mass_listener_config,
            'mRNA_counts_listener': self.get_mrna_counts_listener_config,
            'monomer_counts_listener': self.get_monomer_counts_listener_config,
            'allocator': self.get_allocator_config,
            'ecoli-chromosome-structure': self.get_chromosome_structure_config,
            'ecoli-rna-interference': self.get_rna_interference_config,
            'tetracycline-ribosome-equilibrium': self.get_tetracycline_ribosome_equilibrium_config
        }

        try:
            return name_config_mapping[name](time_step=time_step, parallel=parallel)
        except KeyError:
            raise KeyError(
                f"Process of name {name} is not known to LoadSimData.get_config_by_name")

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
            'seed': self._seedFromName('ChromosomeReplication'),

            'submass_indexes': self.submass_indexes,
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
            'seed': self._seedFromName('TfBinding'),
            'submass_to_idx': self.sim_data.submass_name_to_index}

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
            'seed': self._seedFromName('TranscriptInitiation')
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
            'fragmentBases': self.sim_data.molecule_groups.polymerized_ntps,
            'charged_trnas': self.sim_data.process.transcription.charged_trna_names,

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
            'seed': self._seedFromName('TranscriptElongation'),

            'submass_indexes': self.submass_indexes,
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
            'shuffle_indexes': self.sim_data.process.transcription.rnaDegRateShuffleIdxs
                if hasattr(self.sim_data.process.transcription, "rnaDegRateShuffleIdxs")
                and self.sim_data.process.transcription.rnaDegRateShuffleIdxs is not None
                else None,
            'is_mRNA': self.sim_data.process.transcription.rna_data['is_mRNA'].astype(np.int64),
            'is_rRNA': self.sim_data.process.transcription.rna_data['is_rRNA'].astype(np.int64),
            'is_tRNA': self.sim_data.process.transcription.rna_data['is_tRNA'].astype(np.int64),
            # NEW to vivarium-ecoli, used to degrade duplexes from RNAi
            'is_miscRNA': self.sim_data.process.transcription.rna_data['is_miscRNA'].astype(np.int64),
            'degrade_misc': self.degrade_misc,
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
            'seed': self._seedFromName('RnaDegradation')}

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
            'seed': self._seedFromName('PolypeptideInitiation'),
            'shuffle_indexes': self.sim_data.process.translation.monomer_deg_rate_shuffle_idxs if hasattr(
                self.sim_data.process.translation, "monomer_deg_rate_shuffle_idxs") else None}

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
            'seed': self._seedFromName('PolypeptideElongation'),

            'submass_indexes': self.submass_indexes, }

        return polypeptide_elongation_config

    def get_complexation_config(self, time_step=2, parallel=False):
        complexation_config = {
            'time_step': time_step,
            '_parallel': parallel,

            'stoichiometry': self.sim_data.process.complexation.stoich_matrix().astype(np.int64).T,
            'rates': self.sim_data.process.complexation.rates,
            'molecule_names': self.sim_data.process.complexation.molecule_names,
            'seed': self._seedFromName('Complexation'),
            'numReactions': len(self.sim_data.process.complexation.rates),
        }

        return complexation_config

    def get_two_component_system_config(self, time_step=2, parallel=False):
        two_component_system_config = {
            'time_step': time_step,
            '_parallel': parallel,

            'jit': True,
            # TODO -- wcEcoli has this in 1/mmol, why?
            'n_avogadro': self.sim_data.constants.n_avogadro.asNumber(1 / units.mmol),
            'cell_density': self.sim_data.constants.cell_density.asNumber(units.g / units.L),
            'moleculesToNextTimeStep': self.sim_data.process.two_component_system.molecules_to_next_time_step,
            'moleculeNames': self.sim_data.process.two_component_system.molecule_names,
            'seed': self._seedFromName('TwoComponentSystem')}

        # return two_component_system_config, stoichI, stoichJ, stoichV
        return two_component_system_config

    def get_equilibrium_config(self, time_step=2, parallel=False):
        equilibrium_config = {
            'time_step': time_step,
            '_parallel': parallel,

            'jit': True,
            'n_avogadro': self.sim_data.constants.n_avogadro.asNumber(1 / units.mol),
            'cell_density': self.sim_data.constants.cell_density.asNumber(units.g / units.L),
            'stoichMatrix': self.sim_data.process.equilibrium.stoich_matrix().astype(np.int64),
            'fluxesAndMoleculesToSS': self.sim_data.process.equilibrium.fluxes_and_molecules_to_SS,
            'moleculeNames': self.sim_data.process.equilibrium.molecule_names,
            'seed': self._seedFromName('Equilibrium')}

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
            'seed': self._seedFromName('ProteinDegradation')}

        return protein_degradation_config

    def get_metabolism_gd_config(self, time_step=2, parallel=False, deriver_mode=False):
        # Create reversible reactions from "reaction pairs" similar to original sim_data format.
        stoichiometric_matrix_dict = dict(self.sim_data.process.metabolism.reaction_stoich)
        stoichiometric_matrix_dict = dict(sorted(stoichiometric_matrix_dict.items()))
        rxns = list()
        metabolite_names = set()
        homeostatic_obj_metabolites = \
            self.sim_data.process.metabolism.concentration_updates.concentrations_based_on_nutrients(
                self.sim_data.conditions[self.sim_data.condition]['nutrients']
            ).keys()

        # TODO (Cyrus) Below operations are redundant (renaming, catalyst rearranging) and should just be removed.
        #  from the metabolism dataclass. Are catalysts all required? Or all possible ways to catalyze. Latter.
        reaction_catalysts = self.sim_data.process.metabolism.reaction_catalysts
        catalyst_ids = self.sim_data.process.metabolism.catalyst_ids
        reactions_with_catalyst = self.sim_data.process.metabolism.reactions_with_catalyst

        REVERSE_TAG = ' (reverse)'

        # TODO Consider moving separation of reactions into metabolism reaction. Is it even necessary?
        # Also add check for stoichiometries being equal for removed reverse reactions

        # First pass. Add all reactions without tag.
        # TODO (Cyrus) Investigate how many reactions are supposed to be reversible.
        for key, value in stoichiometric_matrix_dict.items():
            metabolite_names.update(list(value.keys()))

            if not key.endswith(REVERSE_TAG):
                rxns.append({'reaction id': key,
                             'stoichiometry': value,
                             'is reversible': False})
            elif key.endswith(REVERSE_TAG) and rxns[-1]['reaction id'] == key[:-(len(REVERSE_TAG))]:
                rxns[-1]['is reversible'] = True
            # TODO (Cyrus) What to do about reactions with (reverse) tag that actually don't have the original reaction?
            # probably from reactions with multiple forward reactions.
            elif key.endswith(REVERSE_TAG):
                rxns.append({'reaction id': key,
                             'stoichiometry': value,
                             'is reversible': False})

            # Add enzyme to reactions
            if key in reactions_with_catalyst:
                rxns[-1]['enzyme'] = reaction_catalysts[key]
            else:
                rxns[-1]['enzyme'] = []

        rxn_names = [rxn['reaction id'] for rxn in rxns]

        # Avoids issues with index based recognition of reactions later
        kinetic_reactions = [rxn if rxn in rxn_names else None for rxn in self.sim_data.process.metabolism.kinetic_constraint_reactions]

        # Carbon source limitations.
        carbon_source_active_transport = ['TRANS-RXN-157-PTSH-PHOSPHORYLATED/GLC//ALPHA-GLC-6-P/PTSH-MONOMER.52.',
                                           'TRANS-RXN-157-PTSH-PHOSPHORYLATED/GLC//D-glucopyranose-6-phosphate'
                                           '/PTSH-MONOMER.66.',
                                           'TRANS-RXN-157-PTSH-PHOSPHORYLATED/GLC//GLC-6-P/PTSH-MONOMER.46.']

        carbon_source_active_transport_duplicate = ['TRANS-RXN-320-GLC/ATP/WATER//ALPHA-GLUCOSE/ADP/Pi/PROTON.43.',
                                                         'TRANS-RXN-320-GLC/ATP/WATER//GLC/ADP/Pi/PROTON.33.',
                                                         'TRANS-RXN-320-GLC/ATP/WATER//Glucopyranose/ADP/Pi/PROTON.43.']

        carbon_source_facilitated_diffusion = ['RXN0-7077-GLC/PROTON//ALPHA-GLUCOSE/PROTON.33.',
                                                    'RXN0-7077-GLC/PROTON//Glucopyranose/PROTON.33.',
                                                    'RXN0-7077-GLC/PROTON//GLC/PROTON.23.',
                                                    'TRANS-RXN0-574-GLC//GLC.9.',
                                                    'TRANS-RXN0-574-GLC//Glucopyranose.19.']


        # TODO Reconstruct catalysis and annotate.
        # Required:

        metabolism_config = {
            'time_step': time_step,
            '_parallel': parallel,

            # variables
            'stoichiometry': self.sim_data.process.metabolism.reaction_stoich,
            'stoichiometry_r': rxns,
            'metabolite_names': metabolite_names,
            'reaction_catalysts': self.sim_data.process.metabolism.reaction_catalysts,
            'maintenance_reaction': self.sim_data.process.metabolism.maintenance_reaction,
            'aa_names': self.sim_data.molecule_groups.amino_acids,
            'media_id': self.sim_data.conditions[self.sim_data.condition]['nutrients'],
            'avogadro': self.sim_data.constants.n_avogadro,
            'cell_density': self.sim_data.constants.cell_density,
            'nutrientToDoublingTime': self.sim_data.nutrient_to_doubling_time,
            'dark_atp': self.sim_data.constants.darkATP,
            'non_growth_associated_maintenance': self.sim_data.constants.non_growth_associated_maintenance,
            'cell_dry_mass_fraction': self.sim_data.mass.cell_dry_mass_fraction,
            'seed': self.random_state.randint(RAND_MAX),
            'reactions_with_catalyst': self.sim_data.process.metabolism.reactions_with_catalyst,
            'kinetic_constraint_reactions': kinetic_reactions,

            # methods
            'concentration_updates': self.sim_data.process.metabolism.concentration_updates,
            'get_biomass_as_concentrations': self.sim_data.mass.getBiomassAsConcentrations,
            'exchange_data_from_media': self.sim_data.external_state.exchange_data_from_media,
            'doubling_time': self.sim_data.condition_to_doubling_time[self.sim_data.condition],
            'get_kinetic_constraints': self.sim_data.process.metabolism.get_kinetic_constraints,
            'exchange_constraints': self.sim_data.process.metabolism.exchange_constraints,

            # ports schema
            'catalyst_ids': self.sim_data.process.metabolism.catalyst_ids,
            'kinetic_constraint_enzymes': self.sim_data.process.metabolism.kinetic_constraint_enzymes,
            'kinetic_constraint_substrates': self.sim_data.process.metabolism.kinetic_constraint_substrates,
            'deriver_mode': deriver_mode,

            # new parameters
            'carbon_source_active_transport': carbon_source_active_transport,
            'carbon_source_active_transport_duplicate': carbon_source_active_transport_duplicate,
            'carbon_source_facilitated_diffusion': carbon_source_facilitated_diffusion,

        }

        # TODO Create new config-get with only necessary parts.

        return metabolism_config

    def get_metabolism_redux_config(self, time_step=2, parallel=False, deriver_mode=False):

        stoichiometric_matrix_dict = dict(self.sim_data.process.metabolism.reaction_stoich)
        stoichiometric_matrix_dict = dict(sorted(stoichiometric_matrix_dict.items()))

        reaction_catalysts = self.sim_data.process.metabolism.reaction_catalysts
        catalyst_ids = self.sim_data.process.metabolism.catalyst_ids
        reactions_with_catalyst = self.sim_data.process.metabolism.reactions_with_catalyst

        rxns = list()

        # TODO Reconstruct catalysis and annotate.
        for key, value in stoichiometric_matrix_dict.items():

            rxns.append({'reaction id': key, 'stoichiometry': value})

            # Add enzyme to reactions
            if key in reactions_with_catalyst:
                rxns[-1]['enzyme'] = reaction_catalysts[key]
            else:
                rxns[-1]['enzyme'] = []
        # Required:

        metabolism_config = {
            'time_step': time_step,
            '_parallel': parallel,

            # variables
            # 'stoichiometry': self.sim_data.process.metabolism.reaction_stoich,
            'stoichiometry': rxns, # upseparates rxns and catalysts.
            'reaction_catalysts': self.sim_data.process.metabolism.reaction_catalysts,
            'maintenance_reaction': self.sim_data.process.metabolism.maintenance_reaction,
            'aa_names': self.sim_data.molecule_groups.amino_acids,
            'media_id': self.sim_data.conditions[self.sim_data.condition]['nutrients'],
            'avogadro': self.sim_data.constants.n_avogadro,
            'cell_density': self.sim_data.constants.cell_density,
            'nutrientToDoublingTime': self.sim_data.nutrient_to_doubling_time,
            'dark_atp': self.sim_data.constants.darkATP,
            'non_growth_associated_maintenance': self.sim_data.constants.non_growth_associated_maintenance,
            'cell_dry_mass_fraction': self.sim_data.mass.cell_dry_mass_fraction,
            'seed': self.random_state.randint(RAND_MAX),
            'reactions_with_catalyst': self.sim_data.process.metabolism.reactions_with_catalyst,
            'kinetic_constraint_reactions': self.sim_data.process.metabolism.kinetic_constraint_reactions,

            # methods
            'concentration_updates': self.sim_data.process.metabolism.concentration_updates,
            'get_biomass_as_concentrations': self.sim_data.mass.getBiomassAsConcentrations,
            'exchange_data_from_media': self.sim_data.external_state.exchange_data_from_media,
            'doubling_time': self.sim_data.condition_to_doubling_time[self.sim_data.condition],
            'get_kinetic_constraints': self.sim_data.process.metabolism.get_kinetic_constraints,
            'exchange_constraints': self.sim_data.process.metabolism.exchange_constraints,

            # ports schema
            'catalyst_ids': self.sim_data.process.metabolism.catalyst_ids,
            'kinetic_constraint_enzymes': self.sim_data.process.metabolism.kinetic_constraint_enzymes,
            'kinetic_constraint_substrates': self.sim_data.process.metabolism.kinetic_constraint_substrates,
            'deriver_mode': deriver_mode,
        }

        # TODO Create new config-get with only necessary parts.

        return metabolism_config

    def get_metabolism_config(self, time_step=2, parallel=False, deriver_mode=False):

        # bad_rxns = ["RXN-12440", "TRANS-RXN-121", "TRANS-RXN-300"]
        # for rxn in bad_rxns:
        #     self.sim_data.process.metabolism.reaction_stoich.pop(rxn, None)
        #     self.sim_data.process.metabolism.reaction_catalysts.pop(rxn, None)
        #     self.sim_data.process.metabolism.reactions_with_catalyst.remove(rxn) \
        #         if rxn in self.sim_data.process.metabolism.reactions_with_catalyst else None


        metabolism_config = {
            'time_step': time_step,
            '_parallel': parallel,

            # metabolism-gd parameters
            'stoichiometry': self.sim_data.process.metabolism.reaction_stoich,
            'reaction_catalysts': self.sim_data.process.metabolism.reaction_catalysts,
            'catalyst_ids': self.sim_data.process.metabolism.catalyst_ids,
            'concentration_updates': self.sim_data.process.metabolism.concentration_updates,
            'maintenance_reaction': self.sim_data.process.metabolism.maintenance_reaction,

            # wcEcoli parameters
            'get_import_constraints': self.sim_data.external_state.get_import_constraints,
            'nutrientToDoublingTime': self.sim_data.nutrient_to_doubling_time,
            'aa_names': self.sim_data.molecule_groups.amino_acids,

            # these are options given to the wholecell.sim.simulation
            'use_trna_charging': self.trna_charging,
            'include_ppgpp': not self.ppgpp_regulation or not self.trna_charging,

            # these values came from the initialized environment state
            'current_timeline': None,
            'media_id': self.sim_data.conditions[self.sim_data.condition]['nutrients'],

            'condition': self.sim_data.condition,
            'nutrients': self.sim_data.conditions[self.sim_data.condition]['nutrients'],
            # TODO Replace this with media_id
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
            'seed': self._seedFromName('Metabolism'),
            'linked_metabolites': self.sim_data.process.metabolism.linked_metabolites,
            # Whether to use metabolism as a deriver (with t=0 skipped)
            'deriver_mode': deriver_mode,

            # TODO: testing, remove later (perhaps after moving change to sim_data)
            'reduce_murein_objective': False
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
            unique_masses[id_] = (
                mass / self.sim_data.constants.n_avogadro).asNumber(units.fg)

        mass_config = {
            'molecular_weights': molecular_weights,
            'unique_masses': unique_masses,
            'cellDensity': self.sim_data.constants.cell_density.asNumber(units.g / units.L),
            'water_id': 'WATER[c]',
        }
        return mass_config

    def get_mass_listener_config(self, time_step=2, parallel=False):
        mass_config = {
            'cellDensity': self.sim_data.constants.cell_density.asNumber(units.g / units.L),
            'bulk_ids': self.sim_data.internal_state.bulk_molecules.bulk_data['id'],
            'bulk_masses': self.sim_data.internal_state.bulk_molecules.bulk_data['mass'].asNumber(
                units.fg / units.mol) / self.sim_data.constants.n_avogadro.asNumber(1 / units.mol),
            'unique_ids': self.sim_data.internal_state.unique_molecule.unique_molecule_masses['id'],
            'unique_masses': self.sim_data.internal_state.unique_molecule.unique_molecule_masses['mass'].asNumber(
                units.fg / units.mol) / self.sim_data.constants.n_avogadro.asNumber(1/units.mol),
            'compartment_abbrev_to_index': self.sim_data.compartment_abbrev_to_index,
            'expectedDryMassIncreaseDict': self.sim_data.expectedDryMassIncreaseDict,
            'compartment_indices': {
                'projection': self.sim_data.compartment_id_to_index["CCO-CELL-PROJECTION"],
                'cytosol': self.sim_data.compartment_id_to_index["CCO-CYTOSOL"],
                'extracellular': self.sim_data.compartment_id_to_index["CCO-EXTRACELLULAR"],
                'flagellum': self.sim_data.compartment_id_to_index["CCO-FLAGELLUM"],
                'membrane': self.sim_data.compartment_id_to_index["CCO-MEMBRANE"],
                'outer_membrane': self.sim_data.compartment_id_to_index["CCO-OUTER-MEM"],
                'periplasm': self.sim_data.compartment_id_to_index["CCO-PERI-BAC"],
                'pilus': self.sim_data.compartment_id_to_index["CCO-PILUS"],
                'inner_membrane': self.sim_data.compartment_id_to_index["CCO-PM-BAC-NEG"],
            },
            'compartment_id_to_index': self.sim_data.compartment_id_to_index,
            'n_avogadro': self.sim_data.constants.n_avogadro,  # 1/mol
            'time_step': time_step,
            'submass_to_idx': self.sim_data.submass_name_to_index,
            'condition_to_doubling_time': self.sim_data.condition_to_doubling_time,
            'condition': self.sim_data.condition
        }

        return mass_config

    def get_mrna_counts_listener_config(self, time_step=2, parallel=False):
        counts_config = {
            'time_step': time_step,
            '_parallel': parallel,

            'rna_ids': self.sim_data.process.transcription.rna_data['id'],
            'mrna_indexes': np.where(self.sim_data.process.transcription.rna_data['is_mRNA'])[0],
        }

        return counts_config

    def get_monomer_counts_listener_config(self, time_step=2, parallel=False):
        monomer_counts_config = {
            'time_step': time_step,
            '_parallel': parallel,

            # Get IDs of all bulk molecules
            'bulk_molecule_ids': self.sim_data.internal_state.bulk_molecules.bulk_data["id"],
            'unique_ids': self.sim_data.internal_state.unique_molecule.unique_molecule_masses["id"],

            # Get IDs of molecules involved in complexation and equilibrium
            'complexation_molecule_ids': self.sim_data.process.complexation.molecule_names,
            'complexation_complex_ids': self.sim_data.process.complexation.ids_complexes,
            'equilibrium_molecule_ids': self.sim_data.process.equilibrium.molecule_names,
            'equilibrium_complex_ids': self.sim_data.process.equilibrium.ids_complexes,
            'monomer_ids': self.sim_data.process.translation.monomer_data["id"].tolist(),

            # Get IDs of complexed molecules monomers involved in two component system
            'two_component_system_molecule_ids': list(
                self.sim_data.process.two_component_system.molecule_names),
            'two_component_system_complex_ids': list(
                self.sim_data.process.two_component_system.complex_to_monomer.keys()),

            # Get IDs of ribosome subunits
            'ribosome_50s_subunits': self.sim_data.process.complexation.get_monomers(
                self.sim_data.molecule_ids.s50_full_complex),
            'ribosome_30s_subunits': self.sim_data.process.complexation.get_monomers(
                self.sim_data.molecule_ids.s30_full_complex),

            # Get IDs of RNA polymerase subunits
            'rnap_subunits': self.sim_data.process.complexation.get_monomers(
                self.sim_data.molecule_ids.full_RNAP),

            # Get IDs of replisome subunits
            'replisome_trimer_subunits': self.sim_data.molecule_groups.replisome_trimer_subunits,
            'replisome_monomer_subunits': self.sim_data.molecule_groups.replisome_monomer_subunits,

            # Get stoichiometric matrices for complexation, equilibrium, two component system and the
            # assembly of unique molecules
            'complexation_stoich': self.sim_data.process.complexation.stoich_matrix_monomers(),
            'equilibrium_stoich': self.sim_data.process.equilibrium.stoich_matrix_monomers(),
            'two_component_system_stoich': self.sim_data.process.two_component_system.stoich_matrix_monomers(),
        }

        return monomer_counts_config

    def get_allocator_config(self, time_step=2, parallel=False, process_names=None):
        if not process_names:
            process_names = []
        allocator_config = {
            'time_step': time_step,
            '_parallel': parallel,
            'molecule_names': self.sim_data.internal_state.bulk_molecules.bulk_data['id'],
            # Allocator is built into BulkMolecules container in wcEcoli
            'seed': self._seedFromName('BulkMolecules'),
            'process_names': process_names,
            'custom_priorities': {
                'ecoli-rna-degradation': 10,
                'ecoli-protein-degradation': 10,
                'ecoli-two-component-system': -5,
                'ecoli-tf-binding': -10,
                # 'ecoli-metabolism': -10  # Should not be necessary, metabolism runs as deriver (not partitioned)
                }
        }
        return allocator_config

    def get_chromosome_structure_config(self, time_step=2, parallel=False, deriver_mode=False):
        chromosome_structure_config = {
            'time_step': time_step,
            '_parallel': parallel,

            # Load parameters
            'RNA_sequences': self.sim_data.process.transcription.transcription_sequences,
            'protein_sequences': self.sim_data.process.translation.translation_sequences,
            'n_TUs': len(self.sim_data.process.transcription.rna_data),
            'n_TFs': len(self.sim_data.process.transcription_regulation.tf_ids),
            'n_amino_acids': len(self.sim_data.molecule_groups.amino_acids),
            'n_fragment_bases': len(self.sim_data.molecule_groups.polymerized_ntps),
            'replichore_lengths': self.sim_data.process.replication.replichore_lengths,
            'relaxed_DNA_base_pairs_per_turn': self.sim_data.process.chromosome_structure.relaxed_DNA_base_pairs_per_turn,
            'terC_index': self.sim_data.process.chromosome_structure.terC_dummy_molecule_index,

            # TODO: Should be loaded from simulation options
            'calculate_superhelical_densities': False,

            # Get placeholder value for chromosome domains without children
            'no_child_place_holder': self.sim_data.process.replication.no_child_place_holder,

            # Load bulk molecule views
            'inactive_RNAPs': self.sim_data.molecule_ids.full_RNAP,
            'fragmentBases': self.sim_data.molecule_groups.polymerized_ntps,
            'ppi': self.sim_data.molecule_ids.ppi,
            'active_tfs': [x + "[c]" for x in self.sim_data.process.transcription_regulation.tf_ids],
            'ribosome_30S_subunit': self.sim_data.molecule_ids.s30_full_complex,
            'ribosome_50S_subunit': self.sim_data.molecule_ids.s50_full_complex,
            'amino_acids': self.sim_data.molecule_groups.amino_acids,
            'water': self.sim_data.molecule_ids.water,

            'deriver_mode': deriver_mode,
            'seed': self._seedFromName('ChromosomeStructure'),
        }
        return chromosome_structure_config
    
    
    def get_rna_interference_config(self, time_step=2, parallel=False):
        rna_interference_config = {
            'time_step': time_step,
            '_parallel': parallel,
            
            'srna_ids': self.srna_ids,
            'target_tu_ids': self.target_tu_ids,
            'binding_probs': self.binding_probs,
            'duplex_ids': self.duplex_ids,
            
            'ribosome30S': self.sim_data.molecule_ids.s30_full_complex,
            'ribosome50S': self.sim_data.molecule_ids.s50_full_complex,
            
            'seed': self.random_state.randint(RAND_MAX)
        }
        return rna_interference_config

    def get_tetracycline_ribosome_equilibrium_config(self, time_step=2, parallel=False):
        rna_ids = self.sim_data.process.transcription.rna_data['id']
        is_trna = self.sim_data.process.transcription.rna_data['is_tRNA'].astype(np.bool_)
        tetracycline_ribosome_equilibrium_config = {
            'time_step': time_step,
            '_parallel': parallel,
            'trna_ids': rna_ids[is_trna],
            # Ensure that a new seed is set upon division
            'seed': self.random_state.randint(RAND_MAX)
        }
        return tetracycline_ribosome_equilibrium_config

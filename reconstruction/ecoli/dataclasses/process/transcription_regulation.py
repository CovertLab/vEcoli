"""
SimulationData for transcription regulation

"""

from typing import Union

import numpy as np
from scipy import sparse
import itertools
from wholecell.utils import units
from wholecell.utils.fast_nonnegative_least_squares import fast_nnls
from wholecell.utils.fitting import normalize

BASAL_CONDITION = "basal"
# TODO: have this here or in internal_state file?
UNBOUND_TF_BINDING_SITE_IDX = -1

class TranscriptionRegulation(object):
    """
    SimulationData for transcription regulation
    """

    def __init__(self, raw_data, sim_data):
        # Build lookups
        self._build_lookups(raw_data)

        # Store list of transcription factor IDs
        self.tf_ids = list(sorted(sim_data.tf_to_active_inactive_conditions.keys()))

        # Store index of tf_binding_site unique object specifying no TF is bound
        self.unbound_tf_binding_site_idx = UNBOUND_TF_BINDING_SITE_IDX

        # Build tf-binding site dictionaries
        self._build_tf_binding_sites(raw_data)

        # Build tf binding and unbinding rate matrices
        self._build_tf_binding_unbinding_rates(raw_data)

        # Build dictionary mapping RNA targets to its regulators
        self.target_tf = {}

        for tf in sorted(sim_data.tf_to_fold_change):
            targets = sim_data.tf_to_fold_change[tf]
            targetsToRemove = []

            for target in targets:
                if target not in self.target_tf:
                    self.target_tf[target] = []

                self.target_tf[target].append(tf)

            for targetToRemove in targetsToRemove:
                sim_data.tf_to_fold_change[tf].pop(targetToRemove)

        # Build dictionaries mapping transcription factors to their bound form,
        # and to their regulating type
        self.active_to_bound = {
            x["active TF"]: x["metabolite bound form"]
            for x in raw_data.tf_one_component_bound
        }
        self.tf_to_tf_type = {
            x["active TF"]: x["TF type"] for x in raw_data.condition.tf_condition
        }
        self.tf_to_gene_id = {
            x["active TF"]: x["TF"] for x in raw_data.condition.tf_condition
        }

        # Initialize different categories of transcriptional regulation from raw_data.
        # Will be further modified to add simulation parameters during parca.
        self._initialize_one_peak_genes(raw_data)
        self._initialize_two_peak_genes(raw_data)

        # To be set in parca, used for simulations with ppGpp option off
        self.basal_aff = None

    def p_promoter_bound_tf(self, tfActive, tfInactive):
        """
        Computes probability of a transcription factor binding promoter.
        """
        return float(tfActive) / (float(tfActive) + float(tfInactive))

    def p_promoter_bound_SKd(self, signal, Kd, power):
        """
        Computes probability of a one-component transcription factor binding
        promoter.
        """
        return float(signal) ** power / (float(signal) ** power + float(Kd) ** power)

    def get_delta_aff_matrix(
        self, dense=False, ppgpp=False
    ) -> Union[sparse.csr_matrix, np.ndarray]:
        """
        Returns the delta affinity matrix mapping the promoter binding effect
        of each TF to each gene.

        Args:
                dense: If True, returns a dense matrix, otherwise csr sparse
                ppgpp: If True, normalizes delta affinities to be on the same
                        scale as ppGpp normalized affinities since delta_aff is
                        calculated based on basal_aff which is not normalized to 1

        Returns:
                delta_aff: matrix of affinities changes expected with a TF
                        binding to a promoter for each gene (n genes, m TFs)
        """

        ppgpp_scaling = self.basal_aff[self.delta_aff["deltaI"]]
        ppgpp_scaling[ppgpp_scaling == 0] = 1
        scaling_factor = ppgpp_scaling if ppgpp else 1.0
        delta_aff = sparse.csr_matrix(
            (
                self.delta_aff["deltaV"] / scaling_factor,
                (self.delta_aff["deltaI"], self.delta_aff["deltaJ"]),
            ),
            shape=self.delta_aff["shape"],
        )

        if dense:
            delta_aff = delta_aff.toarray()

        return delta_aff

    def get_tf_binding_unbinding_matrices(self, dense=False) -> (Union[sparse.csr_matrix, np.ndarray],
            Union[sparse.csr_matrix, np.ndarray]):
        """
        Returns the binding and unbinding rate matrices mapping each TF to each binding site.
        """
        assert self._binding_rates_shape == self._unbinding_rates_shape

        binding_rates = sparse.csr_matrix(
            (
                self._binding_rates_v,
                (self._binding_rates_i, self._binding_rates_j)
            ),
            shape=self._binding_rates_shape
        )
        unbinding_rates = sparse.csr_matrix(
            (
                self._unbinding_rates_v,
                (self._unbinding_rates_i, self._unbinding_rates_j)
            ),
            shape=self._unbinding_rates_shape
        )

        if dense:
            binding_rates = binding_rates.toarray()
            unbinding_rates = unbinding_rates.toarray()

        return binding_rates, unbinding_rates

    def _build_tf_binding_unbinding_rates(self, raw_data):
        """
        Builds and saves arrays encoding csr matrices of binding and unbinding rates of TFs
        to TF-binding sites. The full matrix can be obtained with self.get_tf_binding_unbinding_matrices.
        """

        self._binding_rates_i = []
        self._binding_rates_j = []
        self._binding_rates_v = []
        self._binding_rates_shape = (len(self.tf_binding_site_ids), len(self.tf_ids))

        self._unbinding_rates_i = []
        self._unbinding_rates_j = []
        self._unbinding_rates_v = []
        self._unbinding_rates_shape = (len(self.tf_binding_site_ids), len(self.tf_ids))
        for row in raw_data.tf_binding_site_rates:
            binding_site_id = row["binding_site_id"]
            tf_id = row["TF"]
            binding_rate = row["binding_rate"]
            unbinding_rate = row["unbinding_rate"]

            binding_site_idx = self.tf_binding_site_ids.index(binding_site_id)
            tf_idx = self.tf_ids.index(tf_id)

            self._binding_rates_i.append(binding_site_idx)
            self._binding_rates_j.append(tf_idx)
            self._binding_rates_v.append(binding_rate)

            self._unbinding_rates_i.append(binding_site_idx)
            self._unbinding_rates_j.append(tf_idx)
            self._unbinding_rates_v.append(unbinding_rate)

    def _build_lookups(self, raw_data):
        """
        Builds dictionaries for mapping transcription factor abbreviations to
        their RNA IDs, and to their active form.
        """
        gene_id_to_cistron_id = {x["id"]: x["rna_ids"][0] for x in raw_data.genes}

        self.abbr_to_rna_id = {}
        for lookupInfo in raw_data.transcription_factors:
            if (
                len(lookupInfo["geneId"]) == 0
                or lookupInfo["geneId"] not in gene_id_to_cistron_id
            ):
                continue
            self.abbr_to_rna_id[lookupInfo["TF"]] = gene_id_to_cistron_id[
                lookupInfo["geneId"]
            ]

        self.abbr_to_active_id = {
            x["TF"]: x["activeId"].split(", ")
            for x in raw_data.transcription_factors
            if len(x["activeId"]) > 0
        }

    def _build_oric_terc_coordinates(self, raw_data, sim_data):
        """
        Builds coordinates of oriC and terC that are used when calculating
        genomic positions of tf-binding sites relative to the origin
        """
        # Get coordinates of oriC and terC
        oric_left, oric_right = sim_data.getter.get_genomic_coordinates(
            sim_data.molecule_ids.oriC_site
        )
        terc_left, terc_right = sim_data.getter.get_genomic_coordinates(
            sim_data.molecule_ids.terC_site
        )
        self._oric_coordinate = round((oric_left + oric_right) / 2)
        self._terc_coordinate = round((terc_left + terc_right) / 2)
        self._genome_length = len(raw_data.genome_sequence)

    def _get_relative_coordinates(self, coordinates):
        """
        Returns the genomic coordinates of a given gene coordinate relative
        to the origin of replication.
        """
        if coordinates < self._terc_coordinate:
            relative_coordinates = (
                self._genome_length - self._oric_coordinate + coordinates
            )
        elif coordinates < self._oric_coordinate:
            relative_coordinates = coordinates - self._oric_coordinate + 1
        else:
            relative_coordinates = coordinates - self._oric_coordinate

        return relative_coordinates

    def _build_tf_binding_sites(self, raw_data):
        """
        Builds dictionaries for mapping binding site ids
        to the TUs they regulate, and to the TF that binds them. These are converted
        into mapping matrices for use in the model in relation.py.
        Also stores a list of binding-site ids, and a list of relative genomic coordinates
        of the ids in the same order.
        """
        tu_to_genes = {
            x["id"]: x["genes"] for x in raw_data.transcription_units
        }
        gene_to_tus = {}
        for tu in tu_to_genes:
            for gene in tu_to_genes[tu]:
                if gene not in gene_to_tus:
                    gene_to_tus[gene] = []

                gene_to_tus[gene].append(tu)

        # Note: have checked that every entity in "regulated_TUs_or_genes"
        # is either a gene or TU within one of the raw data flat files. If
        # it's not in tu_to_genes or gene_to_tus, it's because it corresponds
        # to a removed TU.
        tf_bind_site_to_tus = {}
        tf_bind_site_to_tfs = {}
        for row in raw_data.tf_binding_sites:
            curated_tus = []
            tus_or_genes = row["regulated_TUs_or_genes"]
            for x in tus_or_genes:
                if x in tu_to_genes:
                    curated_tus.append(x)
                elif x in gene_to_tus:
                    curated_tus.extend(gene_to_tus[x])

            tf_bind_site_to_tus[row["binding_site_id"]] = curated_tus
            tf_bind_site_to_tfs[row["binding_site_id"]] = row["binding_TFs"]

        self.tf_bind_site_to_tus = tf_bind_site_to_tus
        self.tf_bind_site_to_tfs = tf_bind_site_to_tfs
        self.tf_binding_site_ids = list(sorted(tf_bind_site_to_tus.keys()))

        tf_bind_site_to_coords = {x["binding_site_id"]: x["coordinates"]
                                  for x in raw_data.tf_binding_sites}
        self.tf_binding_site_coords = np.array([tf_bind_site_to_coords[x] for x in self.tf_binding_site_ids])

    def _initialize_one_peak_genes(self, raw_data):
        gene_symbol_to_id = {x["symbol"]: x["id"] for x in raw_data.genes}
        one_peak_ids = np.array([gene_symbol_to_id[x["gene"]] for x in raw_data.tf_regulations_one_peak])
        ecomac_exp = np.array([x["ECOMAC_expression"] for x in raw_data.tf_regulations_one_peak])

        # Make data objects
        self.one_peak_gene_data = {
            "gene_id": one_peak_ids,
            "mRNA_exp_frac": ecomac_exp,
        }

        # To be set in parca
        self.one_peak_TU_data = {
            "TU_idx": None,
            "mRNA_exp_frac": None,
            "affinity": None
        }

    def _initialize_two_peak_genes(self, raw_data):
        gene_symbol_to_id = {x["symbol"]: x["id"] for x in raw_data.genes}

        two_peak_gene_ids = []
        two_peak_tf_idxs = []
        unbound_ecomac_exp = []
        bound_ecomac_exp = []
        condition = []

        valid_conditions = [x["condition"] for x in raw_data.condition.condition_defs]
        for row in raw_data.tf_regulations_two_peaks:
            two_peak_gene_ids.append(gene_symbol_to_id[row["regulated_gene"]])
            two_peak_tf_idxs.append(self.tf_ids.index(self.abbr_to_active_id[row["TF"]][0]))

            # TODO: should these checks be here or later on? should there be errors or just ignore
            # if it's not valid?
            for cond in itertools.chain(row["low_condition"], row["high_condition"]):
                if cond not in valid_conditions:
                    raise ValueError("Condition {} not in valid conditions".format(
                        cond
                    ))

            # If the TF is a repressor, the lower peak is from the TF being bound. If it's an
            # activator, the lower peak is from the TF being unbound.
            if row["TF_type"] == "repressor":
                unbound_ecomac_exp.append(row["high_ECOMAC_expression"])
                bound_ecomac_exp.append(row["low_ECOMAC_expression"])
                condition.append({
                    "unbound": row["high_condition"],
                    "bound": row["low_condition"],
                })

            elif row["TF_type"] == "activator":
                unbound_ecomac_exp.append(row["low_ECOMAC_expression"])
                bound_ecomac_exp.append(row["high_ECOMAC_expression"])
                condition.append({
                    "unbound": row["low_condition"],
                    "bound": row["high_condition"],
                })

        # Make data object
        self.two_peak_gene_data = {
            "regulated_gene_id": two_peak_gene_ids,
            "tf_idx": two_peak_tf_idxs,
            "unbound_mRNA_exp_frac": unbound_ecomac_exp,
            "bound_mRNA_exp_frac": bound_ecomac_exp,
            "condition": condition,
        }

        # To be set in parca
        self.two_peak_TU_data = {
            "TU_idx": None,
            "tf_idx": None,
            "unbound_mRNA_exp_frac": None,
            "bound_mRNA_exp_frac": None,
            "unbound_affinity": None,
            "bound_affinity": None,
            "condition": None,
            "tf_binding_site_index": None,
        }

    def set_TF_reg_operons(self, sim_data):
        # First, for all one-peak genes, get the operons they're contained in,
        # and get all the TUs within this. Also get the subset of TUs that contain the gene.
        # TODO: maybe restrict to only TUs that contain the one-peak gene?

        transcription = sim_data.process.transcription

        def _solve_basal_nnls(target_gene_ids, target_cistron_mRNA_frac):
            '''
            Solves NNLS and normalizes to get the fraction of mRNA that TUs broadly defined to be
            any TU in an operon containing the indicated target gene ids, should have to match the
            indicated fraction of cistron mRNA counts for target gene ids, and the basal cistron
            expression levels from sim data for other cistrons in the operons that do not correspond to
            the target gene ids. TODO: make this description more readable
            '''
            target_cistron_idxs = [np.where(transcription.cistron_data["gene_id"]
                                          == gene)[0][0] for gene in target_gene_ids]
            cistron_idx_to_mRNA_frac = {
                idx: frac for (idx, frac) in zip(target_cistron_idxs, target_cistron_mRNA_frac)
            }

            cistrons_of_operons = set()
            tus_of_operons = set()
            cistron_idx_to_operon_TUs = {}
            for cistron_idxs, tu_idxs in transcription.operons:
                for cistron in cistron_idxs:
                    if cistron in target_cistron_idxs:
                        cistrons_of_operons.update(cistron_idxs)
                        tus_of_operons.update(tu_idxs)
                        cistron_idx_to_operon_TUs[cistron] = tu_idxs
                        continue

            cistrons_of_operons = sorted(list(cistrons_of_operons))
            tus_of_operons = sorted(list(tus_of_operons))

            # Get the cistron expression values
            cistron_exp = transcription.cistron_expression["basal"]
            cistron_mRNA_sum = np.sum(cistron_exp[transcription.cistron_data["is_mRNA"]])

            cistron_operon_exp = np.zeros(len(cistrons_of_operons))
            for i, cistron in enumerate(cistrons_of_operons):
                if cistron in target_cistron_idxs:
                    cistron_operon_exp[i] = cistron_mRNA_sum * cistron_idx_to_mRNA_frac[cistron]
                else:
                    cistron_operon_exp[i] = cistron_exp[cistron]

            # Solve NNLS for these operons
            mapping_matrix_these_operons = transcription.cistron_tu_mapping_matrix[cistrons_of_operons, :
                                           ][:, tus_of_operons].toarray()

            TU_exp, _ = fast_nnls(mapping_matrix_these_operons, cistron_operon_exp)
            TU_exp /= transcription.unnormalized_mRNA_rna_exp_sum

            return TU_exp, tus_of_operons, cistron_idx_to_operon_TUs

        def _update_basal_expression(TU_idxs, TU_mRNA_frac):
            # Update RNA expression
            basal_TU_exp = transcription.rna_expression["basal"]
            basal_mRNA_TU_exp_sum = np.sum(basal_TU_exp[transcription.rna_data["is_mRNA"]])
            basal_TU_exp[TU_idxs] = TU_mRNA_frac * basal_mRNA_TU_exp_sum
            basal_TU_exp = normalize(basal_TU_exp)
            # Update cistron expression
            cistron_basal_exp = transcription.cistron_tu_mapping_matrix.dot(basal_TU_exp)
            transcription.cistron_expression["basal"] = normalize(cistron_basal_exp)

        # Solve basal NNLS for one-peak genes and update basal expression
        one_peak_TU_mRNA_frac, one_peak_TU_idxs, _ = _solve_basal_nnls(
            self.one_peak_gene_data["gene_id"], self.one_peak_gene_data["mRNA_exp_frac"]
        )
        _update_basal_expression(one_peak_TU_idxs, one_peak_TU_mRNA_frac)

        # Save one-peak TUs and expression values
        self.one_peak_TU_data["TU_idx"] = np.array(one_peak_TU_idxs)
        self.one_peak_TU_data["mRNA_exp_frac"] = one_peak_TU_mRNA_frac

        ## Two-peak genes
        # TODO: a little bit of inaccuracy here, against matching EcoMAC data, for the case where an unregulated TU contains a cistron
        # that a regulated TU does too (e.g. for the cistron trpC): here we subtract the unregulated TU's
        # mRNA expression fraction from the total target mRNA expression fraction to obtain the
        # regulated TU's target mRNA expression fraction during both peaks. However, due to RNAP/r-protein
        # physiological fitting, the mRNA expression fraction of unregulated TUs may change in different
        # conditions, and so they would not contribute the same mRNA expression fraction that is subtracted
        # here. Ideally, we would subtract the cistron contribution from relevant unregulated TUs during
        # each iteration of fitting in the parca to obtain the desired expression of regulated TUs.
        # However, since these cases should be relatively rare (TODO: check that it's rare),
        #  we'll disregard this inaccuracy for now.

        # Update two-peak affinites for basal condition
        cistron_basal_mRNA_frac = []
        cistron_other_mRNA_frac = []
        for i, condition in enumerate(self.two_peak_gene_data["condition"]):
            if BASAL_CONDITION in condition["bound"]:
                cistron_basal_mRNA_frac.append(self.two_peak_gene_data["bound_mRNA_exp_frac"][i])
                cistron_other_mRNA_frac.append(self.two_peak_gene_data["unbound_mRNA_exp_frac"][i])
            elif BASAL_CONDITION in condition["unbound"]:
                cistron_basal_mRNA_frac.append(self.two_peak_gene_data["unbound_mRNA_exp_frac"][i])
                cistron_other_mRNA_frac.append(self.two_peak_gene_data["bound_mRNA_exp_frac"][i])
            else:
                raise ValueError("Basal condition not in bound or unbound conditons for two-peak gene {}".format(
                    self.two_peak_gene_data["regulated_gene_id"][i]
                ))

        # Solve basal NNLS for two-peak genes (considers all TUs in the operons containing the two-peak cistrons),
        # and then update basal expression in transcription for use during the parca
        two_peak_basal_TU_mRNA_frac, two_peak_basal_TU_idxs, two_peak_cistron_idx_to_operon_TUs = _solve_basal_nnls(
            self.two_peak_gene_data["regulated_gene_id"], cistron_basal_mRNA_frac
        )
        two_peak_basal_TU_idx_to_mRNA_frac = {idx: frac for idx, frac in zip(
            two_peak_basal_TU_idxs, two_peak_basal_TU_mRNA_frac)}

        _update_basal_expression(two_peak_basal_TU_idxs, two_peak_basal_TU_mRNA_frac)

        # Get the corresponding tf-binding-site and the TUs that it regulates
        target_cistron_idxs = [np.where(transcription.cistron_data["gene_id"]
                                        == gene)[0][0] for gene in self.two_peak_gene_data["regulated_gene_id"]]
        tf_binding_site_TU_matrix = sim_data.relation.tf_binding_site_to_tus_mapping(dense=True)
        tf_binding_site_TF_matrix = sim_data.relation.tf_binding_site_to_tfs_mapping(dense=True)

        tu_regulation_info = {}
        for i, idx in enumerate(target_cistron_idxs):
            tf_binding_sites = set()
            for TU_idx in transcription.cistron_id_to_rna_indexes(transcription.cistron_data["id"][idx]):
                # Add tf-binding-sites that regulate each TU containing the cistron to a set
                tf_binding_sites.update(np.where(tf_binding_site_TU_matrix[:, TU_idx] == 1)[0])
            tf_binding_sites = list(tf_binding_sites)
            # TODO: maybe could remove this assumption later
            if len(tf_binding_sites) != 1:
                raise ValueError("Two-peak cistron must be regulated by only one tf-binding-site")
            tf_binding_site_idx = tf_binding_sites[0]

            # Get TF that regulates these TUs (should only be one, and matches the TF idx in two_peak_gene_data)
            regulating_TFs = np.where(tf_binding_site_TF_matrix[tf_binding_site_idx, :] == 1)[0]
            regulated_TUs = np.where(tf_binding_site_TU_matrix[tf_binding_site_idx, :] == 1)[0]
            if len(regulating_TFs) != 1:
                raise ValueError("Two-peak cistron must be regulated by only one TF")
            if regulating_TFs[0] != self.two_peak_gene_data["tf_idx"][i]:
                raise ValueError("TF regulating two-peak cistron does not match TF from tf-binding-site data")
            tf_idx = regulating_TFs[0]

            # Get condition data
            condition_data = self.two_peak_gene_data["condition"][i]

            # Store data temporarily in a dictionary
            for TU_idx in regulated_TUs:
                tu_regulation_info[TU_idx] = (tf_idx, tf_binding_site_idx, condition_data)

        two_peak_reg_TU_idxs = np.array(sorted(list(tu_regulation_info.keys())))
        self.two_peak_TU_data["TU_idx"] = two_peak_reg_TU_idxs
        self.two_peak_TU_data["tf_idx"] = np.array(
            [tu_regulation_info[TU_idx][0]] for TU_idx in two_peak_reg_TU_idxs
        )
        self.two_peak_TU_data["tf_binding_site_idx"] = np.array(
            [tu_regulation_info[TU_idx][1]] for TU_idx in two_peak_reg_TU_idxs
        )
        self.two_peak_TU_data["condition"] = np.array(
            [tu_regulation_info[TU_idx][2]] for TU_idx in two_peak_reg_TU_idxs
        )

        # TODO: check that these calculations are right?
        # Solve NNLS for other peak
        # Get all TUs that contain these cistrons
        contain_cistron_TU_idxs = set()
        for idx in target_cistron_idxs:
            contain_cistron_TU_idxs.update(transcription.cistron_id_to_rna_indexes(transcription.cistron_data["id"][idx]))
        contain_cistron_TU_idxs = sorted(list(contain_cistron_TU_idxs))

        # Get the target cistron-level target amount (unnormalized expression, at the level
        # of mapping_matrix * TU expression)
        # We make the assumption that the total amount of mRNA expression changes negligibly due
        # to changes made to TU expression made here
        cistron_exp_unnormalized = transcription.cistron_tu_mapping_matrix.toarray().dot(
            transcription.rna_expression["basal"]
        )
        cistron_exp_unnormalized_mRNA_sum = cistron_exp_unnormalized[transcription.cistron_data["is_mRNA"]]
        cistron_total_target_amount = cistron_exp_unnormalized_mRNA_sum * np.array(cistron_other_mRNA_frac)

        # Subtract away contribution from relevant TUs that are not regulated
        # (at the level of mapping_matrix * TU_expression)
        fixed_TU_idxs = np.array([x for x in contain_cistron_TU_idxs if x not in
                    two_peak_reg_TU_idxs])
        fixed_TU_exp = transcription.rna_expression["basal"][fixed_TU_idxs]

        mapping_matrix_target_cistrons = transcription.cistron_tu_mapping_matrix[target_cistron_idxs, :
                                       ][:, fixed_TU_idxs].toarray()
        fixed_TU_cistron_amount = mapping_matrix_target_cistrons.dot(fixed_TU_exp)
        cistron_target_amount = cistron_total_target_amount - fixed_TU_cistron_amount
        # Fix negative amounts at 0
        # TODO: does this ever happen? check these cases if it does
        cistron_target_amount[(cistron_target_amount < 0)] = 0.

        # Solve for target TU expression fraction of mRNA for the other peak
        # (first get unnormalized TU amount that recreates the previous cistron-level amounts,
        # then divide by the total sum of TU amounts that recreates cistron-level amounts calculated
        # in transcription object, to get the TU-level fraction of mRNA expression. This again makes
        # the assumption that changes made to TU-expression have a negligible impact on the total
        # sum of TU mRNA expression)
        mapping_matrix_target_TU_cistrons = transcription.cistron_tu_mapping_matrix[target_cistron_idxs, :
                                           ][:, two_peak_reg_TU_idxs].toarray()
        two_peak_other_TU_exp, _ = fast_nnls(mapping_matrix_target_TU_cistrons, cistron_target_amount)
        two_peak_other_TU_mRNA_frac = two_peak_other_TU_exp / transcription.unnormalized_mRNA_rna_exp_sum

        # Convert back into bound and unbound mRNA fracs
        unbound_TU_mRNA_frac = []
        bound_TU_mRNA_frac = []
        for i, TU_idx in enumerate(two_peak_reg_TU_idxs):
            # Determine if basal peak is bound or unbound
            condition = self.two_peak_TU_data["condition"][i]
            # Store bound and unbound mRNA fractions for each TU
            if BASAL_CONDITION in condition["bound"]:
                bound_TU_mRNA_frac.append(two_peak_basal_TU_idx_to_mRNA_frac[TU_idx])
                unbound_TU_mRNA_frac.append(two_peak_other_TU_mRNA_frac[i])
            elif BASAL_CONDITION in condition["unbound"]:
                unbound_TU_mRNA_frac.append(two_peak_basal_TU_idx_to_mRNA_frac[TU_idx])
                bound_TU_mRNA_frac.append(two_peak_other_TU_mRNA_frac[i])
            else:
                raise ValueError("Basal condition not in bound or unbound conditions for TU {}".format(
                    transcription.rna_data["id"][TU_idx]
                ))

        # TODO: maybe shorten code by combining some code with _solve_basal_NNLS and other-peak calculations
        # Save data
        self.two_peak_TU_data["unbound_mRNA_exp_frac"] = np.array(unbound_TU_mRNA_frac)
        self.two_peak_TU_data["bound_mRNA_exp_frac"] = np.array(bound_TU_mRNA_frac)

"""
SimulationData for transcription regulation

"""

from typing import Union

import numpy as np
from scipy import sparse
from wholecell.utils import units

class TranscriptionRegulation(object):
    """
    SimulationData for transcription regulation
    """

    def __init__(self, raw_data, sim_data):
        # Build lookups
        self._build_lookups(raw_data)

        # Build tf-binding site dictionaries
        self._build_tf_binding_sites(raw_data)

        # Build tf binding and unbinding rate matrices
        self._build_tf_binding_unbinding_rates(raw_data)

        # Store list of transcription factor IDs
        self.tf_ids = list(sorted(sim_data.tf_to_active_inactive_conditions.keys()))

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

        # Values set after promoter fitting in parca with calculateRnapRecruitment()
        self.basal_aff = None
        self.delta_aff = None

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

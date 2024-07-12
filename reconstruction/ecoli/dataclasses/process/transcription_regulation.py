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
        self.raw_binding_rates = None
        self.raw_unbinding_rates = None

        # TODO: add raw_data files storing the binding and unbinding rates, and a function
        #  to read them in and store them as attributes here

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

    def get_tf_binding_unbinding_matrices(self, sim_data, dense=False) -> (Union[sparse.csr_matrix, np.ndarray],
            Union[sparse.csr_matrix, np.ndarray]):
        """
        Returns the binding and unbinding rate matrices mapping each TF to each binding site (TU promoter for now).
        TODO: change to binding sites instead of by each TU
        """

        # TODO: change this shape when changing to binding sites instead of by each TU
        assert self.raw_binding_rates["shape"] == self.raw_unbinding_rates["shape"]

        binding_rates = sparse.csr_matrix(
            (
                self.raw_binding_rates["bindingV"],
                (self.raw_binding_rates["bindingI"], self.raw_binding_rates["bindingJ"])
            ),
            shape=self.raw_binding_rates["shape"]
        )
        unbinding_rates = sparse.csr_matrix(
            (
                self.raw_unbinding_rates["unbindingV"],
                (self.raw_unbinding_rates["unbindingI"], self.raw_unbinding_rates["unbindingJ"])
            ),
            shape=self.raw_unbinding_rates["shape"]
        )

        if dense:
            binding_rates = binding_rates.toarray()
            unbinding_rates = unbinding_rates.toarray()

        return binding_rates, unbinding_rates

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

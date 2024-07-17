"""
============================
Transcription Factor Binding
============================

This process models how transcription factors bind to promoters on the DNA sequence.
"""

import numpy as np
import warnings
import copy

from vivarium.core.process import Step

from ecoli.library.schema import (
    listener_schema,
    numpy_schema,
    attrs,
    bulk_name_to_idx,
    counts,
)

from wholecell.utils.random import stochasticRound
from wholecell.utils import units

from ecoli.processes.registries import topology_registry


# Register default topology for this process, associating it with process name
NAME = "ecoli-tf-binding"
TOPOLOGY = {
    "promoters": ("unique", "promoter"),
    "bulk": ("bulk",),
    #"bulk_total": ("bulk",),
    "listeners": ("listeners",),
    "timestep": ("timestep",),
    "next_update_time": ("next_update_time", "tf_binding"),
    "global_time": ("global_time",),
}
topology_registry.register(NAME, TOPOLOGY)


class TfBinding(Step):
    """Transcription Factor Binding Step"""

    name = NAME
    topology = TOPOLOGY
    defaults = {
        "tf_ids": [],
        "rna_ids": [],
        "raw_binding_rates": {"bindingI": [], "bindingJ": [], "bindingV": []},
        "raw_unbinding_rates": {"unbindingI": [], "unbindingJ": [], "unbindingV": []},
        "get_binding_unbinding_matrices": None,
        "n_avogadro": 6.02214076e23 / units.mol,
        "cell_density": 1100 * units.g / units.L,
        # Calculate promoter binding probability when not 0CS TF
        #"p_promoter_bound_tf": lambda active, inactive: float(active)
        #/ (float(active) + float(inactive)),
        "tf_to_tf_type": {},
        "active_to_bound": {},
        "get_unbound": lambda tf: "",
        "active_to_inactive_tf": {},
        "bulk_molecule_ids": [],
        "bulk_mass_data": np.array([[]]) * units.g / units.mol,
        "seed": 0,
        "submass_to_idx": {
            "rRNA": 0,
            "tRNA": 1,
            "mRNA": 2,
            "miscRNA": 3,
            "nonspecific_RNA": 4,
            "protein": 5,
            "metabolite": 6,
            "water": 7,
            "DNA": 8,
        },
        "emit_unique": False,
    }

    # Constructor
    def __init__(self, parameters=None):
        super().__init__(parameters)

        # Get IDs of transcription factors
        self.tf_ids = self.parameters["tf_ids"]
        self.n_TF = len(self.tf_ids)

        self.rna_ids = self.parameters["rna_ids"]

        # Build dict that maps TFs to transcription units they regulate
        self.raw_binding_rates = self.parameters["raw_binding_rates"]
        self.raw_unbinding_rates = self.parameters["raw_unbinding_rates"]
        self.get_binding_unbinding_matrices = self.parameters["get_binding_unbinding_matrices"]
        self.TF_to_TU_idx = {}

        # TODO: change this when switching to binding sites
        for i, tf in enumerate(self.tf_ids):
            self.TF_to_TU_idx[tf] = self.raw_binding_rates["bindingI"][
                self.raw_binding_rates["bindingJ"] == i
            ]

        # Get total counts of transcription units
        self.n_TU = self.raw_binding_rates["shape"][1]

        # Get constants
        self.n_avogadro = self.parameters["n_avogadro"]
        self.cell_density = self.parameters["cell_density"]

        # Create dictionaries and method
        #self.p_promoter_bound_tf = self.parameters["p_promoter_bound_tf"]
        self.tf_to_tf_type = self.parameters["tf_to_tf_type"]

        self.active_to_bound = self.parameters["active_to_bound"]
        self.get_unbound = self.parameters["get_unbound"]
        self.active_to_inactive_tf = self.parameters["active_to_inactive_tf"]

        self.active_tfs = {}
        self.inactive_tfs = {}

        for tf in self.tf_ids:
            self.active_tfs[tf] = tf + "[c]"

            if self.tf_to_tf_type[tf] == "1CS":
                if tf == self.active_to_bound[tf]:
                    self.inactive_tfs[tf] = self.get_unbound(tf + "[c]")
                else:
                    self.inactive_tfs[tf] = self.active_to_bound[tf] + "[c]"
            elif self.tf_to_tf_type[tf] == "2CS":
                self.inactive_tfs[tf] = self.active_to_inactive_tf[tf + "[c]"]

        self.bulk_mass_data = self.parameters["bulk_mass_data"]

        # Build array of active TF masses
        self.bulk_molecule_ids = self.parameters["bulk_molecule_ids"]
        tf_indexes = [
            np.where(self.bulk_molecule_ids == tf_id + "[c]")[0][0]
            for tf_id in self.tf_ids
        ]
        self.active_tf_masses = (
            self.bulk_mass_data[tf_indexes] / self.n_avogadro
        ).asNumber(units.fg)

        self.seed = self.parameters["seed"]
        self.random_state = np.random.RandomState(seed=self.seed)

        # Helper indices for Numpy indexing
        self.active_tf_idx = None
        if "PD00365" in self.tf_ids:
            self.marR_name = "CPLX0-7710[c]"
            self.marR_tet = "marR-tet[c]"
        self.submass_indices = self.parameters["submass_indices"]

    def ports_schema(self):
        return {
            "promoters": numpy_schema("promoters", emit=self.parameters["emit_unique"]),
            "bulk": numpy_schema("bulk"),
            "bulk_total": numpy_schema("bulk"),
            "listeners": {
                "rna_synth_prob": listener_schema(
                    {
                        "n_available_promoters": ([0] * self.n_TF, self.tf_ids),
                        "n_bound_promoters": ([0] * self.n_TF, self.tf_ids),
                        "n_binding_events": ([0] * self.n_TF, self.tf_ids),
                        "n_unbinding_events": ([0] * self.n_TF, self.tf_ids),
                        # 900 KB, very large, comment out to halve emit size
                        "n_bound_TF_per_TU": (
                            [[0] * self.n_TF] * self.n_TU,
                            self.rna_ids
                        ),
                        #
                        #
                        # #"p_promoter_bound": ([0.0] * self.n_TF, self.tf_ids),
                        # #"n_promoter_bound": ([0] * self.n_TF, self.tf_ids),
                        # "n_actual_bound": ([0] * self.n_TF, self.tf_ids),
                        # "n_available_promoters": ([0] * self.n_TF, self.tf_ids),
                        # "n_bound_TF_per_TU": (
                        #     [[0] * self.n_TF] * self.n_TU,
                        #     self.rna_ids,
                        #),
                    }
                )
            },
            "next_update_time": {
                "_default": self.parameters["time_step"],
                "_updater": "set",
                "_divider": "set",
            },
            "global_time": {"_default": 0.0},
            "timestep": {"_default": self.parameters["time_step"]},
        }

    def update_condition(self, timestep, states):
        """
        See :py:meth:`~.Requester.update_condition`.
        """
        if states["next_update_time"] <= states["global_time"]:
            if states["next_update_time"] < states["global_time"]:
                warnings.warn(
                    f"{self.name} updated at t="
                    f"{states['global_time']} instead of t="
                    f"{states['next_update_time']}. Decrease the "
                    "timestep for the global clock process for more "
                    "accurate timekeeping."
                )
            return True
        return False

    def next_update(self, timestep, states):
        # At t=0, convert all strings to indices
        if self.active_tf_idx is None:
            bulk_ids = states["bulk"]["id"]
            self.active_tf_idx = {
                tf_id: bulk_name_to_idx(tf_name, bulk_ids)
                for tf_id, tf_name in self.active_tfs.items()
            }
            # self.inactive_tf_idx = {
            #     tf_id: bulk_name_to_idx(tf_name, bulk_ids)
            #     for tf_id, tf_name in self.inactive_tfs.items()
            # }
            if "PD00365" in self.tf_ids:
                self.marR_idx = bulk_name_to_idx(self.marR_name, bulk_ids)
                self.marR_tet_idx = bulk_name_to_idx(self.marR_tet, bulk_ids)

        # If there are no promoters, return immediately
        if states["promoters"]["_entryState"].sum() == 0:
            return {"promoters": {}}

        # Get attributes of all promoters
        TU_index, bound_TF = attrs(states["promoters"], ["TU_index", "bound_TF"])
        #
        # # Calculate number of bound TFs for each TF prior to changes
        # n_bound_TF = bound_TF.sum(axis=0)

        # Initialize new bound_TF array TODO: do you need to copy?
        bound_TF_new = copy.deepcopy(bound_TF)

        # Create vectors for storing values
        #pPromotersBound = np.zeros(self.n_TF, dtype=np.float64)
        #nPromotersBound = np.zeros(self.n_TF, dtype=int)
        n_bound_promoters = np.zeros(self.n_TF, dtype=int)
        n_available_promoters = np.zeros(self.n_TF, dtype=int)
        n_binding_events = np.zeros(self.n_TF, dtype=int)
        n_unbinding_events = np.zeros(self.n_TF, dtype=int)
        n_bound_TF_per_TU = np.zeros((self.n_TU, self.n_TF), dtype=np.int16)

        update = {"bulk": []}


        # Rates of unbinding and binding for a certain TF to a binding site on a certain TU
        # TODO: modify so its for a certain TF onto a binding site, and a second matrix
        #  for associating each binding site to a certain TU
        # TODO: get actual second-order rate constant, also what are units supposed to be?
        # TODO: Consider concentration of binding site on DNA, etc.?

        # Get active tf counts
        active_tf_idxs = np.array([self.active_tf_idx[tf_id] for tf_id in self.tf_ids])
        active_tf_counts_original = counts(states["bulk"], active_tf_idxs)
        active_tf_counts = copy.deepcopy(active_tf_counts_original)
        # TODO: add this check in?
        #if np.sum(active_tf_counts) + np.sum(bound_TF) == 0:

        # Get binding and unbinding rates based on present promoters
        raw_binding_rates_matrix, raw_unbinding_rates_matrix = self.get_binding_unbinding_matrices(dense=True)
        binding_rates = raw_binding_rates_matrix[TU_index, :]
        unbinding_rates = raw_unbinding_rates_matrix[TU_index, :]
        # TODO: figure out whether can simplify using these nonzero stuff?
        # binding_rates_nonzero_idxs = np.nonzero(binding_rates)
        # unbinding_rates_nonzero_idxs = np.nonzero(unbinding_rates)
        # nonzero_binding_rates = binding_rates[binding_rates_nonzero_idxs]
        # nonzero_unbinding_rates = unbinding_rates[unbinding_rates_nonzero_idxs]

        # Calculate final binding rates accounting for reactant concentrations
        binding_rates_final = np.multiply(binding_rates, active_tf_counts)
        # TODO: later, when more than one binding site at once, can do num_total - bound_TF instead of ~bound_TF
        binding_rates_final = np.multiply(binding_rates_final, ~bound_TF_new)
        unbinding_rates_final = np.multiply(unbinding_rates, bound_TF_new)


        # Run Gillespie simulation for the length of timestep
        rxn_time = 0
        while np.sum(binding_rates_final) + np.sum(unbinding_rates_final) > 0:
            # Get binding rate sums
            total_bind_rates = np.sum(binding_rates_final)
            total_unbind_rates = np.sum(unbinding_rates_final)
            total_rates = total_bind_rates + total_unbind_rates

            # Determine reaction time
            r1 = self.random_state.rand()
            rxn_time += 1/total_rates * np.log(1/r1)
            # If we surpass the current timestep, don't perform reaction
            if rxn_time > timestep.asNumber(units.min):
                break

            # Choose whether to bind or unbind
            to_bind = self.random_state.choice([0, 1], [total_unbind_rates / total_rates,
                                                        total_bind_rates / total_rates])

            if to_bind:
                # Choose a reaction to perform
                nonzero_indices = np.nonzero(binding_rates_final)
                nonzero_probas = binding_rates_final[nonzero_indices]
                rxn_index = self.random_state.choice(np.array(x for x in zip(nonzero_indices)),
                                                     nonzero_probas)

                # Decrement the TF (given by column index), and increase the bound TF
                assert active_tf_counts[rxn_index[1]] > 0
                assert bound_TF_new[rxn_index[0], rxn_index[1]] == False
                active_tf_counts[rxn_index[1]] -= 1
                bound_TF_new[rxn_index[0], rxn_index[1]] = True

                # Record the reaction
                n_binding_events[rxn_index[1]] += 1

            else:
                # Choose a reaction to perform
                nonzero_indices = np.nonzero(unbinding_rates_final)
                nonzero_probas = unbinding_rates_final[nonzero_indices]
                rxn_index = self.random_state.choice(np.array(x for x in zip(nonzero_indices)),
                                                     nonzero_probas)

                # Increment the TF (given by column index), and decrease the bound TF
                active_tf_counts[rxn_index[1]] += 1
                bound_TF_new[rxn_index[0], rxn_index[1]] = False

                # Record the reaction
                n_unbinding_events[rxn_index[1]] += 1

            # Recalculate binding rates for next step
            binding_rates_final = np.multiply(binding_rates, active_tf_counts)
            binding_rates_final = np.multiply(binding_rates_final, ~bound_TF_new)
            unbinding_rates_final = np.multiply(unbinding_rates, bound_TF_new)

        # Update promoters bound_TF
        delta_bound_TF = bound_TF_new.astype(np.int8) - bound_TF.astype(np.int8)
        mass_diffs = delta_bound_TF.dot(self.active_tf_masses)
        submass_update = {
            submass: attrs(states["promoters"], [submass])[0] + mass_diffs[:, i]
            for submass, i in self.submass_indices.items()
        }
        update["promoters"] = {"set": {"bound_TF": bound_TF_new.astype(bool), **submass_update}}

        # Update active TFs
        delta_active_TF = active_tf_counts - active_tf_counts_original
        update["bulk"].append((active_tf_idxs, delta_active_TF))


        # So each timestep, certain promoters get bound, certain promoters are unbound.
        # There's a certain number of total available promoters that could be bound.
        # We'd want to get 1. For each TF, how many promoters it could bind to, how many promoters
        # it has currently bound, and how many association/disociation events have occured.
        # 2. For each TU, how many of each kind of TF is bound to it by the end.

        # Record end values for listeners
        for tf_idx, tf_id in enumerate(self.tf_ids):
            available_promoters = np.isin(TU_index, self.TF_to_TU_idx[tf_id])
            n_available_promoters[tf_idx] = np.count_nonzero(available_promoters)
            n_bound_TF_per_TU[:, tf_idx] = np.bincount(
                TU_index[bound_TF_new[:, tf_idx]], minlength=self.n_TU
            )

        # TODO: might have to think about how to do this if more than one binding site per TU, etc.
        n_bound_promoters = bound_TF_new.sum(axis=0)

        # Update listeners
        update["listeners"] = {
            "rna_synth_prob": {
                #"p_promoter_bound": pPromotersBound,
                #"n_promoter_bound": nPromotersBound,
                #"n_actual_bound": nActualBound,
                "n_available_promoters": n_available_promoters,
                "n_bound_promoters": n_bound_promoters,
                "n_binding_events": n_binding_events,
                "n_unbinding_events": n_unbinding_events,
                # 900 KB, very large, comment out to halve emit size
                "n_bound_TF_per_TU": n_bound_TF_per_TU,
            },
        }

        update["next_update_time"] = states["global_time"] + states["timestep"]
        return update

        # TODO: should divide by total dna mass vs normal dna mass?

        # How would I make a stoichiometry? The species are active TFs, bound binding-sites, unbound binding-sites.
        # The reactions are: one more active TF, one less bound binding-site, one more unbound binding-site.
        # and one less active TF, one more bound binding-site, one less unbound binding-site
        # The rates are given in binding_rates and unbinding_rates, which we need to correlate to these reactions.
        # In the end, we need to read how many binding sites there are,

        # # TODO: get actual value from Km from paper or smth like that
        # # TODO: get it from from transcription_regulation reconstruction
        # # (and do sparse matrix if it's too slow but probably fine)
        # unbinding_rate_matrix[purR_idx, purC_idx] = 0.1 / units.min
        #
        # # TODO: right now assumes each TF binds only to one site per TU, and that
        # # all resulting unbindings are independent. Would get more complicated if
        # # a TU can have multiple binding sites for the same TF, or if we take into
        # # account multiple TFs regulating the same TU, etc.
        # unbinding_rates = unbinding_rate_matrix[:, TU_index]
        # unbinding_rates = np.multiply(unbinding_rates, bound_TF)
        # unbinding_events = np.random.poisson(unbinding_rates * timestep.asNumber(units.min))
        # # Can't unbind more TFs than are there (also ensures we don't unbind TFs from places
        # # where they couldn't have bound in the first place)
        # unbound_too_much = (unbinding_events > bound_TF)
        # unbinding_events[unbound_too_much] = bound_TF[unbound_too_much]

        # TODO: think about whether to do anything differently cuz of 0CS, 1CS?

        #
        # for tf_idx, tf_id in enumerate(self.tf_ids):
        #     curr_tf_idx = self.active_tf_idx[tf_id]
        #     active_tf_count = counts(states["bulk"], curr_tf_idx)
        #     bound_tf_counts = n_bound_TF[tf_idx]
        #
        #     binding_promoters = np.isin(TU_index, self.TF_to_TU_idx[tf_id])
        #     available_promoters = (binding_promoters & ~bound_TF)
        #     n_available_promoters = np.count_nonzero(available_promoters)
        #
        #
        #     if active_tf_count > 0 & n_available_promoters > 0:
        #         # TODO: Calculate a rate to bind them and bind as needed
        #         # TODO: figure out if there might be a problem with the allocator?

            #if bound_tf_counts > 0:
            #    unbinding_rates = unbinding_rate_matrix[tf_idx, ]


                # TODO: Calculate a rate to unbind them and unbind as needed

            # TODO: update a bound_tf_new array

            # # Free all DNA-bound transcription factors into free active
            # # transcription factors
            # curr_tf_idx = self.active_tf_idx[tf_id]
            # tf_count = counts(states["bulk"], curr_tf_idx)
            #
            # bound_tf_counts = n_bound_TF[tf_idx]
            # update["bulk"].append((curr_tf_idx, bound_tf_counts))
            #
            # # Get counts of transcription factors
            # active_tf_counts = (
            #     counts(states["bulk_total"], curr_tf_idx) + bound_tf_counts
            # )
            # n_available_active_tfs = tf_count + bound_tf_counts
            #
            # # NEW to vivarium-ecoli
            # # Uncomplexed marR reduces active marA
            # if tf_id == "PD00365":
            #     marR_count = counts(states["bulk_total"], self.marR_idx)
            #     marR_tet_count = counts(states["bulk_total"], self.marR_tet_idx)
            #     # marA activity ramps up as more marR is complexed off
            #     # TODO: Figure out how to modify ParCa so MarA/R are included
            #     # as active TFs so no need to compromise basal or tetracycline
            #     # behavior when total MarR count is zero
            #     ratio = marR_tet_count / max(marR_count + marR_tet_count, 1)
            #     # 34 = # of promoters for genes that marA regulates
            #     n_available_active_tfs = int(34 * ratio)
            #
            # # Determine the number of available promoter sites
            # available_promoters = np.isin(TU_index, self.TF_to_TU_idx[tf_id])
            # n_available_promoters = np.count_nonzero(available_promoters)
            # n_promoters[tf_idx] = n_available_promoters
            #
            # # If there are no active transcription factors to work with,
            # # continue to the next transcription factor
            # if n_available_active_tfs == 0:
            #     continue
            #
            # # Compute probability of binding the promoter
            # if self.tf_to_tf_type[tf_id] == "0CS":
            #     pPromoterBound = 1.0
            # else:
            #     inactive_tf_counts = counts(
            #         states["bulk_total"], self.inactive_tf_idx[tf_id]
            #     )
            #     pPromoterBound = self.p_promoter_bound_tf(
            #         active_tf_counts, inactive_tf_counts
            #     )
            #
            # # Calculate the number of promoters that should be bound
            # n_to_bind = int(
            #     min(
            #         stochasticRound(
            #             self.random_state,
            #             np.full(n_available_promoters, pPromoterBound),
            #         ).sum(),
            #         n_available_active_tfs,
            #     )
            # )
            #
            # bound_locs = np.zeros(n_available_promoters, dtype=bool)
            # if n_to_bind > 0:
            #     # Determine randomly which DNA targets to bind based on which of
            #     # the following is more limiting:
            #     # number of promoter sites to bind, or number of active
            #     # transcription factors
            #     bound_locs[
            #         self.random_state.choice(
            #             n_available_promoters, size=n_to_bind, replace=False
            #         )
            #     ] = True
            #
            #     # Update count of free transcription factors
            #     update["bulk"].append((curr_tf_idx, -bound_locs.sum()))
            #
            #     # Update bound_TF array
            #     bound_TF_new[available_promoters, tf_idx] = bound_locs
            #
            # n_bound_TF_per_TU[:, tf_idx] = np.bincount(
            #     TU_index[bound_TF_new[:, tf_idx]], minlength=self.n_TU
            # )
            #
            # # Record values
            # pPromotersBound[tf_idx] = pPromoterBound
            # nPromotersBound[tf_idx] = n_to_bind
            # nActualBound[tf_idx] = bound_locs.sum()

        #n_bound_TF_per_TU[:, tf_idx] = np.bincount(
            #     TU_index[bound_TF_new[:, tf_idx]], minlength=self.n_TU
            # )
        # available_promoters = np.isin(TU_index, self.TF_to_TU_idx[tf_id])
        # n_available_promoters = np.count_nonzero(available_promoters)
        # n_promoters[tf_idx] = n_available_promoters




def test_tf_binding_listener():
    from ecoli.experiments.ecoli_master_sim import EcoliSim

    sim = EcoliSim.from_file()
    sim.total_time = 2
    sim.raw_output = False
    sim.build_ecoli()
    sim.run()
    data = sim.query()
    assert data is not None


if __name__ == "__main__":
    test_tf_binding_listener()

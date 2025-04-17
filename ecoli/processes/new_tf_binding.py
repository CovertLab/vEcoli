"""
============================
New Transcription Factor Binding
============================

This process models how transcription factors bind to tf-binding sites on the DNA sequence.
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
NAME = "ecoli-new-tf-binding"
TOPOLOGY = {
    "tf_binding_sites": ("unique", "tf_binding_site"),
    "bulk": ("bulk",),
    "listeners": ("listeners",),
    "timestep": ("timestep",),
    "next_update_time": ("next_update_time", "tf_binding"),
    "global_time": ("global_time",),
}
topology_registry.register(NAME, TOPOLOGY)


class NewTfBinding(Step):
    """Transcription Factor Binding Step"""

    name = NAME
    topology = TOPOLOGY
    defaults = {
        "tf_ids": [],
        "rna_ids": [],
        "get_binding_unbinding_matrices": None,
        "tf_binding_site_unbound_idx": -1,
        "get_tf_binding_site_to_TU_matrix": None,
        "n_avogadro": 6.02214076e23 / units.mol,
        "cell_density": 1100 * units.g / units.L,
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


        # TF-regulation gives binding_site_num x tf_num binding and unbinding rates.
        # What we want to do is: for unbinding, since it's slow (we assume for now),
        # can simply get whether the binding site is occupied, and then based on the
        # unbinding rate, each has a probability of falling off during the timestep,
        # and then we can calculate that.
        # For binding: usually it's fast, maybe we could do a Gillespie.

        # Get IDs of transcription factors
        self.tf_ids = self.parameters["tf_ids"]
        self.n_TF = len(self.tf_ids)

        # Get rna_ids
        self.rna_ids = self.parameters["rna_ids"]
        self.n_TU = len(self.rna_ids)

        self.get_binding_unbinding_matrices = self.parameters["get_binding_unbinding_matrices"]
        self.tf_unbound_idx = self.parameters["tf_binding_site_unbound_idx"]

        # Get tf-binding-site to TUs mapping matrix
        self.get_tf_binding_site_to_TU_matrix = self.parameters["get_tf_binding_site_to_TU_matrix"]

        # Get constants
        self.n_avogadro = self.parameters["n_avogadro"]
        self.cell_density = self.parameters["cell_density"]

        # Create dictionaries and method
        # TODO: not sure if will need later
        #self.tf_to_tf_type = self.parameters["tf_to_tf_type"]

        self.active_tfs = {}
        for tf in self.tf_ids:
            self.active_tfs[tf] = tf + "[c]"

        self.bulk_mass_data = self.parameters["bulk_mass_data"]

        # Build array of active TF masses
        self.bulk_molecule_ids = self.parameters["bulk_molecule_ids"]
        tf_indexes = [
            np.where(self.bulk_molecule_ids == self.active_tfs[tf_id])[0][0]
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

        self.time_step = self.parameters["time_step"] # TODO: where to save that units are units.s?

    def ports_schema(self):
        return {
            "tf_binding_sites": numpy_schema("tf_binding_sites", emit=self.parameters["emit_unique"]),
            "bulk": numpy_schema("bulk"),
            "listeners": {
                "rna_synth_prob": listener_schema(
                    {
                        "new_tf_binding": {
                            "n_available_binding_sites": ([0] * self.n_TF, self.tf_ids),
                            "n_bound_binding_sites": ([0] * self.n_TF, self.tf_ids),
                            "n_binding_events": ([0] * self.n_TF, self.tf_ids),
                            "n_unbinding_events": ([0] * self.n_TF, self.tf_ids),
                            # 900 KB, very large, comment out to halve emit size
                            "n_bound_TF_per_TU": (
                                [[0] * self.n_TF] * self.n_TU,
                                self.rna_ids
                            ),
                        }
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

            if "PD00365" in self.tf_ids:
                self.marR_idx = bulk_name_to_idx(self.marR_name, bulk_ids)
                self.marR_tet_idx = bulk_name_to_idx(self.marR_tet, bulk_ids)
        # TODO: restore marR regulation things

        # If there are no tf-binidng-sites, return immediately
        if states["tf_binding_sites"]["_entryState"].sum() == 0:
            return {"tf_binding_sites": {}}

        # Get attributes of all TF-binding sites
        binding_site_index, bound_TF = attrs(states["tf_binding_sites"], [
            "tf_binding_site_index", "bound_TF"])

        new_bound_TF = copy.deepcopy(bound_TF)
        binding_site_is_free = (new_bound_TF == self.tf_unbound_idx)
        n_available_binding_sites = np.zeros(self.n_TF, dtype=int)
        n_bound_binding_sites = np.zeros(self.n_TF, dtype=int)
        n_binding_events = np.zeros(self.n_TF, dtype=int)
        n_unbinding_events = np.zeros(self.n_TF, dtype=int)
        n_bound_TF_per_TU = np.zeros((self.n_TU, self.n_TF), dtype=np.int16)

        update = {"bulk": []}


        # Rates of binding for a certain TF to a binding site on a certain TU
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

        # Get binding and unbinding rates based on present tf-binding sites
        raw_binding_rates_matrix, raw_unbinding_rates_matrix = self.get_binding_unbinding_matrices(dense=True)
        binding_rates = raw_binding_rates_matrix[binding_site_index, :]
        unbinding_rates = raw_unbinding_rates_matrix[binding_site_index, :]

        # For listener: get total number of (bound or unbound) binding sites that a given TF could bind to,
        # by looking at binding matrices
        can_bind = (binding_rates > 0.).astype(int)
        n_binding_sites_per_TF = np.sum(can_bind, axis=0)

        # TODO: figure out whether can simplify using these nonzero stuff?
        # binding_rates_nonzero_idxs = np.nonzero(binding_rates)
        # unbinding_rates_nonzero_idxs = np.nonzero(unbinding_rates)
        # nonzero_binding_rates = binding_rates[binding_rates_nonzero_idxs]
        # nonzero_unbinding_rates = unbinding_rates[unbinding_rates_nonzero_idxs]

        # Calculate final binding rates accounting for reactant concentrations
        binding_rates_final = np.multiply(binding_rates, active_tf_counts)
        # TODO: how to ensure they're multiplying on the right axis?
        binding_rates_final[~binding_site_is_free, :] = 0.

        # Run Gillespie simulation for binding reactions for the length of timestep
        rxn_time = 0
        while np.sum(binding_rates_final) > 0:
            # Get binding rate sums
            total_rates = np.sum(binding_rates_final)

            # Determine reaction time
            r1 = self.random_state.rand()
            rxn_time += 1/total_rates * np.log(1/r1)

            # If we surpass the current timestep, don't perform reaction
            if rxn_time > self.time_step:
                break

            # Choose a reaction to perform
            nonzero_indices = np.nonzero(binding_rates_final)
            nonzero_probas = binding_rates_final[nonzero_indices]
            nonzero_probas = nonzero_probas / np.sum(nonzero_probas)
            nonzero_indices = np.transpose(nonzero_indices)
            chosen_idx = self.random_state.choice(len(nonzero_indices), p=nonzero_probas)
            rxn_index = nonzero_indices[chosen_idx]

            # Decrement the TF (given by column index), change the bound_TF
            assert active_tf_counts[rxn_index[1]] > 0
            assert binding_site_is_free[rxn_index[0]]
            active_tf_counts[rxn_index[1]] -= 1
            new_bound_TF[rxn_index[0]] = rxn_index[1]
            binding_site_is_free[rxn_index[0]] = False

            # Record the reaction
            n_binding_events[rxn_index[1]] += 1

            # Recalculate binding rates for next step
            binding_rates_final = np.multiply(binding_rates, active_tf_counts)
            binding_rates_final[~binding_site_is_free, :] = 0.

        # Unbinding
        # Create unbinding rates array from bound TFs; units are 1/min
        unbinding_rates_final = []
        for i, tf_idx in enumerate(bound_TF):
            if tf_idx != self.tf_unbound_idx:
                unbinding_rates_final.append(unbinding_rates[i, tf_idx])
            else:
                unbinding_rates_final.append(0.)

        # Poisson distribution for probability of event occurring
        unbinding_events = self.random_state.poisson(
            self.time_step / 60 * np.array(unbinding_rates_final), # TODO: where to save units of seconds for self.time_step?
            size=len(unbinding_rates_final))
        # At most one event can occur
        unbinding_events = (unbinding_events > 0)

        # Update molecule counts
        # TODO: remove these asserts when they've been checked to not occur
        assert self.tf_unbound_idx not in new_bound_TF[unbinding_events]
        new_bound_TF[unbinding_events] = self.tf_unbound_idx
        unbound_tfs = bound_TF[unbinding_events]
        assert self.tf_unbound_idx not in unbound_tfs
        unbound_tf_counts = np.bincount(unbound_tfs, minlength=self.n_TF)
        active_tf_counts += unbound_tf_counts

        # Record reactions
        n_unbinding_events += unbound_tf_counts

        # Update tf-binding-site objects
        mass_start = self.active_tf_masses[bound_TF, :]
        mass_start[(bound_TF == self.tf_unbound_idx), :] = 0.
        mass_end = self.active_tf_masses[new_bound_TF, :]
        mass_end[binding_site_is_free, :] = 0.
        mass_diffs = mass_end - mass_start
        submass_update = {
            submass: attrs(states["tf_binding_sites"], [submass])[0] + mass_diffs[:, i]
            for submass, i in self.submass_indices.items()
        }
        update["tf_binding_sites"] = {"set": {"bound_TF": new_bound_TF.astype(int), **submass_update}}

        # Update active TFs
        delta_active_TF = active_tf_counts - active_tf_counts_original
        update["bulk"].append((active_tf_idxs, delta_active_TF))

        # Record end values for listeners
        # Each timestep, certain tf-binding-sites get bound and unbound.
        # There's a certain number of total available binding-sites that could be bound.
        # We get 1. For each TF, how many binding-sites it could bind to, how many binding-sites
        # it has currently bound, and how many binding/unbinding events have occurred.
        # 2. For each TU, how many of each kind of TF is bound to it by the end.
        n_bound_binding_sites = np.array([len(np.where(new_bound_TF == idx)[0])
                                 for idx in range(self.n_TF)])
        n_available_binding_sites = n_binding_sites_per_TF - n_bound_binding_sites

        tf_binding_site_to_TU_matrix = self.get_tf_binding_site_to_TU_matrix()
        for i in range(self.n_TU):
            binding_sites_of_TU = tf_binding_site_to_TU_matrix[:, i].astype(bool)
            tfs_bound = new_bound_TF[binding_sites_of_TU[binding_site_index]]
            for tf_idx in tfs_bound:
                if tf_idx != self.tf_unbound_idx:
                    n_bound_TF_per_TU[i, tf_idx] += 1

        # Update listeners
        update["listeners"] = {
            "rna_synth_prob": {
                "new_tf_binding": {
                    "n_available_binding_sites": n_available_binding_sites,
                    "n_bound_binding_sites": n_bound_binding_sites,
                    "n_binding_events": n_binding_events,
                    "n_unbinding_events": n_unbinding_events,
                    # 900 KB, very large, comment out to halve emit size
                    "n_bound_TF_per_TU": n_bound_TF_per_TU,
                }
            },
        }

        update["next_update_time"] = states["global_time"] + states["timestep"]
        return update

        # TODO: maybe should divide by total dna mass vs normal dna mass?

        # # TODO: get actual binding/unbinding rates from Km from paper or smth like that


def test_new_tf_binding_listener():
    from ecoli.experiments.ecoli_master_sim import EcoliSim

    sim = EcoliSim.from_file()
    sim.total_time = 2
    sim.raw_output = False
    sim.build_ecoli()
    sim.run()
    data = sim.query()
    assert data is not None


if __name__ == "__main__":
    test_new_tf_binding_listener()

"""
===========
TF Ligand Binding
===========

This process models how ligands are bound to or unbound
from their transcription factor binding partners, somewhat
less mechanistically than in Equilibrium, to enable more robust
TF activity across different media.
"""

import numpy as np

from ecoli.library.schema import numpy_schema, bulk_name_to_idx, counts, listener_schema
from ecoli.processes.registries import topology_registry
from ecoli.processes.partition import PartitionedProcess

from wholecell.utils import units
from wholecell.utils.random import stochasticRound


# Register default topology for this process, associating it with process name
NAME = "ecoli-tf-ligand-binding"
TOPOLOGY = {"listeners": ("listeners",), "bulk": ("bulk",), "timestep": ("timestep",)}
topology_registry.register(NAME, TOPOLOGY)


class TFLigandBinding(PartitionedProcess):
    """TF Ligand Binding PartitionedProcess

    molecule_names: list of molecules that are being iterated over size:94
    """

    name = NAME
    topology = TOPOLOGY
    defaults = {
        "n_avogadro": 0.0,
        "cell_density": 0.0,
        "stoich_matrix": [[]],
        "ligand_idxs": [],
        "bound_tf_idxs": [],
        "unbound_tf_idxs": [],
        "molecule_names": [],
        "reaction_ids": [],
        "ligand_bound_fraction": lambda x: [],
        "req_from_fluxes": lambda x: [],
        "seed": 0,
    }

    # Constructor
    def __init__(self, parameters=None):
        super().__init__(parameters)

        # Get constants
        self.n_avogadro = self.parameters["n_avogadro"]
        self.cell_density = self.parameters["cell_density"]

        # Create matrix
        # stoichMatrix: (94, 33), molecule counts are (94,).
        ## TODO: so, for purR, we need smth saying that apopurR and hypoxanthine
        # become holopurR. So I can pass a function the hypoxanthine amount,
        # and it'll return the active fraction of purR. Then I can calculate
        # request based on that.
        # StoichMatrix: have columns as moleculeNames. Then each row is a reaction.
        # Let's say I have two TFs. Then I can have a 2x6 matrix that's, for each reaction,
        # which is the apopurR and which is the holopurR, which is the metabolite and how many.
        # Then a 6-length molecule_names that corresponds to the columns. And then a
        # metabolite set that's a 2-length list showing which of the ones are metabolites.
        self.stoich_matrix = self.parameters["stoich_matrix"]

        # Build views
        # moleculeNames: list of molecules that are being iterated over size: 94
        self.molecule_names = self.parameters["molecule_names"]
        self.molecule_idx = None
        # The indexes of molecules corresponding to the stoichMatrix columns
        # (NOT bulk molecule counts)
        # Assumes these are the same shape as reaction_ids, that is, each
        # reaction involves one of each. TODO: true for now, maybe generalize
        #  to include cases where it is not true?
        self.bound_tf_idxs = self.parameters["bound_tf_idxs"]
        self.unbound_tf_idxs = self.parameters["unbound_tf_idxs"]
        self.ligand_idxs = self.parameters["ligand_idxs"]

        self.bound_tf_ids = self.molecule_names[self.bound_tf_idxs]

        self.seed = self.parameters["seed"]
        self.random_state = np.random.RandomState(seed=self.seed)

        self.reaction_ids = self.parameters["reaction_ids"]
        self.ligand_bound_fraction = self.parameters["ligand_bound_fraction"]
        self.req_from_fluxes = self.parameters["req_from_fluxes"]

        # Values to be passed to listeners
        # The target amount of bound TFs after tf_ligand_binding has run
        self.target_bound_tfs = []
        # The target rate of reactions prior to allocation
        self.target_reaction_rates = []

        # Values to be set during simulation
        self.req = None
        self.rxn_fluxes = None

    def ports_schema(self):
        return {
            "bulk": numpy_schema("bulk"),
            "listeners": {
                "mass": listener_schema({"cell_mass": 0}),
                "tf_ligand_binding": {
                    **listener_schema(
                        {
                            "actual_reaction_rates": (
                                [0.0] * len(self.reaction_ids),
                                self.reaction_ids,
                            ),
                            "target_reaction_rates": (
                                [0.0] * len(self.reaction_ids),
                                self.reaction_ids
                            ),
                            "target_bound_tfs": (
                                [0.0] * len(self.bound_tf_ids),
                                self.bound_tf_ids
                            ),
                        }
                    )
                },
            },
            "timestep": {"_default": self.parameters["time_step"]},
        }

    def calculate_request(self, timestep, states):
        # At t=0, convert all strings to indices
        if self.molecule_idx is None:
            self.molecule_idx = bulk_name_to_idx(
                self.molecule_names, states["bulk"]["id"]
            )

        # Get molecule counts
        moleculeCounts = counts(states["bulk"], self.molecule_idx)
        unbound_tf_counts = moleculeCounts[self.unbound_tf_idxs]
        bound_tf_counts = moleculeCounts[self.bound_tf_idxs]
        total_tf_counts = unbound_tf_counts + bound_tf_counts
        # TODO: might need to change if reactions can have more than one type of ligand
        ligand_counts = moleculeCounts[self.ligand_idxs]

        # Get cell mass and volume to convert ligand counts to concentrations
        cellMass = (states["listeners"]["mass"]["cell_mass"] * units.fg).asNumber(
            units.g
        )
        cellVolume = cellMass / self.cell_density
        counts_to_conc = 1 / (self.n_avogadro * cellVolume)
        ligand_concs = ligand_counts * counts_to_conc

        bound_frac = self.ligand_bound_fraction(ligand_concs)
        target_bound_tfs = total_tf_counts * bound_frac
        target_rxn_fluxes = target_bound_tfs - bound_tf_counts.astype(float)
        bound_tf_coeffs = np.array([self.stoich_matrix[i, idx] for i, idx in enumerate(self.bound_tf_idxs)]).astype(float)
        target_rxn_fluxes /= bound_tf_coeffs
        target_rxn_fluxes = stochasticRound(
            self.random_state, target_rxn_fluxes
        ).astype(np.int64)

        self.rxn_fluxes = target_rxn_fluxes

        # Get requested molecule counts
        self.req = self.req_from_fluxes(target_rxn_fluxes)

        # Values to be passed to listeners in evolve_state
        self.target_bound_tfs = target_bound_tfs
        self.target_reaction_rates = target_rxn_fluxes / timestep

        # Request counts of molecules needed
        requests = {"bulk": [(self.molecule_idx, self.req.astype(int))]}

        return requests

    def evolve_state(self, timestep, states):
        # Get molecule counts
        moleculeCounts = counts(states["bulk"], self.molecule_idx)

        # Get counts of molecules allocated to this process
        rxn_fluxes = self.rxn_fluxes.copy()

        # If we didn't get allocated all the molecules we need, make do with
        # what we have (decrease reaction fluxes so that they make use of what
        # we have, but not more). Reduces at least one reaction every iteration
        # so the max number of iterations is the number of reactions that were
        # originally expected to occur + 1 to reach the break statement.
        max_iterations = int(np.abs(rxn_fluxes).sum()) + 1
        for it in range(max_iterations):
            # Check if any metabolites will have negative counts with current reactions
            negative_metabolite_idxs = np.where(
                np.dot(self.stoich_matrix, rxn_fluxes) + moleculeCounts < 0
            )[0]
            if len(negative_metabolite_idxs) == 0:
                break

            # Reduce reactions that consume metabolites with negative counts
            limited_rxn_stoich = self.stoich_matrix[negative_metabolite_idxs, :]
            fwd_rxn_idxs = np.where(
                np.logical_and(limited_rxn_stoich < 0, rxn_fluxes > 0)
            )[1]
            rev_rxn_idxs = np.where(
                np.logical_and(limited_rxn_stoich > 0, rxn_fluxes < 0)
            )[1]
            rxn_fluxes[fwd_rxn_idxs] -= 1
            rxn_fluxes[rev_rxn_idxs] += 1
            rxn_fluxes[fwd_rxn_idxs] = np.fmax(0, rxn_fluxes[fwd_rxn_idxs])
            rxn_fluxes[rev_rxn_idxs] = np.fmin(0, rxn_fluxes[rev_rxn_idxs])
        else:
            raise ValueError(
                "Could not get positive counts in equilibrium with"
                " allocated molecules."
            )

        # Increment changes in molecule counts
        delta_molecules = np.dot(self.stoich_matrix, rxn_fluxes).astype(int)

        update = {
            "bulk": [(self.molecule_idx, delta_molecules)],
            "listeners": {
                "tf_ligand_binding_listener": {
                    "actual_reaction_rates": delta_molecules[self.bound_tf_idxs]
                    / states["timestep"],
                    "target_bound_tfs": self.target_bound_tfs,
                    "target_reaction_rates": self.target_reaction_rates,
                }
            },
        }

        return update


def test_tf_ligand_binding_listener():
    from ecoli.experiments.ecoli_master_sim import EcoliSim
    # TODO: make listener test
    # sim = EcoliSim.from_file()
    # sim.total_time = 2
    # sim.raw_output = False
    # sim.build_ecoli()
    # sim.run()
    # listeners = sim.query()["agents"]["0"]["listeners"]
    # assert isinstance(listeners["equilibrium_listener"]["reaction_rates"][0], list)
    # assert isinstance(listeners["equilibrium_listener"]["reaction_rates"][1], list)


if __name__ == "__main__":
    test_tf_ligand_binding_listener()

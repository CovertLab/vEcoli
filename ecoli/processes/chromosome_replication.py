"""
======================
Chromosome Replication
======================

Performs initiation, elongation, and termination of active partial chromosomes
that replicate the chromosome.

First, a round of replication is initiated at a ﬁxed cell mass per origin
of replication and generally occurs once per cell cycle. Second, replication
forks are elongated up to the maximal expected elongation rate, dNTP resource
limitations, and template strand sequence but elongation does not take into
account the action of topoisomerases or the enzymes in the replisome. Finally,
replication forks terminate once they reach the end of their template strand
and the chromosome immediately decatenates forming two separate chromosome
molecules.
"""

import numpy as np

from ecoli.library.schema import (
    numpy_schema,
    counts,
    attrs,
    bulk_name_to_idx,
    listener_schema,
)

from wholecell.utils import units
from wholecell.utils.polymerize import buildSequences, polymerize, computeMassIncrease

from ecoli.processes.registries import topology_registry
from ecoli.processes.partition import PartitionedProcess


# Register default topology for this process, associating it with process name
NAME = "ecoli-chromosome-replication"
TOPOLOGY = {
    "bulk": ("bulk",),
    "active_replisomes": ("unique", "active_replisome"),
    "oriCs": ("unique", "oriC"),
    "chromosome_domains": ("unique", "chromosome_domain"),
    "full_chromosomes": ("unique", "full_chromosome"),
    "listeners": ("listeners",),
    "environment": ("environment",),
    "timestep": ("timestep",),
}
topology_registry.register(NAME, TOPOLOGY)


class ChromosomeReplication(PartitionedProcess):
    """Chromosome Replication PartitionedProcess"""

    name = NAME
    topology = TOPOLOGY
    defaults = {
        "max_time_step": 2.0,
        "get_dna_critical_mass": lambda doubling_time: units.Unum,
        "criticalInitiationMass": 975 * units.fg,
        "nutrientToDoublingTime": {},
        "replichore_lengths": np.array([]),
        "sequences": np.array([]),
        "polymerized_dntp_weights": [],
        "replication_coordinate": np.array([]),
        "D_period": np.array([]),
        "replisome_protein_mass": 0,
        "no_child_place_holder": -1,
        "basal_elongation_rate": 967,
        "make_elongation_rates": (
            lambda random, replisomes, base, time_step: units.Unum
        ),
        "mechanistic_replisome": True,
        # molecules
        "replisome_trimers_subunits": [],
        "replisome_monomers_subunits": [],
        "dntps": [],
        "ppi": [],
        # random seed
        "seed": 0,
        "emit_unique": False,
    }

    def __init__(self, parameters=None):
        super().__init__(parameters)

        self.max_time_step = self.parameters["max_time_step"]

        # Load parameters
        self.get_dna_critical_mass = self.parameters["get_dna_critical_mass"]
        self.criticalInitiationMass = self.parameters["criticalInitiationMass"]
        self.nutrientToDoublingTime = self.parameters["nutrientToDoublingTime"]
        self.replichore_lengths = self.parameters["replichore_lengths"]
        self.sequences = self.parameters["sequences"]
        self.polymerized_dntp_weights = self.parameters["polymerized_dntp_weights"]
        self.replication_coordinate = self.parameters["replication_coordinate"]
        self.D_period = self.parameters["D_period"]
        self.replisome_protein_mass = self.parameters["replisome_protein_mass"]
        self.no_child_place_holder = self.parameters["no_child_place_holder"]
        self.basal_elongation_rate = self.parameters["basal_elongation_rate"]
        self.make_elongation_rates = self.parameters["make_elongation_rates"]

        # Sim options
        self.mechanistic_replisome = self.parameters["mechanistic_replisome"]

        # random state
        self.seed = self.parameters["seed"]
        self.random_state = np.random.RandomState(seed=self.seed)

        self.emit_unique = self.parameters.get("emit_unique", True)

        # Bulk molecule names
        self.replisome_trimers_subunits = self.parameters["replisome_trimers_subunits"]
        self.replisome_monomers_subunits = self.parameters[
            "replisome_monomers_subunits"
        ]
        self.dntps = self.parameters["dntps"]
        self.ppi = self.parameters["ppi"]

        self.ppi_idx = None

    def ports_schema(self):
        return {
            # bulk molecules
            "bulk": numpy_schema("bulk"),
            "listeners": {
                "mass": listener_schema({"cell_mass": 0.0}),
                "replication_data": listener_schema(
                    {"critical_initiation_mass": 0.0, "critical_mass_per_oriC": 0.0}
                ),
            },
            "environment": {
                "media_id": {"_default": "", "_updater": "set"},
            },
            "active_replisomes": numpy_schema(
                "active_replisomes", emit=self.parameters["emit_unique"]
            ),
            "oriCs": numpy_schema("oriCs", emit=self.parameters["emit_unique"]),
            "chromosome_domains": numpy_schema(
                "chromosome_domains", emit=self.parameters["emit_unique"]
            ),
            "full_chromosomes": numpy_schema(
                "full_chromosomes", emit=self.parameters["emit_unique"]
            ),
            "timestep": {"_default": self.parameters["time_step"]},
        }

    def calculate_request(self, timestep, states):
        if self.ppi_idx is None:
            self.ppi_idx = bulk_name_to_idx(self.ppi, states["bulk"]["id"])
            self.replisome_trimers_idx = bulk_name_to_idx(
                self.replisome_trimers_subunits, states["bulk"]["id"]
            )
            self.replisome_monomers_idx = bulk_name_to_idx(
                self.replisome_monomers_subunits, states["bulk"]["id"]
            )
            self.dntps_idx = bulk_name_to_idx(self.dntps, states["bulk"]["id"])
        requests = {}
        # Get total count of existing oriC's
        n_oriC = states["oriCs"]["_entryState"].sum()
        # If there are no origins, return immediately
        if n_oriC == 0:
            return requests

        # Get current cell mass
        cellMass = states["listeners"]["mass"]["cell_mass"] * units.fg

        # Get critical initiation mass for current simulation environment
        current_media_id = states["environment"]["media_id"]
        self.criticalInitiationMass = self.get_dna_critical_mass(
            self.nutrientToDoublingTime[current_media_id]
        )

        # Calculate mass per origin of replication, and compare to critical
        # initiation mass. If the cell mass has reached this critical mass,
        # the process will initiate a round of chromosome replication for each
        # origin of replication.
        massPerOrigin = cellMass / n_oriC
        self.criticalMassPerOriC = massPerOrigin / self.criticalInitiationMass

        # If replication should be initiated, request subunits required for
        # building two replisomes per one origin of replication, and edit
        # access to oriC and chromosome domain attributes
        requests["bulk"] = []
        if self.criticalMassPerOriC >= 1.0:
            requests["bulk"].append((self.replisome_trimers_idx, 6 * n_oriC))
            requests["bulk"].append((self.replisome_monomers_idx, 2 * n_oriC))

        # If there are no active forks return
        n_active_replisomes = states["active_replisomes"]["_entryState"].sum()
        if n_active_replisomes == 0:
            return requests

        # Get current locations of all replication forks
        (fork_coordinates,) = attrs(states["active_replisomes"], ["coordinates"])
        sequence_length = np.abs(np.repeat(fork_coordinates, 2))

        self.elongation_rates = self.make_elongation_rates(
            self.random_state,
            len(self.sequences),
            self.basal_elongation_rate,
            states["timestep"],
        )

        sequences = buildSequences(
            self.sequences,
            np.tile(np.arange(4), n_active_replisomes // 2),
            sequence_length,
            self.elongation_rates,
        )

        # Count number of each dNTP in sequences for the next timestep
        sequenceComposition = np.bincount(
            sequences[sequences != polymerize.PAD_VALUE], minlength=4
        )

        # If one dNTP is limiting then limit the request for the other three by
        # the same ratio
        dNtpsTotal = counts(states["bulk"], self.dntps_idx)
        maxFractionalReactionLimit = (
            np.fmin(1, dNtpsTotal / sequenceComposition)
        ).min()

        # Request dNTPs
        requests["bulk"].append(
            (
                self.dntps_idx,
                (maxFractionalReactionLimit * sequenceComposition).astype(int),
            )
        )

        return requests

    def evolve_state(self, timestep, states):
        # Initialize the update dictionary
        update = {
            "bulk": [],
            "active_replisomes": {},
            "oriCs": {},
            "chromosome_domains": {},
            "full_chromosomes": {},
            "listeners": {"replication_data": {}},
        }

        # Module 1: Replication initiation
        # Get number of existing replisomes and oriCs
        n_active_replisomes = states["active_replisomes"]["_entryState"].sum()
        n_oriC = states["oriCs"]["_entryState"].sum()

        # If there are no origins, return immediately
        if n_oriC == 0:
            return update

        # Get attributes of existing chromosome domains
        domain_index_existing_domain, child_domains = attrs(
            states["chromosome_domains"], ["domain_index", "child_domains"]
        )

        initiate_replication = False
        if self.criticalMassPerOriC >= 1.0:
            # Get number of available replisome subunits
            n_replisome_trimers = counts(states["bulk"], self.replisome_trimers_idx)
            n_replisome_monomers = counts(states["bulk"], self.replisome_monomers_idx)
            # Initiate replication only when
            # 1) The cell has reached the critical mass per oriC
            # 2) If mechanistic replisome option is on, there are enough
            # replisome subunits to assemble two replisomes per existing OriC.
            # Note that we assume asynchronous initiation does not happen.
            initiate_replication = not self.mechanistic_replisome or (
                np.all(n_replisome_trimers == 6 * n_oriC)
                and np.all(n_replisome_monomers == 2 * n_oriC)
            )

        # If all conditions are met, initiate a round of replication on every
        # origin of replication
        if initiate_replication:
            # Get attributes of existing oriCs and domains
            (domain_index_existing_oric,) = attrs(states["oriCs"], ["domain_index"])

            # Get indexes of the domains that would be getting child domains
            # (domains that contain an origin)
            new_parent_domains = np.where(
                np.in1d(domain_index_existing_domain, domain_index_existing_oric)
            )[0]

            # Calculate counts of new replisomes and domains to add
            n_new_replisome = 2 * n_oriC
            n_new_domain = 2 * n_oriC

            # Calculate the domain indexes of new domains and oriC's
            max_domain_index = domain_index_existing_domain.max()
            domain_index_new = np.arange(
                max_domain_index + 1, max_domain_index + 2 * n_oriC + 1, dtype=np.int32
            )

            # Add new oriC's, and reset attributes of existing oriC's
            # All oriC's must be assigned new domain indexes
            update["oriCs"]["set"] = {"domain_index": domain_index_new[:n_oriC]}
            update["oriCs"]["add"] = {
                "domain_index": domain_index_new[n_oriC:],
            }

            # Add and set attributes of newly created replisomes.
            # New replisomes inherit the domain indexes of the oriC's they
            # were initiated from. Two replisomes are formed per oriC, one on
            # the right replichore, and one on the left.
            coordinates_replisome = np.zeros(n_new_replisome, dtype=np.int64)
            right_replichore = np.tile(np.array([True, False], dtype=np.bool_), n_oriC)
            right_replichore = right_replichore.tolist()
            domain_index_new_replisome = np.repeat(domain_index_existing_oric, 2)
            massDiff_protein_new_replisome = np.full(
                n_new_replisome,
                self.replisome_protein_mass if self.mechanistic_replisome else 0.0,
            )
            update["active_replisomes"]["add"] = {
                "coordinates": coordinates_replisome,
                "right_replichore": right_replichore,
                "domain_index": domain_index_new_replisome,
                "massDiff_protein": massDiff_protein_new_replisome,
            }

            # Add and set attributes of new chromosome domains. All new domains
            # should have have no children domains.
            new_child_domains = np.full(
                (n_new_domain, 2), self.no_child_place_holder, dtype=np.int32
            )
            new_domains_update = {
                "add": {
                    "domain_index": domain_index_new,
                    "child_domains": new_child_domains,
                }
            }

            # Add new domains as children of existing domains
            child_domains[new_parent_domains] = domain_index_new.reshape(-1, 2)
            existing_domains_update = {"set": {"child_domains": child_domains}}
            update["chromosome_domains"].update(
                {**new_domains_update, **existing_domains_update}
            )

            # Decrement counts of replisome subunits
            if self.mechanistic_replisome:
                update["bulk"].append((self.replisome_trimers_idx, -6 * n_oriC))
                update["bulk"].append((self.replisome_monomers_idx, -2 * n_oriC))

        # Write data from this module to a listener
        update["listeners"]["replication_data"]["critical_mass_per_oriC"] = (
            self.criticalMassPerOriC.asNumber()
        )
        update["listeners"]["replication_data"]["critical_initiation_mass"] = (
            self.criticalInitiationMass.asNumber(units.fg)
        )

        # Module 2: replication elongation
        # If no active replisomes are present, return immediately
        # Note: the new replication forks added in the previous module are not
        # elongated until the next timestep.
        if n_active_replisomes == 0:
            return update

        # Get allocated counts of dNTPs
        dNtpCounts = counts(states["bulk"], self.dntps_idx)

        # Get attributes of existing replisomes
        (
            domain_index_replisome,
            right_replichore,
            coordinates_replisome,
        ) = attrs(
            states["active_replisomes"],
            ["domain_index", "right_replichore", "coordinates"],
        )

        # Build sequences to polymerize
        sequence_length = np.abs(np.repeat(coordinates_replisome, 2))
        sequence_indexes = np.tile(np.arange(4), n_active_replisomes // 2)

        sequences = buildSequences(
            self.sequences, sequence_indexes, sequence_length, self.elongation_rates
        )

        # Use polymerize algorithm to quickly calculate the number of
        # elongations each fork catalyzes
        reactionLimit = dNtpCounts.sum()

        active_elongation_rates = self.elongation_rates[sequence_indexes]

        result = polymerize(
            sequences,
            dNtpCounts,
            reactionLimit,
            self.random_state,
            active_elongation_rates,
        )

        sequenceElongations = result.sequenceElongation
        dNtpsUsed = result.monomerUsages

        # Compute mass increase for each elongated sequence
        mass_increase_dna = computeMassIncrease(
            sequences,
            sequenceElongations,
            self.polymerized_dntp_weights.asNumber(units.fg),
        )

        # Compute masses that should be added to each replisome
        added_dna_mass = mass_increase_dna[0::2] + mass_increase_dna[1::2]

        # Update positions of each fork
        updated_length = sequence_length + sequenceElongations
        updated_coordinates = updated_length[0::2]

        # Reverse signs of fork coordinates on left replichore
        updated_coordinates[~right_replichore] = -updated_coordinates[~right_replichore]

        # Update attributes and submasses of replisomes
        (current_dna_mass,) = attrs(states["active_replisomes"], ["massDiff_DNA"])
        update["active_replisomes"].update(
            {
                "set": {
                    "coordinates": updated_coordinates,
                    "massDiff_DNA": current_dna_mass + added_dna_mass,
                }
            }
        )

        # Update counts of polymerized metabolites
        update["bulk"].append((self.dntps_idx, -dNtpsUsed))
        update["bulk"].append((self.ppi_idx, dNtpsUsed.sum()))

        # Module 3: replication termination
        # Determine if any forks have reached the end of their sequences. If
        # so, delete the replisomes and domains that were terminated.
        terminal_lengths = self.replichore_lengths[
            np.logical_not(right_replichore).astype(np.int64)
        ]
        terminated_replisomes = np.abs(updated_coordinates) == terminal_lengths

        # If any forks were terminated,
        if terminated_replisomes.sum() > 0:
            # Get domain indexes of terminated forks
            terminated_domains = np.unique(
                domain_index_replisome[terminated_replisomes]
            )

            # Get attributes of existing domains and full chromosomes
            (
                domain_index_domains,
                child_domains,
            ) = attrs(states["chromosome_domains"], ["domain_index", "child_domains"])
            (domain_index_full_chroms,) = attrs(
                states["full_chromosomes"], ["domain_index"]
            )

            # Initialize array of replisomes that should be deleted
            replisomes_to_delete = np.zeros_like(domain_index_replisome, dtype=np.bool_)

            # Count number of new full chromosomes that should be created
            n_new_chromosomes = 0

            # Initialize array for domain indexes of new full chromosomes
            domain_index_new_full_chroms = []

            for terminated_domain_index in terminated_domains:
                # Get all terminated replisomes in the terminated domain
                terminated_domain_matching_replisomes = np.logical_and(
                    domain_index_replisome == terminated_domain_index,
                    terminated_replisomes,
                )

                # If both replisomes in the domain have terminated, we are
                # ready to split the chromosome and update the attributes.
                if terminated_domain_matching_replisomes.sum() == 2:
                    # Tag replisomes and domains with the given domain index
                    # for deletion
                    replisomes_to_delete = np.logical_or(
                        replisomes_to_delete, terminated_domain_matching_replisomes
                    )

                    domain_mask = domain_index_domains == terminated_domain_index

                    # Get child domains of deleted domain
                    child_domains_this_domain = child_domains[
                        np.where(domain_mask)[0][0], :
                    ]

                    # Modify domain index of one existing full chromosome to
                    # index of first child domain
                    domain_index_full_chroms = domain_index_full_chroms.copy()
                    domain_index_full_chroms[
                        np.where(domain_index_full_chroms == terminated_domain_index)[0]
                    ] = child_domains_this_domain[0]

                    # Increment count of new full chromosome
                    n_new_chromosomes += 1

                    # Append chromosome index of new full chromosome
                    domain_index_new_full_chroms.append(child_domains_this_domain[1])

            # Delete terminated replisomes
            update["active_replisomes"]["delete"] = np.where(replisomes_to_delete)[0]

            # Generate new full chromosome molecules
            if n_new_chromosomes > 0:
                chromosome_add_update = {
                    "add": {
                        "domain_index": domain_index_new_full_chroms,
                        "division_time": states["global_time"] + self.D_period,
                        "has_triggered_division": False,
                    }
                }

                # Reset domain index of existing chromosomes that have finished
                # replication
                chromosome_existing_update = {
                    "set": {"domain_index": domain_index_full_chroms}
                }

                update["full_chromosomes"].update(
                    {**chromosome_add_update, **chromosome_existing_update}
                )

            # Increment counts of replisome subunits
            if self.mechanistic_replisome:
                update["bulk"].append(
                    (self.replisome_trimers_idx, 3 * replisomes_to_delete.sum())
                )
                update["bulk"].append(
                    (self.replisome_monomers_idx, replisomes_to_delete.sum())
                )

        return update


def test_chromosome_replication():
    test_config = {}
    process = ChromosomeReplication(test_config)
    assert process is not None


if __name__ == "__main__":
    test_chromosome_replication()

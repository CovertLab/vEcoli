"""
====================
Chromosome Structure
====================

- Resolve collisions between molecules and replication forks on the chromosome.
- Remove and replicate promoters and motifs that are traversed by replisomes.
- Reset the boundaries and linking numbers of chromosomal segments.
"""

import numpy as np
import numpy.typing as npt
import warnings
from vivarium.core.process import Step
from vivarium.core.composer import Composer
from vivarium.core.engine import Engine

from ecoli.processes.global_clock import GlobalClock
from ecoli.processes.unique_update import UniqueUpdate
from ecoli.processes.registries import topology_registry
from ecoli.library.schema import (
    listener_schema,
    numpy_schema,
    attrs,
    bulk_name_to_idx,
    get_free_indices,
)
from ecoli.library.json_state import get_state_from_file
from wholecell.utils.polymerize import buildSequences

# Register default topology for this process, associating it with process name
NAME = "ecoli-chromosome-structure"
TOPOLOGY = {
    "bulk": ("bulk",),
    "listeners": ("listeners",),
    "active_replisomes": (
        "unique",
        "active_replisome",
    ),
    "oriCs": (
        "unique",
        "oriC",
    ),
    "chromosome_domains": (
        "unique",
        "chromosome_domain",
    ),
    "active_RNAPs": ("unique", "active_RNAP"),
    "RNAs": ("unique", "RNA"),
    "active_ribosome": ("unique", "active_ribosome"),
    "full_chromosomes": (
        "unique",
        "full_chromosome",
    ),
    "promoters": ("unique", "promoter"),
    "DnaA_boxes": ("unique", "DnaA_box"),
    "genes": ("unique", "gene"),
    "chromosomal_segments": ("unique", "chromosomal_segment"),
    "global_time": ("global_time",),
    "timestep": ("timestep",),
    "next_update_time": ("next_update_time", "chromosome_structure"),
}
topology_registry.register(NAME, TOPOLOGY)


class ChromosomeStructure(Step):
    """Chromosome Structure Process"""

    name = NAME
    topology = TOPOLOGY
    defaults = {
        # Load parameters
        "rna_sequences": [],
        "protein_sequences": [],
        "n_TUs": 1,
        "n_TFs": 1,
        "n_amino_acids": 1,
        "n_fragment_bases": 1,
        "replichore_lengths": [0, 0],
        "relaxed_DNA_base_pairs_per_turn": 1,
        "terC_index": -1,
        "calculate_superhelical_densities": False,
        # Get placeholder value for chromosome domains without children
        "no_child_place_holder": -1,
        # Load bulk molecule views
        "inactive_RNAPs": [],
        "fragmentBases": [],
        "ppi": "ppi",
        "active_tfs": [],
        "ribosome_30S_subunit": "30S",
        "ribosome_50S_subunit": "50S",
        "amino_acids": [],
        "water": "water",
        "seed": 0,
        "emit_unique": False,
        "rna_ids": [],
        "n_mature_rnas": 0,
        "mature_rna_ids": [],
        "mature_rna_end_positions": [],
        "mature_rna_nt_counts": [],
        "unprocessed_rna_index_mapping": {},
        "time_step": 1.0,
    }

    # Constructor
    def __init__(self, parameters=None):
        super().__init__(parameters)
        self.rna_sequences = self.parameters["rna_sequences"]
        self.protein_sequences = self.parameters["protein_sequences"]
        self.n_TUs = self.parameters["n_TUs"]
        self.n_TFs = self.parameters["n_TFs"]
        self.rna_ids = self.parameters["rna_ids"]
        self.n_amino_acids = self.parameters["n_amino_acids"]
        self.n_fragment_bases = self.parameters["n_fragment_bases"]
        replichore_lengths = self.parameters["replichore_lengths"]
        self.min_coordinates = -replichore_lengths[1]
        self.max_coordinates = replichore_lengths[0]
        self.relaxed_DNA_base_pairs_per_turn = self.parameters[
            "relaxed_DNA_base_pairs_per_turn"
        ]
        self.terC_index = self.parameters["terC_index"]

        self.n_mature_rnas = self.parameters["n_mature_rnas"]
        self.mature_rna_ids = self.parameters["mature_rna_ids"]
        self.mature_rna_end_positions = self.parameters["mature_rna_end_positions"]
        self.mature_rna_nt_counts = self.parameters["mature_rna_nt_counts"]
        self.unprocessed_rna_index_mapping = self.parameters[
            "unprocessed_rna_index_mapping"
        ]

        # Load sim options
        self.calculate_superhelical_densities = self.parameters[
            "calculate_superhelical_densities"
        ]

        # Get placeholder value for chromosome domains without children
        self.no_child_place_holder = self.parameters["no_child_place_holder"]

        self.inactive_RNAPs = self.parameters["inactive_RNAPs"]
        self.fragmentBases = self.parameters["fragmentBases"]
        self.ppi = self.parameters["ppi"]
        self.active_tfs = self.parameters["active_tfs"]
        self.ribosome_30S_subunit = self.parameters["ribosome_30S_subunit"]
        self.ribosome_50S_subunit = self.parameters["ribosome_50S_subunit"]
        self.amino_acids = self.parameters["amino_acids"]
        self.water = self.parameters["water"]

        self.inactive_RNAPs_idx = None

        self.emit_unique = self.parameters.get("emit_unique", True)

    def ports_schema(self):
        ports = {
            "listeners": {
                "rnap_data": listener_schema(
                    {
                        "n_total_collisions": 0,
                        "n_headon_collisions": 0,
                        "n_codirectional_collisions": 0,
                        "headon_collision_coordinates": [],
                        "codirectional_collision_coordinates": [],
                        "n_removed_ribosomes": 0,
                        "incomplete_transcription_events": (
                            np.zeros(self.n_TUs, np.int64),
                            self.rna_ids,
                        ),
                        "n_empty_fork_collisions": 0,
                        "empty_fork_collision_coordinates": [],
                    }
                )
            },
            "bulk": numpy_schema("bulk"),
            # Unique molecules
            "active_replisomes": numpy_schema(
                "active_replisomes", emit=self.parameters["emit_unique"]
            ),
            "oriCs": numpy_schema("oriCs", emit=self.parameters["emit_unique"]),
            "chromosome_domains": numpy_schema(
                "chromosome_domains", emit=self.parameters["emit_unique"]
            ),
            "active_RNAPs": numpy_schema(
                "active_RNAPs", emit=self.parameters["emit_unique"]
            ),
            "RNAs": numpy_schema("RNAs", emit=self.parameters["emit_unique"]),
            "active_ribosome": numpy_schema(
                "active_ribosome", emit=self.parameters["emit_unique"]
            ),
            "full_chromosomes": numpy_schema(
                "full_chromosomes", emit=self.parameters["emit_unique"]
            ),
            "promoters": numpy_schema("promoters", emit=self.parameters["emit_unique"]),
            "DnaA_boxes": numpy_schema(
                "DnaA_boxes", emit=self.parameters["emit_unique"]
            ),
            "chromosomal_segments": numpy_schema(
                "chromosomal_segments", emit=self.parameters["emit_unique"]
            ),
            "genes": numpy_schema("genes", emit=self.parameters["emit_unique"]),
            "global_time": {"_default": 0.0},
            "timestep": {"_default": self.parameters["time_step"]},
            "next_update_time": {
                "_default": self.parameters["time_step"],
                "_updater": "set",
                "_divider": "set",
            },
        }

        return ports

    def update_condition(self, timestep, states):
        """
        See :py:meth:`~ecoli.processes.partition.Requester.update_condition`.
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
        if self.inactive_RNAPs_idx is None:
            self.fragmentBasesIdx = bulk_name_to_idx(
                self.fragmentBases, states["bulk"]["id"]
            )
            self.active_tfs_idx = bulk_name_to_idx(
                self.active_tfs, states["bulk"]["id"]
            )
            self.ribosome_30S_subunit_idx = bulk_name_to_idx(
                self.ribosome_30S_subunit, states["bulk"]["id"]
            )
            self.ribosome_50S_subunit_idx = bulk_name_to_idx(
                self.ribosome_50S_subunit, states["bulk"]["id"]
            )
            self.amino_acids_idx = bulk_name_to_idx(
                self.amino_acids, states["bulk"]["id"]
            )
            self.water_idx = bulk_name_to_idx(self.water, states["bulk"]["id"])
            self.ppi_idx = bulk_name_to_idx(self.ppi, states["bulk"]["id"])
            self.inactive_RNAPs_idx = bulk_name_to_idx(
                self.inactive_RNAPs, states["bulk"]["id"]
            )
            self.mature_rna_idx = bulk_name_to_idx(
                self.mature_rna_ids, states["bulk"]["id"]
            )

        # Read unique molecule attributes
        (replisome_domain_indexes, replisome_coordinates, replisome_unique_indexes) = (
            attrs(
                states["active_replisomes"],
                ["domain_index", "coordinates", "unique_index"],
            )
        )
        (all_chromosome_domain_indexes, child_domains) = attrs(
            states["chromosome_domains"], ["domain_index", "child_domains"]
        )
        (
            RNAP_domain_indexes,
            RNAP_coordinates,
            RNAP_is_forward,
            RNAP_unique_indexes,
        ) = attrs(
            states["active_RNAPs"],
            ["domain_index", "coordinates", "is_forward", "unique_index"],
        )
        (origin_domain_indexes,) = attrs(states["oriCs"], ["domain_index"])
        (mother_domain_indexes,) = attrs(states["full_chromosomes"], ["domain_index"])
        (
            RNA_TU_indexes,
            transcript_lengths,
            RNA_RNAP_indexes,
            RNA_full_transcript,
            RNA_unique_indexes,
        ) = attrs(
            states["RNAs"],
            [
                "TU_index",
                "transcript_length",
                "RNAP_index",
                "is_full_transcript",
                "unique_index",
            ],
        )
        (ribosome_protein_indexes, ribosome_peptide_lengths, ribosome_mRNA_indexes) = (
            attrs(
                states["active_ribosome"],
                ["protein_index", "peptide_length", "mRNA_index"],
            )
        )
        (
            promoter_TU_indexes,
            promoter_domain_indexes,
            promoter_coordinates,
            promoter_bound_TFs,
        ) = attrs(
            states["promoters"], ["TU_index", "domain_index", "coordinates", "bound_TF"]
        )
        (gene_cistron_indexes, gene_domain_indexes, gene_coordinates) = attrs(
            states["genes"], ["cistron_index", "domain_index", "coordinates"]
        )
        (DnaA_box_domain_indexes, DnaA_box_coordinates, DnaA_box_bound) = attrs(
            states["DnaA_boxes"], ["domain_index", "coordinates", "DnaA_bound"]
        )

        # Build dictionary of replisome coordinates with domain indexes as keys
        replisome_coordinates_from_domains = {
            domain_index: replisome_coordinates[
                replisome_domain_indexes == domain_index
            ]
            for domain_index in np.unique(replisome_domain_indexes)
        }

        def get_removed_molecules_mask(domain_indexes, coordinates):
            """
            Computes the boolean mask of unique molecules that should be
            removed based on the progression of the replication forks.
            """
            mask = np.zeros_like(domain_indexes, dtype=np.bool_)

            # Loop through all domains
            for domain_index in np.unique(domain_indexes):
                # Domain has active replisomes
                if domain_index in replisome_coordinates_from_domains:
                    domain_replisome_coordinates = replisome_coordinates_from_domains[
                        domain_index
                    ]

                    # Get mask for molecules on this domain that are out of range
                    # It's rare but we have to remove molecules at the exact same
                    # coordinates as the replisomes as well so that they do not break
                    # the chromosome segment calculations if they are removed by a
                    # different process (hence, >= and <= instead of > and <)
                    domain_mask = np.logical_and.reduce(
                        (
                            domain_indexes == domain_index,
                            coordinates >= domain_replisome_coordinates.min(),
                            coordinates <= domain_replisome_coordinates.max(),
                        )
                    )

                # Domain has no active replisomes
                else:
                    children_of_domain = child_domains[
                        all_chromosome_domain_indexes == domain_index
                    ]
                    # Child domains are full chromosomes (domain has finished replicating)
                    if np.all(np.isin(children_of_domain, mother_domain_indexes)):
                        # Remove all molecules on this domain
                        domain_mask = domain_indexes == domain_index
                    # Domain has not started replication or replication was interrupted
                    else:
                        continue

                mask[domain_mask] = True

            return mask

        # Build mask for molecules that should be removed
        removed_RNAPs_mask = get_removed_molecules_mask(
            RNAP_domain_indexes, RNAP_coordinates
        )
        removed_promoters_mask = get_removed_molecules_mask(
            promoter_domain_indexes, promoter_coordinates
        )
        removed_genes_mask = get_removed_molecules_mask(
            gene_domain_indexes, gene_coordinates
        )
        removed_DnaA_boxes_mask = get_removed_molecules_mask(
            DnaA_box_domain_indexes, DnaA_box_coordinates
        )

        # Build masks for head-on and co-directional collisions between RNAPs
        # and replication forks
        RNAP_headon_collision_mask = np.logical_and(
            removed_RNAPs_mask, np.logical_xor(RNAP_is_forward, RNAP_coordinates > 0)
        )
        RNAP_codirectional_collision_mask = np.logical_and(
            removed_RNAPs_mask, np.logical_not(RNAP_headon_collision_mask)
        )

        n_total_collisions = np.count_nonzero(removed_RNAPs_mask)
        n_headon_collisions = np.count_nonzero(RNAP_headon_collision_mask)
        n_codirectional_collisions = np.count_nonzero(RNAP_codirectional_collision_mask)

        # Write values to listeners
        update = {
            "listeners": {
                "rnap_data": {
                    "n_total_collisions": n_total_collisions,
                    "n_headon_collisions": n_headon_collisions,
                    "n_codirectional_collisions": n_codirectional_collisions,
                    "headon_collision_coordinates": RNAP_coordinates[
                        RNAP_headon_collision_mask
                    ],
                    "codirectional_collision_coordinates": RNAP_coordinates[
                        RNAP_codirectional_collision_mask
                    ],
                }
            },
            "bulk": [],
            "active_replisomes": {},
            "oriCs": {},
            "chromosome_domains": {},
            "active_RNAPs": {},
            "RNAs": {},
            "active_ribosome": {},
            "full_chromosomes": {},
            "chromosomal_segments": {},
            "promoters": {},
            "genes": {},
            "DnaA_boxes": {},
        }

        if self.calculate_superhelical_densities:
            # Get attributes of existing segments
            (
                boundary_molecule_indexes,
                boundary_coordinates,
                segment_domain_indexes,
                linking_numbers,
            ) = attrs(
                states["chromosomal_segments"],
                [
                    "boundary_molecule_indexes",
                    "boundary_coordinates",
                    "domain_index",
                    "linking_number",
                ],
            )

            # Initialize new attributes of chromosomal segments
            all_new_boundary_molecule_indexes = np.empty((0, 2), dtype=np.int64)
            all_new_boundary_coordinates = np.empty((0, 2), dtype=np.int64)
            all_new_segment_domain_indexes = np.array([], dtype=np.int32)
            all_new_linking_numbers = np.array([], dtype=np.float64)

            # Iteratively tally RNAPs that were removed due to collisions with
            # replication forks with or without replisomes on each domain
            removed_RNAP_masks_all_domains = np.full_like(removed_RNAPs_mask, False)
            for domain_index in np.unique(all_chromosome_domain_indexes):
                # Skip domains that have completed replication
                if np.all(domain_index < mother_domain_indexes):
                    continue

                domain_spans_oriC = domain_index in origin_domain_indexes
                domain_spans_terC = domain_index in mother_domain_indexes

                # Parse attributes of remaining RNAPs in this domain
                RNAPs_domain_mask = RNAP_domain_indexes == domain_index
                RNAP_coordinates_this_domain = RNAP_coordinates[RNAPs_domain_mask]
                RNAP_unique_indexes_this_domain = RNAP_unique_indexes[RNAPs_domain_mask]
                domain_remaining_RNAPs_mask = ~removed_RNAPs_mask[RNAPs_domain_mask]

                # Parse attributes of segments in this domain
                segments_domain_mask = segment_domain_indexes == domain_index
                boundary_molecule_indexes_this_domain = boundary_molecule_indexes[
                    segments_domain_mask, :
                ]
                boundary_coordinates_this_domain = boundary_coordinates[
                    segments_domain_mask, :
                ]
                linking_numbers_this_domain = linking_numbers[segments_domain_mask]

                new_molecule_coordinates_this_domain = np.array([], dtype=np.int64)
                new_molecule_indexes_this_domain = np.array([], dtype=np.int64)
                # Append coordinates and indexes of replisomes on this domain,
                # if any
                if not domain_spans_oriC:
                    replisome_domain_mask = replisome_domain_indexes == domain_index
                    replisome_coordinates_this_domain = replisome_coordinates[
                        replisome_domain_mask
                    ]
                    replisome_molecule_indexes_this_domain = replisome_unique_indexes[
                        replisome_domain_mask
                    ]

                    # If one or more replisomes was removed in the last time step,
                    # use the last known location and molecule index.
                    if len(replisome_molecule_indexes_this_domain) != 2:
                        assert len(replisome_molecule_indexes_this_domain) < 2
                        (
                            replisome_coordinates_this_domain,
                            replisome_molecule_indexes_this_domain,
                        ) = get_last_known_replisome_data(
                            boundary_coordinates_this_domain,
                            boundary_molecule_indexes_this_domain,
                            replisome_coordinates_this_domain,
                            replisome_molecule_indexes_this_domain,
                        )
                        # Assume that RNAPs that run into a replication fork
                        # are removed even if there is no replisome
                        RNAPs_on_forks = np.isin(
                            RNAP_coordinates_this_domain,
                            replisome_coordinates_this_domain,
                        )
                        domain_remaining_RNAPs_mask = np.logical_and(
                            domain_remaining_RNAPs_mask, ~RNAPs_on_forks
                        )
                        full_removed_RNAPs_mask = np.full_like(
                            removed_RNAPs_mask, False
                        )
                        full_removed_RNAPs_mask[RNAPs_domain_mask] = RNAPs_on_forks
                        removed_RNAP_masks_all_domains = np.logical_or(
                            removed_RNAP_masks_all_domains, full_removed_RNAPs_mask
                        )

                    new_molecule_coordinates_this_domain = np.concatenate(
                        (
                            new_molecule_coordinates_this_domain,
                            replisome_coordinates_this_domain,
                        )
                    )
                    new_molecule_indexes_this_domain = np.concatenate(
                        (
                            new_molecule_indexes_this_domain,
                            replisome_molecule_indexes_this_domain,
                        )
                    )

                # Append coordinates and indexes of parent domain replisomes,
                # if any
                if not domain_spans_terC:
                    parent_domain_index = all_chromosome_domain_indexes[
                        np.where(child_domains == domain_index)[0][0]
                    ]
                    replisome_parent_domain_mask = (
                        replisome_domain_indexes == parent_domain_index
                    )
                    replisome_coordinates_parent_domain = replisome_coordinates[
                        replisome_parent_domain_mask
                    ]
                    replisome_molecule_indexes_parent_domain = replisome_unique_indexes[
                        replisome_parent_domain_mask
                    ]

                    # If one or more replisomes was removed in the last time step,
                    # use the last known location and molecule index.
                    if len(replisome_molecule_indexes_parent_domain) != 2:
                        assert len(replisome_molecule_indexes_parent_domain) < 2
                        # Parse attributes of segments in parent domain
                        parent_segments_domain_mask = (
                            segment_domain_indexes == parent_domain_index
                        )
                        boundary_molecule_indexes_parent_domain = (
                            boundary_molecule_indexes[parent_segments_domain_mask, :]
                        )
                        boundary_coordinates_parent_domain = boundary_coordinates[
                            parent_segments_domain_mask, :
                        ]
                        (
                            replisome_coordinates_parent_domain,
                            replisome_molecule_indexes_parent_domain,
                        ) = get_last_known_replisome_data(
                            boundary_coordinates_parent_domain,
                            boundary_molecule_indexes_parent_domain,
                            replisome_coordinates_parent_domain,
                            replisome_molecule_indexes_parent_domain,
                        )
                        # Assume that RNAPs that run into a replication fork
                        # are removed even if there is no replisome
                        RNAPs_on_forks = np.isin(
                            RNAP_coordinates_this_domain,
                            replisome_coordinates_parent_domain,
                        )
                        domain_remaining_RNAPs_mask = np.logical_and(
                            domain_remaining_RNAPs_mask, ~RNAPs_on_forks
                        )
                        full_removed_RNAPs_mask = np.full_like(
                            removed_RNAPs_mask, False
                        )
                        full_removed_RNAPs_mask[RNAPs_domain_mask] = RNAPs_on_forks
                        removed_RNAP_masks_all_domains = np.logical_or(
                            removed_RNAP_masks_all_domains, full_removed_RNAPs_mask
                        )

                    new_molecule_coordinates_this_domain = np.concatenate(
                        (
                            new_molecule_coordinates_this_domain,
                            replisome_coordinates_parent_domain,
                        )
                    )
                    new_molecule_indexes_this_domain = np.concatenate(
                        (
                            new_molecule_indexes_this_domain,
                            replisome_molecule_indexes_parent_domain,
                        )
                    )

                # Add remaining RNAPs in this domain after accounting for removals
                # due to collisions with replication forks
                new_molecule_coordinates_this_domain = np.concatenate(
                    (
                        new_molecule_coordinates_this_domain,
                        RNAP_coordinates_this_domain[domain_remaining_RNAPs_mask],
                    )
                )
                new_molecule_indexes_this_domain = np.concatenate(
                    (
                        new_molecule_indexes_this_domain,
                        RNAP_unique_indexes_this_domain[domain_remaining_RNAPs_mask],
                    )
                )

                # If there are no molecules left on this domain, continue
                if len(new_molecule_indexes_this_domain) == 0:
                    continue

                # Calculate attributes of new segments
                new_segment_attrs = self._compute_new_segment_attributes(
                    boundary_molecule_indexes_this_domain,
                    boundary_coordinates_this_domain,
                    linking_numbers_this_domain,
                    new_molecule_indexes_this_domain,
                    new_molecule_coordinates_this_domain,
                    domain_spans_oriC,
                    domain_spans_terC,
                )

                # Append to existing array of new segment attributes
                all_new_boundary_molecule_indexes = np.vstack(
                    (
                        all_new_boundary_molecule_indexes,
                        new_segment_attrs["boundary_molecule_indexes"],
                    )
                )
                all_new_boundary_coordinates = np.vstack(
                    (
                        all_new_boundary_coordinates,
                        new_segment_attrs["boundary_coordinates"],
                    )
                )
                all_new_segment_domain_indexes = np.concatenate(
                    (
                        all_new_segment_domain_indexes,
                        np.full(
                            len(new_segment_attrs["linking_numbers"]),
                            domain_index,
                            dtype=np.int32,
                        ),
                    )
                )
                all_new_linking_numbers = np.concatenate(
                    (all_new_linking_numbers, new_segment_attrs["linking_numbers"])
                )

            # Delete all existing chromosomal segments
            if len(boundary_molecule_indexes) > 0:
                update["chromosomal_segments"].update(
                    {"delete": np.arange(len(boundary_molecule_indexes))}
                )

            # Add new chromosomal segments
            update["chromosomal_segments"].update(
                {
                    "add": {
                        "boundary_molecule_indexes": all_new_boundary_molecule_indexes,
                        "boundary_coordinates": all_new_boundary_coordinates,
                        "domain_index": all_new_segment_domain_indexes,
                        "linking_number": all_new_linking_numbers,
                    }
                }
            )

            # Figure out if any additional RNAPs were removed due to collisions with
            # replication forks where there were no replisomes
            empty_fork_RNAP_collision_mask = np.logical_and(
                removed_RNAP_masks_all_domains,
                np.logical_not(
                    np.logical_or(
                        RNAP_headon_collision_mask, RNAP_codirectional_collision_mask
                    )
                ),
            )
            update["listeners"]["rnap_data"].update(
                {
                    "n_empty_fork_collisions": empty_fork_RNAP_collision_mask.sum(),
                    "empty_fork_collision_coordinates": RNAP_coordinates[
                        empty_fork_RNAP_collision_mask
                    ],
                }
            )

        # Get mask for RNAs that are transcribed from removed RNAPs
        removed_RNAs_mask = np.isin(
            RNA_RNAP_indexes, RNAP_unique_indexes[removed_RNAPs_mask]
        )

        # Initialize counts of incomplete transcription events
        incomplete_transcription_event = np.zeros(self.n_TUs)

        # Remove RNAPs and RNAs that have collided with replisomes
        if n_total_collisions > 0:
            if removed_RNAPs_mask.sum() > 0:
                update["active_RNAPs"].update(
                    {"delete": np.where(removed_RNAPs_mask)[0]}
                )
            if removed_RNAs_mask.sum() > 0:
                update["RNAs"].update({"delete": np.where(removed_RNAs_mask)[0]})

            # Increment counts of inactive RNAPs
            update["bulk"].append((self.inactive_RNAPs_idx, n_total_collisions))

            # Get sequences of incomplete transcripts
            incomplete_sequence_lengths = transcript_lengths[removed_RNAs_mask]
            # Under resource-limited conditions, some transcripts may be
            # initiated but not elongated (zero length). Include them in the count.
            n_initiated_sequences = (~RNA_full_transcript[removed_RNAs_mask]).sum()
            n_ppi_added = n_initiated_sequences

            if n_initiated_sequences > 0:
                incomplete_rna_indexes = RNA_TU_indexes[removed_RNAs_mask]
                incomplete_transcription_event = np.bincount(
                    incomplete_rna_indexes, minlength=self.n_TUs
                )

                incomplete_sequences = buildSequences(
                    self.rna_sequences,
                    incomplete_rna_indexes,
                    np.zeros(n_total_collisions, dtype=np.int64),
                    np.full(n_total_collisions, incomplete_sequence_lengths.max()),
                )

                mature_rna_counts = np.zeros(self.n_mature_rnas, dtype=np.int64)
                base_counts = np.zeros(self.n_fragment_bases, dtype=np.int64)

                for ri, sl, seq in zip(
                    incomplete_rna_indexes,
                    incomplete_sequence_lengths,
                    incomplete_sequences,
                ):
                    # Check if incomplete RNA is an unprocessed RNA
                    if ri in self.unprocessed_rna_index_mapping:
                        # Find mature RNA molecules that would need to be added
                        # given the length of the incomplete RNA
                        mature_rna_end_pos = self.mature_rna_end_positions[
                            :, self.unprocessed_rna_index_mapping[ri]
                        ]
                        mature_rnas_produced = np.logical_and(
                            mature_rna_end_pos != 0, mature_rna_end_pos < sl
                        )

                        # Increment counts of mature RNAs
                        mature_rna_counts += mature_rnas_produced

                        # Increment counts of fragment NTPs, but exclude bases
                        # that are part of the mature RNAs generated
                        base_counts += np.bincount(
                            seq[:sl], minlength=self.n_fragment_bases
                        ) - self.mature_rna_nt_counts[mature_rnas_produced, :].sum(
                            axis=0
                        )

                        # Exclude ppi molecules that are part of mature RNAs
                        n_ppi_added -= mature_rnas_produced.sum()
                    else:
                        base_counts += np.bincount(
                            seq[:sl], minlength=self.n_fragment_bases
                        )

                # Increment counts of mature RNAs, fragment NTPs and phosphates
                update["bulk"].append((self.mature_rna_idx, mature_rna_counts))
                update["bulk"].append((self.fragmentBasesIdx, base_counts))
                update["bulk"].append((self.ppi_idx, n_ppi_added))

            assert n_initiated_sequences == incomplete_transcription_event.sum()

        update["listeners"]["rnap_data"]["incomplete_transcription_event"] = (
            incomplete_transcription_event
        )

        # Get mask for ribosomes that are bound to nonexisting mRNAs
        remaining_RNA_unique_indexes = RNA_unique_indexes[
            np.logical_not(removed_RNAs_mask)
        ]
        removed_ribosomes_mask = np.logical_not(
            np.isin(ribosome_mRNA_indexes, remaining_RNA_unique_indexes)
        )
        n_removed_ribosomes = np.count_nonzero(removed_ribosomes_mask)

        # Remove ribosomes that are bound to missing RNA molecules. This
        # includes both RNAs removed by this function and RNAs removed
        # by other processes (e.g. RNA degradation).
        if n_removed_ribosomes > 0:
            update["active_ribosome"].update(
                {"delete": np.where(removed_ribosomes_mask)[0]}
            )

            # Increment counts of inactive ribosomal subunits
            update["bulk"].extend(
                [
                    (self.ribosome_30S_subunit_idx, n_removed_ribosomes),
                    (self.ribosome_50S_subunit_idx, n_removed_ribosomes),
                ]
            )

            # Get amino acid sequences of incomplete polypeptides
            incomplete_sequence_lengths = ribosome_peptide_lengths[
                removed_ribosomes_mask
            ]
            n_initiated_sequences = np.count_nonzero(incomplete_sequence_lengths)

            if n_initiated_sequences > 0:
                incomplete_sequences = buildSequences(
                    self.protein_sequences,
                    ribosome_protein_indexes[removed_ribosomes_mask],
                    np.zeros(n_removed_ribosomes, dtype=np.int64),
                    np.full(n_removed_ribosomes, incomplete_sequence_lengths.max()),
                )

                amino_acid_counts = np.zeros(self.n_amino_acids, dtype=np.int64)

                for sl, seq in zip(incomplete_sequence_lengths, incomplete_sequences):
                    amino_acid_counts += np.bincount(
                        seq[:sl], minlength=self.n_amino_acids
                    )

                # Increment counts of free amino acids and decrease counts of
                # free water molecules
                update["bulk"].append((self.amino_acids_idx, amino_acid_counts))
                update["bulk"].append(
                    (
                        self.water_idx,
                        (n_initiated_sequences - incomplete_sequence_lengths.sum()),
                    )
                )

        # Write to listener
        update["listeners"]["rnap_data"]["n_removed_ribosomes"] = n_removed_ribosomes

        def get_replicated_motif_attributes(old_coordinates, old_domain_indexes):
            """
            Computes the attributes of replicated motifs on the chromosome,
            given the old coordinates and domain indexes of the original motifs.
            """
            # Coordinates are simply repeated
            new_coordinates = np.repeat(old_coordinates, 2)

            # Domain indexes are set to the child indexes of the original index
            new_domain_indexes = child_domains[
                np.array(
                    [
                        np.where(all_chromosome_domain_indexes == idx)[0][0]
                        for idx in old_domain_indexes
                    ]
                ),
                :,
            ].flatten()

            return new_coordinates, new_domain_indexes

        #######################
        # Replicate promoters #
        #######################
        n_new_promoters = 2 * np.count_nonzero(removed_promoters_mask)

        if n_new_promoters > 0:
            # Delete original promoters
            update["promoters"].update({"delete": np.where(removed_promoters_mask)[0]})

            # Add freed active tfs
            update["bulk"].append(
                (
                    self.active_tfs_idx,
                    promoter_bound_TFs[removed_promoters_mask, :].sum(axis=0),
                )
            )

            # Set up attributes for the replicated promoters
            promoter_TU_indexes_new = np.repeat(
                promoter_TU_indexes[removed_promoters_mask], 2
            )
            (promoter_coordinates_new, promoter_domain_indexes_new) = (
                get_replicated_motif_attributes(
                    promoter_coordinates[removed_promoters_mask],
                    promoter_domain_indexes[removed_promoters_mask],
                )
            )

            # Add new promoters with new domain indexes
            update["promoters"].update(
                {
                    "add": {
                        "TU_index": promoter_TU_indexes_new,
                        "coordinates": promoter_coordinates_new,
                        "domain_index": promoter_domain_indexes_new,
                        "bound_TF": np.zeros(
                            (n_new_promoters, self.n_TFs), dtype=np.bool_
                        ),
                    }
                }
            )

        # Replicate genes
        n_new_genes = 2 * np.count_nonzero(removed_genes_mask)

        if n_new_genes > 0:
            # Delete original genes
            update["genes"].update({"delete": np.where(removed_genes_mask)[0]})

            # Set up attributes for the replicated genes
            gene_cistron_indexes_new = np.repeat(
                gene_cistron_indexes[removed_genes_mask], 2
            )
            gene_coordinates_new, gene_domain_indexes_new = (
                get_replicated_motif_attributes(
                    gene_coordinates[removed_genes_mask],
                    gene_domain_indexes[removed_genes_mask],
                )
            )

            # Add new genes with new domain indexes
            update["genes"].update(
                {
                    "add": {
                        "cistron_index": gene_cistron_indexes_new,
                        "coordinates": gene_coordinates_new,
                        "domain_index": gene_domain_indexes_new,
                    }
                }
            )

        ########################
        # Replicate DnaA boxes #
        ########################
        n_new_DnaA_boxes = 2 * np.count_nonzero(removed_DnaA_boxes_mask)

        if n_new_DnaA_boxes > 0:
            # Delete original DnaA boxes
            if removed_DnaA_boxes_mask.sum() > 0:
                update["DnaA_boxes"].update(
                    {"delete": np.where(removed_DnaA_boxes_mask)[0]}
                )

            # Set up attributes for the replicated boxes
            (DnaA_box_coordinates_new, DnaA_box_domain_indexes_new) = (
                get_replicated_motif_attributes(
                    DnaA_box_coordinates[removed_DnaA_boxes_mask],
                    DnaA_box_domain_indexes[removed_DnaA_boxes_mask],
                )
            )

            # Add new DnaA boxes with new domain indexes
            dict_dna = {
                "add": {
                    "coordinates": DnaA_box_coordinates_new,
                    "domain_index": DnaA_box_domain_indexes_new,
                    "DnaA_bound": np.zeros(n_new_DnaA_boxes, dtype=np.bool_),
                }
            }
            update["DnaA_boxes"].update(dict_dna)

        update["next_update_time"] = states["global_time"] + states["timestep"]
        return update

    def _compute_new_segment_attributes(
        self,
        old_boundary_molecule_indexes: npt.NDArray[np.int64],
        old_boundary_coordinates: npt.NDArray[np.int64],
        old_linking_numbers: npt.NDArray[np.int64],
        new_molecule_indexes: npt.NDArray[np.int64],
        new_molecule_coordinates: npt.NDArray[np.int64],
        spans_oriC: bool,
        spans_terC: bool,
    ) -> dict[str, npt.NDArray[np.int64 | np.float64]]:
        """
        Calculates the updated attributes of chromosomal segments belonging to
        a specific chromosomal domain, given the previous and current
        coordinates of molecules bound to the chromosome.

        Args:
            old_boundary_molecule_indexes: (N, 2) array of unique
                indexes of molecules that formed the boundaries of each
                chromosomal segment in the previous timestep.
            old_boundary_coordinates: (N, 2) array of chromosomal
                coordinates of molecules that formed the boundaries of each
                chromosomal segment in the previous timestep.
            old_linking_numbers: (N,) array of linking numbers of each
                chromosomal segment in the previous timestep.
            new_molecule_indexes: (N,) array of unique indexes of all
                molecules bound to the domain at the current timestep.
            new_molecule_coordinates: (N,) array of chromosomal
                coordinates of all molecules bound to the domain at the current
                timestep.
            spans_oriC: True if the domain spans the origin.
            spans_terC: True if the domain spans the terminus.

        Returns:
            Dictionary of the following format::

                {
                    'boundary_molecule_indexes': (M, 2) array of unique
                        indexes of molecules that form the boundaries of new
                        chromosomal segments,
                    'boundary_coordinates': (M, 2) array of chromosomal
                        coordinates of molecules that form the boundaries of
                        new chromosomal segments,
                    'linking_numbers': (M,) array of linking numbers of new
                        chromosomal segments
                }

        """
        # Sort old segment arrays by coordinates of left boundary
        old_coordinates_argsort = np.argsort(old_boundary_coordinates[:, 0])
        old_boundary_coordinates_sorted = old_boundary_coordinates[
            old_coordinates_argsort, :
        ]
        old_boundary_molecule_indexes_sorted = old_boundary_molecule_indexes[
            old_coordinates_argsort, :
        ]
        old_linking_numbers_sorted = old_linking_numbers[old_coordinates_argsort]

        # Sort new segment arrays by molecular coordinates
        new_coordinates_argsort = np.argsort(new_molecule_coordinates)
        new_molecule_coordinates_sorted = new_molecule_coordinates[
            new_coordinates_argsort
        ]
        new_molecule_indexes_sorted = new_molecule_indexes[new_coordinates_argsort]

        # Domain does not span the origin
        if not spans_oriC:
            # A fragment spans oriC if two boundaries have opposite signs,
            # or both are equal to zero
            oriC_fragment_counts = np.count_nonzero(
                np.logical_not(
                    np.logical_xor(
                        old_boundary_coordinates_sorted[:, 0] < 0,
                        old_boundary_coordinates_sorted[:, 1] > 0,
                    )
                )
            )

            # if oriC fragment did not exist in the domain in the previous
            # timestep, add a dummy fragment that covers the origin with
            # linking number zero. This is done to generalize the
            # implementation of this method.
            if oriC_fragment_counts == 0:
                # Index of first segment where left boundary is nonnegative
                oriC_fragment_index = np.argmax(
                    old_boundary_coordinates_sorted[:, 0] >= 0
                )

                # Get indexes of boundary molecules for this dummy segment
                oriC_fragment_boundary_molecule_indexes = np.array(
                    [
                        old_boundary_molecule_indexes_sorted[
                            oriC_fragment_index - 1, 1
                        ],
                        old_boundary_molecule_indexes_sorted[oriC_fragment_index, 0],
                    ]
                )

                # Insert dummy segment to array
                old_boundary_molecule_indexes_sorted = np.insert(
                    old_boundary_molecule_indexes_sorted,
                    oriC_fragment_index,
                    oriC_fragment_boundary_molecule_indexes,
                    axis=0,
                )
                old_linking_numbers_sorted = np.insert(
                    old_linking_numbers_sorted, oriC_fragment_index, 0
                )
            else:
                # There should not be more than one fragment that spans oriC
                assert oriC_fragment_counts == 1

        # Domain spans the terminus
        if spans_terC:
            # If the domain spans the terminus, dummy molecules are added to
            # each end of the chromosome s.t. the segment that spans terC is
            # split to two segments and we can maintain a linear representation
            # for the circular chromosome. These two segments are later
            # adjusted to have the same superhelical densities.
            new_molecule_coordinates_sorted = np.insert(
                new_molecule_coordinates_sorted,
                [0, len(new_molecule_coordinates_sorted)],
                [self.min_coordinates, self.max_coordinates],
            )

            new_molecule_indexes_sorted = np.insert(
                new_molecule_indexes_sorted,
                [0, len(new_molecule_indexes_sorted)],
                self.terC_index,
            )

            # Add dummy molecule to old segments if they do not already exist
            if old_boundary_molecule_indexes_sorted[0, 0] != self.terC_index:
                old_boundary_molecule_indexes_sorted = np.vstack(
                    (
                        np.array(
                            [
                                self.terC_index,
                                old_boundary_molecule_indexes_sorted[0, 0],
                            ]
                        ),
                        old_boundary_molecule_indexes_sorted,
                        np.array(
                            [
                                old_boundary_molecule_indexes_sorted[-1, 1],
                                self.terC_index,
                            ]
                        ),
                    )
                )
                old_linking_numbers_sorted = np.insert(
                    old_linking_numbers_sorted, [0, len(old_linking_numbers_sorted)], 0
                )

        # Recalculate linking numbers of each segment after accounting for
        # boundary molecules that were removed in the current timestep
        linking_numbers_after_removal = []
        right_boundaries_retained = np.isin(
            old_boundary_molecule_indexes_sorted[:, 1], new_molecule_indexes_sorted
        )

        # Add up linking numbers of each segment until each retained boundary
        ln_this_fragment = 0.0
        for retained, ln in zip(right_boundaries_retained, old_linking_numbers_sorted):
            ln_this_fragment += ln

            if retained:
                linking_numbers_after_removal.append(ln_this_fragment)
                ln_this_fragment = 0.0

        # Number of segments should be equal to number of retained boundaries
        assert len(linking_numbers_after_removal) == right_boundaries_retained.sum()

        # Redistribute linking numbers of the two terC segments such that the
        # segments have same superhelical densities
        if spans_terC and np.count_nonzero(right_boundaries_retained) > 1:
            # Get molecule indexes of the boundaries of the two terC segments
            # left and right of terC
            retained_boundary_indexes = np.where(right_boundaries_retained)[0]
            left_segment_boundary_index = old_boundary_molecule_indexes_sorted[
                retained_boundary_indexes[0], 1
            ]
            right_segment_boundary_index = old_boundary_molecule_indexes_sorted[
                retained_boundary_indexes[-2], 1
            ]

            # Get mapping from molecule index to chromosomal coordinates
            molecule_index_to_coordinates = {
                index: coordinates
                for index, coordinates in zip(
                    new_molecule_indexes_sorted, new_molecule_coordinates_sorted
                )
            }

            # Distribute linking number between two segments proportional to
            # the length of each segment
            left_segment_length = (
                molecule_index_to_coordinates[left_segment_boundary_index]
                - self.min_coordinates
            )
            right_segment_length = (
                self.max_coordinates
                - molecule_index_to_coordinates[right_segment_boundary_index]
            )
            full_segment_length = left_segment_length + right_segment_length
            full_linking_number = (
                linking_numbers_after_removal[0] + linking_numbers_after_removal[-1]
            )

            linking_numbers_after_removal[0] = (
                full_linking_number * left_segment_length / full_segment_length
            )
            linking_numbers_after_removal[-1] = (
                full_linking_number * right_segment_length / full_segment_length
            )

        # Get mask for molecules that already existed in the previous timestep
        existing_molecules_mask = np.isin(
            new_molecule_indexes_sorted, old_boundary_molecule_indexes_sorted
        )

        # Get numbers and lengths of new segments that each segment will be
        # split into
        segment_split_sizes = np.diff(np.where(existing_molecules_mask)[0])
        segment_lengths = np.diff(new_molecule_coordinates_sorted)

        assert len(segment_split_sizes) == len(linking_numbers_after_removal)

        # Calculate linking numbers of each segment after accounting for new
        # boundaries that were added
        new_linking_numbers = []
        i = 0

        for ln, size in zip(linking_numbers_after_removal, segment_split_sizes):
            if size == 1:
                new_linking_numbers.append(ln)
            else:
                # Split linking number proportional to length of segment
                total_length = segment_lengths[i : i + size].sum()
                new_linking_numbers.extend(
                    list(ln * segment_lengths[i : i + size] / total_length)
                )
            i += size

        # Handle edge case where a domain was just initialized, and two
        # replisomes are bound to the origin
        if len(new_linking_numbers) == 0:
            new_linking_numbers = [np.float64(0)]

        # Build Mx2 array for boundary indexes and coordinates
        new_boundary_molecule_indexes = np.hstack(
            (
                new_molecule_indexes_sorted[:-1, np.newaxis],
                new_molecule_indexes_sorted[1:, np.newaxis],
            )
        )
        new_boundary_coordinates = np.hstack(
            (
                new_molecule_coordinates_sorted[:-1, np.newaxis],
                new_molecule_coordinates_sorted[1:, np.newaxis],
            )
        )
        new_linking_numbers = np.array(new_linking_numbers)

        # If domain does not span oriC, remove new segment that spans origin
        if not spans_oriC:
            oriC_fragment_mask = np.logical_not(
                np.logical_xor(
                    new_boundary_coordinates[:, 0] < 0,
                    new_boundary_coordinates[:, 1] > 0,
                )
            )

            assert oriC_fragment_mask.sum() == 1

            new_boundary_molecule_indexes = new_boundary_molecule_indexes[
                np.logical_not(oriC_fragment_mask), :
            ]
            new_boundary_coordinates = new_boundary_coordinates[
                np.logical_not(oriC_fragment_mask), :
            ]
            new_linking_numbers = new_linking_numbers[
                np.logical_not(oriC_fragment_mask)
            ]

        return {
            "boundary_molecule_indexes": new_boundary_molecule_indexes,
            "boundary_coordinates": new_boundary_coordinates,
            "linking_numbers": new_linking_numbers,
        }


def get_last_known_replisome_data(
    boundary_coordinates: np.ndarray,
    boundary_molecule_indexes: np.ndarray,
    replisome_coordinates: np.ndarray,
    replisome_molecule_indexes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Gets the last known coordinates and molecule indexes of both replisomes
    for a chromosome domain.

    Args:
        boundary_coordinates: (N, 2) array of chromosomal coordinates of
            all boundary molecules in the domain during the last time step.
        boundary_molecule_indexes: (N, 2) array of unique indexes of all
            boundary molecules in the domain during the last time step.
        replisome_coordinates: (1,) or (0,) array of chromosomal coordinates of
            the replisomes in the domain in the current time step.
        replisome_molecule_indexes: (1,) or (0,) array of unique indexes of the
            replisomes in the domain in the current time step.

    Returns:
        Tuple of the following format::

            (
                (2,) array of last known replisome coordinates in domain,
                (2,) array of last known replisome molecule indexes in domain
            )
    """
    # Sort old boundary coordinates and molecule indexes to find first index
    # where left boundary is non-negative
    boundary_coordinates_argsort = np.argsort(boundary_coordinates[:, 0])
    boundary_coordinates_sorted = boundary_coordinates[boundary_coordinates_argsort]
    boundary_molecule_indexes_sorted = boundary_molecule_indexes[
        boundary_coordinates_argsort
    ]
    replisome_index = np.argmax(boundary_coordinates_sorted[:, 0] >= 0)
    # If positive coordinate replisome still exists, add
    # last known info on negative coordinate replisome.
    if np.any(replisome_coordinates > 0):
        replisome_coordinates = np.insert(
            replisome_coordinates,
            0,
            boundary_coordinates_sorted[replisome_index - 1, 1],
        )
        replisome_molecule_indexes = np.insert(
            replisome_molecule_indexes,
            0,
            boundary_molecule_indexes_sorted[replisome_index - 1, 1],
        )
    # If negative coordinate replisome still exists, add
    # last known info on positive coordinate replisome.
    elif np.any(replisome_coordinates < 0):
        replisome_coordinates = np.insert(
            replisome_coordinates, 0, boundary_coordinates_sorted[replisome_index, 0]
        )
        replisome_molecule_indexes = np.insert(
            replisome_molecule_indexes,
            0,
            boundary_molecule_indexes_sorted[replisome_index, 0],
        )
    # If neither replisomes exist, use last known info on both.
    else:
        replisome_coordinates = np.array(
            [
                boundary_coordinates_sorted[replisome_index - 1, 1],
                boundary_coordinates_sorted[replisome_index, 0],
            ]
        )
        replisome_molecule_indexes = np.array(
            [
                boundary_molecule_indexes_sorted[replisome_index - 1, 1],
                boundary_molecule_indexes_sorted[replisome_index, 0],
            ]
        )

    return replisome_coordinates, replisome_molecule_indexes


def test_superhelical_removal_sim():
    """
    Run a single time step simulation of :py:class:`~.ChromosomeStructure`
    that tests some edge cases in superhelical density calculations. Start with
    a chromosome that has four active replication forks for a total of 4 replisomes
    and 5 chromosome domains. There are 4 RNAPs per domain and 1 to 2 of those
    will be intentionally placed at the same coordinates as a replisome per domain.
    They should be removed, causing some segment boundaries to be redefined.
    Additionally, 3 of the replisomes will be removed. This should be detected
    and superhelical density calculations should still work using the last known
    information for the removed replisomes.
    """
    # Get topology for UniqueUpdate Steps
    unique_topology = TOPOLOGY.copy()
    for non_unique in [
        "bulk",
        "listeners",
        "global_time",
        "timestep",
        "next_update_time",
    ]:
        unique_topology.pop(non_unique)

    class TestComposer(Composer):
        def generate_processes(self, config):
            return {
                "chromosome_structure": ChromosomeStructure(
                    {
                        "inactive_RNAPs": "APORNAP-CPLX[c]",
                        "ppi": "PPI[c]",
                        "active_tfs": ["CPLX-125[c]"],
                        "ribosome_30S_subunit": "CPLX0-3953[c]",
                        "ribosome_50S_subunit": "CPLX0-3962[c]",
                        "amino_acids": ["L-ALPHA-ALANINE[c]"],
                        "water": "WATER[c]",
                        "mature_rna_ids": ["alaT-tRNA[c]"],
                        "fragmentBases": ["polymerized_ATP[c]"],
                        "replichore_lengths": [100000, 100000],
                        "calculate_superhelical_densities": True,
                    }
                ),
                "unique_update": UniqueUpdate({"unique_topo": unique_topology}),
                "global_clock": GlobalClock(),
            }

        def generate_topology(self, config):
            return {
                "chromosome_structure": TOPOLOGY,
                "unique_update": unique_topology,
                "global_clock": {
                    "global_time": ("global_time",),
                    "next_update_time": ("next_update_time",),
                },
            }

        def generate_flow(self, config):
            return {
                "chromosome_structure": [],
                "unique_update": [("chromosome_structure",)],
            }

    composer = TestComposer()
    template_initial_state = get_state_from_file("data/vivecoli_t2526.json")["agents"][
        "0"
    ]
    # Zero out all unique molecules
    for unique_mol in template_initial_state["unique"].values():
        unique_mol.flags.writeable = True
        unique_mol["_entryState"] = 0
        unique_mol.flags.writeable = False
    # Set up a single full chromosome
    full_chromosomes = template_initial_state["unique"]["full_chromosome"]
    full_chromosomes.flags.writeable = True
    full_chromosomes["_entryState"][0] = 1
    full_chromosomes["domain_index"][0] = 0
    full_chromosomes["unique_index"][0] = 0
    full_chromosomes.flags.writeable = False
    # Set up chromosome domains
    chromosome_domains, replisome_idx = get_free_indices(
        template_initial_state["unique"]["chromosome_domain"], 5
    )
    chromosome_domains.flags.writeable = True
    chromosome_domains["_entryState"][replisome_idx] = 1
    chromosome_domains["domain_index"][replisome_idx] = np.arange(5)
    chromosome_domains["unique_index"][replisome_idx] = np.arange(5)
    chromosome_domains["child_domains"][replisome_idx] = [
        [1, 2],
        [3, 4],
        [-1, -1],
        [-1, -1],
        [-1, -1],
    ]
    chromosome_domains.flags.writeable = False
    template_initial_state["unique"]["chromosome_domain"] = chromosome_domains
    # Set up 1 oriC per domain that is not actively replicating
    oriCs, oriC_idx = get_free_indices(template_initial_state["unique"]["oriC"], 3)
    oriCs.flags.writeable = True
    oriCs["_entryState"][oriC_idx] = 1
    oriCs["domain_index"][oriC_idx] = [2, 3, 4]
    oriCs["unique_index"][oriC_idx] = np.arange(3)
    oriCs.flags.writeable = False
    template_initial_state["unique"]["oriC"] = oriCs
    # Set up replisome for actively replicating domain
    # Notice that the replisomes previously at 45000, 20000, and -20000
    # when the chromosomal segment data below was tabulated are removed
    active_replisomes, replisome_idx = get_free_indices(
        template_initial_state["unique"]["active_replisome"], 1
    )
    active_replisomes.flags.writeable = True
    active_replisomes["_entryState"][replisome_idx] = 1
    active_replisomes["domain_index"][replisome_idx] = 0
    active_replisomes["coordinates"][replisome_idx] = -50000
    active_replisomes["unique_index"][replisome_idx] = 0
    active_replisomes.flags.writeable = False
    template_initial_state["unique"]["active_replisome"] = active_replisomes
    # Set up 4 RNAPs per domain, some of which will intentionally have
    # the same coordinates as replisomes (either active or removed)
    active_RNAP = template_initial_state["unique"]["active_RNAP"]
    active_RNAP.flags.writeable = True
    coordinates_per_domain = [
        [-65000, -50000, 60000, 75000],
        [-40000, -26000, 25000, 35000],
        [-30000, -10000, 20000, 40000],
        [-20000, -15000, 10000, 15000],
    ]
    for i in range(4):
        active_RNAP["_entryState"][i * 4 : (i + 1) * 4] = 1
        active_RNAP["domain_index"][i * 4 : (i + 1) * 4] = i
        active_RNAP["is_forward"][i * 4 : (i + 1) * 4] = True
        active_RNAP["coordinates"][i * 4 : (i + 1) * 4] = coordinates_per_domain[i]
        active_RNAP["unique_index"][i * 4 : (i + 1) * 4] = np.arange(
            4 + i * 4, 4 + (i + 1) * 4
        )
    # Special domain 4 that will be left with no molecules
    active_RNAP["_entryState"][16:18] = 1
    active_RNAP["domain_index"][16:18] = 4
    active_RNAP["is_forward"][16:18] = True
    active_RNAP["coordinates"][16:18] = [-20000, 20000]
    active_RNAP["unique_index"][16:18] = np.arange(20, 22)
    active_RNAP.flags.writeable = False
    # Construct chromosomal segments by domain
    # Assume that RNAPs have advanced 1000 bp from their initial coordinates
    # and replisomes have advanced 5000 bp
    boundary_coordinates = []
    boundary_molecule_indexes = []
    domain_index = []
    linking_number = []
    # Segments for domain 0
    # - Includes replisome at 45000 (index 1) that will be removed
    # - Includes RNAP (index 5) that will conflict with replisome at -50000 (index 0)
    boundary_coordinates.extend(
        [[-64000, -49000], [-49000, -45000], [45000, 59000], [59000, 74000]]
    )
    boundary_molecule_indexes.extend([[4, 5], [5, 0], [1, 6], [6, 7]])
    domain_index.extend([0, 0, 0, 0])
    linking_number.extend([1, 1, 1, 1])
    # Segments for domain 1
    # - Includes replisome at 20000 (index 2) that will be removed
    # - Includes replisome at -20000 (index 3) that will be removed
    # - Parent domain replisome at 45000 (index 1) will be removed
    boundary_coordinates.extend(
        [
            [-45000, -39000],
            [-39000, -25000],
            [-25000, -20000],
            [20000, 24000],
            [24000, 34000],
            [34000, 45000],
        ]
    )
    boundary_molecule_indexes.extend(
        [[0, 8], [8, 9], [9, 3], [2, 10], [10, 11], [11, 1]]
    )
    domain_index.extend([1, 1, 1, 1, 1, 1])
    linking_number.extend([1, 1, 1, 1, 1, 1])
    # Segments for domain 2
    # - Parent domain replisome at 45000 (index 1) will be removed
    boundary_coordinates.extend(
        [
            [-45000, -29000],
            [-29000, -9000],
            [-9000, 19000],
            [19000, 39000],
            [39000, 45000],
        ]
    )
    boundary_molecule_indexes.extend([[0, 12], [12, 13], [13, 14], [14, 15], [15, 1]])
    domain_index.extend([2, 2, 2, 2, 2])
    linking_number.extend([1, 1, 1, 1, 1])
    # Segments for domain 3
    # - Includes RNAP (index 16) that will conflict with replisome that will
    # be removed at -20000 (index 3)
    # - Includes replisome at 20000 (index 2) that will be removed
    boundary_coordinates.extend(
        [
            [-20000, -19000],
            [-19000, -14000],
            [-14000, 9000],
            [9000, 14000],
            [14000, 20000],
        ]
    )
    boundary_molecule_indexes.extend([[3, 16], [16, 17], [17, 18], [18, 19], [19, 2]])
    domain_index.extend([3, 3, 3, 3, 3])
    linking_number.extend([1, 1, 1, 1, 1])
    # Segments for domain 4
    # - Includes RNAP (index 20) that will conflict with replisome that will
    # be removed at -20000 (index 3)
    # - Includes RNAP (index 23) that will conflict with replisome that will
    # be removed at 20000 (index 2)
    boundary_coordinates.extend([[-20000, -19000], [-19000, 19000], [19000, 20000]])
    boundary_molecule_indexes.extend([[3, 20], [20, 21], [21, 2]])
    domain_index.extend([4, 4, 4])
    linking_number.extend([1, 1, 1])
    chromosomal_segments, segment_idx = get_free_indices(
        template_initial_state["unique"]["chromosomal_segment"], len(linking_number)
    )
    chromosomal_segments.flags.writeable = True
    chromosomal_segments["_entryState"][segment_idx] = 1
    chromosomal_segments["boundary_coordinates"][segment_idx] = boundary_coordinates
    chromosomal_segments["boundary_molecule_indexes"][segment_idx] = (
        boundary_molecule_indexes
    )
    chromosomal_segments["domain_index"][segment_idx] = domain_index
    chromosomal_segments["linking_number"][segment_idx] = linking_number
    chromosomal_segments["unique_index"][segment_idx] = np.arange(len(linking_number))
    chromosomal_segments.flags.writeable = False
    template_initial_state["unique"]["chromosomal_segment"] = chromosomal_segments
    # Since unique numpy updater is an class method, internal
    # deepcopying in vivarium-core causes this warning to appear
    warnings.filterwarnings(
        "ignore",
        message="Incompatible schema "
        "assignment at .+ Trying to assign the value <bound method "
        r"UniqueNumpyUpdater\.updater .+ to key updater, which already "
        r"has the value <bound method UniqueNumpyUpdater\.updater",
    )
    engine = Engine(
        composite=composer.generate(),
        initial_state=template_initial_state,
    )
    engine.update(1)
    state = engine.state.get_value()
    # Check that right number of collisions happened at right coordinates
    rnap_data = state["listeners"]["rnap_data"]
    assert rnap_data["n_total_collisions"] == 1
    assert rnap_data["n_headon_collisions"] == 1
    assert rnap_data["n_codirectional_collisions"] == 0
    assert np.array_equal(
        rnap_data["headon_collision_coordinates"], np.array([-50000], dtype=int)
    )
    assert rnap_data["n_empty_fork_collisions"] == 3
    assert np.array_equal(
        rnap_data["empty_fork_collision_coordinates"],
        np.array([-20000, -20000, 20000], dtype=int),
    )
    # Check chromosomal segments
    chromosomal_segments = state["unique"]["chromosomal_segment"][
        state["unique"]["chromosomal_segment"]["_entryState"].view(np.bool_)
    ]
    assert np.array_equal(
        chromosomal_segments["boundary_coordinates"],
        np.array(
            [
                # Domain 0
                [-100000, -65000],
                [-65000, -50000],
                [45000, 60000],
                [60000, 75000],
                [75000, 100000],
                # Domain 1
                [-50000, -40000],
                [-40000, -26000],
                [-26000, -20000],
                [20000, 25000],
                [25000, 35000],
                [35000, 45000],
                # Domain 2
                [-50000, -30000],
                [-30000, -10000],
                [-10000, 20000],
                [20000, 40000],
                [40000, 45000],
                # Domain 3
                [-20000, -15000],
                [-15000, 10000],
                [10000, 15000],
                [15000, 20000],
                # Domain 4
                [-20000, 20000],
            ],
            dtype=int,
        ),
    )
    assert np.array_equal(
        chromosomal_segments["boundary_molecule_indexes"],
        np.array(
            [
                # Domain 0: index 5 is gone due to conflict with replisome,
                # segments were added flanking terC (-1 placeholder index),
                # index 1 replisome is gone but still exists as placeholder
                # for replication fork in our superhelical density calculations
                [-1, 4],
                [4, 0],
                [1, 6],
                [6, 7],
                [7, -1],
                # Domain 1: index 1, 2, and 3 replisomes are removed but still exist
                # here as placeholders
                [0, 8],
                [8, 9],
                [9, 3],
                [2, 10],
                [10, 11],
                [11, 1],
                # Domain 2: index 1 replisome was removed but still exists here
                # as a placeholder
                [0, 12],
                [12, 13],
                [13, 14],
                [14, 15],
                [15, 1],
                # Domain 3: index 2 and 3 replisomes are removed but still exist
                # here as placeholders, index 16 RNAP was removed due to conflict
                # with replisome placeholder index 3
                [3, 17],
                [17, 18],
                [18, 19],
                [19, 2],
                # Domain 4: index 2 and 3 replisomes were removed but still exist
                # here as placeholders, index 20 and 21 RNAPs were removed due to
                # conflicts with replisome placeholders
                [3, 2],
            ],
            dtype=int,
        ),
    )
    assert np.array_equal(
        chromosomal_segments["domain_index"],
        np.array(
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4], dtype=int
        ),
    )
    assert np.array_equal(
        chromosomal_segments["linking_number"],
        np.array(
            [
                # Domain 0: terC segments have 0 linking number, second segment
                # combines two original segments due to removed RNAP index 5
                0,
                2,
                1,
                1,
                0,
                # Domain 1: No molecule boundaries were changed
                1,
                1,
                1,
                1,
                1,
                1,
                # Domain 2: No molecule boundaries were changed
                1,
                1,
                1,
                1,
                1,
                # Domain 3: First segment combines two original segments due to
                # removed RNAP index 16
                2,
                1,
                1,
                1,
                # Domain 4: Sole segment combines all three original segments
                # due to removed RNAP index 20 and 21, leaving only a single
                # segment between two unoccupied replication forks
                3,
            ],
            dtype=float,
        ),
    )


if __name__ == "__main__":
    test_superhelical_removal_sim()

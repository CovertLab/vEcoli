"""
====================
Chromosome Structure
====================

- Resolve collisions between molecules and replication forks on the chromosome.
- Remove and replicate promoters and motifs that are traversed by replisomes.
- Reset the boundaries and linking numbers of chromosomal segments.
"""

import numpy as np

from vivarium.core.process import Process
from ecoli.library.unique_indexes import create_unique_indexes
from ecoli.processes.registries import topology_registry
from ecoli.library.schema import (
    add_elements, arrays_from, bulk_schema,
    arrays_to, array_to, dict_value_schema, listener_schema)

from wholecell.utils.polymerize import buildSequences


# Register default topology for this process, associating it with process name
NAME = 'ecoli-chromosome-structure'
TOPOLOGY = {
    "listeners": ("listeners",),
    "fragmentBases": ("bulk",),
    "molecules": ("bulk",),
    "active_tfs": ("bulk",),
    "subunits": ("bulk",),
    "amino_acids": ("bulk",),
    "active_replisomes": ("unique", "active_replisome",),
    "oriCs": ("unique", "oriC",),
    "chromosome_domains": ("unique", "chromosome_domain",),
    "active_RNAPs": ("unique", "active_RNAP"),
    "RNAs": ("unique", "RNA"),
    "active_ribosome": ("unique", "active_ribosome"),
    "full_chromosomes": ("unique", "full_chromosome",),
    "promoters": ("unique", "promoter"),
    "DnaA_boxes": ("unique", "DnaA_box"),
    # TODO(vivarium): Only include if superhelical density flag is passed
    # "chromosomal_segments": ("unique", "chromosomal_segment")
}
topology_registry.register(NAME, TOPOLOGY)

class ChromosomeStructure(Process):
    """ Chromosome Structure Process """

    name = NAME
    topology = TOPOLOGY
    defaults = {
            # Load parameters
            'RNA_sequences': [],
            'protein_sequences': [],
            'n_TUs': 1,
            'n_TFs': 1,
            'n_amino_acids': 1,
            'n_fragment_bases': 1,
            'replichore_lengths': [0,0],
            'relaxed_DNA_base_pairs_per_turn': 1,
            'terC_index': 1,

            'calculate_superhelical_densities': False,

            # Get placeholder value for chromosome domains without children
            'no_child_place_holder': -1,

            # Load bulk molecule views
            'inactive_RNAPs': [],
            'fragmentBases': [],
            'ppi': 'ppi',
            'active_tfs': [],
            'ribosome_30S_subunit': '30S',
            'ribosome_50S_subunit': '50S',
            'amino_acids': [],
            'water': 'water',
            'deriver_mode': True
        }

    # Constructor
    def __init__(self, parameters=None):
        super().__init__(parameters)
        self.RNA_sequences = self.parameters['RNA_sequences']
        self.protein_sequences = self.parameters['protein_sequences']
        self.n_TUs = self.parameters['n_TUs']
        self.n_TFs = self.parameters['n_TFs']
        self.n_amino_acids = self.parameters['n_amino_acids']
        self.n_fragment_bases = self.parameters['n_fragment_bases']
        replichore_lengths = self.parameters['replichore_lengths']
        self.min_coordinates = -replichore_lengths[1]
        self.max_coordinates = replichore_lengths[0]
        self.relaxed_DNA_base_pairs_per_turn = self.parameters['relaxed_DNA_base_pairs_per_turn']
        self.terC_index = self.parameters['terC_index']

        # Load sim options
        self.calculate_superhelical_densities = self.parameters['calculate_superhelical_densities']

        # Get placeholder value for chromosome domains without children
        self.no_child_place_holder = self.parameters['no_child_place_holder']

        self.inactive_RNAPs = self.parameters['inactive_RNAPs']
        self.fragmentBases = self.parameters['fragmentBases']
        self.ppi = self.parameters['ppi']
        self.active_tfs = self.parameters['active_tfs']
        self.ribosome_30S_subunit = self.parameters['ribosome_30S_subunit']
        self.ribosome_50S_subunit = self.parameters['ribosome_50S_subunit']
        self.amino_acids = self.parameters['amino_acids']
        self.water = self.parameters['water']

        self.emit_unique = self.parameters.get('emit_unique', True)

        self.chromosome_segment_index = 0
        self.promoter_index = 60000
        self.DnaA_box_index = 60000
        self.deriver_mode = self.parameters['deriver_mode']

    def is_deriver(self):
        return self.deriver_mode

    def ports_schema(self):
        default_unique_schema = {
            '_default': 0, '_updater': 'set', '_emit': self.emit_unique}

        ports = {
            'listeners': {
                'RnapData': listener_schema(
                    {'n_total_collisions': 0,
                    'n_headon_collisions': 0,
                    'n_codirectional_collisions': 0,
                    'headon_collision_coordinates': 0,
                    'codirectional_collision_coordinates': 0,
                    'n_removed_ribosomes': 0})},
            # Bulk molecules
            'fragmentBases': bulk_schema(self.fragmentBases),
            'molecules': bulk_schema([self.ppi, self.water,
                                      self.inactive_RNAPs]),
            'active_tfs': bulk_schema(self.active_tfs),
            'subunits': bulk_schema(
                [self.ribosome_30S_subunit,
                 self.ribosome_50S_subunit]),
            'amino_acids': bulk_schema(self.amino_acids),

            # Unique molecules
            'active_replisomes': dict_value_schema('active_replisomes'),
            'oriCs': dict_value_schema('oriCs'),
            'chromosome_domains': dict_value_schema('chromosome_domains'),
            'active_RNAPs': {
                '_divider': 'by_domain',
                **dict_value_schema('active_RNAPs')},
            'RNAs': {
                '_divider': {
                    'divider': 'rna_by_domain',
                    'topology': {'active_RNAP': ('..', 'active_RNAP',)}},
                **dict_value_schema('RNAs')},
            'active_ribosome': dict_value_schema('active_ribosome'),
            'full_chromosomes': dict_value_schema('full_chromosomes'),
            'promoters': {
                '_divider': 'by_domain',
                **dict_value_schema('promoters')},
            'DnaA_boxes': {
                '_divider': 'by_domain',
                **dict_value_schema('DnaA_boxes')}
        }

        if self.calculate_superhelical_densities:
            ports['chromosomal_segments'] = {
                '*': {
                    'boundary_molecule_indexes': {'_default': np.empty((0, 2), dtype=np.int64)},
                    'boundary_coordinates': {'_default': np.empty((0, 2), dtype=np.int64)},
                    'domain_index': {'_default': 0},
                    'linking_number': {'_default': 0}}}

        return ports

    def next_update(self, timestep, states):
        # Skip t=0 if a deriver
        if self.deriver_mode:
            self.deriver_mode = False
            return {}

        # Read unique molecule attributes
        if states['active_replisomes'].values():
            replisome_domain_indexes, replisome_coordinates, replisome_unique_indexes = arrays_from(
                states['active_replisomes'].values(),
                ['domain_index', 'coordinates', 'unique_index'])
        else:
            replisome_domain_indexes, replisome_coordinates, replisome_unique_indexes = (
                np.array([]), np.array([]), np.array([]))
        all_chromosome_domain_indexes, child_domains = arrays_from(
            states['chromosome_domains'].values(),
            ['domain_index', 'child_domains'])
        RNAP_domain_indexes, RNAP_coordinates, RNAP_directions, RNAP_unique_indexes = arrays_from(
            states['active_RNAPs'].values(),
            ['domain_index', 'coordinates', 'direction', 'unique_index'])
        origin_domain_indexes = arrays_from(states['oriCs'].values(), ['domain_index'])[0]
        mother_domain_indexes = arrays_from(states['full_chromosomes'].values(), ['domain_index'])[0]
        RNA_TU_indexes, transcript_lengths, RNA_RNAP_indexes, RNA_unique_indexes = arrays_from(
            states['RNAs'].values(),
            ['TU_index', 'transcript_length', 'RNAP_index', 'unique_index'])
        ribosome_protein_indexes, ribosome_peptide_lengths, ribosome_mRNA_indexes = arrays_from(
            states['active_ribosome'].values(),
            ['protein_index', 'peptide_length', 'mRNA_index'])
        promoter_TU_indexes, promoter_domain_indexes, promoter_coordinates, promoter_bound_TFs = arrays_from(
            states['promoters'].values(),
            ['TU_index', 'domain_index', 'coordinates', 'bound_TF'])
        DnaA_box_domain_indexes, DnaA_box_coordinates, DnaA_box_bound = arrays_from(
            states['DnaA_boxes'].values(),
            ['domain_index', 'coordinates', 'DnaA_bound'])

        # Build dictionary of replisome coordinates with domain indexes as keys
        replisome_coordinates_from_domains = {
            domain_index: replisome_coordinates[replisome_domain_indexes == domain_index]
            for domain_index in np.unique(replisome_domain_indexes)
            }


        def get_removed_molecules_mask(domain_indexes, coordinates):
            """
            Computes the boolean mask of unique molecules that should be
            removed based on the progression of the replication forks.
            """
            mask = np.zeros_like(domain_indexes, dtype=np.bool)

            # Loop through all domains
            for domain_index in np.unique(domain_indexes):
                # Domain has active replisomes
                if domain_index in replisome_coordinates_from_domains:
                    domain_replisome_coordinates = replisome_coordinates_from_domains[
                        domain_index]

                    # Get mask for molecules on this domain that are out of range
                    domain_mask = np.logical_and.reduce((
                        domain_indexes == domain_index,
                        coordinates > domain_replisome_coordinates.min(),
                        coordinates < domain_replisome_coordinates.max()))

                # Domain has no active replisomes
                else:
                    # Domain has child domains (has finished replicating)
                    if (child_domains[all_chromosome_domain_indexes == domain_index, 0]
                            != self.no_child_place_holder):
                        # Remove all molecules on this domain
                        domain_mask = (domain_indexes == domain_index)
                    # Domain has not started replication
                    else:
                        continue

                mask[domain_mask] = True

            return mask


        # Build mask for molecules that should be removed
        removed_RNAPs_mask = get_removed_molecules_mask(
            RNAP_domain_indexes, RNAP_coordinates)
        removed_promoters_mask = get_removed_molecules_mask(
            promoter_domain_indexes, promoter_coordinates)
        removed_DnaA_boxes_mask = get_removed_molecules_mask(
            DnaA_box_domain_indexes, DnaA_box_coordinates)

        # Get attribute arrays of remaining RNAPs
        remaining_RNAPs_mask = np.logical_not(removed_RNAPs_mask)
        remaining_RNAP_domain_indexes = RNAP_domain_indexes[remaining_RNAPs_mask]
        remaining_RNAP_coordinates = RNAP_coordinates[remaining_RNAPs_mask]
        remaining_RNAP_unique_indexes = RNAP_unique_indexes[remaining_RNAPs_mask]

        # Build masks for head-on and co-directional collisions between RNAPs
        # and replication forks
        RNAP_headon_collision_mask = np.logical_and(
            removed_RNAPs_mask,
            np.logical_xor(RNAP_directions, RNAP_coordinates > 0))
        RNAP_codirectional_collision_mask = np.logical_and(
            removed_RNAPs_mask, np.logical_not(RNAP_headon_collision_mask))

        n_total_collisions = np.count_nonzero(removed_RNAPs_mask)
        n_headon_collisions = np.count_nonzero(RNAP_headon_collision_mask)
        n_codirectional_collisions = np.count_nonzero(
            RNAP_codirectional_collision_mask)

        # Write values to listeners
        update = {
            'listeners': {
                'RnapData': {
                    'n_total_collisions': n_total_collisions,
                    'n_headon_collisions': n_headon_collisions,
                    'n_codirectional_collisions': n_codirectional_collisions,
                    'headon_collision_coordinates': RNAP_coordinates[RNAP_headon_collision_mask],
                    'codirectional_collision_coordinates': RNAP_coordinates[RNAP_codirectional_collision_mask]
                }
            },
            'molecules': {},
            'fragmentBases': {},
            'active_tfs': {},
            'subunits': {},
            'amino_acids': {},
            'active_replisomes': {},
            'oriCs': {},
            'chromosome_domains': {},
            'active_RNAPs': {},
            'RNAs': {},
            'active_ribosome': {},
            'full_chromosomes': {},
            'promoters': {},
            'DnaA_boxes': {}
        }

        if self.calculate_superhelical_densities:
            # Get attributes of existing segments
            boundary_molecule_indexes, boundary_coordinates, segment_domain_indexes, linking_numbers = arrays_from(
                states['chromosomal_segments'].values(),
                ['boundary_molecule_indexes', 'boundary_coordinates',
                'domain_index', 'linking_number'])

            # Initialize new attributes of chromosomal segments
            all_new_boundary_molecule_indexes = np.empty((0, 2), dtype=np.int64)
            all_new_boundary_coordinates = np.empty((0, 2), dtype=np.int64)
            all_new_segment_domain_indexes = np.array([], dtype=np.int32)
            all_new_linking_numbers = np.array([], dtype=np.float64)

            for domain_index in np.unique(all_chromosome_domain_indexes):
                # Skip domains that have completed replication
                if np.all(domain_index < mother_domain_indexes):
                    continue

                domain_spans_oriC = (domain_index in origin_domain_indexes)
                domain_spans_terC = (domain_index in mother_domain_indexes)

                # Get masks for segments and RNAPs in this domain
                segments_domain_mask = (segment_domain_indexes == domain_index)
                RNAP_domain_mask = (remaining_RNAP_domain_indexes == domain_index)

                # Parse attributes of segments in this domain
                boundary_molecule_indexes_this_domain = boundary_molecule_indexes[segments_domain_mask, :]
                boundary_coordinates_this_domain = boundary_coordinates[segments_domain_mask, :]
                linking_numbers_this_domain = linking_numbers[segments_domain_mask]

                # Parse attributes of remaining RNAPs in this domain
                new_molecule_coordinates_this_domain = remaining_RNAP_coordinates[RNAP_domain_mask]
                new_molecule_indexes_this_domain = remaining_RNAP_unique_indexes[RNAP_domain_mask]

                # Append coordinates and indexes of replisomes on this domain, if any
                if not domain_spans_oriC:
                    replisome_domain_mask = (replisome_domain_indexes == domain_index)

                    new_molecule_coordinates_this_domain = np.concatenate((
                        new_molecule_coordinates_this_domain,
                        replisome_coordinates[replisome_domain_mask]
                        ))
                    new_molecule_indexes_this_domain = np.concatenate((
                        new_molecule_indexes_this_domain,
                        replisome_unique_indexes[replisome_domain_mask]
                        ))

                # Append coordinates and indexes of parent domain replisomes, if any
                if not domain_spans_terC:
                    parent_domain_index = all_chromosome_domain_indexes[
                        np.where(child_domains == domain_index)[0][0]]
                    replisome_parent_domain_mask = (replisome_domain_indexes == parent_domain_index)

                    new_molecule_coordinates_this_domain = np.concatenate((
                        new_molecule_coordinates_this_domain,
                        replisome_coordinates[replisome_parent_domain_mask]
                        ))
                    new_molecule_indexes_this_domain = np.concatenate((
                        new_molecule_indexes_this_domain,
                        replisome_unique_indexes[replisome_parent_domain_mask]
                        ))

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
                    domain_spans_oriC, domain_spans_terC
                    )

                # Append to existing array of new segment attributes
                all_new_boundary_molecule_indexes = np.vstack((
                    all_new_boundary_molecule_indexes,
                    new_segment_attrs["boundary_molecule_indexes"]))
                all_new_boundary_coordinates = np.vstack((
                    all_new_boundary_coordinates,
                    new_segment_attrs["boundary_coordinates"]))
                all_new_segment_domain_indexes = np.concatenate((
                    all_new_segment_domain_indexes,
                    np.full(len(new_segment_attrs["linking_numbers"]), domain_index,
                        dtype=np.int32)))
                all_new_linking_numbers = np.concatenate((
                    all_new_linking_numbers, new_segment_attrs["linking_numbers"]))

            # Delete all existing chromosomal segments
            chromosomal_segments_delete_update = [
                key for key in states['chromosonal_segments'].keys()]
            if chromosomal_segments_delete_update:
                update['chromosomal_segments'] = {
                    '_delete': chromosomal_segments_delete_update}

            # Add new chromosomal segments
            n_segments = len(all_new_linking_numbers)
            new_chromosome_segments = arrays_to(
            n_segments, {
                'unique_index': np.arange(
                    self.chromosome_segment_index, self.chromosome_segment_index +
                    n_segments).astype(str),
                'boundary_molecule_indexes': all_new_boundary_molecule_indexes,
                'boundary_coordinates': all_new_boundary_coordinates,
                'domain_index': all_new_segment_domain_indexes,
                'linking_number': all_new_linking_numbers})
            update['chromosomal_segments'].update(add_elements(
                new_chromosome_segments, 'unique_index'))
            self.chromosome_segment_index += n_segments

        # Get mask for RNAs that are transcribed from removed RNAPs
        removed_RNAs_mask = np.isin(
            RNA_RNAP_indexes, RNAP_unique_indexes[removed_RNAPs_mask].astype(int))

        # Remove RNAPs and RNAs that have collided with replisomes
        if n_total_collisions > 0:
            active_RNAP_delete_update = [
                key for index, key in enumerate(states['active_RNAPs'].keys())
                if removed_RNAPs_mask[index]]
            if active_RNAP_delete_update:
                update['active_RNAPs'] = {
                    '_delete': active_RNAP_delete_update}

            RNA_delete_update = [
                key for index, key in enumerate(states['RNAs'].keys())
                if removed_RNAs_mask[index]]
            if RNA_delete_update:
                update['RNAs'] = {'_delete': RNA_delete_update}

            # Increment counts of inactive RNAPs
            update['molecules'][self.inactive_RNAPs] = n_total_collisions

            # Get sequences of incomplete transcripts
            incomplete_sequence_lengths = transcript_lengths[
                removed_RNAs_mask]
            n_initiated_sequences = np.count_nonzero(incomplete_sequence_lengths)

            if n_initiated_sequences > 0:
                incomplete_sequences = buildSequences(
                    self.RNA_sequences,
                    RNA_TU_indexes[removed_RNAs_mask],
                    np.zeros(n_total_collisions, dtype=np.int64),
                    np.full(n_total_collisions, incomplete_sequence_lengths.max()))

                base_counts = np.zeros(self.n_fragment_bases, dtype=np.int64)

                for sl, seq in zip(incomplete_sequence_lengths, incomplete_sequences):
                    base_counts += np.bincount(seq[:sl], minlength=self.n_fragment_bases)

                # Increment counts of fragment NTPs and phosphates
                update['fragmentBases'] = array_to(
                    self.fragmentBases, base_counts)
                update['molecules'] = {self.ppi: n_initiated_sequences}

        # Get mask for ribosomes that are bound to nonexisting mRNAs
        remaining_RNA_unique_indexes = RNA_unique_indexes[
            np.logical_not(removed_RNAs_mask)]
        removed_ribosomes_mask = np.logical_not(np.isin(
            ribosome_mRNA_indexes, remaining_RNA_unique_indexes.astype(int)))
        n_removed_ribosomes = np.count_nonzero(removed_ribosomes_mask)

        # Remove ribosomes that are bound to removed mRNA molecules
        if n_removed_ribosomes > 0:
            active_ribosome_delete_update = [
                key for index, key in enumerate(states['active_ribosome'].keys())
                if removed_ribosomes_mask[index]]
            if active_ribosome_delete_update:
                update['active_ribosome'] = {
                    '_delete': active_ribosome_delete_update}

            # Increment counts of inactive ribosomal subunits
            update['subunits'] = {
                self.ribosome_30S_subunit: n_removed_ribosomes,
                self.ribosome_50S_subunit: n_removed_ribosomes}

            # Get amino acid sequences of incomplete polypeptides
            incomplete_sequence_lengths = ribosome_peptide_lengths[
                removed_ribosomes_mask]
            n_initiated_sequences = np.count_nonzero(incomplete_sequence_lengths)

            if n_initiated_sequences > 0:
                incomplete_sequences = buildSequences(
                    self.protein_sequences,
                    ribosome_protein_indexes[removed_ribosomes_mask],
                    np.zeros(n_removed_ribosomes, dtype=np.int64),
                    np.full(n_removed_ribosomes, incomplete_sequence_lengths.max()))

                amino_acid_counts = np.zeros(
                    self.n_amino_acids, dtype=np.int64)

                for sl, seq in zip(incomplete_sequence_lengths, incomplete_sequences):
                    amino_acid_counts += np.bincount(
                        seq[:sl], minlength=self.n_amino_acids)

                # Increment counts of free amino acids and decrease counts of
                # free water molecules
                update['amino_acids'] = array_to(self.amino_acids, amino_acid_counts)
                update['molecules'][self.water] = (
                    n_initiated_sequences - incomplete_sequence_lengths.sum())

        # Write to listener
        update['listeners']['RnapData']['n_removed_ribosomes'] = n_removed_ribosomes


        def get_replicated_motif_attributes(old_coordinates, old_domain_indexes):
            """
            Computes the attributes of replicated motifs on the chromosome,
            given the old coordinates and domain indexes of the original motifs.
            """
            # Coordinates are simply repeated
            new_coordinates = np.repeat(old_coordinates, 2)

            # Domain indexes are set to the child indexes of the original index
            new_domain_indexes = child_domains[
                np.array([np.where(all_chromosome_domain_indexes == idx)[0][0]
                    for idx in old_domain_indexes]),
                :].flatten()

            return new_coordinates, new_domain_indexes


        #######################
        # Replicate promoters #
        #######################
        n_new_promoters = 2*np.count_nonzero(removed_promoters_mask)

        if n_new_promoters > 0:
            # Delete original promoters
            promoter_delete_update = [
                key for index, key in enumerate(states['promoters'].keys())
                if removed_promoters_mask[index]]
            if promoter_delete_update:
                update['promoters'] = {'_delete': promoter_delete_update}

            # Add freed active tfs
            update['active_tfs'] = array_to(
                self.active_tfs, promoter_bound_TFs[removed_promoters_mask, :].sum(axis=0))

            # Set up attributes for the replicated promoters
            promoter_TU_indexes_new = np.repeat(promoter_TU_indexes[removed_promoters_mask], 2)
            promoter_coordinates_new, promoter_domain_indexes_new = get_replicated_motif_attributes(
                promoter_coordinates[removed_promoters_mask],
                promoter_domain_indexes[removed_promoters_mask])

            # Add new promoters with new domain indexes
            new_promoters = arrays_to(
                n_new_promoters, {
                    'unique_index': np.array(create_unique_indexes(n_new_promoters)),
                    'TU_index': promoter_TU_indexes_new,
                    'coordinates': promoter_coordinates_new,
                    'domain_index': promoter_domain_indexes_new,
                    'bound_TF': np.zeros((n_new_promoters, self.n_TFs), dtype=np.bool).tolist()})
            update['promoters'].update(add_elements(
                new_promoters, 'unique_index'))
            self.promoter_index += n_new_promoters


        ########################
        # Replicate DnaA boxes #
        ########################
        n_new_DnaA_boxes = 2*np.count_nonzero(removed_DnaA_boxes_mask)

        if n_new_DnaA_boxes > 0:
            # Delete original DnaA boxes
            DnaA_box_delete_update = [
                key for index, key in enumerate(states['DnaA_boxes'].keys())
                if removed_DnaA_boxes_mask[index]]
            if DnaA_box_delete_update:
                update['DnaA_boxes'] = {'_delete': DnaA_box_delete_update}

            # Set up attributes for the replicated boxes
            DnaA_box_coordinates_new, DnaA_box_domain_indexes_new = get_replicated_motif_attributes(
                DnaA_box_coordinates[removed_DnaA_boxes_mask],
                DnaA_box_domain_indexes[removed_DnaA_boxes_mask])

            # Add new promoters with new domain indexes
            new_DnaA_boxes = arrays_to(
                n_new_DnaA_boxes, {
                    'unique_index': np.array(create_unique_indexes(n_new_DnaA_boxes)),
                    'coordinates': DnaA_box_coordinates_new,
                    'domain_index': DnaA_box_domain_indexes_new,
                    'DnaA_bound': np.zeros(n_new_DnaA_boxes, dtype=np.bool).tolist()})
            update['DnaA_boxes'].update(add_elements(
                new_DnaA_boxes, 'unique_index'))
            self.DnaA_box_index += n_new_DnaA_boxes

        return update


    def _compute_new_segment_attributes(self, old_boundary_molecule_indexes,
            old_boundary_coordinates, old_linking_numbers,
            new_molecule_indexes, new_molecule_coordinates,
            spans_oriC, spans_terC):
        # type: (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool, bool) -> dict
        """
        Calculates the updated attributes of chromosomal segments belonging to
        a specific chromosomal domain, given the previous and current
        coordinates of molecules bound to the chromosome.
        Args:
            old_boundary_molecule_indexes (np.ndarray, (N, 2)): Array of unique
                indexes of molecules that formed the boundaries of each
                chromosomal segment in the previous timestep.
            old_boundary_coordinates (np.ndarray, (N, 2)): Array of chromosomal
                 coordinates of molecules that formed the boundaries of each
                 chromosomal segment in the previous timestep.
            old_linking_numbers (np.ndarray, (N,)): Linking numbers of each
                chromosomal segment in the previous timestep.
            new_molecule_indexes (np.ndarray, (N,)): Unique indexes of all
                molecules bound to the domain at the current timestep.
            new_molecule_coordinates (np.ndarray, (N,)): Chromosomal
                coordinates of all molecules bound to the domain at the current
                timestep.
            spans_oriC (bool): True if the domain spans the origin.
            spans_terC (bool): True if the domain spans the terminus.
        Returns (wrapped as dict):
            boundary_molecule_indexes (np.ndarray, (M, 2)): Array of unique
                indexes of molecules that form the boundaries of new
                chromosomal segments.
            boundary_coordinates (np.ndarray, (M, 2)): Array of chromosomal
                coordinates of molecules that form the boundaries of new
                chromosomal segments.
            linking_numbers (np.ndarray, (M,)): Linking numbers of new
                chromosomal segments.
        """
        # Sort old segment arrays by coordinates of left boundary
        old_coordinates_argsort = np.argsort(old_boundary_coordinates[:, 0])
        old_boundary_coordinates_sorted = old_boundary_coordinates[old_coordinates_argsort, :]
        old_boundary_molecule_indexes_sorted = old_boundary_molecule_indexes[old_coordinates_argsort, :]
        old_linking_numbers_sorted = old_linking_numbers[old_coordinates_argsort]

        # Sort new segment arrays by molecular coordinates
        new_coordinates_argsort = np.argsort(new_molecule_coordinates)
        new_molecule_coordinates_sorted = new_molecule_coordinates[new_coordinates_argsort]
        new_molecule_indexes_sorted = new_molecule_indexes[new_coordinates_argsort]

        # Domain does not span the origin
        if not spans_oriC:
            # A fragment spans oriC if two boundaries have opposite signs,
            # or both are equal to zero
            oriC_fragment_counts = np.count_nonzero(
                np.logical_not(np.logical_xor(
                    old_boundary_coordinates_sorted[:, 0] < 0,
                    old_boundary_coordinates_sorted[:, 1] > 0
                    )))

            # if oriC fragment did not exist in the domain in the previous
            # timestep, add a dummy fragment that covers the origin with
            # linking number zero. This is done to generalize the
            # implementation of this method.
            if oriC_fragment_counts == 0:
                # Index of first segment where left boundary is nonnegative
                oriC_fragment_index = np.argmax(
                    old_boundary_coordinates_sorted[:, 0] >= 0)

                # Get indexes of boundary molecules for this dummy segment
                oriC_fragment_boundary_molecule_indexes = np.array([
                    old_boundary_molecule_indexes_sorted[oriC_fragment_index - 1, 1],
                    old_boundary_molecule_indexes_sorted[oriC_fragment_index, 0]
                    ])

                # Insert dummy segment to array
                old_boundary_molecule_indexes_sorted = np.insert(
                    old_boundary_molecule_indexes_sorted,
                    oriC_fragment_index, oriC_fragment_boundary_molecule_indexes,
                    axis=0)
                old_linking_numbers_sorted = np.insert(
                    old_linking_numbers_sorted,
                    oriC_fragment_index, 0)
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
                [self.min_coordinates, self.max_coordinates]
                )

            new_molecule_indexes_sorted = np.insert(
                new_molecule_indexes_sorted,
                [0, len(new_molecule_indexes_sorted)], self.terC_index
                )

            # Add dummy molecule to old segments if they do not already exist
            if old_boundary_molecule_indexes_sorted[0, 0] != self.terC_index:
                old_boundary_molecule_indexes_sorted = np.vstack((
                    np.array([self.terC_index, old_boundary_molecule_indexes_sorted[0, 0]]),
                    old_boundary_molecule_indexes_sorted,
                    np.array([old_boundary_molecule_indexes_sorted[-1, 1], self.terC_index])
                    ))
                old_linking_numbers_sorted = np.insert(
                    old_linking_numbers_sorted,
                    [0, len(old_linking_numbers_sorted)], 0
                    )

        # Recalculate linking numbers of each segment after accounting for
        # boundary molecules that were removed in the current timestep
        linking_numbers_after_removal = []
        right_boundaries_retained = np.isin(
            old_boundary_molecule_indexes_sorted[:, 1],
            new_molecule_indexes_sorted)

        # Add up linking numbers of each segment until each retained boundary
        ln_this_fragment = 0.
        for retained, ln in zip(right_boundaries_retained, old_linking_numbers_sorted):
            ln_this_fragment += ln

            if retained:
                linking_numbers_after_removal.append(ln_this_fragment)
                ln_this_fragment = 0.

        # Number of segments should be equal to number of retained boundaries
        assert len(linking_numbers_after_removal) == right_boundaries_retained.sum()

        # Redistribute linking numbers of the two terC segments such that the
        # segments have same superhelical densities
        if spans_terC and np.count_nonzero(right_boundaries_retained) > 1:
            # Get molecule indexes of the boundaries of the two terC segments
            # left and right of terC
            retained_boundary_indexes = np.where(right_boundaries_retained)[0]
            left_segment_boundary_index = old_boundary_molecule_indexes_sorted[retained_boundary_indexes[0], 1]
            right_segment_boundary_index = old_boundary_molecule_indexes_sorted[retained_boundary_indexes[-2], 1]

            # Get mapping from molecule index to chromosomal coordinates
            molecule_index_to_coordinates = {
                index: coordinates for index, coordinates
                in zip(new_molecule_indexes_sorted, new_molecule_coordinates_sorted)
                }

            # Distribute linking number between two segments proportional to
            # the length of each segment
            left_segment_length = molecule_index_to_coordinates[left_segment_boundary_index] - self.min_coordinates
            right_segment_length = self.max_coordinates - molecule_index_to_coordinates[right_segment_boundary_index]
            full_segment_length = left_segment_length + right_segment_length
            full_linking_number = linking_numbers_after_removal[0] + linking_numbers_after_removal[-1]

            linking_numbers_after_removal[0] = full_linking_number * left_segment_length/full_segment_length
            linking_numbers_after_removal[-1] = full_linking_number * right_segment_length/full_segment_length

        # Get mask for molecules that already existed in the previous timestep
        existing_molecules_mask = np.isin(
            new_molecule_indexes_sorted, old_boundary_molecule_indexes_sorted)

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
                total_length = segment_lengths[i:i + size].sum()
                new_linking_numbers.extend(
                    list(ln*segment_lengths[i:i + size]/total_length)
                    )
            i += size

        # Handle edge case where a domain was just initialized, and two
        # replisomes are bound to the origin
        if len(new_linking_numbers) == 0:
            new_linking_numbers = [0]

        # Build Mx2 array for boundary indexes and coordinates
        new_boundary_molecule_indexes = np.hstack((
            new_molecule_indexes_sorted[:-1, np.newaxis],
            new_molecule_indexes_sorted[1:, np.newaxis]
            ))
        new_boundary_coordinates = np.hstack((
            new_molecule_coordinates_sorted[:-1, np.newaxis],
            new_molecule_coordinates_sorted[1:, np.newaxis]
            ))
        new_linking_numbers = np.array(new_linking_numbers)

        # If domain does not span oriC, remove new segment that spans origin
        if not spans_oriC:
            oriC_fragment_mask = np.logical_not(np.logical_xor(
                new_boundary_coordinates[:, 0] < 0,
                new_boundary_coordinates[:, 1] > 0
                ))

            assert oriC_fragment_mask.sum() == 1

            new_boundary_molecule_indexes = new_boundary_molecule_indexes[
                np.logical_not(oriC_fragment_mask), :]
            new_boundary_coordinates = new_boundary_coordinates[
                np.logical_not(oriC_fragment_mask), :]
            new_linking_numbers = new_linking_numbers[
                np.logical_not(oriC_fragment_mask)]

        return {
            "boundary_molecule_indexes": new_boundary_molecule_indexes,
            "boundary_coordinates": new_boundary_coordinates,
            "linking_numbers": new_linking_numbers
            }

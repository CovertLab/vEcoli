"""
SimulationData for replication process
"""

import numpy as np
import collections

from wholecell.utils import units
from wholecell.utils.polymerize import polymerize
from wholecell.utils.random import stochasticRound


# Maximum allowed simulation time step. Used to create buffer for sequences.
MAX_TIMESTEP = 2


class SiteNotFoundError(Exception):
    pass


class Replication(object):
    """
    SimulationData for the replication process
    """

    def __init__(self, raw_data, sim_data):
        self._n_nt_types = len(sim_data.dntp_code_to_id_ordered)

        self._build_sequence(raw_data, sim_data)
        self._build_gene_data(raw_data, sim_data)
        self._build_sites(raw_data, sim_data)
        self._build_replication(raw_data, sim_data)
        self._build_motifs(raw_data, sim_data)
        self._build_elongation_rates(raw_data, sim_data)

        self.c_period = sim_data.constants.c_period
        self.d_period = sim_data.constants.d_period
        self.c_period_in_mins = self.c_period.asNumber(units.min)
        self.d_period_in_mins = self.d_period.asNumber(units.min)

    def _build_sequence(self, raw_data, sim_data):
        self.genome_sequence = raw_data.genome_sequence
        self.genome_sequence_rc = self.genome_sequence.reverse_complement()
        self.genome_length = len(self.genome_sequence)
        self.genome_A_count = self.genome_sequence.count("A")
        self.genome_T_count = self.genome_sequence.count("T")
        self.genome_G_count = self.genome_sequence.count("G")
        self.genome_C_count = self.genome_sequence.count("C")

    def _build_gene_data(self, raw_data, sim_data):
        """
        Build gene-associated simulation data from raw data.
        """

        def extract_data(raw, key, use_first_from_list=False):
            if use_first_from_list:
                data = [row[key][0] for row in raw]
            else:
                data = [row[key] for row in raw]
            dtype = "U{}".format(max(len(d) for d in data if d is not None))
            return data, dtype

        names, name_dtype = extract_data(raw_data.genes, "id")
        symbols, symbol_dtype = extract_data(raw_data.genes, "symbol")
        cistron_ids, cistron_dtype = extract_data(
            raw_data.genes, "rna_ids", use_first_from_list=True
        )

        self.gene_data = np.zeros(
            len(raw_data.genes),
            dtype=[
                ("name", name_dtype),
                ("symbol", symbol_dtype),
                ("cistron_id", cistron_dtype),
            ],
        )

        self.gene_data["name"] = names
        self.gene_data["symbol"] = symbols
        self.gene_data["cistron_id"] = cistron_ids

    def _build_sites(self, raw_data, sim_data):
        """
        Build simulation data associated with DNA sites from raw_data.
        """

        def get_site_center_coordinates(site_id):
            """
            Calculate the center coordinates (rounded average of left and right
            end coordinates) of a given DNA site.
            """
            try:
                left, right = sim_data.getter.get_genomic_coordinates(site_id)
                center_coordinate = round((left + right) / 2)
            except (KeyError, TypeError):
                raise SiteNotFoundError(
                    f"Coordinates of DNA site with ID {site_id} were not found in raw_data."
                )

            return center_coordinate

        # Get coordinates of oriC and terC
        self.oric_coordinate = get_site_center_coordinates(
            sim_data.molecule_ids.oriC_site
        )
        self.terc_coordinate = get_site_center_coordinates(
            sim_data.molecule_ids.terC_site
        )

    def _build_replication(self, raw_data, sim_data):
        """
        Build replication-associated simulation data from raw data.
        """
        # Map ATGC to 8 bit integers
        numerical_sequence = np.empty(self.genome_length, np.int8)
        ntMapping = collections.OrderedDict(
            [
                (ntp_code, i)
                for i, ntp_code in enumerate(sim_data.dntp_code_to_id_ordered.keys())
            ]
        )
        for i, letter in enumerate(raw_data.genome_sequence):
            numerical_sequence[i] = ntMapping[
                letter
            ]  # Build genome sequence as small integers

        # Create 4 possible polymerization sequences
        # Forward sequence includes oriC
        self.forward_sequence = numerical_sequence[
            np.hstack(
                (
                    np.arange(self.oric_coordinate, self.genome_length),
                    np.arange(0, self.terc_coordinate),
                )
            )
        ]

        # Reverse sequence includes terC
        self.reverse_sequence = numerical_sequence[
            np.arange(self.oric_coordinate - 1, self.terc_coordinate - 1, -1)
        ]

        self.forward_complement_sequence = self._get_complement_sequence(
            self.forward_sequence
        )
        self.reverse_complement_sequence = self._get_complement_sequence(
            self.reverse_sequence
        )

        assert (
            self.forward_sequence.size + self.reverse_sequence.size
            == self.genome_length
        )

        # Log array of lengths of each replichore
        self.replichore_lengths = np.array(
            [
                self.forward_sequence.size,
                self.reverse_sequence.size,
            ]
        )

        # Determine size of the matrix used by polymerize function
        maxLen = np.int64(
            self.replichore_lengths.max()
            # Add buffer to account for elongation by max timestep
            + MAX_TIMESTEP
            * sim_data.constants.replisome_elongation_rate.asNumber(units.nt / units.s)
        )

        self.replication_sequences = np.empty((4, maxLen), np.int8)
        self.replication_sequences.fill(polymerize.PAD_VALUE)

        self.replication_sequences[0, : self.forward_sequence.size] = (
            self.forward_sequence
        )
        self.replication_sequences[1, : self.forward_complement_sequence.size] = (
            self.forward_complement_sequence
        )
        self.replication_sequences[2, : self.reverse_sequence.size] = (
            self.reverse_sequence
        )
        self.replication_sequences[3, : self.reverse_complement_sequence.size] = (
            self.reverse_complement_sequence
        )

        # Get polymerized nucleotide weights
        self.replication_monomer_weights = (
            sim_data.getter.get_masses(sim_data.molecule_groups.dntps)
            - sim_data.getter.get_masses([sim_data.molecule_ids.ppi])
        ) / sim_data.constants.n_avogadro

        # Placeholder value for "child_domains" attribute of domains without
        # children domains
        self.no_child_place_holder = -1

    def _build_motifs(self, raw_data, sim_data):
        """
        Build simulation data associated with sequence motifs from raw_data.
        Coordinates of all motifs are calculated based on the given sequences
        of the genome and the motifs.
        """
        # Initialize dictionary of motif coordinates
        self.motif_coordinates = dict()

        for motif in raw_data.sequence_motifs:
            # Keys are the IDs of the motifs; Values are arrays of the motif's
            # coordinates.
            self.motif_coordinates[motif["id"]] = self._get_motif_coordinates(
                motif["length"], motif["sequences"]
            )

    def _get_complement_sequence(self, sequenceVector):
        """
        Calculates the vector for a complement sequence of a DNA sequence given
        in vector form.
        """
        return (self._n_nt_types - 1) - sequenceVector

    def _get_motif_coordinates(self, motif_length, motif_sequences):
        """
        Finds the coordinates of all sequence motifs of a specific type. The
        coordinates are given as the positions of the midpoint of the motif
        relative to the oriC in base pairs.
        """
        # Append the first n bases to the end of the sequences to account for
        # motifs that span the two endpoints
        extended_sequence = (
            self.genome_sequence + self.genome_sequence[: (motif_length - 1)]
        )
        extended_rc_sequence = (
            self.genome_sequence_rc + self.genome_sequence_rc[: (motif_length - 1)]
        )

        # Initialize list for coordinates of motifs
        coordinates = []

        # Loop through all possible motif sequences
        for sequence in motif_sequences:
            # Find occurrences of the motif in the original sequence
            loc = extended_sequence.find(sequence)  # Returns -1 if not found

            while loc != -1:
                coordinates.append(loc + motif_length // 2)
                loc = extended_sequence.find(sequence, loc + 1)

            # Find occurrences of the motif in the reverse complement
            loc = extended_rc_sequence.find(sequence)

            while loc != -1:
                coordinates.append(
                    self.genome_length - loc - motif_length + motif_length // 2
                )
                loc = extended_rc_sequence.find(sequence, loc + 1)

        # Compute coordinates relative to oriC
        motif_coordinates = self._get_relative_coordinates(np.array(coordinates))
        motif_coordinates.sort()

        return motif_coordinates

    def _get_relative_coordinates(self, coordinates):
        """
        Converts an array of genomic coordinates into coordinates relative to
        the origin of replication.
        """
        relative_coordinates = (
            (coordinates - self.terc_coordinate) % self.genome_length
            + self.terc_coordinate
            - self.oric_coordinate
        )

        relative_coordinates[relative_coordinates < 0] += 1

        return relative_coordinates

    def _build_elongation_rates(self, raw_data, sim_data):
        self.basal_elongation_rate = int(
            round(
                sim_data.constants.replisome_elongation_rate.asNumber(
                    units.nt / units.s
                )
            )
        )

    def make_elongation_rates(self, random, replisomes, base, time_step):
        rates = np.full(
            replisomes, stochasticRound(random, base * time_step), dtype=np.int64
        )

        return rates

    def get_average_copy_number(self, tau, coords):
        """
        Calculates the average copy number of a gene throughout the cell cycle
        given the location of the gene in coordinates.

        Args:
                tau (float): expected doubling time in minutes
                coords (int or ndarray[int]): chromosome coordinates of genes

        Returns:
                float or ndarray[float] (matches length of coords): average copy
                number of each gene expected at a doubling time, tau
        """

        right_replichore_length = self.replichore_lengths[0]
        left_replichore_length = self.replichore_lengths[1]

        # Calculate the relative position of the gene along the chromosome
        # from its coordinate
        relative_pos = np.array(coords, float)
        relative_pos[coords > 0] = relative_pos[coords > 0] / right_replichore_length
        relative_pos[coords < 0] = -relative_pos[coords < 0] / left_replichore_length

        # Return the predicted average copy number
        n_avg_copy = 2 ** (
            ((1 - relative_pos) * self.c_period_in_mins + self.d_period_in_mins) / tau
        )
        return n_avg_copy

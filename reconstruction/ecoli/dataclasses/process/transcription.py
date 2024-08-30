"""
SimulationData for transcription process

TODO: add mapping of tRNA to charged tRNA if allowing more than one modified form of tRNA and separate mappings for tRNA and charged tRNA to AA
TODO: handle ppGpp and DksA-ppGpp regulation separately
"""

from functools import cache
from typing import cast
import copy

import numpy as np

from scipy.sparse import csr_matrix
import sympy as sp

from reconstruction.ecoli.dataclasses.getter_functions import EXCLUDED_RNA_TYPES
from ecoli.library.sim_data import MAX_TIME_STEP
from ecoli.library.schema import bulk_name_to_idx, counts
from wholecell.utils import data, fitting, units
from wholecell.utils.fast_nonnegative_least_squares import fast_nnls
from wholecell.utils.fitting import normalize
from wholecell.utils.unit_struct_array import UnitStructArray
from wholecell.utils.polymerize import polymerize
from wholecell.utils.random import make_elongation_rates


PROCESS_MAX_TIME_STEP = 2.0
RNA_SEQ_ANALYSIS = "rsem_tpm"
PPGPP_CONC_UNITS = units.umol / units.L
PRINT_VALUES = False  # print values for supplemental table if True


class TranscriptionDirectionError(Exception):
    pass


class Transcription(object):
    """
    SimulationData for the transcription process
    """

    def __init__(self, raw_data, sim_data):
        self.max_time_step = min(MAX_TIME_STEP, PROCESS_MAX_TIME_STEP)

        self._build_oric_terc_coordinates(raw_data, sim_data)
        self._build_cistron_data(raw_data, sim_data)
        self._build_rna_data(raw_data, sim_data)
        self._build_mature_rna_data(raw_data, sim_data)
        self._build_transcription(raw_data, sim_data)
        self._build_charged_trna(raw_data, sim_data)
        self._build_attenuation(raw_data, sim_data)
        self._build_elongation_rates(raw_data, sim_data)
        self._build_new_gene_data(raw_data, sim_data)


    def __getstate__(self):
        """Return the state to pickle with transcriptionSequences removed and
        only storing data from transcriptionSequences with pad values stripped.
        """

        state = data.dissoc_strict(self.__dict__, ("transcription_sequences",))
        state["sequences"] = np.array(
            [seq[seq != polymerize.PAD_VALUE] for seq in self.transcription_sequences],
            dtype=object,
        )
        state["sequence_shape"] = self.transcription_sequences.shape
        return state

    def __setstate__(self, state):
        """Restore transcriptionSequences and remove processed versions of the data."""
        sequences = state.pop("sequences")
        sequence_shape = state.pop("sequence_shape")
        self.__dict__.update(state)

        self.transcription_sequences = np.full(
            sequence_shape, polymerize.PAD_VALUE, dtype=np.int8
        )
        for i, seq in enumerate(sequences):
            self.transcription_sequences[i, : len(seq)] = seq

    def _build_ppgpp_regulation(self, raw_data, sim_data):
        """
        Determine which genes are regulated by ppGpp and store the fold
        change in expression associated with each gene.

        Attributes set:
                ppgpp_regulated_genes (ndarray[str]): cistron ID of regulated genes
                ppgpp_fold_changes (ndarray[float]): log2 fold change for each gene
                        in ppgpp_regulated_genes
                _ppgpp_unset_mask (list[bool]): flag for genes with unset ppGpp fold changes
                        that are mainly rRNAs, tRNAs, etc. that have synthesis affinity
                        ratios directly set during ppGpp K_m fitting
                _ppgpp_growth_parameters: parameters for interpolate.splev
                        to estimate growth rate from ppGpp concentration
        """

        def read_value(d, k):
            """Handle empty values from raw_data as 0"""
            val = d[k]
            return 0 if val == "" else val

        # Flag to check when ppGpp regulated expression has been set
        # see set_ppgpp_expression()
        self._ppgpp_expression_set = False

        # Read regulation data from raw_data
        # Treats ppGpp and DksA-ppGpp regulation the same
        gene_to_rna = {g["symbol"]: g["rna_ids"][0] for g in raw_data.genes}
        rna_id_to_rna_type = {r["id"]: r["type"] for r in raw_data.rnas}
        regulation = {}
        for reg in raw_data.ppgpp_regulation:
            # Convert to regulated RNA
            gene = reg["Gene"]
            rna = gene_to_rna.get(gene, None)
            if rna is None or rna_id_to_rna_type[rna] in EXCLUDED_RNA_TYPES:
                continue

            # Add additional gene symbols for matching FC data
            curated_gene = reg["Curated Gene"]
            if gene != curated_gene:
                gene_to_rna[curated_gene] = rna

            # Update value (some genes are repeated in raw_data))
            direction = read_value(reg, "ppGpp") + read_value(reg, "DksA-ppGpp")
            regulation[rna] = regulation.get(rna, 0) + direction

        # Read fold change data from raw_data
        ## Categories A-D are statistically significant fold changes
        ## Categories E-G are not significant or data is not usable
        valid_categories = {"A", "B", "C", "D"}
        sample_time = (
            5  # Could also be 10 (5 min minimizes downstream regulation impacts)
        )
        sample_id = "1+2+ {} min".format(
            sample_time
        )  # Column contains FC data for given time
        rna_fold_changes = {}
        for fc in raw_data.ppgpp_fc:
            # Convert to regulated RNA
            gene = fc["Gene"]
            rna = gene_to_rna.get(gene, None)
            if rna is None:
                continue

            category = fc["{} Category".format(sample_id)]
            if category not in valid_categories:
                continue
            rna_fold_changes[rna] = fc[sample_id]

        # Store arrays of regulation
        regulated_genes = []
        regulation_direction = []
        fold_changes = []
        for rna in sorted(regulation):
            reg_dir = regulation[rna]
            fc_dir = rna_fold_changes.get(rna, 0)

            # Ignore inconsistent regulatory directions
            if reg_dir == 0:
                continue

            # Use default value if annotated direction does not match data direction
            if reg_dir * fc_dir < 0:
                fc_dir = 0

            regulated_genes.append(rna)
            regulation_direction.append(np.sign(reg_dir))
            fold_changes.append(fc_dir)
        self.ppgpp_regulated_genes = np.array(regulated_genes)
        regulation_direction = np.array(regulation_direction)

        # Replace fold changes without data with the average
        fold_changes = np.array(fold_changes)
        average_positive_fc = fold_changes[fold_changes > 0].mean()
        fold_changes[(fold_changes == 0) & (regulation_direction > 0)] = (
            average_positive_fc
        )
        unset_mask = (fold_changes == 0) & (regulation_direction < 0)
        self._ppgpp_unset_mask = unset_mask
        self.ppgpp_fold_changes = fold_changes

        # Predict growth rate from ppGpp level
        # Transforms selected for good fit and to keep the growth rate positive
        # even at high ppGpp concentrations.
        per_dry_mass_to_per_volume = (
            sim_data.constants.cell_density * sim_data.mass.cell_dry_mass_fraction
        )
        ppgpp = np.array(
            [
                (d["ppGpp_conc"] * per_dry_mass_to_per_volume).asNumber(
                    PPGPP_CONC_UNITS
                )
                for d in raw_data.growth_rate_dependent_parameters
            ]
        )
        growth_rates = np.log(2) / np.array(
            [
                d["doublingTime"].asNumber(units.s)
                for d in raw_data.growth_rate_dependent_parameters
            ]
        )
        self._ppgpp_growth_parameters = fitting.fit_linearized_transforms(
            ppgpp, growth_rates, x_fun=["none"], y_fun=["1/sqrt"]
        )

        if PRINT_VALUES:
            print(
                "Supplement value (KM): {:.1f}".format(np.sqrt(self._ppgpp_km_squared))
            )
            print(
                "Supplement value (FC): [{:.2f}, {:.2f}]".format(
                    fold_changes.min(), fold_changes.max()
                )
            )
            #print("Supplement value (FC-): {:.2f}".format(self._fit_ppgpp_fc)) TODO: what to do about this?
            print("Supplement value (FC+): {:.2f}".format(average_positive_fc))


    def _build_oric_terc_coordinates(self, raw_data, sim_data):
        """
        Builds coordinates of oriC and terC that are used when calculating
        genomic positions of cistrons and RNAs relative to the origin
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

    def _build_cistron_data(self, raw_data, sim_data):
        """
        Build cistron-associated simulation data from raw data. Cistrons are
        sections of RNAs that encode for a specific polypeptide. A single RNA
        molecule may contain one or more cistrons.
        """
        # Get list of all cistrons with an associated gene and right and left
        # end positions
        cistron_id_to_gene_id = {
            gene["rna_ids"][0]: gene["id"] for gene in raw_data.genes
        }
        gene_id_to_left_end_pos = {
            gene["id"]: gene["left_end_pos"] for gene in raw_data.genes
        }
        gene_id_to_right_end_pos = {
            gene["id"]: gene["right_end_pos"] for gene in raw_data.genes
        }

        all_cistrons = [
            rna
            for rna in raw_data.rnas
            if rna["id"] in cistron_id_to_gene_id
            and gene_id_to_left_end_pos[cistron_id_to_gene_id[rna["id"]]] is not None
            and gene_id_to_right_end_pos[cistron_id_to_gene_id[rna["id"]]] is not None
            and rna["type"] not in EXCLUDED_RNA_TYPES
        ]
        all_cistron_ids = [cistron["id"] for cistron in all_cistrons]

        # Load gene IDs associated with each cistron
        gene_id = np.array([cistron_id_to_gene_id[rna["id"]] for rna in all_cistrons])
        gene_id_to_cistron_id = {g: c for (c, g) in cistron_id_to_gene_id.items()}

        # Calculate lengths of each cistron from their gene end positions
        cistron_lengths = np.array(
            [
                np.abs(
                    gene_id_to_right_end_pos[cistron_id_to_gene_id[cistron["id"]]]
                    - gene_id_to_left_end_pos[cistron_id_to_gene_id[cistron["id"]]]
                )
                + 1
                for cistron in all_cistrons
            ]
        )

        # Get mapping from cistron IDs to coordinate and direction
        cistron_id_to_coordinate = {}
        cistron_id_to_direction = {}
        for gene in raw_data.genes:
            cistron_id_to_direction[gene["rna_ids"][0]] = gene["direction"]
            if gene["direction"] == "+":
                cistron_id_to_coordinate[gene["rna_ids"][0]] = gene["left_end_pos"]
            else:
                cistron_id_to_coordinate[gene["rna_ids"][0]] = gene["right_end_pos"]

        # Get location of each cistron on the chromosome relative to the origin
        replication_coordinate = [
            self._get_relative_coordinates(cistron_id_to_coordinate[cistron["id"]])
            for cistron in all_cistrons
        ]

        # Get direction of each cistron
        is_forward = [
            cistron_id_to_direction[cistron["id"]] == "+" for cistron in all_cistrons
        ]

        # Get cistron nucleotide compositions
        genome_sequence = raw_data.genome_sequence

        def parse_sequence(cistron_id, left_end_pos, right_end_pos, direction):
            """
            Parses genome sequence to get the sequence of the RNA transcribed
            from the cistron, given left and right end positions and
            transcription direction (Note: the left and right end positions in
            the raw data files are given as 1-indexed coordinates)
            """
            if direction == "+":
                return genome_sequence[left_end_pos - 1 : right_end_pos].transcribe()
            elif direction == "-":
                return (
                    genome_sequence[left_end_pos - 1 : right_end_pos]
                    .reverse_complement()
                    .transcribe()
                )
            else:
                raise TranscriptionDirectionError(
                    f"Unidentified transcription direction given for {cistron_id}"
                )

        rna_seqs = [
            parse_sequence(
                cistron_id,
                gene_id_to_left_end_pos[cistron_id_to_gene_id[cistron_id]],
                gene_id_to_right_end_pos[cistron_id_to_gene_id[cistron_id]],
                cistron_id_to_direction[cistron_id],
            )
            for cistron_id in all_cistron_ids
        ]

        nt_counts = []
        for seq in rna_seqs:
            nt_counts.append(
                [seq.count(letter) for letter in sim_data.ntp_code_to_id_ordered.keys()]
            )
        nt_counts = np.array(nt_counts)

        # Calculate molecular weights of the RNAs corresponding to each cistron
        ppi_mw = sim_data.getter.get_mass(sim_data.molecule_ids.ppi[:-3]).asNumber(
            units.g / units.mol
        )
        polymerized_ntp_mws = np.array(
            [
                sim_data.getter.get_mass(met_id[:-3]).asNumber(units.g / units.mol)
                for met_id in sim_data.molecule_groups.polymerized_ntps
            ]
        )

        mws = nt_counts.dot(polymerized_ntp_mws) + ppi_mw  # Add end weight

        # Get boolean arrays for each RNA type
        is_mRNA = [rna["type"] == "mRNA" for rna in all_cistrons]
        is_miscRNA = [rna["type"] == "miscRNA" for rna in all_cistrons]
        is_rRNA = [rna["type"] == "rRNA" for rna in all_cistrons]
        is_tRNA = [rna["type"] == "tRNA" for rna in all_cistrons]

        # Load set of mRNA cistron ids
        mRNA_cistron_ids = set(np.array(all_cistron_ids)[is_mRNA])

        # Load cistron half lives
        cistron_id_to_half_life = {}
        reported_mRNA_cistron_half_lives = []

        for gene in raw_data.rna_half_lives:
            if (
                gene["id"] in gene_id_to_cistron_id
                and gene["half_life"].asNumber(units.s) > 0
            ):
                cistron_id = gene_id_to_cistron_id[gene["id"]]
                cistron_id_to_half_life[cistron_id] = gene["half_life"]
                if cistron_id in mRNA_cistron_ids:
                    reported_mRNA_cistron_half_lives.append(gene["half_life"])

        # Calculate averaged reported half life of mRNAs
        self.average_mRNA_cistron_half_life = np.mean(reported_mRNA_cistron_half_lives)

        # Half-lives of rRNAs are set to be equal to the average reported half
        # life of mRNAs
        # Note: rRNAs complexed into ribosomal subunits will not degrade, so
        # this will only significantly affect excess rRNAs
        rRNA_cistron_ids = np.array(all_cistron_ids)[is_rRNA]
        for cistron_id in rRNA_cistron_ids:
            cistron_id_to_half_life[cistron_id] = self.average_mRNA_cistron_half_life

        # Half-life of tRNAs are set to the stable RNA half life value defined
        # in sim_data.constants
        tRNA_cistron_ids = np.array(all_cistron_ids)[is_tRNA]
        for cistron_id in tRNA_cistron_ids:
            cistron_id_to_half_life[cistron_id] = (
                sim_data.constants.stable_RNA_half_life
            )

        # Get half life of each RNA cistron - if the half life is not given, use
        # the averaged reported half life of mRNAs
        cistron_half_lives = np.array(
            [
                cistron_id_to_half_life.get(
                    cistron_id, self.average_mRNA_cistron_half_life
                ).asNumber(units.s)
                for cistron_id in all_cistron_ids
            ]
        )

        # Calculate expected first-order degradation rates of each cistron
        cistron_deg_rates = np.log(2) / cistron_half_lives

        # Construct boolean arrays for ribosomal protein and RNAP-encoding
        # cistrons
        n_cistrons = len(all_cistrons)

        is_ribosomal_protein = np.zeros(n_cistrons, dtype=bool)
        is_RNAP = np.zeros(n_cistrons, dtype=bool)
        for i, cistron in enumerate(all_cistrons):
            for monomer_id in cistron["monomer_ids"]:
                if monomer_id + "[c]" in sim_data.molecule_groups.ribosomal_proteins:
                    is_ribosomal_protein[i] = True
                if monomer_id + "[c]" in sim_data.molecule_groups.RNAP_subunits:
                    is_RNAP[i] = True

        # Construct boolean arrays and index arrays for each rRNA type
        is_23S = np.zeros(n_cistrons, dtype=bool)
        is_16S = np.zeros(n_cistrons, dtype=bool)
        is_5S = np.zeros(n_cistrons, dtype=bool)
        idx_23S = []
        idx_16S = []
        idx_5S = []

        for rnaIndex, cistron in enumerate(all_cistrons):
            if cistron["id"] + "[c]" in sim_data.molecule_groups.s50_23s_rRNA:
                is_23S[rnaIndex] = True
                idx_23S.append(rnaIndex)
            if cistron["id"] + "[c]" in sim_data.molecule_groups.s30_16s_rRNA:
                is_16S[rnaIndex] = True
                idx_16S.append(rnaIndex)
            if cistron["id"] + "[c]" in sim_data.molecule_groups.s50_5s_rRNA:
                is_5S[rnaIndex] = True
                idx_5S.append(rnaIndex)

        max_cistron_id_length = max(len(rna["id"]) for rna in all_cistrons)
        max_gene_id_length = max(len(id_) for id_ in gene_id)

        cistron_data = np.zeros(
            n_cistrons,
            dtype=[
                ("id", "U{}".format(max_cistron_id_length)),
                ("gene_id", "U{}".format(max_gene_id_length)),
                ("length", "i8"),
                ("replication_coordinate", "i8"),
                ("is_forward", "bool"),
                ("mw", "f8"),
                ("deg_rate", "f8"),
                ("is_mRNA", "bool"),
                ("is_miscRNA", "bool"),
                ("is_rRNA", "bool"),
                ("is_tRNA", "bool"),
                ("is_23S_rRNA", "bool"),
                ("is_16S_rRNA", "bool"),
                ("is_5S_rRNA", "bool"),
                ("is_ribosomal_protein", "bool"),
                ("is_RNAP", "bool"),
                ("uses_corrected_seq_counts", "bool"),
                ("is_new_gene", "bool"),
            ],
        )

        cistron_data["id"] = [rna["id"] for rna in all_cistrons]
        cistron_data["gene_id"] = gene_id
        cistron_data["length"] = cistron_lengths
        cistron_data["replication_coordinate"] = replication_coordinate
        cistron_data["is_forward"] = is_forward
        cistron_data["mw"] = mws
        cistron_data["deg_rate"] = cistron_deg_rates
        cistron_data["is_mRNA"] = is_mRNA
        cistron_data["is_miscRNA"] = is_miscRNA
        cistron_data["is_rRNA"] = is_rRNA
        cistron_data["is_tRNA"] = is_tRNA
        cistron_data["is_23S_rRNA"] = is_23S
        cistron_data["is_16S_rRNA"] = is_16S
        cistron_data["is_5S_rRNA"] = is_5S
        cistron_data["is_ribosomal_protein"] = is_ribosomal_protein
        cistron_data["is_RNAP"] = is_RNAP
        cistron_data["uses_corrected_seq_counts"] = np.zeros(n_cistrons, dtype=bool)
        cistron_data["is_new_gene"] = [k.startswith("NG") for k in gene_id]

        cistron_field_units = {
            "id": None,
            "gene_id": None,
            "length": units.nt,
            "replication_coordinate": None,
            "is_forward": None,
            "mw": units.g / units.mol,
            "deg_rate": 1 / units.s,
            "is_mRNA": None,
            "is_miscRNA": None,
            "is_rRNA": None,
            "is_tRNA": None,
            "is_23S_rRNA": None,
            "is_16S_rRNA": None,
            "is_5S_rRNA": None,
            "is_ribosomal_protein": None,
            "is_RNAP": None,
            "uses_corrected_seq_counts": None,
            "is_new_gene": None,
        }

        self.cistron_data = UnitStructArray(cistron_data, cistron_field_units)
        self._cistron_id_to_index = {
            cistron_id: i for (i, cistron_id) in enumerate(self.cistron_data["id"])
        }

        # Load expression levels of individual cistrons from sequencing data
        cistron_expression = []
        cistron_id_to_gene_id = {
            gene["rna_ids"][0]: gene["id"] for gene in raw_data.genes
        }
        seq_data = {
            x["Gene"]: x[sim_data.basal_expression_condition]
            for x in getattr(raw_data.rna_seq_data, f"rnaseq_{RNA_SEQ_ANALYSIS}_mean")
        }

        cistron_rnaseq_coverage = []
        for cistron_id in self.cistron_data["id"]:
            gene_id = cistron_id_to_gene_id[cistron_id]
            # If sequencing data is not found, initialize expression to zero.
            cistron_expression.append(seq_data.get(gene_id, 0.0))
            cistron_rnaseq_coverage.append(gene_id in seq_data)

        cistron_expression = np.array(cistron_expression)
        self._cistron_is_rnaseq_covered = np.array(cistron_rnaseq_coverage)

        # Set basal expression levels of each cistron - conditional values are
        # set in the parca.
        self.cistron_expression = {}
        self.cistron_expression["basal"] = cistron_expression / cistron_expression.sum()

        # Initialize dictionary for fitted cistron expression levels. Values for
        # this dictionary are set in the parca.
        self.fit_cistron_expression = {}

    def _build_rna_data(self, raw_data, sim_data):
        """
        Build RNA-associated simulation data from raw data.
        """
        self._basal_rna_fractions = sim_data.mass.get_basal_rna_fractions()

        # Get list of transcription units used by the model
        all_valid_tus = [
            tu
            for tu in raw_data.transcription_units
            if sim_data.getter.is_valid_molecule(tu["id"])
        ]

        # Get mapping from transcription unit IDs to list of constituent
        # cistrons
        gene_id_to_rna_id = {gene["id"]: gene["rna_ids"][0] for gene in raw_data.genes}
        tu_id_to_cistron_ids = {
            tu["id"]: [
                gene_id_to_rna_id[gene]
                for gene in tu["genes"]
                if gene_id_to_rna_id[gene] in self.cistron_data["id"]
            ]
            for tu in all_valid_tus
        }

        # Get list of cistrons that are covered by one or more TUs
        cistrons_covered_by_tus = []
        for cistrons in tu_id_to_cistron_ids.values():
            cistrons_covered_by_tus.extend(cistrons)
        cistrons_covered_by_tus = set(cistrons_covered_by_tus)

        # Compile IDs of all RNAs that should be directly transcribed in the
        # model, including RNAs for genes that are not covered by any
        # transcription units
        rna_ids = [tu["id"] for tu in all_valid_tus]
        rna_ids.extend(
            [
                cistron_id
                for cistron_id in self.cistron_data["id"]
                if sim_data.getter.is_valid_molecule(cistron_id)
                and cistron_id not in cistrons_covered_by_tus
            ]
        )
        n_rnas = len(rna_ids)

        # Build mapping matrix between transcription units and constituent
        # cistrons
        cistron_indexes = []
        rna_indexes = []
        v = []

        # Mapping from cistron ID to index
        cistron_id_to_index = {
            cistron_id: cistron_index
            for (cistron_index, cistron_id) in enumerate(self.cistron_data["id"])
        }

        for rna_index, rna_id in enumerate(rna_ids):
            if rna_id in tu_id_to_cistron_ids:
                for mc_rna_id in tu_id_to_cistron_ids[rna_id]:
                    cistron_indexes.append(cistron_id_to_index[mc_rna_id])
                    rna_indexes.append(rna_index)
                    v.append(1)
            else:
                cistron_indexes.append(cistron_id_to_index[rna_id])
                rna_indexes.append(rna_index)
                v.append(1)

        cistron_indexes = np.array(cistron_indexes)
        rna_indexes = np.array(rna_indexes)
        v = np.array(v)
        shape = (cistron_indexes.max() + 1, rna_indexes.max() + 1)

        # Build sparse mapping matrix
        self.cistron_tu_mapping_matrix = csr_matrix(
            (v, (cistron_indexes, rna_indexes)), shape=shape
        )

        # Find groups of cistrons and TUs that belong to the same operons.
        # self.operons is a list of tuples that each specify an operon with
        # a list of cistron indexes and a list of RNA indexes
        # (e.g. ([380, 379, 377], [2356, 2433]))
        visited_cistron_indexes = set()
        visited_rna_indexes = set()
        self.operons = []

        def rna_DFS(rna_index, operon_cistron_indexes, operon_rna_indexes):
            """
            Recursive function to look for indexes of RNAs (transcription units)
            and cistrons that belong to the same operon as the RNA with the
            given index.
            """
            visited_rna_indexes.add(rna_index)
            operon_rna_indexes.append(rna_index)

            for i in cistron_indexes[rna_indexes == rna_index]:
                if i not in visited_cistron_indexes:
                    cistron_DFS(i, operon_cistron_indexes, operon_rna_indexes)

        def cistron_DFS(cistron_index, operon_cistron_indexes, operon_rna_indexes):
            """
            Recursive function to look for indexes of RNAs (transcription units)
            and cistrons that belong to the same operon as the cistron with the
            given index.
            """
            visited_cistron_indexes.add(cistron_index)
            operon_cistron_indexes.append(cistron_index)

            for i in rna_indexes[cistron_indexes == cistron_index]:
                if i not in visited_rna_indexes:
                    rna_DFS(i, operon_cistron_indexes, operon_rna_indexes)

        # Loop through each RNA index
        for rna_index in range(n_rnas):
            # Search for cistrons and RNAs that can be grouped together into the
            # same operon
            if rna_index not in visited_rna_indexes:
                operon_cistron_indexes = []
                operon_rna_indexes = []
                rna_DFS(rna_index, operon_cistron_indexes, operon_rna_indexes)

                # Sort cistron indexes by coordinates
                operon_cistron_indexes = sorted(
                    operon_cistron_indexes,
                    key=lambda i: self.cistron_data["replication_coordinate"][i],
                )

                self.operons.append((operon_cistron_indexes, operon_rna_indexes))

        # Build list of all RNA IDs with compartment tags
        compartments = sim_data.getter.get_compartments(rna_ids)
        rna_ids_with_compartments = [
            f"{rna_id}[{loc[0]}]" for (rna_id, loc) in zip(rna_ids, compartments)
        ]

        # Apply RNAseq corrections to shorter genes if required by operon version
        if sim_data.operons_on:
            self._apply_rnaseq_correction()

        expression, _ = self.fit_rna_expression(self.cistron_expression["basal"])

        # TODO (Albert): should modify more when other types of hybrid RNAs
        #  are introduced (only rRNA-tRNA hybrids are included currently)
        # Determine type of each RNA
        is_mRNA = (
            self.cistron_data["is_mRNA"] @ self.cistron_tu_mapping_matrix
        ).astype(bool)
        is_miscRNA = (
            self.cistron_data["is_miscRNA"] @ self.cistron_tu_mapping_matrix
        ).astype(bool)
        is_rRNA = (
            self.cistron_data["is_rRNA"] @ self.cistron_tu_mapping_matrix
        ).astype(bool)
        # All hybrid TUs containing both rRNAs and tRNAs are assumed to be rRNAs
        includes_tRNA = (
            self.cistron_data["is_tRNA"] @ self.cistron_tu_mapping_matrix
        ).astype(bool)
        is_tRNA = np.logical_and(includes_tRNA, ~is_rRNA)
        is_rtRNA = np.logical_or(is_rRNA, is_tRNA)

        # Get boolean array for unprocessed rRNA/tRNA molecules
        mature_cistron_ids = set(
            self.cistron_data["id"][
                self.cistron_data["is_tRNA"] | self.cistron_data["is_rRNA"]
            ]
        )
        is_mature_rtRNA = np.array([rna_id in mature_cistron_ids for rna_id in rna_ids])
        is_unprocessed = is_rtRNA & ~is_mature_rtRNA

        # Determine if each RNA contains cistrons that encode for special
        # components
        includes_ribosomal_protein = (
            self.cistron_data["is_ribosomal_protein"] @ self.cistron_tu_mapping_matrix
        ).astype(bool)
        includes_RNAP = (
            self.cistron_data["is_RNAP"] @ self.cistron_tu_mapping_matrix
        ).astype(bool)

        # Build the relative abundance matrix between transcription units and
        # constituent cistrons
        cistron_indexes = []
        rna_indexes = []
        v = []

        for cistron_index, cistron_id in enumerate(self.cistron_data["id"]):
            rna_indexes_this_cistron = self.cistron_id_to_rna_indexes(cistron_id)
            v_this_cistron = np.zeros(len(rna_indexes_this_cistron))
            for i, rna_index in enumerate(rna_indexes_this_cistron):
                cistron_indexes.append(cistron_index)
                rna_indexes.append(rna_index)
                v_this_cistron[i] = expression[rna_index]

            # Assume uniform distribution if cistron is not expressed
            if v_this_cistron.sum() == 0:
                v_this_cistron[:] = 1.0 / len(v_this_cistron)
            else:
                v_this_cistron = v_this_cistron / v_this_cistron.sum()

            v.extend(v_this_cistron)

        cistron_indexes = np.array(cistron_indexes)
        rna_indexes = np.array(rna_indexes)
        v = np.array(v)
        shape = (cistron_indexes.max() + 1, rna_indexes.max() + 1)

        cistron_tu_relative_abundancy_matrix = csr_matrix(
            (v, (cistron_indexes, rna_indexes)), shape=shape
        )

        rna_deg_rates = np.zeros(n_rnas)
        # If a measured half-life for the transcription unit exists, use the
        # measured value to calculate the degradation rate
        rna_id_to_index = {
            rna_id: cistron_index for (cistron_index, rna_id) in enumerate(rna_ids)
        }
        for rna in raw_data.rna_half_lives:
            if rna["id"] in rna_id_to_index and rna["id"] not in cistron_id_to_index:
                rna_deg_rates[rna_id_to_index[rna["id"]]] = np.log(2) / rna[
                    "half_life"
                ].asNumber(units.s)

        # Get mask for RNAs with measured deg rates
        mask_measured_deg_rate = rna_deg_rates > 0

        # Set the minimum possible degradation rates of mRNAs to the minimum of
        # the measured degradation rates of mRNA cistrons
        min_deg_rates = np.zeros(n_rnas)
        cistron_deg_rates = self.cistron_data["deg_rate"].asNumber(1 / units.s)
        mRNA_cistron_deg_rates = cistron_deg_rates[self.cistron_data["is_mRNA"]]
        min_deg_rates[is_mRNA] = mRNA_cistron_deg_rates.min()
        min_deg_rates = min_deg_rates[~mask_measured_deg_rate]

        # Solve NNLS for unmeasured degredation rates, using the fact that
        # A(x + m) = b is equivalent to Ax = b - Am
        abundancy_no_measurements = cistron_tu_relative_abundancy_matrix[
            :, ~mask_measured_deg_rate
        ]
        abundancy_with_measurements = cistron_tu_relative_abundancy_matrix[
            :, mask_measured_deg_rate
        ]
        deg_rates_from_min_rates = abundancy_no_measurements.dot(min_deg_rates)
        deg_rates_from_measured_rates = abundancy_with_measurements.dot(
            rna_deg_rates[mask_measured_deg_rate]
        )
        rna_deg_rates_estimated_minus_min, _ = fast_nnls(
            abundancy_no_measurements,
            cistron_deg_rates
            - deg_rates_from_measured_rates
            - deg_rates_from_min_rates,
        )

        rna_deg_rates[~mask_measured_deg_rate] = (
            rna_deg_rates_estimated_minus_min + min_deg_rates
        )

        # Clip mRNA degradation rates that are higher than the maximum measured
        # degradation rate
        max_mRNA_deg_rate = mRNA_cistron_deg_rates.max()
        rna_deg_rates[np.logical_and(is_mRNA, rna_deg_rates > max_mRNA_deg_rate)] = (
            max_mRNA_deg_rate
        )

        # Set degradation rates of rRNAs and tRNAs from the stable RNA half life
        # value defined in sim_data.constants
        rna_deg_rates[is_rtRNA] = np.log(
            2
        ) / sim_data.constants.stable_RNA_half_life.asNumber(units.s)

        # Calculate synthesis probabilities from expression and normalize
        synth_prob = expression * (
            np.log(2) / sim_data.doubling_time.asNumber(units.s) + rna_deg_rates
        )
        synth_prob /= synth_prob.sum()

        # Load RNA sequences and molecular weights from getter functions
        rna_seqs = sim_data.getter.get_sequences(rna_ids)
        mws = sim_data.getter.get_masses(rna_ids).asNumber(units.g / units.mol)

        # Calculate the masses of component rRNA and tRNA cistrons of each RNA
        rRNA_cistron_mws = self.cistron_data["mw"].asNumber(units.g / units.mol)
        rRNA_cistron_mws[~self.cistron_data["is_rRNA"]] = 0.0
        tRNA_cistron_mws = self.cistron_data["mw"].asNumber(units.g / units.mol)
        tRNA_cistron_mws[~self.cistron_data["is_tRNA"]] = 0.0

        rRNA_mws = self.cistron_tu_mapping_matrix.T.dot(rRNA_cistron_mws)
        tRNA_mws = self.cistron_tu_mapping_matrix.T.dot(tRNA_cistron_mws)

        # Calculate lengths and nt counts from sequence
        rna_lengths = np.array([len(seq) for seq in rna_seqs])

        # Get RNA nucleotide compositions
        ntp_abbreviations = [ntp_id[0] for ntp_id in sim_data.molecule_groups.ntps]
        nt_counts = []
        for seq in rna_seqs:
            nt_counts.append([seq.count(letter) for letter in ntp_abbreviations])
        nt_counts = np.array(nt_counts)

        # Get mapping from cistron IDs to coordinate and direction
        rna_id_to_coordinate = {}
        rna_id_to_direction = {}
        for gene in raw_data.genes:
            rna_id_to_direction[gene["rna_ids"][0]] = gene["direction"]
            if gene["direction"] == "+":
                rna_id_to_coordinate[gene["rna_ids"][0]] = gene["left_end_pos"]
            else:
                rna_id_to_coordinate[gene["rna_ids"][0]] = gene["right_end_pos"]

        # Further extend the dictionaries to include mappings from transcription
        # unit IDs to coordinate and direction
        for tu in all_valid_tus:
            rna_id_to_direction[tu["id"]] = tu["direction"]
            if tu["direction"] == "+":
                rna_id_to_coordinate[tu["id"]] = tu["left_end_pos"]
            else:
                rna_id_to_coordinate[tu["id"]] = tu["right_end_pos"]

        # Get mapping from cistron IDs to lengths
        cistron_id_to_length = {
            cistron["id"]: cistron["length"] for cistron in self.cistron_data
        }

        # Get location of transcription initiation relative to origin and the
        # transcription direction for each transcription unit
        replication_coordinate = [
            self._get_relative_coordinates(rna_id_to_coordinate[rna_id])
            for rna_id in rna_ids
        ]
        is_forward = [rna_id_to_direction[rna_id] == "+" for rna_id in rna_ids]

        # Calculate relative start and end positions of each cistron within each
        # transcription unit
        all_cistron_ids = self.cistron_data["id"]
        self.cistron_start_end_pos_in_tu = {}

        for rna_idx, rna_id in enumerate(rna_ids):
            rna_coordinate = rna_id_to_coordinate[rna_id]
            constituent_cistron_indexes = self.cistron_tu_mapping_matrix.getcol(
                rna_idx
            ).nonzero()[0]

            for cistron_idx in constituent_cistron_indexes:
                cistron_id = all_cistron_ids[cistron_idx]
                cistron_coordinate = rna_id_to_coordinate[cistron_id]

                start_pos = np.abs(cistron_coordinate - rna_coordinate)
                end_pos = start_pos + cistron_id_to_length[cistron_id] - 1

                # End position should stay within length of entire RNA
                assert end_pos < rna_lengths[rna_idx]

                # Key: (index of cistron, index of RNA)
                # Value: (start position, end position)
                self.cistron_start_end_pos_in_tu[(cistron_idx, rna_idx)] = (
                    start_pos,
                    end_pos,
                )

        max_rna_id_length = max(len(id_) for id_ in rna_ids_with_compartments)

        # Get evidence codes for each transcription unit from raw data
        self.rna_id_to_evidence_codes = {
            tu["id"]: sorted(tu["evidence"]) for tu in raw_data.transcription_units
        }

        rna_data = np.zeros(
            n_rnas,
            dtype=[
                ("id", "U{}".format(max_rna_id_length)),
                ("deg_rate", "f8"),
                ("deg_rate_is_measured", "bool"),
                ("length", "i8"),
                ("counts_ACGU", "4i8"),
                ("mw", "f8"),
                ("rRNA_mw", "f8"),
                ("tRNA_mw", "f8"),
                ("Km_endoRNase", "f8"),
                ("replication_coordinate", "int64"),
                ("wt_replication_coordinate", "int64"),
                ("is_forward", "bool"),
                ("is_mRNA", "bool"),
                ("is_miscRNA", "bool"),
                ("is_rRNA", "bool"),
                ("is_tRNA", "bool"),
                ("includes_tRNA", "bool"),
                ("is_unprocessed", "bool"),
                ("includes_ribosomal_protein", "bool"),
                ("includes_RNAP", "bool"),
            ],
        )

        rna_data["id"] = rna_ids_with_compartments
        rna_data["deg_rate"] = rna_deg_rates
        rna_data["deg_rate_is_measured"] = mask_measured_deg_rate
        rna_data["length"] = rna_lengths
        rna_data["counts_ACGU"] = nt_counts
        rna_data["mw"] = mws
        rna_data["rRNA_mw"] = rRNA_mws
        rna_data["tRNA_mw"] = tRNA_mws
        rna_data["Km_endoRNase"] = np.zeros(
            len(rna_ids_with_compartments)
        )  # Set later in ParCa
        rna_data["replication_coordinate"] = replication_coordinate
        rna_data["wt_replication_coordinate"] = replication_coordinate
        rna_data["is_forward"] = is_forward
        rna_data["is_mRNA"] = is_mRNA
        rna_data["is_miscRNA"] = is_miscRNA
        rna_data["is_rRNA"] = is_rRNA
        rna_data["is_tRNA"] = is_tRNA
        rna_data["includes_tRNA"] = includes_tRNA
        rna_data["is_unprocessed"] = is_unprocessed
        rna_data["includes_ribosomal_protein"] = includes_ribosomal_protein
        rna_data["includes_RNAP"] = includes_RNAP

        field_units = {
            "id": None,
            "deg_rate": 1 / units.s,
            "deg_rate_is_measured": None,
            "length": units.nt,
            "counts_ACGU": units.nt,
            "mw": units.g / units.mol,
            "rRNA_mw": units.g / units.mol,
            "tRNA_mw": units.g / units.mol,
            "Km_endoRNase": units.mol / units.L,
            "replication_coordinate": None,
            "wt_replication_coordinate": None,
            "is_forward": None,
            "is_mRNA": None,
            "is_miscRNA": None,
            "is_rRNA": None,
            "is_tRNA": None,
            "includes_tRNA": None,
            "is_unprocessed": None,
            "includes_ribosomal_protein": None,
            "includes_RNAP": None,
        }

        self.rna_data = UnitStructArray(rna_data, field_units)
        self._rna_id_to_index = {
            rna_id: cistron_index
            for (cistron_index, rna_id) in enumerate(self.rna_data["id"])
        }

        # For use in transcription_regulation process
        self.unnormalized_mRNA_rna_exp_sum = np.sum(expression[is_mRNA])

        # Set basal expression and synthesis probabilities - conditional values
        # are set in the parca.
        self.rna_expression = {}
        self.rna_synth_prob = {}
        self.rna_expression["basal"] = expression / expression.sum()
        self.rna_synth_prob["basal"] = synth_prob / synth_prob.sum()
        # Initialize rna synthesis affinities, values are set in the parca
        # (can't set basal affinities here since depends on copy numbers from
        # replication dataclass)
        self.rna_synth_aff = {}


    def cistron_id_to_rna_indexes(self, cistron_id):
        """
        Returns the indexes of transcription units containing the given RNA
        cistron given the ID of the cistron.
        """
        return self.cistron_tu_mapping_matrix.getrow(
            self._cistron_id_to_index[cistron_id]
        ).nonzero()[1]

    @cache
    def rna_id_to_cistron_indexes(self, rna_id):
        """
        Returns the indexes of cistrons that constitute the given transcription
        unit given the ID of the RNA transcription unit.
        """
        return self.cistron_tu_mapping_matrix.getcol(
            self._rna_id_to_index[rna_id]
        ).nonzero()[0]

    def fit_rna_expression(self, cistron_expression):
        """
        Calculates the expression of RNA transcription units that best fits the
        given expression levels of cistrons using nonnegative least squares.
        """
        rna_exp, res = fast_nnls(self.cistron_tu_mapping_matrix, cistron_expression)
        return rna_exp, res

    def fit_trna_expression(self, tRNA_cistron_expression):
        """
        Calculates the expression of tRNA transcription units that best fits the
        given expression levels of tRNA cistrons using nonnegative least
        squares.
        """
        tRNA_exp, res = fast_nnls(
            self.tRNA_cistron_tu_mapping_matrix, tRNA_cistron_expression
        )
        return tRNA_exp, res

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

    def _apply_rnaseq_correction(self):
        """
        Applies correction to RNAseq data for shorter genes as required when
        operon structure is included in the model.
        """
        cistron_expression = self.cistron_expression["basal"].copy()
        zero_exp_mask = cistron_expression == 0

        # Find minimum length of cistron with nonzero expression
        cistron_lengths = self.cistron_data["length"].asNumber(units.nt)
        length_threshold = cistron_lengths[~zero_exp_mask].min()

        # Get mask for cistrons that are mRNAs, shorter than the threshold
        # length, covered by RNAseq data, and with zero expression
        correction_mask = np.logical_and.reduce(
            (
                self.cistron_data["is_mRNA"],
                zero_exp_mask,
                cistron_lengths < length_threshold,
                self._cistron_is_rnaseq_covered,
            )
        )
        corrected_indexes = []

        for cistron_index in np.where(correction_mask)[0]:
            # Get indexes of cistrons in the same operon
            cistrons_in_operon = None
            rnas_in_operon = None
            for operon in self.operons:
                if cistron_index in operon[0]:
                    cistrons_in_operon = operon[0]
                    rnas_in_operon = operon[1]
                    break
            assert cistrons_in_operon is not None

            # Skip monocistronic operons
            if len(cistrons_in_operon) == 1:
                continue

            # Get cistron-TU mapping matrix for this operon
            mapping_matrix_this_operon = self.cistron_tu_mapping_matrix[
                cistrons_in_operon, :
            ][:, rnas_in_operon].toarray()

            # Remove given gene from operon and run NNLS
            pos_in_operon = cistrons_in_operon.index(cistron_index)
            mapping_matrix_gene_removed = mapping_matrix_this_operon.copy()
            mapping_matrix_gene_removed[pos_in_operon, :] = 0

            rna_exp, _ = fast_nnls(
                mapping_matrix_gene_removed, cistron_expression[cistrons_in_operon]
            )

            # Use solution to get expected expression for given gene
            exp = mapping_matrix_this_operon.dot(rna_exp)[pos_in_operon]
            cistron_expression[cistron_index] = exp
            corrected_indexes.append(cistron_index)

        # Reset cistron_expression to new values
        self.cistron_expression["basal"] = cistron_expression / cistron_expression.sum()

        # Keep record of cistrons whose expression was corrected
        self.cistron_data["uses_corrected_seq_counts"][np.array(corrected_indexes)] = (
            True
        )

    def _build_mature_rna_data(self, raw_data, sim_data):
        """
        Build mature RNA-associated simulation data from raw data.
        """
        unprocessed_rna_indexes = np.where(self.rna_data["is_unprocessed"])[0]

        # Get IDs of all mature RNAs that are derived from unprocessed RNAs
        mature_rna_cistron_indexes = np.unique(
            self.cistron_tu_mapping_matrix[:, unprocessed_rna_indexes].nonzero()[0]
        )
        mature_rna_ids = self.cistron_data["id"][mature_rna_cistron_indexes]
        n_mature_rnas = len(mature_rna_ids)
        compartments = sim_data.getter.get_compartments(mature_rna_ids)
        mature_rna_ids_with_compartments = [
            f"{rna_id}[{loc[0]}]" for (rna_id, loc) in zip(mature_rna_ids, compartments)
        ]

        # Get stoichiometric matrix for RNA maturation process
        self.rna_maturation_stoich_matrix = self.cistron_tu_mapping_matrix[
            :, unprocessed_rna_indexes
        ][mature_rna_cistron_indexes, :]

        # Build matrix of the end positions of each mature RNA within each
        # unprocessed RNA
        self.mature_rna_end_positions = np.zeros(
            self.rna_maturation_stoich_matrix.shape
        )
        (rows, columns) = self.rna_maturation_stoich_matrix.nonzero()

        for i, j in zip(rows, columns):
            self.mature_rna_end_positions[i, j] = self.cistron_start_end_pos_in_tu[
                (mature_rna_cistron_indexes[i], unprocessed_rna_indexes[j])
            ][1]

        # Get mapping matrix from mature RNA to processing enzymes that are
        # needed to get the mature forms
        mature_rna_to_enzyme_list = {
            row["rna_id"]: row["enzymes"] for row in raw_data.rna_maturation_enzymes
        }
        mature_rna_id_to_index = {
            rna_id: i for (i, rna_id) in enumerate(mature_rna_ids)
        }

        all_enzymes = []
        mature_rna_indexes = []
        enzyme_indexes = []

        for rna_id, enzyme_list in mature_rna_to_enzyme_list.items():
            # Skip if RNA is not a mature RNA
            try:
                mature_rna_index = mature_rna_id_to_index[rna_id]
            except KeyError:
                continue

            for enzyme in enzyme_list:
                mature_rna_indexes.append(mature_rna_index)
                if enzyme in all_enzymes:
                    enzyme_indexes.append(all_enzymes.index(enzyme))
                else:
                    enzyme_indexes.append(len(all_enzymes))
                    all_enzymes.append(enzyme)

        mature_rna_indexes = np.array(mature_rna_indexes, dtype=int)
        enzyme_indexes = np.array(enzyme_indexes, dtype=int)
        mature_rna_to_enzyme_mapping_matrix = np.zeros(
            (n_mature_rnas, len(all_enzymes)), dtype=bool
        )
        mature_rna_to_enzyme_mapping_matrix[mature_rna_indexes, enzyme_indexes] = True

        # Convert to mapping matrix between unprocessed RNAs and enzymes
        self.rna_maturation_enzyme_matrix = self.rna_maturation_stoich_matrix.T.dot(
            mature_rna_to_enzyme_mapping_matrix
        ).astype(bool)
        enzyme_compartments = sim_data.getter.get_compartments(all_enzymes)
        self.rna_maturation_enzymes = [
            f"{enzyme_id}[{loc[0]}]"
            for (enzyme_id, loc) in zip(all_enzymes, enzyme_compartments)
        ]

        # Get mapping matrix between rtRNA cistrons and TUs
        rRNA_indexes = np.where(self.rna_data["is_rRNA"])[0]
        rRNA_cistron_indexes = np.where(self.cistron_data["is_rRNA"])[0]
        self.rRNA_cistron_tu_mapping_matrix = self.cistron_tu_mapping_matrix[
            :, rRNA_indexes
        ][rRNA_cistron_indexes, :]

        tRNA_indexes = np.where(self.rna_data["includes_tRNA"])[0]
        tRNA_cistron_indexes = np.where(self.cistron_data["is_tRNA"])[0]
        self.tRNA_cistron_tu_mapping_matrix = self.cistron_tu_mapping_matrix[
            :, tRNA_indexes
        ][tRNA_cistron_indexes, :]

        # Get RNA nucleotide compositions of unprocessed and processed RNAs
        unprocessed_rna_nt_counts = self.rna_data["counts_ACGU"][
            unprocessed_rna_indexes
        ].asNumber(units.nt)

        mature_rna_seqs = sim_data.getter.get_sequences(mature_rna_ids)
        lengths = np.array([len(seq) for seq in mature_rna_seqs])
        ntp_abbreviations = [ntp_id[0] for ntp_id in sim_data.molecule_groups.ntps]

        mature_rna_nt_counts = np.zeros((0, 4))
        for seq in mature_rna_seqs:
            mature_rna_nt_counts = np.vstack(
                (
                    mature_rna_nt_counts,
                    np.array([seq.count(letter) for letter in ntp_abbreviations]),
                )
            )

        # Calculate number of nucleotides that are degraded as part of the
        # maturation process for each unprocessed RNA
        degraded_nt_counts = unprocessed_rna_nt_counts.copy()
        rows, cols = self.rna_maturation_stoich_matrix.nonzero()

        for i, j in zip(rows, cols):
            degraded_nt_counts[j, :] -= mature_rna_nt_counts[i, :]

        assert np.all(degraded_nt_counts >= 0)
        self.rna_maturation_degraded_nt_counts = degraded_nt_counts

        # Get identities of each stable RNA
        is_rRNA = self.cistron_data["is_rRNA"][mature_rna_cistron_indexes]
        is_tRNA = self.cistron_data["is_tRNA"][mature_rna_cistron_indexes]
        is_23S_rRNA = self.cistron_data["is_23S_rRNA"][mature_rna_cistron_indexes]
        is_16S_rRNA = self.cistron_data["is_16S_rRNA"][mature_rna_cistron_indexes]
        is_5S_rRNA = self.cistron_data["is_5S_rRNA"][mature_rna_cistron_indexes]

        rna_deg_rates = np.zeros(n_mature_rnas)
        if sim_data.stable_rrna:
            # If stable rRNA option is on, set degradation rates of mature rRNAs
            # to the values calculated from the half-life in sim_data.constants
            rna_deg_rates[is_rRNA] = np.log(
                2
            ) / sim_data.constants.stable_RNA_half_life.asNumber(units.s)
        else:
            # Default: Set degradation rates of mature rRNAs to the average
            # reported degradation rates of mRNAs
            # Note: rRNAs complexed into ribosomal subunits will not degrade, so
            # this will only significantly affect excess rRNAs
            rna_deg_rates[is_rRNA] = np.log(
                2
            ) / self.average_mRNA_cistron_half_life.asNumber(units.s)

        # Set degradation rates of tRNAs to the values calculated from the
        # half-life in sim_data.constants
        rna_deg_rates[is_tRNA] = np.log(
            2
        ) / sim_data.constants.stable_RNA_half_life.asNumber(units.s)

        # Get MWs of mature RNA molecules
        mws = sim_data.getter.get_masses(mature_rna_ids).asNumber(units.g / units.mol)

        if n_mature_rnas > 0:
            max_rna_id_length = max(
                len(id_) for id_ in mature_rna_ids_with_compartments
            )
        else:
            max_rna_id_length = 1

        mature_rna_data = np.zeros(
            n_mature_rnas,
            dtype=[
                ("id", "U{}".format(max_rna_id_length)),
                ("deg_rate", "f8"),
                ("length", "i8"),
                ("counts_ACGU", "4i8"),
                ("mw", "f8"),
                ("Km_endoRNase", "f8"),
                ("is_rRNA", "bool"),
                ("is_tRNA", "bool"),
                ("is_23S_rRNA", "bool"),
                ("is_16S_rRNA", "bool"),
                ("is_5S_rRNA", "bool"),
            ],
        )

        mature_rna_data["id"] = mature_rna_ids_with_compartments
        mature_rna_data["deg_rate"] = rna_deg_rates
        mature_rna_data["length"] = lengths
        mature_rna_data["counts_ACGU"] = mature_rna_nt_counts
        mature_rna_data["mw"] = mws
        mature_rna_data["Km_endoRNase"] = np.zeros(
            len(mature_rna_ids_with_compartments)
        )  # Set later in ParCa
        mature_rna_data["is_rRNA"] = is_rRNA
        mature_rna_data["is_tRNA"] = is_tRNA
        mature_rna_data["is_23S_rRNA"] = is_23S_rRNA
        mature_rna_data["is_16S_rRNA"] = is_16S_rRNA
        mature_rna_data["is_5S_rRNA"] = is_5S_rRNA

        field_units = {
            "id": None,
            "deg_rate": 1 / units.s,
            "length": units.nt,
            "counts_ACGU": units.nt,
            "mw": units.g / units.mol,
            "Km_endoRNase": units.mol / units.L,
            "is_rRNA": None,
            "is_tRNA": None,
            "is_23S_rRNA": None,
            "is_16S_rRNA": None,
            "is_5S_rRNA": None,
        }

        self.mature_rna_data = UnitStructArray(mature_rna_data, field_units)

    def _build_transcription(self, raw_data, sim_data):
        """
        Build transcription-associated simulation data from raw data.
        """
        # Load sequence data
        rna_seqs = sim_data.getter.get_sequences(
            [rna_id[:-3] for rna_id in self.rna_data["id"]]
        )

        # Construct transcription sequence matrix
        maxLen = np.int64(
            self.rna_data["length"].asNumber().max()
            + self.max_time_step
            * sim_data.constants.RNAP_elongation_rate_for_stable_RNA.asNumber(
                units.nt / units.s
            )
        )

        self.transcription_sequences = np.full(
            (len(rna_seqs), maxLen), polymerize.PAD_VALUE, dtype=np.int8
        )
        ntMapping = {ntpId: i for i, ntpId in enumerate(["A", "C", "G", "U"])}
        for i, sequence in enumerate(rna_seqs):
            for j, letter in enumerate(sequence):
                self.transcription_sequences[i, j] = ntMapping[letter]

        # Calculate weights of transcript nucleotide monomers
        self.transcription_monomer_weights = (
            (
                sim_data.getter.get_masses(sim_data.molecule_groups.ntps)
                - sim_data.getter.get_masses([sim_data.molecule_ids.ppi])
            )
            / sim_data.constants.n_avogadro
        ).asNumber(units.fg)

        self.transcription_end_weight = (
            sim_data.getter.get_masses([sim_data.molecule_ids.ppi])
            / sim_data.constants.n_avogadro
        ).asNumber(units.fg)

        # Load active RNAP footprint on DNA
        molecule_id_to_footprint_sizes = {
            row["molecule_id"]: row["footprint_size"]
            for row in raw_data.footprint_sizes
        }
        try:
            self.active_rnap_footprint_size = molecule_id_to_footprint_sizes[
                sim_data.molecule_ids.full_RNAP[:-3]
            ]
        except KeyError:
            raise ValueError("DNA footprint size for RNA polymerses not found.")

    def _build_charged_trna(self, raw_data, sim_data):
        """
        Loads information and creates data structures necessary for charging of tRNA

        Note:
                Requires self.rna_data so can't be built in translation even if some
                data structures would be more appropriate there.
        """
        # Create list of charged tRNAs
        uncharged_trna_names = [
            x + "[c]" for x in self.cistron_data["id"][self.cistron_data["is_tRNA"]]
        ]

        charged_trnas = [
            x["modified_forms"]
            for x in raw_data.rnas
            if x["id"] + "[c]" in uncharged_trna_names
        ]

        filtered_charged_trna = []
        for charged_list in charged_trnas:
            for trna in charged_list:
                # Skip modified forms so only one charged tRNA per uncharged tRNA
                if "FMET" in trna or "modified" in trna:
                    continue

                assert "c" in sim_data.getter.get_compartment(trna)
                filtered_charged_trna += [trna + "[c]"]

        self.uncharged_trna_names = uncharged_trna_names
        self.charged_trna_names = filtered_charged_trna
        assert len(self.charged_trna_names) == len(self.uncharged_trna_names)

        # Create mapping of each tRNA/charged tRNA to associated AA
        trna_dict = {
            "RNA0-300[c]": "VAL",
            "RNA0-301[c]": "LYS",
            "RNA0-302[c]": "LYS",
            "RNA0-303[c]": "LYS",
            "RNA0-304[c]": "ASN",
            "RNA0-305[c]": "ILE",
            "RNA0-306[c]": "MET",
        }
        aa_names = sim_data.molecule_groups.amino_acids
        aa_indices = {aa: i for i, aa in enumerate(aa_names)}
        trna_indices = {trna: i for i, trna in enumerate(self.uncharged_trna_names)}
        self.aa_from_trna = np.zeros((len(aa_names), len(self.uncharged_trna_names)))
        for trna in self.uncharged_trna_names:
            aa = trna[:3].upper()
            if aa == "ALA":
                aa = "L-ALPHA-ALANINE"
            elif aa == "ASP":
                aa = "L-ASPARTATE"
            elif aa == "SEL":
                aa = "L-SELENOCYSTEINE"
            elif aa == "RNA":
                aa = trna_dict[trna]

            assert "c" in sim_data.getter.get_compartment(aa)
            aa += "[c]"
            if aa in aa_names:
                aa_idx = aa_indices[aa]
                trna_idx = trna_indices[trna]
                self.aa_from_trna[aa_idx, trna_idx] = 1

        # Arrays for stoichiometry and synthetase mapping matrices
        molecules = []

        # Sparse matrix representation - i, j are row/column indices and v is value
        stoich_matrix_i = []
        stoich_matrix_j = []
        stoich_matrix_v = []

        synthetase_names = []
        synthetase_mapping_aa = []
        synthetase_mapping_syn = []
        synthetase_metabolites = {}
        # Get IDs of all metabolites
        metabolite_ids = {met["id"] for met in raw_data.metabolites}

        # Create stoichiometry matrix for charging reactions
        for reaction in raw_data.trna_charging_reactions:
            # Get uncharged tRNA name for the given reaction
            trna = None
            for mol_id in reaction["stoichiometry"].keys():
                if f"{mol_id}[c]" in self.uncharged_trna_names:
                    trna = f"{mol_id}[c]"
                    break

            if trna is None:
                continue

            trna_index = trna_indices[trna]

            # Get molecule information
            aa_idx = None
            for mol_id, coeff in reaction["stoichiometry"].items():
                if mol_id in metabolite_ids:
                    molecule_name = "{}[{}]".format(
                        mol_id,
                        "c",
                        # Assume all metabolites are in cytosol
                    )
                else:
                    molecule_name = "{}[{}]".format(
                        mol_id, sim_data.getter.get_compartment(mol_id)[0]
                    )

                if molecule_name not in molecules:
                    molecules.append(molecule_name)
                    molecule_index = len(molecules) - 1
                else:
                    molecule_index = molecules.index(molecule_name)

                aa_idx = aa_indices.get(molecule_name, aa_idx)

                # Assume coefficents given as null are -1
                if coeff is None:
                    coeff = -1
                assert coeff % 1 == 0

                stoich_matrix_i.append(molecule_index)
                stoich_matrix_j.append(trna_index)
                stoich_matrix_v.append(coeff)

            assert aa_idx is not None

            # Create mapping for synthetases catalyzing charging
            for synthetase in reaction["catalyzed_by"]:
                synthetase_metabolites[synthetase] = (
                    synthetase_metabolites.get(synthetase, set())
                    | reaction["stoichiometry"].keys()
                )
                synthetase = "{}[{}]".format(
                    synthetase, sim_data.getter.get_compartment(synthetase)[0]
                )

                if synthetase not in synthetase_names:
                    synthetase_names.append(synthetase)

                synthetase_mapping_aa.append(aa_idx)
                synthetase_mapping_syn.append(synthetase_names.index(synthetase))

        # Extract KM data for amino acids and tRNA in charging reactions
        synthetase_names_without_tag = {name[:-3] for name in synthetase_names}
        aa_names_without_tag = [aa[:-3] for aa in sim_data.molecule_groups.amino_acids]
        aa_kms = {}
        trna_kms = {}
        skipped_reactions = {"RXN-16165"}  # Not correct MET charging reaction
        for row in raw_data.metabolism_kinetics:
            # Only look at data for charging reactions
            if (
                row["reactionID"] in skipped_reactions
                or row["enzymeID"] not in synthetase_names_without_tag
            ):
                continue

            for met, km in zip(row["substrateIDs"], row["kM"]):
                if met in aa_names_without_tag:
                    # Prevent data from mismatched amino acid/synthetases being used
                    if met not in synthetase_metabolites[row["enzymeID"]]:
                        continue
                    aa_kms[met] = aa_kms.get(met, []) + [km]
                elif "tRNA" in met:
                    # Exclude suspiciously high data
                    if km > 5 * sim_data.constants.Km_synthetase_uncharged_trna:
                        continue
                    aa = met.split("-")[0]
                    if aa == "ALA":
                        aa = "L-ALPHA-ALANINE"
                    elif aa == "ASP":
                        aa = "L-ASPARTATE"
                    elif aa == "Elongation":
                        aa = "MET"
                    trna_kms[aa] = trna_kms.get(aa, []) + [km]

        # Save average KM values and use the default value if no data is available
        km_units = units.umol / units.L
        average_aa_kms = []
        average_trna_kms = []
        for aa_id in aa_names_without_tag:
            average_aa_kms.append(
                np.mean(
                    aa_kms.get(aa_id, sim_data.constants.Km_synthetase_amino_acid)
                ).asNumber(km_units)
            )
            average_trna_kms.append(
                np.mean(
                    trna_kms.get(aa_id, sim_data.constants.Km_synthetase_uncharged_trna)
                ).asNumber(km_units)
            )
        self.aa_kms = km_units * np.array(average_aa_kms)
        self.trna_kms = km_units * np.array(average_trna_kms)

        # Save matrices and related lists of names
        self._stoich_matrix_i = np.array(stoich_matrix_i)
        self._stoich_matrix_j = np.array(stoich_matrix_j)
        self._stoich_matrix_v = np.array(stoich_matrix_v)

        self.aa_from_synthetase = np.zeros((len(aa_names), len(synthetase_names)))
        self.aa_from_synthetase[synthetase_mapping_aa, synthetase_mapping_syn] = 1

        self.synthetase_names = synthetase_names
        self.charging_molecules = molecules

    def charging_stoich_matrix(self):
        """
        Creates stoich matrix from i, j, v arrays

        Returns 2D array with rows of metabolites for each tRNA charging reaction on the column
        """
        shape = (self._stoich_matrix_i.max() + 1, self._stoich_matrix_j.max() + 1)

        out = np.zeros(shape, np.float64)
        out[self._stoich_matrix_i, self._stoich_matrix_j] = self._stoich_matrix_v

        return out

    def _build_attenuation(self, raw_data, sim_data):
        """
        Load fold changes related to transcriptional attenuation.
        """
        # Load data from file
        aa_rna_pair_to_log_fcs = {}
        gene_symbol_to_cistron_id = {
            g["symbol"]: g["rna_ids"][0] for g in raw_data.genes
        }
        for row in raw_data.transcriptional_attenuation:
            trna_aa = row["tRNA"].split("-")[1].upper() + "[c]"
            gene = row["Target"]
            cistron_id = gene_symbol_to_cistron_id[gene]

            # Map FC to all RNAs that cover the cistron
            rna_indexes_with_cistron = self.cistron_id_to_rna_indexes(cistron_id)
            for rna_idx in rna_indexes_with_cistron:
                rna_id = self.rna_data["id"][rna_idx]
                if (trna_aa, self.rna_data["id"][rna_idx]) in aa_rna_pair_to_log_fcs:
                    aa_rna_pair_to_log_fcs[(trna_aa, rna_id)].append(row["log2 FC"])
                else:
                    aa_rna_pair_to_log_fcs[(trna_aa, rna_id)] = [row["log2 FC"]]

        aa_trnas = []
        attenuated_rnas = []
        fold_changes = []

        for (trna_aa, rna_id), all_log_fcs in aa_rna_pair_to_log_fcs.items():
            aa_trnas.append(trna_aa)
            attenuated_rnas.append(rna_id)

            # Take the average of the reported FCs of each constituent cistron
            fold_changes.append(2 ** np.mean(all_log_fcs))

        self.attenuated_rna_ids = np.unique(attenuated_rnas)

        # Convert data to matrix mapping tRNA to genes with a fold change
        trna_to_row = {t: i for i, t in enumerate(sim_data.molecule_groups.amino_acids)}
        rna_to_col = {r: i for i, r in enumerate(self.attenuated_rna_ids)}
        n_aas = len(sim_data.molecule_groups.amino_acids)
        n_rnas = len(self.attenuated_rna_ids)
        self._attenuation_rna_fold_changes = np.ones((n_aas, n_rnas))
        for trna, cistron_id, fc in zip(aa_trnas, attenuated_rnas, fold_changes):
            i = trna_to_row[trna]
            j = rna_to_col[cistron_id]
            self._attenuation_rna_fold_changes[i, j] = fc

        # Attenuated cistron index mapping
        self.attenuated_rna_indices = np.array(
            [self._rna_id_to_index[r] for r in self.attenuated_rna_ids]
        )

        # Specify location in gene where attenuation will occur
        # Currently just assumes before a transcript begins elongation (position < 1)
        # TODO: base this on specific locations for each gene
        locations = np.ones(len(self.attenuated_rna_indices))
        self.attenuation_location = {
            idx: loc for idx, loc in zip(self.attenuated_rna_indices, locations)
        }

    def calculate_attenuation(self, sim_data, cell_specs):
        """
        Calculate constants for each attenuated gene.

        TODO:
                Calculate estimated charged tRNA concentration to use instead of all tRNA
        """

        def get_trna_conc(condition):
            spec = cell_specs[condition]
            unprocessed_trna_ids = self.rna_data["id"][self.rna_data["includes_tRNA"]]
            unprocessed_trna_idx = bulk_name_to_idx(
                unprocessed_trna_ids, spec["bulkAverageContainer"]["id"]
            )
            unprocessed_counts = counts(
                spec["bulkAverageContainer"], unprocessed_trna_idx
            )
            trna_counts = self.tRNA_cistron_tu_mapping_matrix.dot(unprocessed_counts)
            volume = (
                spec["avgCellDryMassInit"]
                / sim_data.constants.cell_density
                / sim_data.mass.cell_dry_mass_fraction
            )
            # Order of operations for conc (counts last) is to get units to work well
            conc = 1 / sim_data.constants.n_avogadro / volume * trna_counts
            return conc

        k_units = units.umol / units.L
        trna_conc = self.aa_from_trna @ get_trna_conc("with_aa").asNumber(k_units)

        # Calculate constant for stop probability
        self.attenuation_k = np.zeros_like(self._attenuation_rna_fold_changes)
        for i, j in zip(*np.where(self._attenuation_rna_fold_changes != 1)):
            k = trna_conc[i] / np.log(self._attenuation_rna_fold_changes[i, j])
            self.attenuation_k[i, j] = 1 / k
        self.attenuation_k = 1 / k_units * self.attenuation_k

        # Adjust basal transcription affinities to account for less synthesis
        # due to attenuation
        condition = "basal"
        basal_aff = sim_data.process.transcription_regulation.basal_aff
        delta_aff = sim_data.process.transcription_regulation.get_delta_aff_matrix()
        p_promoter_bound = np.array(
            [
                sim_data.pPromoterBound[condition][tf]
                for tf in sim_data.process.transcription_regulation.tf_ids
            ]
        )
        delta = delta_aff @ p_promoter_bound
        basal_stop_prob = self.get_attenuation_stop_probabilities(
            get_trna_conc(condition)
        )
        basal_synth_aff = (basal_aff + delta)[self.attenuated_rna_indices]
        self.attenuation_basal_aff_adjustments = basal_synth_aff * (
            1 / (1 - basal_stop_prob) - 1
        )

        # Store expected readthrough fraction for each condition to use in initial conditions
        self.attenuation_readthrough = {}
        for condition in sim_data.conditions:
            self.attenuation_readthrough[condition] = (
                1 - self.get_attenuation_stop_probabilities(get_trna_conc(condition))
            )

    def get_attenuation_stop_probabilities(self, trna_conc):
        """
        Calculate the probability of a transcript stopping early due to attenuation.

        TODO:
                Consider a maximum stop probability factor (eg can only attenuate up to 90% of RNAs)
        """

        trna_by_aa = units.matmul(self.aa_from_trna, trna_conc)
        return 1 - np.exp(units.strip_empty_units(trna_by_aa @ self.attenuation_k))

    def _build_elongation_rates(self, raw_data, sim_data):
        self.stable_RNA_elongation_rate = (
            sim_data.constants.RNAP_elongation_rate_for_stable_RNA.asNumber(
                units.nt / units.s
            )
        )

        # rRNAs are set to have higher elongation rates
        # TODO (ggsun): Consider adding tRNAs
        self.rRNA_indexes = np.where(self.rna_data["is_rRNA"])[0]

    def make_elongation_rates(self, random, base, time_step, variable_elongation=False):
        return make_elongation_rates(
            random,
            self.transcription_sequences.shape[0],
            base,
            self.rRNA_indexes,
            self.stable_RNA_elongation_rate,
            time_step,
            variable_elongation,
        )

    def set_ppgpp_parameters(self, raw_data, sim_data):
        """
        Solves for ppGpp parameters like Km and fraction active RNAP.

        Attributes set:
                self.fraction_active_rnap_bound (float)
                self.fraction_active_rnap_free (float)
        """
        # Solves for ppGpp parameters
        self._solve_ppgpp_km(raw_data, sim_data)

        # Calculate the expected active fraction when RNAP is bound to ppGpp or free
        doubling_times = units.min * np.linspace(25, 100, 10)
        ppgpp = sim_data.growth_rate_parameters.get_ppGpp_conc(doubling_times)
        fraction_active = sim_data.growth_rate_parameters.get_fraction_active_rnap(
            doubling_times
        )
        fraction_bound = self.fraction_rnap_bound_ppgpp(ppgpp)
        A = np.vstack((fraction_bound, 1 - fraction_bound)).T
        self.fraction_active_rnap_bound, self.fraction_active_rnap_free = (
            np.linalg.lstsq(A, fraction_active, rcond=None)[0]
        )
        assert 0 < self.fraction_active_rnap_bound < 1
        assert 0 < self.fraction_active_rnap_free < 1
        # TODO: eventually merge with _solve_ppgpp_km

    def _solve_ppgpp_km(self, raw_data, sim_data):
        """
        Solves for general expression rates for bound and free RNAP and
        a KM for ppGpp to RNAP based on global cellular measurements.
        Parameters are solved for at different doubling times using a
        gradient descent method to minimize the difference in expression
        of stable RNA compared to the measured RNA in a cell.
        Assumes a Hill coefficient of 2 for ppGpp binding to RNAP.

        Attributes set:
                _fit_ppgpp_aff_fc (float): log2 fold change in stable RNA
                        affinity between ppGpp-bound and free RNAP based on
                        the rates of bound and free RNAP expression found
                _ppgpp_km_squared (float): squared and unitless KM value for
                        to limit computation needed for fraction bound
                ppgpp_km (float with mol / volume units): KM for ppGpp binding
                        to RNAP

        Notes: This is done at the cistron-level. ppGpp fold changes are converted into
        TU-level fold changes later in set_ppgpp_expression().
        """

        # Data for different doubling times (100, 60, 40, 30, 24 min)
        per_dry_mass_to_per_volume = (
            sim_data.constants.cell_density * sim_data.mass.cell_dry_mass_fraction
        )
        ppgpp = (
            np.array([
                    (d["ppGpp_conc"] * per_dry_mass_to_per_volume).asNumber(
                        PPGPP_CONC_UNITS)
                    for d in raw_data.growth_rate_dependent_parameters
                ]) ** 2
        )
        rna = np.array([d["rnaMassFraction"] for d in raw_data.dry_mass_composition])
        minimal_rRNA_mass_fraction = self._basal_rna_fractions["rRNA"]
        rRNA_mass_frac = rna * minimal_rRNA_mass_fraction
        # TODO: account for tRNAs here too maybe

        # Get approximate number of rRNAs per cell
        mass_per_cell = np.array(
            [
                d["averageDryMass"].asNumber(units.fg)
                for d in raw_data.dry_mass_composition
            ]
        )
        total_rRNA_mass = rRNA_mass_frac * mass_per_cell
        # Assumes there's an equal number of each type of rRNA, which is not true but
        # should approximately work because the seven operons are quite similar to each other
        # in sequence and length, and we expect 5S, 16S, and 23S rRNAs to be produced proportionally
        # to their abundance in each operon.
        average_rRNA_mass = (np.mean(self.cistron_data['rRNA_mw'][self.cistron_data['is_rRNA']])
                             / sim_data.constants.n_avogadro).asNumber(units.fg)
        rRNA_per_cell = total_rRNA_mass / average_rRNA_mass

        rnap_per_cell = np.array(
            [d["RNAP_per_cell"] for d in raw_data.growth_rate_dependent_parameters]
        )

        RNAP_active_fraction = np.array(
            [d["fractionActiveRnap"] for d in raw_data.growth_rate_dependent_parameters]
        )
        rRNA_elong_rate = self.stable_RNA_elongation_rate
        # Also assumes there's an equal number of each type of rRNA, see note above
        rRNA_length = np.mean(self.cistron_data["length"][self.cistron_data["is_rRNA"]]).asNumber(units.nt)

        # Get summed copy numbers of rRNAs and mRNAs
        RNA_coords = self.rna_data["replication_coordinate"]
        rRNA_coords = RNA_coords[self.rna_data['is_rRNA']]
        mRNA_coords = RNA_coords[self.rna_data['is_mRNA']]
        doubling_times = np.array(
            [d["doublingTime"].asNumber(units.min) for d in raw_data.growth_rate_dependent_parameters]
        )
        copy_number = sim_data.process.replication.get_average_copy_number
        sum_rRNA_copy = np.array([np.sum(
            copy_number(
                tau, rRNA_coords
            ))
            for tau in doubling_times]
        )
        sum_mRNA_copy = np.array([np.sum(
            copy_number(
                tau, mRNA_coords
            ))
            for tau in doubling_times]
        )
        mRNA_rRNA_copy_ratio = sum_mRNA_copy / sum_rRNA_copy

        # Variables for the objective
        ## y1: ratio of affinity for rRNAs to affinity for mRNAs for RNAP bound to ppGpp
        ## y2: ratio of affinity for rRNAs to affinity for mRNAs for free RNAP
        ## km: KM of ppGpp binding to RNAP
        y1s, y2s, kms = sp.symbols("y1 y2 km")

        # Create objective to minimize
        ## Objective is squared difference between RNA created with different rates for RNAP bound
        ## to ppGpp and free RNAP compared to measured RNA in the cell for each measured doubling time.
        ## Use sp.exp to prevent negative parameter values, also improves stability for larger step size.
        synth_capacity = rnap_per_cell * RNAP_active_fraction * rRNA_elong_rate / rRNA_length

        difference = (
            rRNA_per_cell * np.log(2) / (doubling_times * 60)
            - synth_capacity / (
                    1 + mRNA_rRNA_copy_ratio /
                    (
                            sp.exp(y1s) * ppgpp / (sp.exp(kms) + ppgpp)
                            + sp.exp(y2s) *
                            (1 -
                             (ppgpp / (sp.exp(kms) + ppgpp))
                             )
                     )
            )
        )

        J = difference.dot(difference)

        # Convert to functions for faster performance
        dJdy1 = sp.lambdify((y1s, y2s, kms), J.diff(y1s))
        dJdy2 = sp.lambdify((y1s, y2s, kms), J.diff(y2s))
        dJdkm = sp.lambdify((y1s, y2s, kms), J.diff(kms))
        J = sp.lambdify((y1s, y2s, kms), J)

        # Initial parameters
        y1 = np.log(25)
        y2 = np.log(720)
        km = np.log(1200)
        step_size = 1e-3

        # Use gradient descent to find
        obj = J(y1, y2, km)
        old_obj = 100
        steps = 0
        max_step = 1e5
        tol = 1e-4
        rel_tol = 1e-9
        while obj > tol and np.abs(1 - obj / old_obj) > rel_tol:
            y1 -= dJdy1(y1, y2, km) * step_size
            y2 -= dJdy2(y1, y2, km) * step_size
            km -= dJdkm(y1, y2, km) * step_size

            old_obj = obj
            obj = J(y1, y2, km)

            steps += 1
            if steps > max_step:
                import ipdb; ipdb.set_trace()
                raise RuntimeError(
                    "Fitting ppGpp binding KM failed to converge."
                    " Check tolerances or maximum number of steps."
                )

        y1 = np.exp(y1)
        y2 = np.exp(y2)
        km = np.exp(km)

        self._fit_ppGpp_aff_fc = np.log2(y1 / y2)
        self._ppgpp_km_squared = km
        self.ppgpp_km = np.sqrt(km) * PPGPP_CONC_UNITS
        # TODO: optimize step_size, tol, rel_tol to get faster convergence
        # TODO: modify so that we have active_fraction_ppgpp and active_fraction_free, and these are fit in this function as well
        import ipdb; ipdb.set_trace()

    def get_rna_fractions(self, sim_data, doubling_time):
        """
        Calculates expected RNA subgroup mass fractions based doubling time,
        using the corresponding ppGpp concentration. If ppGpp expression
        has not been set yet, uses default measured fractions.

        Args:
                doubling_time (float with units of time) - doubling time of the condition

        Returns:
                dict[str, float]: mass fraction for each subgroup mass, values sum
                to 1
        """
        if self._ppgpp_expression_set:
            rna_exp = self.expression_from_doubling_time(sim_data, doubling_time)
            cistron_exp = self.cistron_tu_mapping_matrix.dot(rna_exp)
            mass = self.cistron_data["mw"] * cistron_exp
            mass = (mass / units.sum(mass)).asNumber()

            fractions = {
                "rRNA": mass[self.cistron_data["is_rRNA"]].sum(),
                "tRNA": mass[self.cistron_data["is_tRNA"]].sum(),
                "mRNA": mass[self.cistron_data["is_mRNA"]].sum(),
            }
        else:
            fractions = self._basal_rna_fractions

        return fractions

    def set_ppgpp_expression(self, sim_data):
        """
        Called during the parca to determine expression of each transcription
        unit for ppGpp bound and free RNAP.

        Attributes set:
                exp_ppgpp (ndarray[float]): expression for each TU when RNAP is
                        bound to ppGpp
                exp_free (ndarray[float]): expression for each TU when RNAP is not
                        bound to ppGpp
        """
        minimal_dt = sim_data.condition_to_doubling_time["basal"]
        rich_dt = sim_data.condition_to_doubling_time["with_aa"]

        ppgpp_aa = sim_data.growth_rate_parameters.get_ppGpp_conc(
            rich_dt
        )
        ppgpp_minimal = sim_data.growth_rate_parameters.get_ppGpp_conc(
            minimal_dt
        )
        f_ppgpp_aa = self.fraction_rnap_bound_ppgpp(ppgpp_aa)
        f_ppgpp_minimal = self.fraction_rnap_bound_ppgpp(ppgpp_minimal)
        cistron_id_to_idx = {
            cistron: i for i, cistron in enumerate(self.cistron_data["id"])
        }

        # Since fold changes are reported for each cistron (gene), the FCs are
        # applied first to the expression levels of individual cistrons which
        # are converted back to TU expression levels through NNLS
        minimal_aff_TUs = self.rna_synth_aff["basal"]
        minimal_aff_cistrons = self.cistron_tu_mapping_matrix.dot(minimal_aff_TUs)

        # Ratio of synthprobs, x_ppGpp/x_free in doc. Assumes exp is proportional
        # to synthprob, and so exp_fc = synthprob_ppGpp / synthprob_rich
        # = x_ppGpp / (f_ppgpp x_ppgpp + (1-f_ppgpp) x_free)
        synthprob_ratio = (1-f_ppgpp_aa) / (
                (1 / 2**self.ppgpp_fold_changes) - f_ppgpp_aa
        )

        def _affinities_from_fc_f_ppgpp(affs, fcs, f_ppgpp):
            '''
            Computes affinities for certain RNAs from log2 fold changes and affinities
            in a certain condition.

            Since aff = f_ppgpp * aff_ppgpp + (1-f_ppgpp) * aff_free
            = f_ppgpp * aff_free * fc + (1-f_ppgpp) * aff_free
            we can solve for aff_free and aff_ppgpp.

            Inputs:
                affs (ndarray[float]) - affinities in a particular condition
                f_ppgpp (float) - fraction ppGpp-bound RNAP in a particular condition
                fcs (ndarray[float]) or float - aff_ppgpp / aff_free for these RNAs


            '''
            aff_free = affs / (f_ppgpp * 2**fcs + (1 - f_ppgpp))
            aff_ppgpp = aff_free * fcs

            return aff_ppgpp, aff_free

        # Calculate aff_ppgpp and aff_free for stable RNA cistrons using previously fit log2(fc)
        unset_ids = self.ppgpp_regulated_genes[self._ppgpp_unset_mask]
        unset_idxs = np.array([cistron_id_to_idx[cistron] for cistron in unset_ids])
        unset_gene_minimal_affs = minimal_aff_cistrons[unset_idxs]

        stable_RNA_aff_ppGpp, stable_RNA_aff_free = _affinities_from_fc_f_ppgpp(
            unset_gene_minimal_affs, self._fit_ppGpp_aff_fc, f_ppgpp_minimal
        )

        # Calculate average affinity of mRNAs in minimal media
        copy_numbers = sim_data.process.replication.get_average_copy_number
        mRNA_cistron_coords = self.cistron_data["replication_coordinate"][
            self.cistron_data["is_mRNA"]]
        minimal_mRNA_copy = copy_numbers(minimal_dt, mRNA_cistron_coords)
        avg_aff_minimal_mRNA = minimal_aff_cistrons[self.cistron_data["is_mRNA"]].dot(
            minimal_mRNA_copy
        ) / np.sum(minimal_mRNA_copy)

        # Calculate relevant copy numbers
        unset_gene_cistron_coords = self.cistron_data["replication_coordinate"][
            unset_idxs
        ]
        stable_RNA_aa_copies = copy_numbers(rich_dt, unset_gene_cistron_coords)
        sum_mRNA_aa_copies = np.sum(copy_numbers(rich_dt, mRNA_cistron_coords))

        # Calculate the fcs in affinity for all ppGpp-regulated cistrons,
        # inserting the stable RNA fc for the marked cistrons
        # TODO: check what these marked cistrons are made of (self._ppgpp_unset_mask),
        # anything important to growth, etc. other than stable RNAs?
        # TODO: check all the log2's are correct throughout this
        factor = (
                avg_aff_minimal_mRNA * sum_mRNA_aa_copies +
                stable_RNA_aff_free.dot(stable_RNA_aa_copies)
                  ) / (
                avg_aff_minimal_mRNA * sum_mRNA_aa_copies +
                stable_RNA_aff_ppGpp.dot(stable_RNA_aa_copies)
                  )
        aff_fcs = np.log2(synthprob_ratio / factor)
        aff_fcs[self._ppgpp_unset_mask] = self._fit_ppGpp_aff_fc

        # Get array containing fcs for all cistrons overall
        all_aff_fcs = np.zeros(len(self.cistron_data))
        for cistron_id, fc in zip(self.ppgpp_regulated_genes, aff_fcs):
            all_aff_fcs[cistron_id_to_idx[cistron_id]] = fc

        # Calculate cistron-level aff_ppgpp and aff_free
        all_cistron_aff_ppgpp, all_cistron_aff_free = _affinities_from_fc_f_ppgpp(
            minimal_aff_cistrons, all_aff_fcs, f_ppgpp_minimal
        )

        # Convert to TU-level affinities with NNLS (overall scale for mRNAs shouldn't have
        # changed too much since we got cistron-level affinities by multiplying by
        # conversion matrix in the first place, without normalizing)
        self.aff_ppgpp, _ = self.fit_rna_expression(all_cistron_aff_ppgpp)
        self.aff_free, _ = self.fit_rna_expression(all_cistron_aff_free)

        # Scale TU-level affinities to match original minimal-condition affinities
        # TODO: check does anything change drastically here? might want to change if so (probably don't change even if so)
        self.match_ppgpp_expression_to_target(minimal_aff_TUs, ppgpp_minimal)

        self._ppgpp_expression_set = True

    def match_ppgpp_expression_to_target(self, target_affs, ppgpp_conc, target_idxs=None):
        '''
        Scales ppGpp-bound and free RNAP affinities equally to produce target_affs when ppGpp
        levels are at ppGpp_conc. If target_idxs are provided, scales affinities for these
        TUs; if not, scales for all (len(target_affs) must be total number of TUs then)

        Modifies: TODO: write description here
            aff_free
            aff_ppgpp

        Notes:
            Uses synth_aff_from_ppgpp function.
        '''

        # Current synthesis affinities from ppGpp regulation
        old_aff = self.synth_aff_from_ppgpp(ppgpp_conc)

        if target_idxs is not None:
            old_aff = old_aff[target_idxs]

        # Determine adjustments to the current ppGpp affinities to scale
        # to the expected affinities
        with np.errstate(invalid="ignore", divide="ignore"):
            adjustment = target_affs / old_aff
        adjustment[~np.isfinite(adjustment)] = 1

        # Scale free and bound expression
        if target_idxs is not None:
            self.aff_free[target_idxs] *= adjustment
            self.aff_ppgpp[target_idxs] *= adjustment
            # TODO: check that this modifies the ndarray
        else:
            self.aff_free *= adjustment
            self.aff_ppgpp *= adjustment
        # TODO: look at what's actually changing here, anything drastic?

        # TODO: do we need these checks?
        self.aff_free[self.aff_free < 0] = 0
        self.aff_ppgpp[self.aff_ppgpp < 0] = 0

    def adjust_polymerizing_ppgpp_expression(self, sim_data):
        """
        Adjust ppGpp expression based on fit for ribosome and RNAP physiological
        constraints using least squares fit for 3 conditions with different
        growth rates/ppGpp.

        Modifies attributes:
                exp_ppgpp (ndarray[float]): expression for each gene when RNAP
                        is bound to ppGpp, adjusted for necessary RNAP and ribosome
                        expression, normalized to 1
                exp_free (ndarray[float]): expression for each gene when RNAP
                        is not bound to ppGpp, adjusted for necessary RNAP and ribosome
                        expression, normalized to 1

        Note:
                See docs/processes/transcription_regulation.pdf for a description
                of the math used in this section.

        TODO:
                Try should it be just three conditions or all conditions?
                Figure out how to adjust for TFs (or maybe we don't need to if
                TF adjustments take place before this?)
        """

        # Fraction RNAP bound to ppGpp in different conditions
        ppgpp_concs = np.array([
            sim_data.growth_rate_parameters.get_ppGpp_conc(
                sim_data.condition_to_doubling_time[condition]
            )
            for condition in sim_data.conditions
        ])

        f_ppgpps = np.array([
            self.fraction_rnap_bound_ppgpp(ppgpp) for ppgpp in ppgpp_concs
        ])

        adjusted_mask = (
            self.rna_data["includes_RNAP"]
            | self.rna_data["includes_ribosomal_protein"]
            | self.rna_data["is_rRNA"]
        )

        # Solve least squares fit for affinity of each component of RNAP and ribosomes
        F = np.array([
                [1 - f, f]
                for f in f_ppgpps
            ])
        Flst = np.linalg.inv(F.T.dot(F)).dot(F.T)
        affinity = np.array([
                self.rna_synth_aff[condition][adjusted_mask]
                for condition in sim_data.conditions
            ])
        adjusted_free_aff, adjusted_ppgpp_aff = Flst.dot(affinity)

        # Set free and ppGpp affinities
        self.aff_free[adjusted_mask] = adjusted_free_aff
        self.aff_ppgpp[adjusted_mask] = adjusted_ppgpp_aff

        self.aff_free[self.aff_free < 0] = 0
        self.aff_ppgpp[self.aff_ppgpp < 0] = 0

    def adjust_ppgpp_expression_for_tfs(self, sim_data):
        # Get one-peak and two-peak data (without TFs bound, one-peak TUs should have
        # their set affinity, and two-peak TUs should be set at their unbound affinity
        one_peak_affs = sim_data.process.transcription_regulation["affinity"]
        one_peak_TUs = sim_data.process.transcription_regulation["TU_idxs"]
        two_peak_unbound_affs = sim_data.process.transcription_regulation["unbound_affinity"]
        two_peak_TUs = sim_data.process.transcription_regulation["TU_idxs"]

        # Combine into single arrays
        target_affs = copy.deepcopy(one_peak_affs)
        target_affs.extend(two_peak_unbound_affs)
        target_TUs = copy.deepcopy(one_peak_TUs)
        target_TUs.extend(two_peak_TUs)

        # Let basal condition ppGpp-derived affinity for one-peak and two-peak genes match the targets
        basal_ppgpp = sim_data.growth_rate_parameters.get_ppGpp_conc(
            sim_data.condition_to_doubling_time["basal"]
        )
        self.match_ppgpp_expression_to_target(target_affs, basal_ppgpp, target_idxs=target_TUs)

    # def adjust_ppgpp_expression_for_tfs(self, sim_data):
    #     """
    #     Adjusts ppGpp regulated expression to get expression with and without
    #     ppGpp regulation to match in basal condition and taking into account
    #     the effect transcription factors will have.
    #
    #     TODO:
    #             Should this not adjust polymerizing genes (adjusted_mask in
    #                     adjust_polymerizing_ppgpp_expression) since they have already
    #                     been adjusted for transcription factor effects?
    #     """
    #
    #     condition = "basal"
    #
    #     # Current (unnormalized) probabilities from ppGpp regulation
    #     ppgpp_conc = sim_data.growth_rate_parameters.get_ppGpp_conc(
    #         sim_data.condition_to_doubling_time[condition]
    #     )
    #     old_aff, factor = self.synth_aff_from_ppgpp(
    #         ppgpp_conc, sim_data.process.replication.get_average_copy_number
    #     )
    #
    #     # Calculate the average expected effect of TFs in basal condition
    #     p_promoter_bound = np.array(
    #         [
    #             sim_data.pPromoterBound[condition][tf]
    #             for tf in sim_data.process.transcription_regulation.tf_ids
    #         ]
    #     )
    #     delta_aff_no_ppgpp = (
    #         sim_data.process.transcription_regulation.get_delta_aff_matrix(ppgpp=False)
    #     )
    #     delta_aff_with_ppgpp = (
    #         sim_data.process.transcription_regulation.get_delta_aff_matrix(ppgpp=True)
    #     )
    #     delta_no_ppgpp = delta_aff_no_ppgpp @ p_promoter_bound
    #     delta_with_ppgpp = delta_aff_with_ppgpp @ p_promoter_bound
    #
    #     # Calculate the required affinity to match expression without ppGpp
    #     new_aff = copy.deepcopy(sim_data.process.transcription_regulation.basal_aff)
    #         #(sim_data.process.transcription_regulation.basal_aff
    #          #       + delta_no_ppgpp) / (1 + delta_with_ppgpp)
    #
    #     new_aff[new_aff < 0] = old_aff[new_aff < 0]
    #
    #     # Determine adjustments to the current ppGpp expression to scale
    #     # to the expected expression
    #     with np.errstate(invalid="ignore", divide="ignore"):
    #         adjustment = new_aff / old_aff
    #     adjustment[~np.isfinite(adjustment)] = 1
    #
    #     # TODO: probably change this part when changing to using affinities instead?
    #     # Scale free and bound expression and renormalize ppGpp regulated expression
    #     self.exp_free *= adjustment
    #     self.exp_ppgpp *= adjustment
    #     self._normalize_ppgpp_expression()
    #     # TODO: what to do if exp_free has a gene already at 0, but it's not at 0 in basal_aff?

    def _normalize_ppgpp_expression(self):
        """
        Normalize both free and ppGpp bound expression values to 1.
        """

        self.exp_free[self.exp_free < 0] = 0
        self.exp_ppgpp[self.exp_ppgpp < 0] = 0
        self.exp_free /= self.exp_free.sum()
        self.exp_ppgpp /= self.exp_ppgpp.sum()

    def set_ppgpp_kinetics_parameters(self, init_container, constants):
        uncharged_trna_idx = bulk_name_to_idx(
            self.uncharged_trna_names, init_container["id"]
        )
        trna_counts = self.aa_from_trna @ counts(init_container, uncharged_trna_idx)
        trna_ratio = trna_counts / trna_counts.sum()
        adjustment_fraction = trna_ratio / trna_ratio.mean()

        self.KD_RelA = constants.KD_RelA_ribosome * adjustment_fraction
        self.KI_SpoT = constants.KI_SpoT_ppGpp_degradation * adjustment_fraction

    def fraction_rnap_bound_ppgpp(self, ppgpp):
        """
        Calculates the fraction of RNAP expected to be bound to ppGpp
        at a given concentration of ppGpp.

        Args:
                ppgpp (float with or without mol / volume units): concentration of ppGpp,
                        if unitless, should represent the concentration of PPGPP_CONC_UNITS

        Returns:
                float: fraction of RNAP that will be bound to ppGpp
        """

        # TODO: incorporate active fraction into this
        if units.hasUnit(ppgpp):
            ppgpp = ppgpp.asNumber(PPGPP_CONC_UNITS)

        return ppgpp**2 / (self._ppgpp_km_squared + ppgpp**2)

    def expression_from_doubling_time(self, sim_data, doubling_time, avgCellDryMassInit=None, Km=None):
        '''
        Calculates expression of each gene at the ppGpp concentration corresponding to a given doubling time.
        First calculates affinities from ppgpp, converts to synthesis
        probabilities with copy numbers from doubling time, and synthesis probabilities to
        expression with loss rates of RNAs.

        Inputs
        ------
        - doubling_time (float with units of time) - doubling time of the condition

        Returns
        --------
        - expression (array of floats) - derived expression for each RNA,
        normalized to 1

        Notes
        ------
        - uses synth_aff_from_ppgpp function
        TODO: use avgCellDryMassInit and Km to account for endoRNAses (used where Km is not None),
        see synth_aff_prob_from_exp(). Need to get loss rates, but without actual counts somehow,
        maybe some iterative procedure if not overkill? Also need to get avgCellDryMassInit before
        actually having expression.
        - copy_number should be sim_data.process.replication.get_average_copy_number, sending this
        improves efficiency by getting rid of pickling step
        '''
        transcription = sim_data.process.transcription
        # Calculate synthesis affinities
        ppgpp = sim_data.growth_rate_parameters.get_ppGpp_conc(doubling_time)
        synth_aff = sim_data.process.transcription.synth_aff_from_ppgpp(ppgpp)

        # Calculate synthesis probabilities
        rna_coords = transcription.rna_data["replication_coordinate"]
        tau = doubling_time.asNumber(units.min)
        copy_numbers = sim_data.process.replication.get_average_copy_number(tau, rna_coords)

        synth_prob = synth_aff * copy_numbers
        synth_prob = normalize(synth_prob)

        ## Calculate expression
        # TODO: account for endoRNAses, right now it makes the assumption
        #  that Km=None and follows the loss rate assumption from
        #  netLossRateFromDilutionDegradationRNALinear()
        deg_rates = transcription.rna_data["deg_rate"]
        loss_rate = (np.log(2) / doubling_time + deg_rates)

        expression = synth_prob / loss_rate
        expression = normalize(expression)

        return expression

    def synth_aff_from_ppgpp(self, ppgpp, balanced_rRNA_prob=True):
        """
        Calculates the synthesis affinity of each gene at a given concentration
        of ppGpp.

        Args:
                ppgpp (float with mol / volume units): concentration of ppGpp
                balanced_rRNA_prob (bool): if True, set synthesis affinities
                    of rRNA promoters equal to one another

        Returns
                aff (ndarray[float]): synthesis affinity for each gene
        """

        ppgpp = ppgpp.asNumber(PPGPP_CONC_UNITS)
        f_ppgpp = self.fraction_rnap_bound_ppgpp(ppgpp)

        # TODO: make this involve active fraction of RNAPs too
        aff = self.aff_free * f_ppgpp + self.aff_ppgpp * (1 - f_ppgpp)

        if balanced_rRNA_prob:
            aff[self.rna_data["is_rRNA"]] = aff[self.rna_data["is_rRNA"]].mean()

        return aff

    def get_rnap_active_fraction_from_ppGpp(self, ppgpp):
        f_ppgpp = self.fraction_rnap_bound_ppgpp(ppgpp)
        return (
            self.fraction_active_rnap_bound * f_ppgpp
            + self.fraction_active_rnap_free * (1 - f_ppgpp)
        )

    def _build_new_gene_data(self, raw_data, sim_data):
        """
        Load baseline values for new gene expression in all simulations.
        """

        self.new_gene_expression_baselines = (
            raw_data.new_gene_data.new_gene_baseline_expression_parameters
        )

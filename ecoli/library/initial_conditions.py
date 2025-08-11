"""
Functions to initialize molecule states from sim_data.
"""

import numpy as np
import numpy.typing as npt
from numpy.lib import recfunctions as rfn
from typing import Any
from unum import Unum

from ecoli.library.schema import (
    attrs,
    bulk_name_to_idx,
    counts,
    MetadataArray,
)
from ecoli.processes.polypeptide_elongation import (
    calculate_trna_charging,
    REMOVED_FROM_CHARGING,
    MICROMOLAR_UNITS,
)
from wholecell.utils import units
from wholecell.utils.fitting import (
    countsFromMassAndExpression,
    masses_and_counts_for_homeostatic_target,
    normalize,
)

try:
    from wholecell.utils.mc_complexation import mccFormComplexesWithPrebuiltMatrices
except ImportError as exc:
    raise RuntimeError(
        "Failed to import Cython module. Try running 'make clean compile'."
    ) from exc
from wholecell.utils.polymerize import computeMassIncrease
from wholecell.utils.random import stochasticRound

RAND_MAX = 2**31


def create_bulk_container(
    sim_data,
    n_seeds=1,
    condition=None,
    seed=0,
    ppgpp_regulation=True,
    trna_attenuation=True,
    mass_coeff=1,
    form_complexes=True,
):
    try:
        old_condition = sim_data.condition
        if condition is not None:
            sim_data.condition = condition
        media_id = sim_data.conditions[sim_data.condition]["nutrients"]
        exchange_data = sim_data.external_state.exchange_data_from_media(media_id)
        import_molecules = set(
            exchange_data["importUnconstrainedExchangeMolecules"]
        ) | set(exchange_data["importConstrainedExchangeMolecules"])

        random_state = np.random.RandomState(seed=seed)

        # Construct bulk container
        ids_molecules = sim_data.internal_state.bulk_molecules.bulk_data["id"]
        average_container = np.array(
            [mol_data for mol_data in zip(ids_molecules, np.zeros(len(ids_molecules)))],
            dtype=[("id", ids_molecules.dtype), ("count", np.float64)],
        )

        for n in range(n_seeds):
            random_state = np.random.RandomState(seed=seed + n)
            average_container["count"] += initialize_bulk_counts(
                sim_data,
                media_id,
                import_molecules,
                random_state,
                mass_coeff,
                ppgpp_regulation,
                trna_attenuation,
                form_complexes=form_complexes,
            )["count"]
    except Exception:
        raise RuntimeError(
            "sim_data might not be fully initialized. "
            "Make sure all attributes have been set before "
            "using this function."
        )

    sim_data.condition = old_condition
    average_container["count"] = average_container["count"] / n_seeds
    return average_container


def initialize_bulk_counts(
    sim_data,
    media_id,
    import_molecules,
    random_state,
    mass_coeff,
    ppgpp_regulation,
    trna_attenuation,
    form_complexes=True,
):
    # Allocate count array to populate
    bulk_counts = np.zeros(
        len(sim_data.internal_state.bulk_molecules.bulk_data["id"]), dtype=int
    )

    # Set protein counts from expression
    initialize_protein_monomers(
        bulk_counts,
        sim_data,
        random_state,
        mass_coeff,
        ppgpp_regulation,
        trna_attenuation,
    )

    # Set RNA counts from expression
    initialize_rna(
        bulk_counts,
        sim_data,
        random_state,
        mass_coeff,
        ppgpp_regulation,
        trna_attenuation,
    )

    # Set mature RNA counts
    initialize_mature_RNA(bulk_counts, sim_data)

    # Set other biomass components
    set_small_molecule_counts(
        bulk_counts, sim_data, media_id, import_molecules, mass_coeff
    )

    # Form complexes
    if form_complexes:
        initialize_complexation(bulk_counts, sim_data, random_state)

    bulk_masses = sim_data.internal_state.bulk_molecules.bulk_data["mass"].asNumber(
        units.fg / units.mol
    ) / sim_data.constants.n_avogadro.asNumber(1 / units.mol)
    bulk_submasses = []
    bulk_submass_dtypes = []
    for submass, idx in sim_data.submass_name_to_index.items():
        bulk_submasses.append(bulk_masses[:, idx])
        bulk_submass_dtypes.append((f"{submass}_submass", np.float64))
    bulk_ids = sim_data.internal_state.bulk_molecules.bulk_data.struct_array["id"]
    bulk_array = np.array(
        [mol_data for mol_data in zip(bulk_ids, bulk_counts, *bulk_submasses)],
        dtype=[("id", bulk_ids.dtype), ("count", int)] + bulk_submass_dtypes,
    )

    return bulk_array


def initialize_unique_molecules(
    bulk_state,
    sim_data,
    cell_mass,
    random_state,
    unique_id_rng,
    superhelical_density,
    ppgpp_regulation,
    trna_attenuation,
    mechanistic_replisome,
):
    unique_molecules = {}

    # Initialize counts of full chromosomes
    initialize_full_chromosome(unique_molecules, sim_data, unique_id_rng)

    # Initialize unique molecules relevant to replication
    initialize_replication(
        bulk_state,
        unique_molecules,
        sim_data,
        cell_mass,
        mechanistic_replisome,
        unique_id_rng,
    )

    # Initialize bound transcription factors
    initialize_transcription_factors(
        bulk_state, unique_molecules, sim_data, random_state
    )

    # Initialize active RNAPs and unique molecule representations of RNAs
    initialize_transcription(
        bulk_state,
        unique_molecules,
        sim_data,
        random_state,
        unique_id_rng,
        ppgpp_regulation,
        trna_attenuation,
    )

    # Initialize linking numbers of chromosomal segments
    if superhelical_density:
        initialize_chromosomal_segments(unique_molecules, sim_data, unique_id_rng)
    else:
        unique_molecules["chromosomal_segment"] = create_new_unique_molecules(
            "chromosomal_segment", 0, sim_data, unique_id_rng
        )

    # Initialize active ribosomes
    initialize_translation(
        bulk_state, unique_molecules, sim_data, random_state, unique_id_rng
    )

    return unique_molecules


def create_new_unique_molecules(name, n_mols, sim_data, random_state, **attrs):
    """
    Helper function to create a new Numpy structured array with n_mols
    instances of the unique molecule called name. Accepts keyword arguments
    that become initial values for specified attributes of the new molecules.
    """
    dtypes = list(
        sim_data.internal_state.unique_molecule.unique_molecule_definitions[
            name
        ].items()
    )
    submasses = list(sim_data.submass_name_to_index)
    dtypes += [(f"massDiff_{submass}", "<f8") for submass in submasses]
    dtypes += [("_entryState", "i1"), ("unique_index", "<i8")]
    unique_mols = np.zeros(n_mols, dtype=dtypes)
    for attr_name, attr_value in attrs.items():
        unique_mols[attr_name] = attr_value
    # Each unique molecule has unique prefix for indices to prevent conflicts
    unique_mol_names = list(
        sim_data.internal_state.unique_molecule.unique_molecule_definitions.keys()
    )
    unique_prefix = unique_mol_names.index(name) << 59
    unique_mols["unique_index"] = np.arange(unique_prefix, unique_prefix + n_mols)
    unique_mols["_entryState"] = 1
    unique_mols = MetadataArray(unique_mols, unique_prefix + n_mols)
    return unique_mols


def initialize_protein_monomers(
    bulk_counts, sim_data, random_state, mass_coeff, ppgpp_regulation, trna_attenuation
):
    monomer_mass = (
        mass_coeff
        * sim_data.mass.get_component_masses(
            sim_data.condition_to_doubling_time[sim_data.condition]
        )["proteinMass"]
        / sim_data.mass.avg_cell_to_initial_cell_conversion_factor
    )
    # TODO: unify this logic with the parca so it doesn]t fall out of step
    # again (look at teh calProteinCounts function)

    transcription = sim_data.process.transcription
    if ppgpp_regulation:
        rna_expression = sim_data.calculate_ppgpp_expression(sim_data.condition)
    else:
        rna_expression = transcription.rna_expression[sim_data.condition]

    if trna_attenuation:
        # Need to adjust expression (calculated without attenuation) by basal_adjustment
        # to get the expected expression without any attenuation and then multiply
        # by the condition readthrough probability to get the condition specific expression
        readthrough = transcription.attenuation_readthrough[sim_data.condition]
        basal_adjustment = transcription.attenuation_readthrough["basal"]
        rna_expression[transcription.attenuated_rna_indices] *= (
            readthrough / basal_adjustment
        )

    monomer_expression = normalize(
        sim_data.process.transcription.cistron_tu_mapping_matrix.dot(rna_expression)[
            sim_data.relation.cistron_to_monomer_mapping
        ]
        * sim_data.process.translation.translation_efficiencies_by_monomer
        / (
            np.log(2)
            / sim_data.condition_to_doubling_time[sim_data.condition].asNumber(units.s)
            + sim_data.process.translation.monomer_data["deg_rate"].asNumber(
                1 / units.s
            )
        )
    )

    n_monomers = countsFromMassAndExpression(
        monomer_mass.asNumber(units.g),
        sim_data.process.translation.monomer_data["mw"].asNumber(units.g / units.mol),
        monomer_expression,
        sim_data.constants.n_avogadro.asNumber(1 / units.mol),
    )

    # Get indices for monomers in bulk counts array
    monomer_ids = sim_data.process.translation.monomer_data["id"]
    bulk_ids = sim_data.internal_state.bulk_molecules.bulk_data["id"]
    monomer_idx = bulk_name_to_idx(monomer_ids, bulk_ids)
    # Calculate initial counts of each monomer from mutinomial distribution
    bulk_counts[monomer_idx] = random_state.multinomial(n_monomers, monomer_expression)


def initialize_rna(
    bulk_counts, sim_data, random_state, mass_coeff, ppgpp_regulation, trna_attenuation
):
    """
    Initializes counts of RNAs in the bulk molecule container using RNA
    expression data. mRNA counts are also initialized here, but is later reset
    to zero when the representations for mRNAs are moved to the unique molecule
    container.
    """

    transcription = sim_data.process.transcription

    rna_mass = (
        mass_coeff
        * sim_data.mass.get_component_masses(
            sim_data.condition_to_doubling_time[sim_data.condition]
        )["rnaMass"]
        / sim_data.mass.avg_cell_to_initial_cell_conversion_factor
    )

    if ppgpp_regulation:
        rna_expression = sim_data.calculate_ppgpp_expression(sim_data.condition)
    else:
        rna_expression = normalize(transcription.rna_expression[sim_data.condition])

    if trna_attenuation:
        # Need to adjust expression (calculated without attenuation) by basal_adjustment
        # to get the expected expression without any attenuation and then multiply
        # by the condition readthrough probability to get the condition specific expression
        readthrough = transcription.attenuation_readthrough[sim_data.condition]
        basal_adjustment = transcription.attenuation_readthrough["basal"]
        rna_expression[transcription.attenuated_rna_indices] *= (
            readthrough / basal_adjustment
        )
        rna_expression /= rna_expression.sum()

    n_rnas = countsFromMassAndExpression(
        rna_mass.asNumber(units.g),
        transcription.rna_data["mw"].asNumber(units.g / units.mol),
        rna_expression,
        sim_data.constants.n_avogadro.asNumber(1 / units.mol),
    )

    # Get indices for monomers in bulk counts array
    rna_ids = transcription.rna_data["id"]
    bulk_ids = sim_data.internal_state.bulk_molecules.bulk_data["id"]
    rna_idx = bulk_name_to_idx(rna_ids, bulk_ids)
    # Calculate initial counts of each RNA from mutinomial distribution
    bulk_counts[rna_idx] = random_state.multinomial(n_rnas, rna_expression)


def initialize_mature_RNA(bulk_counts, sim_data):
    """
    Initializes counts of mature RNAs in the bulk molecule container using the
    counts of unprocessed RNAs. Also consolidates the different variants of each
    rRNA molecule into the main type.
    """
    transcription = sim_data.process.transcription
    rna_data = transcription.rna_data
    unprocessed_rna_ids = rna_data["id"][rna_data["is_unprocessed"]]
    bulk_ids = sim_data.internal_state.bulk_molecules.bulk_data["id"]
    unprocessed_rna_idx = bulk_name_to_idx(unprocessed_rna_ids, bulk_ids)

    # Skip if there are no unprocessed RNAs represented
    if len(unprocessed_rna_ids) > 0:
        mature_rna_ids = transcription.mature_rna_data["id"]
        maturation_stoich_matrix = transcription.rna_maturation_stoich_matrix
        mature_rna_idx = bulk_name_to_idx(mature_rna_ids, bulk_ids)

        # Get counts of unprocessed RNAs
        unprocessed_rna_counts = bulk_counts[unprocessed_rna_idx]

        # Assume all unprocessed RNAs are converted to mature RNAs
        bulk_counts[unprocessed_rna_idx] = 0
        bulk_counts[mature_rna_idx] += maturation_stoich_matrix.dot(
            unprocessed_rna_counts
        )

    # Get IDs of rRNAs
    main_23s_rRNA_id = sim_data.molecule_groups.s50_23s_rRNA[0]
    main_16s_rRNA_id = sim_data.molecule_groups.s30_16s_rRNA[0]
    main_5s_rRNA_id = sim_data.molecule_groups.s50_5s_rRNA[0]
    variant_23s_rRNA_ids = sim_data.molecule_groups.s50_23s_rRNA[1:]
    variant_16s_rRNA_ids = sim_data.molecule_groups.s30_16s_rRNA[1:]
    variant_5s_rRNA_ids = sim_data.molecule_groups.s50_5s_rRNA[1:]

    # Get indices of main and variant rRNAs
    main_23s_rRNA_idx = bulk_name_to_idx(main_23s_rRNA_id, bulk_ids)
    main_16s_rRNA_idx = bulk_name_to_idx(main_16s_rRNA_id, bulk_ids)
    main_5s_rRNA_idx = bulk_name_to_idx(main_5s_rRNA_id, bulk_ids)
    variant_23s_rRNA_idx = bulk_name_to_idx(variant_23s_rRNA_ids, bulk_ids)
    variant_16s_rRNA_idx = bulk_name_to_idx(variant_16s_rRNA_ids, bulk_ids)
    variant_5s_rRNA_idx = bulk_name_to_idx(variant_5s_rRNA_ids, bulk_ids)

    # Evolve states
    bulk_counts[main_23s_rRNA_idx] += bulk_counts[variant_23s_rRNA_idx].sum()
    bulk_counts[main_16s_rRNA_idx] += bulk_counts[variant_16s_rRNA_idx].sum()
    bulk_counts[main_5s_rRNA_idx] += bulk_counts[variant_5s_rRNA_idx].sum()
    bulk_counts[variant_23s_rRNA_idx] -= bulk_counts[variant_23s_rRNA_idx]
    bulk_counts[variant_16s_rRNA_idx] -= bulk_counts[variant_16s_rRNA_idx]
    bulk_counts[variant_5s_rRNA_idx] -= bulk_counts[variant_5s_rRNA_idx]


# TODO: remove checks for zero concentrations (change to assertion)
# TODO: move any rescaling logic to KB/fitting
def set_small_molecule_counts(
    bulk_counts, sim_data, media_id, import_molecules, mass_coeff, cell_mass=None
):
    doubling_time = sim_data.condition_to_doubling_time[sim_data.condition]

    conc_dict = sim_data.process.metabolism.concentration_updates.concentrations_based_on_nutrients(
        media_id=media_id, imports=import_molecules
    )
    conc_dict.update(sim_data.mass.getBiomassAsConcentrations(doubling_time))
    conc_dict[sim_data.molecule_ids.ppGpp] = (
        sim_data.growth_rate_parameters.get_ppGpp_conc(doubling_time)
    )
    molecule_ids = sorted(conc_dict)
    bulk_ids = sim_data.internal_state.bulk_molecules.bulk_data["id"]
    molecule_concentrations = (units.mol / units.L) * np.array(
        [conc_dict[key].asNumber(units.mol / units.L) for key in molecule_ids]
    )

    if cell_mass is None:
        avg_cell_fraction_mass = sim_data.mass.get_component_masses(doubling_time)
        other_dry_mass = (
            mass_coeff
            * (
                avg_cell_fraction_mass["proteinMass"]
                + avg_cell_fraction_mass["rnaMass"]
                + avg_cell_fraction_mass["dnaMass"]
            )
            / sim_data.mass.avg_cell_to_initial_cell_conversion_factor
        )
    else:
        small_molecule_mass = 0 * units.fg
        for mol in conc_dict:
            mol_idx = bulk_name_to_idx(mol, bulk_ids)
            small_molecule_mass += (
                bulk_counts[mol_idx]
                * sim_data.getter.get_mass(mol)
                / sim_data.constants.n_avogadro
            )
        other_dry_mass = cell_mass - small_molecule_mass

    masses_to_add, counts_to_add = masses_and_counts_for_homeostatic_target(
        other_dry_mass,
        molecule_concentrations,
        sim_data.getter.get_masses(molecule_ids),
        sim_data.constants.cell_density,
        sim_data.constants.n_avogadro,
    )

    molecule_idx = bulk_name_to_idx(molecule_ids, bulk_ids)
    bulk_counts[molecule_idx] = counts_to_add


def initialize_complexation(bulk_counts, sim_data, random_state):
    molecule_names = sim_data.process.complexation.molecule_names
    bulk_ids = sim_data.internal_state.bulk_molecules.bulk_data["id"]
    molecule_idx = bulk_name_to_idx(molecule_names, bulk_ids)

    stoich_matrix = sim_data.process.complexation.stoich_matrix().astype(
        np.int64, order="F"
    )

    molecule_counts = bulk_counts[molecule_idx]
    updated_molecule_counts, complexation_events = mccFormComplexesWithPrebuiltMatrices(
        molecule_counts,
        random_state.randint(1000),
        stoich_matrix,
        *sim_data.process.complexation.prebuilt_matrices,
    )

    bulk_counts[molecule_idx] = updated_molecule_counts

    if np.any(updated_molecule_counts < 0):
        raise ValueError("Negative counts after complexation")


def initialize_full_chromosome(unique_molecules, sim_data, unique_id_rng):
    """
    Initializes the counts of full chromosomes to one. The division_time of
    this initial chromosome is set to be zero for consistency.
    """
    unique_molecules["full_chromosome"] = create_new_unique_molecules(
        "full_chromosome",
        1,
        sim_data,
        unique_id_rng,
        division_time=0.0,
        has_triggered_division=True,
        domain_index=0,
    )


def initialize_replication(
    bulk_state,
    unique_molecules,
    sim_data,
    cell_mass,
    mechanistic_replisome,
    unique_id_rng,
):
    """
    Initializes replication by creating an appropriate number of replication
    forks given the cell growth rate. This also initializes the gene dosage
    bulk counts using the initial locations of the forks.
    """
    # Determine the number and location of replication forks at the start of
    # the cell cycle
    # Get growth rate constants
    tau = sim_data.condition_to_doubling_time[sim_data.condition].asUnit(units.min)
    critical_mass = sim_data.mass.get_dna_critical_mass(tau)
    replication_rate = sim_data.process.replication.basal_elongation_rate

    # Calculate length of replichore
    genome_length = sim_data.process.replication.genome_length
    replichore_length = np.ceil(0.5 * genome_length) * units.nt

    # Calculate the maximum number of replisomes that could be formed with
    # the existing counts of replisome subunits. If mechanistic_replisome option
    # is off, set to an arbitrary high number.
    replisome_trimer_idx = bulk_name_to_idx(
        sim_data.molecule_groups.replisome_trimer_subunits, bulk_state["id"]
    )
    replisome_monomer_idx = bulk_name_to_idx(
        sim_data.molecule_groups.replisome_monomer_subunits, bulk_state["id"]
    )
    if mechanistic_replisome:
        n_max_replisomes = np.min(
            np.concatenate(
                (
                    bulk_state["count"][replisome_trimer_idx] // 3,
                    bulk_state["count"][replisome_monomer_idx],
                )
            )
        )
    else:
        n_max_replisomes = 1000

    # Generate arrays specifying appropriate initial replication conditions
    oric_state, replisome_state, domain_state = determine_chromosome_state(
        tau,
        replichore_length,
        n_max_replisomes,
        sim_data.process.replication.no_child_place_holder,
        cell_mass,
        critical_mass,
        replication_rate,
    )

    n_oric = oric_state["domain_index"].size
    n_replisome = replisome_state["domain_index"].size
    n_domain = domain_state["domain_index"].size

    # Add OriC molecules with the proposed attributes
    unique_molecules["oriC"] = create_new_unique_molecules(
        "oriC", n_oric, sim_data, unique_id_rng, domain_index=oric_state["domain_index"]
    )

    # Add chromosome domain molecules with the proposed attributes
    unique_molecules["chromosome_domain"] = create_new_unique_molecules(
        "chromosome_domain",
        n_domain,
        sim_data,
        unique_id_rng,
        domain_index=domain_state["domain_index"],
        child_domains=domain_state["child_domains"],
    )

    if n_replisome != 0:
        # Update mass of replisomes if the mechanistic replisome option is set
        if mechanistic_replisome:
            replisome_trimer_subunit_masses = np.vstack(
                [
                    sim_data.getter.get_submass_array(x).asNumber(
                        units.fg / units.count
                    )
                    for x in sim_data.molecule_groups.replisome_trimer_subunits
                ]
            )
            replisome_monomer_subunit_masses = np.vstack(
                [
                    sim_data.getter.get_submass_array(x).asNumber(
                        units.fg / units.count
                    )
                    for x in sim_data.molecule_groups.replisome_monomer_subunits
                ]
            )
            replisome_mass_array = 3 * replisome_trimer_subunit_masses.sum(
                axis=0
            ) + replisome_monomer_subunit_masses.sum(axis=0)
            replisome_protein_mass = replisome_mass_array.sum()
        else:
            replisome_protein_mass = 0.0

        # Update mass to account for DNA strands that have already been
        # elongated.
        sequences = sim_data.process.replication.replication_sequences
        fork_coordinates = replisome_state["coordinates"]
        sequence_elongations = np.abs(np.repeat(fork_coordinates, 2))

        mass_increase_dna = computeMassIncrease(
            np.tile(sequences, (n_replisome // 2, 1)),
            sequence_elongations,
            sim_data.process.replication.replication_monomer_weights.asNumber(units.fg),
        )

        # Add active replisomes as unique molecules and set attributes
        unique_molecules["active_replisome"] = create_new_unique_molecules(
            "active_replisome",
            n_replisome,
            sim_data,
            unique_id_rng,
            domain_index=replisome_state["domain_index"],
            coordinates=replisome_state["coordinates"],
            right_replichore=replisome_state["right_replichore"],
            massDiff_DNA=mass_increase_dna[0::2] + mass_increase_dna[1::2],
            massDiff_protein=replisome_protein_mass,
        )

        if mechanistic_replisome:
            # Remove replisome subunits from bulk molecules
            bulk_state["count"][replisome_trimer_idx] -= 3 * n_replisome
            bulk_state["count"][replisome_monomer_idx] -= n_replisome
    else:
        # For n_replisome = 0, still create an empty structured array with
        # the expected fields
        unique_molecules["active_replisome"] = create_new_unique_molecules(
            "active_replisome", n_replisome, sim_data, unique_id_rng
        )

    # Get coordinates of all genes, promoters and DnaA boxes
    all_gene_coordinates = sim_data.process.transcription.cistron_data[
        "replication_coordinate"
    ]
    all_promoter_coordinates = sim_data.process.transcription.rna_data[
        "replication_coordinate"
    ]
    all_DnaA_box_coordinates = sim_data.process.replication.motif_coordinates[
        "DnaA_box"
    ]

    # Define function that initializes attributes of sequence motifs given the
    # initial state of the chromosome
    def get_motif_attributes(all_motif_coordinates):
        """
        Using the initial positions of replication forks, calculate attributes
        of unique molecules representing DNA motifs, given their positions on
        the genome.

        Args:
            all_motif_coordinates (ndarray): Genomic coordinates of DNA motifs,
            represented in a specific order

        Returns:
            motif_index: Indices of all motif copies, in the case where
            different indexes imply a different functional role
            motif_coordinates: Genomic coordinates of all motif copies
            motif_domain_index: Domain indexes of the chromosome domain that
            each motif copy belongs to
        """
        motif_index, motif_coordinates, motif_domain_index = [], [], []

        def in_bounds(coordinates, lb, ub):
            return np.logical_and(coordinates < ub, coordinates > lb)

        # Loop through all chromosome domains
        for domain_idx in domain_state["domain_index"]:
            # If the domain is the mother domain of the initial chromosome,
            if domain_idx == 0:
                if n_replisome == 0:
                    # No replisomes - all motifs should fall in this domain
                    motif_mask = np.ones_like(all_motif_coordinates, dtype=bool)

                else:
                    # Get domain boundaries
                    domain_boundaries = replisome_state["coordinates"][
                        replisome_state["domain_index"] == 0
                    ]

                    # Add motifs outside of this boundary
                    motif_mask = np.logical_or(
                        all_motif_coordinates > domain_boundaries.max(),
                        all_motif_coordinates < domain_boundaries.min(),
                    )

            # If the domain contains the origin,
            elif np.isin(domain_idx, oric_state["domain_index"]):
                # Get index of the parent domain
                parent_domain_idx = domain_state["domain_index"][
                    np.where(domain_state["child_domains"] == domain_idx)[0]
                ]

                # Get domain boundaries of the parent domain
                parent_domain_boundaries = replisome_state["coordinates"][
                    replisome_state["domain_index"] == parent_domain_idx
                ]

                # Add motifs inside this boundary
                motif_mask = in_bounds(
                    all_motif_coordinates,
                    parent_domain_boundaries.min(),
                    parent_domain_boundaries.max(),
                )

            # If the domain neither contains the origin nor the terminus,
            else:
                # Get index of the parent domain
                parent_domain_idx = domain_state["domain_index"][
                    np.where(domain_state["child_domains"] == domain_idx)[0]
                ]

                # Get domain boundaries of the parent domain
                parent_domain_boundaries = replisome_state["coordinates"][
                    replisome_state["domain_index"] == parent_domain_idx
                ]

                # Get domain boundaries of this domain
                domain_boundaries = replisome_state["coordinates"][
                    replisome_state["domain_index"] == domain_idx
                ]

                # Add motifs between the boundaries
                motif_mask = np.logical_or(
                    in_bounds(
                        all_motif_coordinates,
                        domain_boundaries.max(),
                        parent_domain_boundaries.max(),
                    ),
                    in_bounds(
                        all_motif_coordinates,
                        parent_domain_boundaries.min(),
                        domain_boundaries.min(),
                    ),
                )

            # Append attributes to existing list
            motif_index.extend(np.nonzero(motif_mask)[0])
            motif_coordinates.extend(all_motif_coordinates[motif_mask])
            motif_domain_index.extend(np.full(motif_mask.sum(), domain_idx))

        return motif_index, motif_coordinates, motif_domain_index

    # Use function to get attributes for promoters and DnaA boxes
    TU_index, promoter_coordinates, promoter_domain_index = get_motif_attributes(
        all_promoter_coordinates
    )
    cistron_index, gene_coordinates, gene_domain_index = get_motif_attributes(
        all_gene_coordinates
    )
    _, DnaA_box_coordinates, DnaA_box_domain_index = get_motif_attributes(
        all_DnaA_box_coordinates
    )

    # Add promoters as unique molecules and set attributes
    # Note: the bound_TF attribute is properly initialized in the function
    # initialize_transcription_factors
    n_promoter = len(TU_index)
    n_tf = len(sim_data.process.transcription_regulation.tf_ids)

    unique_molecules["promoter"] = create_new_unique_molecules(
        "promoter",
        n_promoter,
        sim_data,
        unique_id_rng,
        domain_index=promoter_domain_index,
        coordinates=promoter_coordinates,
        TU_index=TU_index,
        bound_TF=np.zeros((n_promoter, n_tf), dtype=bool),
    )

    # Add genes as unique molecules and set attributes
    n_gene = len(cistron_index)

    unique_molecules["gene"] = create_new_unique_molecules(
        "gene",
        n_gene,
        sim_data,
        unique_id_rng,
        cistron_index=cistron_index,
        coordinates=gene_coordinates,
        domain_index=gene_domain_index,
    )

    # Add DnaA boxes as unique molecules and set attributes
    n_DnaA_box = len(DnaA_box_coordinates)

    unique_molecules["DnaA_box"] = create_new_unique_molecules(
        "DnaA_box",
        n_DnaA_box,
        sim_data,
        unique_id_rng,
        domain_index=DnaA_box_domain_index,
        coordinates=DnaA_box_coordinates,
        DnaA_bound=np.zeros(n_DnaA_box, dtype=bool),
    )


def initialize_transcription_factors(
    bulk_state, unique_molecules, sim_data, random_state
):
    """
    Initialize transcription factors that are bound to the chromosome. For each
    type of transcription factor, this function calculates the total number of
    transcription factors that should be bound to the chromosome using the
    binding probabilities of each transcription factor and the number of
    available promoter sites. The calculated number of transcription factors
    are then distributed randomly to promoters, whose bound_TF attributes and
    submasses are updated correspondingly.
    """
    # Get transcription factor properties from sim_data
    tf_ids = sim_data.process.transcription_regulation.tf_ids
    tf_to_tf_type = sim_data.process.transcription_regulation.tf_to_tf_type
    p_promoter_bound_TF = sim_data.process.transcription_regulation.p_promoter_bound_tf

    # Build dict that maps TFs to transcription units they regulate
    delta_prob = sim_data.process.transcription_regulation.delta_prob
    TF_to_TU_idx = {}

    for i, tf in enumerate(tf_ids):
        TF_to_TU_idx[tf] = delta_prob["deltaI"][delta_prob["deltaJ"] == i]

    # Get views into bulk molecule representations of transcription factors
    active_tf_view = {}
    inactive_tf_view = {}
    active_tf_view_idx = {}
    inactive_tf_view_idx = {}

    for tf in tf_ids:
        tf_idx = bulk_name_to_idx(tf + "[c]", bulk_state["id"])
        active_tf_view[tf] = bulk_state["count"][tf_idx]
        active_tf_view_idx[tf] = tf_idx

        if tf_to_tf_type[tf] == "1CS":
            if tf == sim_data.process.transcription_regulation.active_to_bound[tf]:
                inactive_tf_idx = bulk_name_to_idx(
                    sim_data.process.equilibrium.get_unbound(tf + "[c]"),
                    bulk_state["id"],
                )
                inactive_tf_view[tf] = bulk_state["count"][inactive_tf_idx]
            else:
                inactive_tf_idx = bulk_name_to_idx(
                    sim_data.process.transcription_regulation.active_to_bound[tf]
                    + "[c]",
                    bulk_state["id"],
                )
                inactive_tf_view[tf] = bulk_state["count"][inactive_tf_idx]
        elif tf_to_tf_type[tf] == "2CS":
            inactive_tf_idx = bulk_name_to_idx(
                sim_data.process.two_component_system.active_to_inactive_tf[tf + "[c]"],
                bulk_state["id"],
            )
            inactive_tf_view[tf] = bulk_state["count"][inactive_tf_idx]
        inactive_tf_view_idx[tf] = inactive_tf_idx

    # Get masses of active transcription factors
    tf_indexes = [np.where(bulk_state["id"] == tf_id + "[c]")[0][0] for tf_id in tf_ids]
    active_tf_masses = (
        sim_data.internal_state.bulk_molecules.bulk_data["mass"][tf_indexes]
        / sim_data.constants.n_avogadro
    ).asNumber(units.fg)

    # Get TU indices of promoters
    TU_index = unique_molecules["promoter"]["TU_index"]

    # Initialize bound_TF array
    bound_TF = np.zeros((len(TU_index), len(tf_ids)), dtype=bool)

    for tf_idx, tf_id in enumerate(tf_ids):
        # Get counts of transcription factors
        active_tf_counts = active_tf_view[tf_id]

        # If there are no active transcription factors at initialization,
        # continue to the next transcription factor
        if active_tf_counts == 0:
            continue

        # Compute probability of binding the promoter
        if tf_to_tf_type[tf_id] == "0CS":
            p_promoter_bound = 1.0
        else:
            inactive_tf_counts = inactive_tf_view[tf_id]
            p_promoter_bound = p_promoter_bound_TF(active_tf_counts, inactive_tf_counts)

        # Determine the number of available promoter sites
        available_promoters = np.isin(TU_index, TF_to_TU_idx[tf_id])
        n_available_promoters = available_promoters.sum()

        # Calculate the number of promoters that should be bound
        n_to_bind = int(
            stochasticRound(
                random_state, np.full(n_available_promoters, p_promoter_bound)
            ).sum()
        )

        bound_locs = np.zeros(n_available_promoters, dtype=bool)
        if n_to_bind > 0:
            # Determine randomly which DNA targets to bind based on which of
            # the following is more limiting:
            # number of promoter sites to bind, or number of active
            # transcription factors
            bound_locs[
                random_state.choice(
                    n_available_promoters,
                    size=min(n_to_bind, active_tf_view[tf_id]),
                    replace=False,
                )
            ] = True

            # Update count of free transcription factors
            bulk_state["count"][active_tf_view_idx[tf_id]] -= bound_locs.sum()

            # Update bound_TF array
            bound_TF[available_promoters, tf_idx] = bound_locs

    # Calculate masses of bound TFs
    mass_diffs = bound_TF.dot(active_tf_masses)

    # Reset bound_TF attribute of promoters
    unique_molecules["promoter"]["bound_TF"] = bound_TF

    # Add mass_diffs array to promoter submass
    for submass, i in sim_data.submass_name_to_index.items():
        unique_molecules["promoter"][f"massDiff_{submass}"] = mass_diffs[:, i]


def initialize_transcription(
    bulk_state,
    unique_molecules,
    sim_data,
    random_state,
    unique_id_rng,
    ppgpp_regulation,
    trna_attenuation,
):
    """
    Activate RNA polymerases as unique molecules, and distribute them along
    lengths of trancription units, while decreasing counts of unactivated RNA
    polymerases (APORNAP-CPLX[c]). Also initialize unique molecule
    representations of fully transcribed mRNAs and partially transcribed RNAs,
    using counts of mRNAs initialized as bulk molecules, and the attributes of
    initialized RNA polymerases. The counts of full mRNAs represented as bulk
    molecules are reset to zero.

    RNA polymerases are placed randomly across the length of each transcription
    unit, with the synthesis probabilities for each TU determining the number of
    RNA polymerases placed at each gene.
    """
    # Load parameters
    rna_lengths = sim_data.process.transcription.rna_data["length"].asNumber()
    rna_masses = (
        sim_data.process.transcription.rna_data["mw"] / sim_data.constants.n_avogadro
    ).asNumber(units.fg)
    current_media_id = sim_data.conditions[sim_data.condition]["nutrients"]
    frac_active_rnap = sim_data.process.transcription.rnapFractionActiveDict[
        current_media_id
    ]
    inactive_rnap_idx = bulk_name_to_idx(
        sim_data.molecule_ids.full_RNAP, bulk_state["id"]
    )
    inactive_RNAP_counts = bulk_state["count"][inactive_rnap_idx]
    rna_sequences = sim_data.process.transcription.transcription_sequences
    nt_weights = sim_data.process.transcription.transcription_monomer_weights
    end_weight = sim_data.process.transcription.transcription_end_weight
    replichore_lengths = sim_data.process.replication.replichore_lengths
    chromosome_length = replichore_lengths.sum()

    # Number of rnaPoly to activate
    n_RNAPs_to_activate = np.int64(frac_active_rnap * inactive_RNAP_counts)

    # Get attributes of promoters
    TU_index, bound_TF, domain_index_promoters = attrs(
        unique_molecules["promoter"], ["TU_index", "bound_TF", "domain_index"]
    )

    # Parameters for rnaSynthProb
    if ppgpp_regulation:
        doubling_time = sim_data.condition_to_doubling_time[sim_data.condition]
        ppgpp_conc = sim_data.growth_rate_parameters.get_ppGpp_conc(doubling_time)
        basal_prob, _ = sim_data.process.transcription.synth_prob_from_ppgpp(
            ppgpp_conc, sim_data.process.replication.get_average_copy_number
        )
        ppgpp_scale = basal_prob[TU_index]
        # Use original delta prob if no ppGpp basal prob
        ppgpp_scale[ppgpp_scale == 0] = 1
    else:
        basal_prob = sim_data.process.transcription_regulation.basal_prob.copy()
        ppgpp_scale = 1

    if trna_attenuation:
        basal_prob[sim_data.process.transcription.attenuated_rna_indices] += (
            sim_data.process.transcription.attenuation_basal_prob_adjustments
        )
    n_TUs = len(basal_prob)
    delta_prob_matrix = sim_data.process.transcription_regulation.get_delta_prob_matrix(
        dense=True, ppgpp=ppgpp_regulation
    )

    # Synthesis probabilities for different categories of genes
    rna_synth_prob_fractions = sim_data.process.transcription.rnaSynthProbFraction
    rna_synth_prob_R_protein = sim_data.process.transcription.rnaSynthProbRProtein
    rna_synth_prob_rna_polymerase = (
        sim_data.process.transcription.rnaSynthProbRnaPolymerase
    )

    # Get coordinates and transcription directions of transcription units
    replication_coordinate = sim_data.process.transcription.rna_data[
        "replication_coordinate"
    ]
    transcription_direction = sim_data.process.transcription.rna_data["is_forward"]

    # Determine changes from genetic perturbations
    genetic_perturbations = {}
    perturbations = getattr(sim_data, "genetic_perturbations", {})

    if len(perturbations) > 0:
        probability_indexes = [
            (index, sim_data.genetic_perturbations[rna_data["id"]])
            for index, rna_data in enumerate(sim_data.process.transcription.rna_data)
            if rna_data["id"] in sim_data.genetic_perturbations
        ]

        genetic_perturbations = {
            "fixedRnaIdxs": [pair[0] for pair in probability_indexes],
            "fixedSynthProbs": [pair[1] for pair in probability_indexes],
        }

    # ID Groups
    idx_rRNA = np.where(sim_data.process.transcription.rna_data["is_rRNA"])[0]
    idx_mRNA = np.where(sim_data.process.transcription.rna_data["is_mRNA"])[0]
    idx_tRNA = np.where(sim_data.process.transcription.rna_data["is_tRNA"])[0]
    idx_rprotein = np.where(
        sim_data.process.transcription.rna_data["includes_ribosomal_protein"]
    )[0]
    idx_rnap = np.where(sim_data.process.transcription.rna_data["includes_RNAP"])[0]

    # Calculate probabilities of the RNAP binding to the promoters
    promoter_init_probs = basal_prob[TU_index] + ppgpp_scale * np.multiply(
        delta_prob_matrix[TU_index, :], bound_TF
    ).sum(axis=1)

    if len(genetic_perturbations) > 0:
        rescale_initiation_probs(
            promoter_init_probs,
            TU_index,
            genetic_perturbations["fixedSynthProbs"],
            genetic_perturbations["fixedRnaIdxs"],
        )

    # Adjust probabilities to not be negative
    promoter_init_probs[promoter_init_probs < 0] = 0.0
    promoter_init_probs /= promoter_init_probs.sum()
    if np.any(promoter_init_probs < 0):
        raise Exception("Have negative RNA synthesis probabilities")

    # Adjust synthesis probabilities depending on environment
    synth_prob_fractions = rna_synth_prob_fractions[current_media_id]

    # Create masks for different types of RNAs
    is_mRNA = np.isin(TU_index, idx_mRNA)
    is_tRNA = np.isin(TU_index, idx_tRNA)
    is_rRNA = np.isin(TU_index, idx_rRNA)
    is_rprotein = np.isin(TU_index, idx_rprotein)
    is_rnap = np.isin(TU_index, idx_rnap)
    is_fixed = is_tRNA | is_rRNA | is_rprotein | is_rnap

    # Rescale initiation probabilities based on type of RNA
    promoter_init_probs[is_mRNA] *= (
        synth_prob_fractions["mRna"] / promoter_init_probs[is_mRNA].sum()
    )
    promoter_init_probs[is_tRNA] *= (
        synth_prob_fractions["tRna"] / promoter_init_probs[is_tRNA].sum()
    )
    promoter_init_probs[is_rRNA] *= (
        synth_prob_fractions["rRna"] / promoter_init_probs[is_rRNA].sum()
    )

    # Set fixed synthesis probabilities for RProteins and RNAPs
    rescale_initiation_probs(
        promoter_init_probs,
        TU_index,
        np.concatenate(
            (
                rna_synth_prob_R_protein[current_media_id],
                rna_synth_prob_rna_polymerase[current_media_id],
            )
        ),
        np.concatenate((idx_rprotein, idx_rnap)),
    )

    assert promoter_init_probs[is_fixed].sum() < 1.0

    # Adjust for attenuation that will stop transcription after initiation
    if trna_attenuation:
        attenuation_readthrough = {
            idx: prob
            for idx, prob in zip(
                sim_data.process.transcription.attenuated_rna_indices,
                sim_data.process.transcription.attenuation_readthrough[
                    sim_data.condition
                ],
            )
        }
        readthrough_adjustment = np.array(
            [attenuation_readthrough.get(idx, 1) for idx in TU_index]
        )
        promoter_init_probs *= readthrough_adjustment

    scale_the_rest_by = (
        1.0 - promoter_init_probs[is_fixed].sum()
    ) / promoter_init_probs[~is_fixed].sum()
    promoter_init_probs[~is_fixed] *= scale_the_rest_by

    # normalize to length of rna
    init_prob_length_adjusted = promoter_init_probs * rna_lengths[TU_index]
    init_prob_normalized = init_prob_length_adjusted / init_prob_length_adjusted.sum()

    # Sample a multinomial distribution of synthesis probabilities to determine
    # what RNA are initialized
    n_initiations = random_state.multinomial(n_RNAPs_to_activate, init_prob_normalized)

    # Build array of transcription unit indexes for partially transcribed mRNAs
    # and domain indexes for RNAPs
    TU_index_partial_RNAs = np.repeat(TU_index, n_initiations)
    domain_index_rnap = np.repeat(domain_index_promoters, n_initiations)

    # Build arrays of starting coordinates and transcription directions
    starting_coordinates = replication_coordinate[TU_index_partial_RNAs]
    is_forward = transcription_direction[TU_index_partial_RNAs]

    # Randomly advance RNAPs along the transcription units
    # TODO (Eran): make sure there aren't any RNAPs at same location on same TU
    updated_lengths = np.array(
        random_state.rand(n_RNAPs_to_activate) * rna_lengths[TU_index_partial_RNAs],
        dtype=int,
    )

    # Rescale boolean array of directions to an array of 1's and -1's.
    direction_rescaled = (2 * (is_forward - 0.5)).astype(np.int64)

    # Compute the updated coordinates of RNAPs. Coordinates of RNAPs moving in
    # the positive direction are increased, whereas coordinates of RNAPs moving
    # in the negative direction are decreased.
    updated_coordinates = starting_coordinates + np.multiply(
        direction_rescaled, updated_lengths
    )

    # Reset coordinates of RNAPs that cross the boundaries between right and
    # left replichores
    updated_coordinates[updated_coordinates > replichore_lengths[0]] -= (
        chromosome_length
    )
    updated_coordinates[updated_coordinates < -replichore_lengths[1]] += (
        chromosome_length
    )

    # Update mass
    sequences = rna_sequences[TU_index_partial_RNAs]
    added_mass = computeMassIncrease(sequences, updated_lengths, nt_weights)
    added_mass[updated_lengths != 0] += end_weight  # add endWeight to all new Rna

    # Masses of partial mRNAs are counted as mRNA mass as they are already
    # functional, but the masses of other types of partial RNAs are counted as
    # generic RNA mass.
    added_RNA_mass = added_mass.copy()
    added_mRNA_mass = added_mass.copy()

    is_mRNA_partial_RNAs = np.isin(TU_index_partial_RNAs, idx_mRNA)
    added_RNA_mass[is_mRNA_partial_RNAs] = 0
    added_mRNA_mass[np.logical_not(is_mRNA_partial_RNAs)] = 0

    # Add active RNAPs and get their unique indexes
    unique_molecules["active_RNAP"] = create_new_unique_molecules(
        "active_RNAP",
        n_RNAPs_to_activate,
        sim_data,
        unique_id_rng,
        domain_index=domain_index_rnap,
        coordinates=updated_coordinates,
        is_forward=is_forward,
    )

    # Decrement counts of bulk inactive RNAPs
    rnap_idx = bulk_name_to_idx(sim_data.molecule_ids.full_RNAP, bulk_state["id"])
    bulk_state["count"][rnap_idx] = inactive_RNAP_counts - n_RNAPs_to_activate

    # Add partially transcribed RNAs
    partial_rnas = create_new_unique_molecules(
        "RNA",
        n_RNAPs_to_activate,
        sim_data,
        unique_id_rng,
        TU_index=TU_index_partial_RNAs,
        transcript_length=updated_lengths,
        is_mRNA=is_mRNA_partial_RNAs,
        is_full_transcript=np.zeros(n_RNAPs_to_activate, dtype=bool),
        can_translate=is_mRNA_partial_RNAs,
        RNAP_index=unique_molecules["active_RNAP"]["unique_index"],
        massDiff_nonspecific_RNA=added_RNA_mass,
        massDiff_mRNA=added_mRNA_mass,
    )

    # Get counts of mRNAs initialized as bulk molecules
    mRNA_ids = sim_data.process.transcription.rna_data["id"][
        sim_data.process.transcription.rna_data["is_mRNA"]
    ]
    mRNA_idx = bulk_name_to_idx(mRNA_ids, bulk_state["id"])
    mRNA_counts = bulk_state["count"][mRNA_idx]

    # Subtract number of partially transcribed mRNAs that were initialized.
    # Note: some mRNAs with high degradation rates have more partial mRNAs than
    # the expected total number of mRNAs - for these mRNAs we simply set the
    # initial full mRNA counts to be zero.
    partial_mRNA_counts = np.bincount(
        TU_index_partial_RNAs[is_mRNA_partial_RNAs], minlength=n_TUs
    )[idx_mRNA]
    full_mRNA_counts = (mRNA_counts - partial_mRNA_counts).clip(min=0)

    # Get array of TU indexes for each full mRNA
    TU_index_full_mRNAs = np.repeat(idx_mRNA, full_mRNA_counts)

    # Add fully transcribed mRNAs. The RNAP_index attribute of these molecules
    # are set to -1.
    full_rnas = create_new_unique_molecules(
        "RNA",
        len(TU_index_full_mRNAs),
        sim_data,
        unique_id_rng,
        TU_index=TU_index_full_mRNAs,
        transcript_length=rna_lengths[TU_index_full_mRNAs],
        is_mRNA=np.ones_like(TU_index_full_mRNAs, dtype=bool),
        is_full_transcript=np.ones_like(TU_index_full_mRNAs, dtype=bool),
        can_translate=np.ones_like(TU_index_full_mRNAs, dtype=bool),
        RNAP_index=np.full(TU_index_full_mRNAs.shape, -1, dtype=np.int64),
        massDiff_mRNA=rna_masses[TU_index_full_mRNAs],
    )
    unique_molecules["RNA"] = np.concatenate((partial_rnas, full_rnas))
    # Have to recreate unique indices or else there will be conflicts between
    # full and partial RNAs
    unique_prefix = np.min(unique_molecules["RNA"]["unique_index"])
    unique_molecules["RNA"]["unique_index"] = np.arange(
        unique_prefix, unique_prefix + len(unique_molecules["RNA"])
    )
    unique_molecules["RNA"] = MetadataArray(
        unique_molecules["RNA"],
        unique_prefix + len(unique_molecules["RNA"]),
    )

    # Reset counts of bulk mRNAs to zero
    bulk_state["count"][mRNA_idx] = 0


def initialize_chromosomal_segments(unique_molecules, sim_data, unique_id_rng):
    """
    Initialize unique molecule representations of chromosomal segments. All
    chromosomal segments are assumed to be at their relaxed states upon
    initialization.
    """
    # Load parameters
    relaxed_DNA_base_pairs_per_turn = (
        sim_data.process.chromosome_structure.relaxed_DNA_base_pairs_per_turn
    )
    terC_index = sim_data.process.chromosome_structure.terC_dummy_molecule_index
    replichore_lengths = sim_data.process.replication.replichore_lengths
    min_coordinates = -replichore_lengths[1]
    max_coordinates = replichore_lengths[0]

    # Get attributes of replisomes, active RNAPs, chromosome domains, full
    # chromosomes, and oriCs
    (replisome_coordinates, replisome_domain_indexes, replisome_unique_indexes) = attrs(
        unique_molecules["active_replisome"],
        ["coordinates", "domain_index", "unique_index"],
    )

    (
        active_RNAP_coordinates,
        active_RNAP_domain_indexes,
        active_RNAP_unique_indexes,
    ) = attrs(
        unique_molecules["active_RNAP"], ["coordinates", "domain_index", "unique_index"]
    )

    chromosome_domain_domain_indexes, child_domains = attrs(
        unique_molecules["chromosome_domain"], ["domain_index", "child_domains"]
    )

    (full_chromosome_domain_indexes,) = attrs(
        unique_molecules["full_chromosome"], ["domain_index"]
    )

    (origin_domain_indexes,) = attrs(unique_molecules["oriC"], ["domain_index"])

    # Initialize chromosomal segment attributes
    all_boundary_molecule_indexes = np.empty((0, 2), dtype=np.int64)
    all_boundary_coordinates = np.empty((0, 2), dtype=np.int64)
    all_segment_domain_indexes = np.array([], dtype=np.int32)
    all_linking_numbers = np.array([], dtype=np.float64)

    def get_chromosomal_segment_attributes(
        coordinates, unique_indexes, spans_oriC, spans_terC
    ):
        """
        Returns the attributes of all chromosomal segments from a continuous
        stretch of DNA, given the coordinates and unique indexes of all
        boundary molecules.
        """
        coordinates_argsort = np.argsort(coordinates)
        coordinates_sorted = coordinates[coordinates_argsort]
        unique_indexes_sorted = unique_indexes[coordinates_argsort]

        # Add dummy molecule at terC if domain spans terC
        if spans_terC:
            coordinates_sorted = np.insert(
                coordinates_sorted,
                [0, len(coordinates_sorted)],
                [min_coordinates, max_coordinates],
            )
            unique_indexes_sorted = np.insert(
                unique_indexes_sorted, [0, len(unique_indexes_sorted)], terC_index
            )

        boundary_molecule_indexes = np.hstack(
            (
                unique_indexes_sorted[:-1][:, np.newaxis],
                unique_indexes_sorted[1:][:, np.newaxis],
            )
        )
        boundary_coordinates = np.hstack(
            (
                coordinates_sorted[:-1][:, np.newaxis],
                coordinates_sorted[1:][:, np.newaxis],
            )
        )

        # Remove segment that spans oriC if the domain does not span oriC
        if not spans_oriC:
            oriC_segment_index = np.where(
                np.sign(boundary_coordinates).sum(axis=1) == 0
            )[0]
            assert len(oriC_segment_index) == 1

            boundary_molecule_indexes = np.delete(
                boundary_molecule_indexes, oriC_segment_index, 0
            )
            boundary_coordinates = np.delete(
                boundary_coordinates, oriC_segment_index, 0
            )

        # Assumes all segments are at their relaxed state at initialization
        linking_numbers = (
            boundary_coordinates[:, 1] - boundary_coordinates[:, 0]
        ) / relaxed_DNA_base_pairs_per_turn

        return boundary_molecule_indexes, boundary_coordinates, linking_numbers

    # Loop through each domain index
    for domain_index in chromosome_domain_domain_indexes:
        domain_spans_oriC = domain_index in origin_domain_indexes
        domain_spans_terC = domain_index in full_chromosome_domain_indexes

        # Get coordinates and indexes of all RNAPs on this domain
        RNAP_domain_mask = active_RNAP_domain_indexes == domain_index
        molecule_coordinates_this_domain = active_RNAP_coordinates[RNAP_domain_mask]
        molecule_indexes_this_domain = active_RNAP_unique_indexes[RNAP_domain_mask]

        # Append coordinates and indexes of replisomes on this domain, if any
        if not domain_spans_oriC:
            replisome_domain_mask = replisome_domain_indexes == domain_index
            molecule_coordinates_this_domain = np.concatenate(
                (
                    molecule_coordinates_this_domain,
                    replisome_coordinates[replisome_domain_mask],
                )
            )
            molecule_indexes_this_domain = np.concatenate(
                (
                    molecule_indexes_this_domain,
                    replisome_unique_indexes[replisome_domain_mask],
                )
            )

        # Append coordinates and indexes of parent domain replisomes, if any
        if not domain_spans_terC:
            parent_domain_index = chromosome_domain_domain_indexes[
                np.where(child_domains == domain_index)[0][0]
            ]
            replisome_parent_domain_mask = (
                replisome_domain_indexes == parent_domain_index
            )
            molecule_coordinates_this_domain = np.concatenate(
                (
                    molecule_coordinates_this_domain,
                    replisome_coordinates[replisome_parent_domain_mask],
                )
            )
            molecule_indexes_this_domain = np.concatenate(
                (
                    molecule_indexes_this_domain,
                    replisome_unique_indexes[replisome_parent_domain_mask],
                )
            )

        # Get attributes of chromosomal segments on this domain
        (
            boundary_molecule_indexes_this_domain,
            boundary_coordinates_this_domain,
            linking_numbers_this_domain,
        ) = get_chromosomal_segment_attributes(
            molecule_coordinates_this_domain,
            molecule_indexes_this_domain,
            domain_spans_oriC,
            domain_spans_terC,
        )

        # Append to existing array of attributes
        all_boundary_molecule_indexes = np.vstack(
            [all_boundary_molecule_indexes, boundary_molecule_indexes_this_domain]
        )
        all_boundary_coordinates = np.vstack(
            (all_boundary_coordinates, boundary_coordinates_this_domain)
        )
        all_segment_domain_indexes = np.concatenate(
            (
                all_segment_domain_indexes,
                np.full(len(linking_numbers_this_domain), domain_index, dtype=np.int32),
            )
        )
        all_linking_numbers = np.concatenate(
            (all_linking_numbers, linking_numbers_this_domain)
        )

    # Confirm total counts of all segments
    n_segments = len(all_linking_numbers)
    assert (
        n_segments
        == len(active_RNAP_unique_indexes) + 1.5 * len(replisome_unique_indexes) + 1
    )

    # Add chromosomal segments
    unique_molecules["chromosomal_segment"] = create_new_unique_molecules(
        "chromosomal_segment",
        n_segments,
        sim_data,
        unique_id_rng,
        boundary_molecule_indexes=all_boundary_molecule_indexes,
        boundary_coordinates=all_boundary_coordinates,
        domain_index=all_segment_domain_indexes,
        linking_number=all_linking_numbers,
    )


def initialize_translation(
    bulk_state, unique_molecules, sim_data, random_state, unique_id_rng
):
    """
    Activate ribosomes as unique molecules, and distribute them along lengths
    of mRNAs, while decreasing counts of unactivated ribosomal subunits (30S
    and 50S).

    Ribosomes are placed randomly across the lengths of each mRNA.
    """
    # Load translation parameters
    current_nutrients = sim_data.conditions[sim_data.condition]["nutrients"]
    frac_active_ribosome = sim_data.process.translation.ribosomeFractionActiveDict[
        current_nutrients
    ]
    protein_sequences = sim_data.process.translation.translation_sequences
    protein_lengths = sim_data.process.translation.monomer_data["length"].asNumber()
    translation_efficiencies = normalize(
        sim_data.process.translation.translation_efficiencies_by_monomer
    )
    aa_weights_incorporated = sim_data.process.translation.translation_monomer_weights
    end_weight = sim_data.process.translation.translation_end_weight
    cistron_lengths = sim_data.process.transcription.cistron_data["length"].asNumber(
        units.nt
    )
    TU_ids = sim_data.process.transcription.rna_data["id"]
    monomer_index_to_tu_indexes = sim_data.relation.monomer_index_to_tu_indexes
    monomer_index_to_cistron_index = {
        i: sim_data.process.transcription._cistron_id_to_index[monomer["cistron_id"]]
        for (i, monomer) in enumerate(sim_data.process.translation.monomer_data)
    }

    # Get attributes of RNAs
    (
        TU_index_all_RNAs,
        length_all_RNAs,
        is_mRNA,
        is_full_transcript_all_RNAs,
        unique_index_all_RNAs,
    ) = attrs(
        unique_molecules["RNA"],
        [
            "TU_index",
            "transcript_length",
            "is_mRNA",
            "is_full_transcript",
            "unique_index",
        ],
    )
    TU_index_mRNAs = TU_index_all_RNAs[is_mRNA]
    length_mRNAs = length_all_RNAs[is_mRNA]
    is_full_transcript_mRNAs = is_full_transcript_all_RNAs[is_mRNA]
    unique_index_mRNAs = unique_index_all_RNAs[is_mRNA]

    # Calculate available template lengths of each mRNA cistron from fully
    # transcribed mRNA transcription units
    TU_index_full_mRNAs = TU_index_mRNAs[is_full_transcript_mRNAs]
    TU_counts_full_mRNAs = np.bincount(TU_index_full_mRNAs, minlength=len(TU_ids))
    cistron_counts_full_mRNAs = (
        sim_data.process.transcription.cistron_tu_mapping_matrix.dot(
            TU_counts_full_mRNAs
        )
    )
    available_cistron_lengths = np.multiply(cistron_counts_full_mRNAs, cistron_lengths)

    # Add available template lengths from each partially transcribed mRNAs
    TU_index_incomplete_mRNAs = TU_index_mRNAs[np.logical_not(is_full_transcript_mRNAs)]
    length_incomplete_mRNAs = length_mRNAs[np.logical_not(is_full_transcript_mRNAs)]

    TU_index_to_mRNA_lengths = {}
    for TU_index, length in zip(TU_index_incomplete_mRNAs, length_incomplete_mRNAs):
        TU_index_to_mRNA_lengths.setdefault(TU_index, []).append(length)

    for TU_index, available_lengths in TU_index_to_mRNA_lengths.items():
        cistron_indexes = sim_data.process.transcription.rna_id_to_cistron_indexes(
            TU_ids[TU_index]
        )
        cistron_start_positions = np.array(
            [
                sim_data.process.transcription.cistron_start_end_pos_in_tu[
                    (cistron_index, TU_index)
                ][0]
                for cistron_index in cistron_indexes
            ]
        )

        for length in available_lengths:
            available_cistron_lengths[cistron_indexes] += np.clip(
                length - cistron_start_positions, 0, cistron_lengths[cistron_indexes]
            )

    # Find number of ribosomes to activate
    ribosome30S_idx = bulk_name_to_idx(
        sim_data.molecule_ids.s30_full_complex, bulk_state["id"]
    )
    ribosome30S = bulk_state["count"][ribosome30S_idx]
    ribosome50S_idx = bulk_name_to_idx(
        sim_data.molecule_ids.s50_full_complex, bulk_state["id"]
    )
    ribosome50S = bulk_state["count"][ribosome50S_idx]
    inactive_ribosome_count = np.minimum(ribosome30S, ribosome50S)
    n_ribosomes_to_activate = np.int64(frac_active_ribosome * inactive_ribosome_count)

    # Add total available template lengths as weights and normalize
    protein_init_probs = normalize(
        available_cistron_lengths[sim_data.relation.cistron_to_monomer_mapping]
        * translation_efficiencies
    )

    # Sample a multinomial distribution of synthesis probabilities to determine
    # which types of mRNAs are initialized
    n_new_proteins = random_state.multinomial(
        n_ribosomes_to_activate, protein_init_probs
    )

    # Build attributes for active ribosomes
    protein_indexes = np.empty(n_ribosomes_to_activate, np.int64)
    cistron_start_positions_on_mRNA = np.empty(n_ribosomes_to_activate, np.int64)
    positions_on_mRNA_from_cistron_start_site = np.empty(
        n_ribosomes_to_activate, np.int64
    )
    mRNA_indexes = np.empty(n_ribosomes_to_activate, np.int64)
    start_index = 0
    nonzero_count = n_new_proteins > 0

    for protein_index, protein_counts in zip(
        np.arange(n_new_proteins.size)[nonzero_count], n_new_proteins[nonzero_count]
    ):
        # Set protein index
        protein_indexes[start_index : start_index + protein_counts] = protein_index

        # Get index of cistron corresponding to this protein
        cistron_index = monomer_index_to_cistron_index[protein_index]

        # Initialize list of available lengths for each transcript and the
        # indexes of each transcript in the list of mRNA attributes
        available_lengths = []
        attribute_indexes = []
        cistron_start_positions = []

        # Distribute ribosomes among mRNAs that produce this protein, weighted
        # by their lengths
        for TU_index in monomer_index_to_tu_indexes[protein_index]:
            attribute_indexes_this_TU = np.where(TU_index_mRNAs == TU_index)[0]
            cistron_start_position = (
                sim_data.process.transcription.cistron_start_end_pos_in_tu[
                    (cistron_index, TU_index)
                ][0]
            )
            available_lengths.extend(
                np.clip(
                    length_mRNAs[attribute_indexes_this_TU] - cistron_start_position,
                    0,
                    cistron_lengths[cistron_index],
                )
            )
            attribute_indexes.extend(attribute_indexes_this_TU)
            cistron_start_positions.extend(
                [cistron_start_position] * len(attribute_indexes_this_TU)
            )

        available_lengths = np.array(available_lengths)
        attribute_indexes = np.array(attribute_indexes)
        cistron_start_positions = np.array(cistron_start_positions)

        n_ribosomes_per_RNA = random_state.multinomial(
            protein_counts, normalize(available_lengths)
        )

        # Get unique indexes of each mRNA
        mRNA_indexes[start_index : start_index + protein_counts] = np.repeat(
            unique_index_mRNAs[attribute_indexes], n_ribosomes_per_RNA
        )

        # Get full length of this polypeptide
        peptide_full_length = protein_lengths[protein_index]

        # Randomly place ribosomes along the length of each mRNA, capped by the
        # mRNA length expected from the full polypeptide length to prevent
        # ribosomes from overshooting full peptide lengths
        cistron_start_positions_on_mRNA[start_index : start_index + protein_counts] = (
            np.repeat(cistron_start_positions, n_ribosomes_per_RNA)
        )
        positions_on_mRNA_from_cistron_start_site[
            start_index : start_index + protein_counts
        ] = np.floor(
            random_state.rand(protein_counts)
            * np.repeat(
                np.minimum(available_lengths, peptide_full_length * 3),
                n_ribosomes_per_RNA,
            )
        )

        start_index += protein_counts

    # Calculate the lengths of the partial polypeptide, and rescale position on
    # mRNA to be a multiple of three using this peptide length
    peptide_lengths = np.floor_divide(positions_on_mRNA_from_cistron_start_site, 3)
    positions_on_mRNA = cistron_start_positions_on_mRNA + 3 * peptide_lengths

    # Update masses of partially translated proteins
    sequences = protein_sequences[protein_indexes]
    mass_increase_protein = computeMassIncrease(
        sequences, peptide_lengths, aa_weights_incorporated
    )

    # Add end weight
    mass_increase_protein[peptide_lengths != 0] += end_weight

    # Add active ribosomes
    unique_molecules["active_ribosome"] = create_new_unique_molecules(
        "active_ribosome",
        n_ribosomes_to_activate,
        sim_data,
        unique_id_rng,
        protein_index=protein_indexes,
        peptide_length=peptide_lengths,
        mRNA_index=mRNA_indexes,
        pos_on_mRNA=positions_on_mRNA,
        massDiff_protein=mass_increase_protein,
    )

    # Decrease counts of free 30S and 50S ribosomal subunits
    bulk_state["count"][ribosome30S_idx] = ribosome30S - n_ribosomes_to_activate
    bulk_state["count"][ribosome50S_idx] = ribosome50S - n_ribosomes_to_activate


def determine_chromosome_state(
    tau: Unum,
    replichore_length: Unum,
    n_max_replisomes: int,
    place_holder: int,
    cell_mass: Unum,
    critical_mass: Unum,
    replication_rate: float,
) -> tuple[
    dict[str, npt.NDArray[np.int32]],
    dict[str, npt.NDArray[Any]],
    dict[str, npt.NDArray[np.int32]],
]:
    """
    Calculates the attributes of oriC's, replisomes, and chromosome domains on
    the chromosomes at the beginning of the cell cycle.

    Args:
        tau: the doubling time of the cell (with Unum time unit)
        replichore_length: the amount of DNA to be replicated per fork, usually
            half of the genome, in base-pairs (with Unum nucleotide unit)
        n_max_replisomes: the maximum number of replisomes that can be formed
            given the initial counts of replisome subunits
        place_holder: placeholder value for chromosome domains without child
            domains
        cell_mass: total mass of the cell with mass units (with Unum mass unit)
        critical_mass: mass per oriC before replication is initiated
            (with Unum mass unit)
        replication_rate: rate of nucleotide elongation
            (with Unum nucleotides per time unit)

    Returns:
        Three dictionaries, each containing updates to attributes of a unique molecule type.

        - ``oric_state``: dictionary of the following format::

            {'domain_index': a vector of integers indicating which chromosome domain the
                oriC sequence belongs to.}

        - ``replisome_state``: dictionary of the following format::

            {'coordinates': a vector of integers that indicates where the replisomes
                are located on the chromosome relative to the origin in base pairs,
            'right_replichore': a vector of boolean values that indicates whether the
                replisome is on the right replichore (True) or the left replichore (False),
            'domain_index': a vector of integers indicating which chromosome domain the
                replisomes belong to. The index of the "mother" domain of the replication
                fork is assigned to the replisome}

        - ``domain_state``: dictionary of the following format::

            {'domain_index': the indexes of the domains,
            'child_domains': the (n_domain X 2) array of the domain indexes of the two
                children domains that are connected on the oriC side with the given domain.}

    """

    # All inputs must be positive numbers
    unitless_tau = tau.asNumber(units.s)
    unitless_replichore_length = replichore_length.asNumber(units.nt)
    assert unitless_tau >= 0, "tau value can't be negative."
    assert unitless_replichore_length > 0, "replichore_length must be positive."

    # Convert to unitless
    unitless_cell_mass = cell_mass.asNumber(units.fg)
    unitless_critical_mass = critical_mass.asNumber(units.fg)

    # Calculate the maximum number of replication rounds given the maximum
    # count of replisomes
    n_max_rounds = int(np.log2(n_max_replisomes / 2 + 1))

    # Calculate the number of active replication rounds
    n_rounds = min(
        n_max_rounds,
        max(0, int(np.ceil(np.log2(unitless_cell_mass / unitless_critical_mass)))),
    )

    # Initialize arrays for replisomes
    n_replisomes = 2 * (2**n_rounds - 1)
    coordinates = np.zeros(n_replisomes, dtype=np.int64)
    right_replichore_replisome = np.zeros(n_replisomes, dtype=bool)
    domain_index_replisome = np.zeros(n_replisomes, dtype=np.int32)

    # Initialize child domain array for chromosome domains
    n_domains = 2 ** (n_rounds + 1) - 1
    child_domains = np.full((n_domains, 2), place_holder, dtype=np.int32)

    # Set domain_index attribute of oriC's and chromosome domains
    domain_index_oric = np.arange(
        2**n_rounds - 1, 2 ** (n_rounds + 1) - 1, dtype=np.int32
    )
    domain_index_domains = np.arange(0, n_domains, dtype=np.int32)

    def n_events_before_this_round(round_idx):
        """
        Calculates the number of replication events that happen before the
        replication round index given as an argument. Since 2**i events happen
        at each round i = 0, 1, ..., the sum of the number of events before
        round j is 2**j - 1.
        """
        return 2**round_idx - 1

    # Loop through active replication rounds, starting from the oldest round.
    # If n_round = 0 skip loop entirely - no active replication round.
    for round_idx in np.arange(n_rounds):
        # Determine at which location (base) of the chromosome the replication
        # forks should be initialized to
        round_critical_mass = 2**round_idx * unitless_critical_mass
        growth_rate = np.log(2) / unitless_tau
        replication_time = (
            np.log(unitless_cell_mass / round_critical_mass) / growth_rate
        )
        # TODO: this should handle completed replication (instead of taking min)
        # for accuracy but will likely never start with multiple chromosomes
        fork_location = min(
            np.floor(replication_time * replication_rate),
            unitless_replichore_length - 1,
        )

        # Add 2^n initiation events per round. A single initiation event
        # generates two replication forks.
        n_events_this_round = int(2**round_idx)

        # Set attributes of replisomes for this replication round
        coordinates[
            2 * n_events_before_this_round(round_idx) : 2
            * n_events_before_this_round(round_idx + 1)
        ] = np.tile(np.array([fork_location, -fork_location]), n_events_this_round)

        right_replichore_replisome[
            2 * n_events_before_this_round(round_idx) : 2
            * n_events_before_this_round(round_idx + 1)
        ] = np.tile(np.array([True, False]), n_events_this_round)

        for i, domain_index in enumerate(
            np.arange(
                n_events_before_this_round(round_idx),
                n_events_before_this_round(round_idx + 1),
            )
        ):
            domain_index_replisome[
                2 * n_events_before_this_round(round_idx) + 2 * i : 2
                * n_events_before_this_round(round_idx)
                + 2 * (i + 1)
            ] = np.repeat(domain_index, 2)

        # Set attributes of chromosome domains for this replication round
        for i, domain_index in enumerate(
            np.arange(
                n_events_before_this_round(round_idx + 1),
                n_events_before_this_round(round_idx + 2),
                2,
            )
        ):
            child_domains[n_events_before_this_round(round_idx) + i, :] = np.array(
                [domain_index, domain_index + 1]
            )

    # Convert to numpy arrays and wrap into dictionaries
    oric_state: dict[str, npt.NDArray[np.int32]] = {"domain_index": domain_index_oric}

    replisome_state = {
        "coordinates": coordinates,
        "right_replichore": right_replichore_replisome,
        "domain_index": domain_index_replisome,
    }

    domain_state = {
        "child_domains": child_domains,
        "domain_index": domain_index_domains,
    }

    return oric_state, replisome_state, domain_state


def rescale_initiation_probs(init_probs, TU_index, fixed_synth_probs, fixed_TU_indexes):
    """
    Rescales the initiation probabilities of each promoter such that the total
    synthesis probabilities of certain types of RNAs are fixed to a
    predetermined value. For instance, if there are two copies of promoters for
    RNA A, whose synthesis probability should be fixed to 0.1, each promoter is
    given an initiation probability of 0.05.
    """
    for rna_idx, synth_prob in zip(fixed_TU_indexes, fixed_synth_probs):
        fixed_rna_mask = TU_index == rna_idx
        init_probs[fixed_rna_mask] = synth_prob / fixed_rna_mask.sum()


def calculate_cell_mass(bulk_state, unique_molecules, sim_data):
    """
    Calculates cell mass in femtograms.
    """
    bulk_submass_names = [
        f"{submass}_submass" for submass in sim_data.submass_name_to_index.keys()
    ]
    cell_mass = (
        bulk_state["count"]
        .dot(rfn.structured_to_unstructured(bulk_state[bulk_submass_names]))
        .sum()
    )

    if len(unique_molecules) > 0:
        unique_masses = sim_data.internal_state.unique_molecule.unique_molecule_masses[
            "mass"
        ].asNumber(units.fg / units.mol) / sim_data.constants.n_avogadro.asNumber(
            1 / units.mol
        )
        unique_ids = sim_data.internal_state.unique_molecule.unique_molecule_masses[
            "id"
        ]
        unique_submass_names = [
            f"massDiff_{submass}" for submass in sim_data.submass_name_to_index.keys()
        ]
        for unique_id, unique_submasses in zip(unique_ids, unique_masses):
            if unique_id in unique_molecules:
                cell_mass += (
                    unique_molecules[unique_id]["_entryState"].sum() * unique_submasses
                ).sum()
                cell_mass += rfn.structured_to_unstructured(
                    unique_molecules[unique_id][unique_submass_names]
                ).sum()

    return units.fg * cell_mass


def initialize_trna_charging(
    bulk_state: np.ndarray,
    unique_molecules: dict[str, np.ndarray],
    sim_data: Any,
    variable_elongation: bool,
):
    """
    Initializes charged tRNA from uncharged tRNA and amino acids

    Args:
        bulk_state: Structured array with IDs and counts of all bulk molecules
        unique_molecules: Mapping of unique molecule names to structured
            arrays of their current simulation states
        sim_data: Simulation data loaded from pickle generated by ParCa
        variable_elongation: Sets max elongation higher if True

    .. note::
        Does not adjust for mass of amino acids on charged tRNA (~0.01% of cell mass)
    """
    # Calculate cell volume for concentrations
    cell_volume = (
        calculate_cell_mass(bulk_state, unique_molecules, sim_data)
        / sim_data.constants.cell_density
    )
    counts_to_molar = 1 / (sim_data.constants.n_avogadro * cell_volume)

    # Get molecule views and concentrations
    transcription = sim_data.process.transcription
    aa_from_synthetase = transcription.aa_from_synthetase
    aa_from_trna = transcription.aa_from_trna
    synthetases = counts(
        bulk_state, bulk_name_to_idx(transcription.synthetase_names, bulk_state["id"])
    )
    uncharged_trna_idx = bulk_name_to_idx(
        transcription.uncharged_trna_names, bulk_state["id"]
    )
    uncharged_trna = counts(bulk_state, uncharged_trna_idx)
    charged_trna_idx = bulk_name_to_idx(
        transcription.charged_trna_names, bulk_state["id"]
    )
    charged_trna = counts(bulk_state, charged_trna_idx)
    aas = counts(
        bulk_state,
        bulk_name_to_idx(sim_data.molecule_groups.amino_acids, bulk_state["id"]),
    )

    ribosome_counts = unique_molecules["active_ribosome"]["_entryState"].sum()

    synthetase_conc = counts_to_molar * np.dot(aa_from_synthetase, synthetases)
    uncharged_trna_conc = counts_to_molar * np.dot(aa_from_trna, uncharged_trna)
    charged_trna_conc = counts_to_molar * np.dot(aa_from_trna, charged_trna)
    aa_conc = counts_to_molar * aas
    ribosome_conc = counts_to_molar * ribosome_counts

    # Estimate fraction of amino acids from sequences, excluding first index for padding of -1
    _, aas_in_sequences = np.unique(
        sim_data.process.translation.translation_sequences, return_counts=True
    )
    f = aas_in_sequences[1:] / np.sum(aas_in_sequences[1:])

    # Estimate initial charging state
    constants = sim_data.constants
    transcription = sim_data.process.transcription
    metabolism = sim_data.process.metabolism
    elongation_max = (
        constants.ribosome_elongation_rate_max
        if variable_elongation
        else constants.ribosome_elongation_rate_basal
    )
    charging_params = {
        "kS": constants.synthetase_charging_rate.asNumber(1 / units.s),
        "KMaa": transcription.aa_kms.asNumber(MICROMOLAR_UNITS),
        "KMtf": transcription.trna_kms.asNumber(MICROMOLAR_UNITS),
        "krta": constants.Kdissociation_charged_trna_ribosome.asNumber(
            MICROMOLAR_UNITS
        ),
        "krtf": constants.Kdissociation_uncharged_trna_ribosome.asNumber(
            MICROMOLAR_UNITS
        ),
        "max_elong_rate": float(elongation_max.asNumber(units.aa / units.s)),
        "charging_mask": np.array(
            [
                aa not in REMOVED_FROM_CHARGING
                for aa in sim_data.molecule_groups.amino_acids
            ]
        ),
        "unit_conversion": metabolism.get_amino_acid_conc_conversion(MICROMOLAR_UNITS),
    }
    fraction_charged, *_ = calculate_trna_charging(
        synthetase_conc,
        uncharged_trna_conc,
        charged_trna_conc,
        aa_conc,
        ribosome_conc,
        f,
        charging_params,
    )

    # Update counts of tRNA to match charging
    total_trna_counts = uncharged_trna + charged_trna
    charged_trna_counts = np.round(
        total_trna_counts * np.dot(fraction_charged, aa_from_trna)
    )
    uncharged_trna_counts = total_trna_counts - charged_trna_counts
    bulk_state["count"][charged_trna_idx] = charged_trna_counts
    bulk_state["count"][uncharged_trna_idx] = uncharged_trna_counts
